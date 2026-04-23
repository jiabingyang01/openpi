from cv2 import norm
import numpy as np
import os
import math
import warnings
from typing import List, Optional, Tuple, Union
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
import random

from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.generation.logits_process import LogitsProcessorList

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# from llava.mm_utils import get_anyres_image_grid_shape

# import llava.model.llava_arch

class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # UAOR
        self.apply_uaor = False
        self.vision_retracing_layer = 0
        self.visual_token = None
        self.retracing_ratio = 0
        self.entropy_threshold = 1
        self.starting_layer = 0
        self.ending_layer = 0
        self.adpt_sign = 0




    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        elif self.adpt_sign == 0:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        elif self.adpt_sign == 1:
            ffn_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            adapter_out = torch.matmul(torch.matmul(x, self.adpt_w1.T), self.adpt_w2.T)
            norm_adapter_out = (torch.mean(torch.abs(ffn_out)) / torch.mean(torch.abs((adapter_out)))) * adapter_out
            return (ffn_out*(1-self.retracing_ratio) + norm_adapter_out*self.retracing_ratio)
            

        return down_proj


def forward(
    self,
    input_ids: torch.LongTensor, #= None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_processor = LogitsProcessorList() ,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # UAOR
    layer = 0
    entropy_list = []
    apply_uaor = self.layers[0].mlp.apply_uaor
    visual_token = self.layers[0].mlp.visual_token
    retracing_ratio = self.layers[0].mlp.retracing_ratio
    entropy_threshold = self.layers[0].mlp.entropy_threshold
    starting_layer = self.layers[0].mlp.starting_layer
    ending_layer = self.layers[0].mlp.ending_layer
    visual_retracing_event = False # to prevent multiple retracing event
    vision_retracing_sign  = False # to decide whether to add visual token in the next layer


    for decoder_layer in self.layers:
        # print("/n calculating hidden states at layer: ", layer)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # print("\n calculating logits at layer: ", layer)

        norm_hidden_states = self.norm(hidden_states)
        logits = self.lm_head(norm_hidden_states)
        # print(logits.shape)
        logits = logits[:, -57:-1, :]
        logits = logits.float()
        logits = logits_processor(input_ids, logits)
        # print(logits.shape)         # torch.Size([1, 56, 32064])
        
        # n_action_bins = 256
        # pad_to_multiple_of = 64        

        # # logits: [B, 56, 32064]
        # probs = F.softmax(logits, dim=-1)  # 对 vocab 做 softmax

        # 取动作 token 区间概率并求和
        # action_probs_sum = probs[:, :, -(n_action_bins + pad_to_multiple_of):-pad_to_multiple_of].sum(dim=-1)  # [B, 56]
        # action_probs_sum = torch.topk(probs, 256, dim=-1).values.sum(dim=-1)

        # 对 56 个 token 求平均
        # mean_action_prob = action_probs_sum.mean().item()  
        # print(mean_action_prob)      

        # if layer==1:
        #     probs = F.softmax(logits, dim=-1)
        #     max_probs, max_indices = probs.max(dim=-1)  # [batch, 8]
        #     print(max_probs, max_indices)

        # # Calculate the layer entropy
        # action_logits = logits[:, :, - (n_action_bins + pad_to_multiple_of) : -pad_to_multiple_of] 
        # probabilities = F.softmax(action_logits, dim=-1)

        top_k = 256
        top_k_scores, top_k_indices = torch.topk(logits, top_k)
        probabilities = F.softmax(top_k_scores, dim=-1)

        # print(probabilities.shape)
        entropy = torch.sum(-probabilities * torch.log(probabilities), dim=-1)/np.log(top_k)
        entropy = entropy.mean().item()

        # formatted_top_k_scores = [f"{score:.3f}" for score in top_k_scores.flatten().tolist()]
        # formatted_top_k_indices = [f"{index:.3f}" for index in top_k_indices.flatten().tolist()]
        # formatted_probabilities = [f"{prob:.3f}" for prob in probabilities.flatten().tolist()]
        formatted_entropy = f"{entropy:.3f}"


        # round n+1
        # vision_retracing_sign is true, meaning that the visual token has been added. Now, clear the adaptation channel, reset the adpt_sign and vision_retracing_sign.
        if vision_retracing_sign == True:

            self.layers[layer].mlp.adpt_sign = 0
            self.layers[layer].mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(visual_token))
            self.layers[layer].mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(visual_token.T))
            # print("\n added visual token with adatption channel at layer ", layer)
                
            vision_retracing_sign = False


            
        # round n
        # calculate the entropy of the top 10 logits. if the entropy is greater than the threshold, and the visual retracing event is not happening, and the layer is within the range of starting and ending layer, then add the visual token to the next layer with adaptation channel
        # initialize the adaptation channel with the visual token
        if entropy > entropy_threshold and visual_retracing_event == False and layer >= starting_layer-1 and layer <= ending_layer-1:
            
            vision_retracing_sign = True
            visual_retracing_event = False

            self.layers[layer+1].mlp.adpt_sign = 1 # triggers the UAOR adaptation channel in MLP of the next layer
            self.layers[layer+1].mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(visual_token))
            self.layers[layer+1].mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(visual_token.T))
            self.layers[layer+1].mlp.adpt_w1 += (torch.mean(torch.abs(self.layers[layer+1].mlp.up_proj.weight)) / (torch.mean(torch.abs(visual_token))))  * visual_token
            self.layers[layer+1].mlp.adpt_w2 += (torch.mean(torch.abs(self.layers[layer+1].mlp.down_proj.weight)) / (torch.mean(torch.abs(visual_token))))  * visual_token.T

            
        # print("Extracted Top 10 largest scores:", formatted_top_k_scores)
        # print("Extracted Indices of top 10 largest scores:", formatted_top_k_indices)
        # print("Probabilities of top 10 logits:", formatted_probabilities)
        # print("Entropy of top 10 logits:", formatted_entropy, "\n")
        # entropy_list.append(formatted_entropy)
        entropy_list.append(round(entropy, 4))
        layer += 1
    # print(entropy_list)
    save_path = "/mnt/data-1/data/yangjiabing/projects/VLA/openvla-oft/experiments/entropy/10_uaor_0.95_0.05_entropy.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "a") as f:
        f.write(str(entropy_list) + "\n")



    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
    vision_tower = self.get_vision_tower()
    # rank_print(modalities)
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    # import pdb; pdb.set_trace()
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        encoded_image_features = self.encode_images(concat_images)
        # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

        # This is a list, each element is [num_images, patch * patch, dim]
        # rank_print(f"Concat images : {concat_images.shape}")
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
        # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
        # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
        # image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)
                # rank0_print("At least we are reaching here")
                # import pdb; pdb.set_trace()
                if image_idx in video_idx_in_batch:  # video operations
                    # rank0_print("Video")
                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        if getattr(self.config, "add_faster_video", False):
                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # Add a token for each frame
                            concat_slow_fater_token = []
                            # import pdb; pdb.set_trace()
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                else:
                                    concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # import pdb; pdb.set_trace()
                            image_feature = torch.cat(concat_slow_fater_token)

                            # print("!!!!!!!!!!!!")
                    
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature)

                        new_image_features.append(image_feature.flatten(0, 1))
                        
                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1)
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        new_image_features.append(image_feature)      
                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
            image_features = new_image_features
            print("image_features[0]:",image_features[0].shape)
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)
    # print("image_features:",image_features.shape)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError
    # rank_print(f"Total images : {len(image_features)}")
    # rank_print("image_features[0].shape:",image_features[0].shape)

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    # --- UAOR: 预留变量
    _robot_obs_features = None
    _ROBOT_LEN = 15
    # --- /UAOR

    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1

            # --- UAOR: 按 198（换行）定位倒数第二个换行前的 15 个 token
            try:
                idx_198 = torch.nonzero(cur_input_ids == 198, as_tuple=False).squeeze(-1)
                if idx_198.numel() >= 2:
                    start = int(idx_198[-2].item() - _ROBOT_LEN)
                    end = int(idx_198[-2].item() - 1)   # 右开
                    if end > start and start >= 0:
                        _ids = cur_input_ids[start:end].to(self.device)
                        _robot_obs_features = self.get_model().embed_tokens(_ids)  # [15, hidden]
            except Exception:
                pass
            # --- /UAOR

            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        # --- UAOR: 严格按 198 寻址，取倒数第二个换行前 15 个 token 的嵌入
        try:
            idx_198 = torch.nonzero(cur_input_ids == 198, as_tuple=False).squeeze(-1)
            if idx_198.numel() >= 2:
                start = int(idx_198[-2].item() - _ROBOT_LEN)
                end = int(idx_198[-2].item() - 1)   # 右开
                if end > start and start >= 0:
                    _ids = cur_input_ids[start:end].to(self.device)
                    _robot_obs_features = self.get_model().embed_tokens(_ids)  # [15, hidden]
        except Exception:
            pass
        # --- /UAOR

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # import pdb; pdb.set_trace()
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    # rank_print("Finishing Inserting")

    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
    # TODO: Hard code for control loss spike
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    # rank0_print("tokenizer padding")

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # import pdb; pdb.set_trace()
    # rank0_print("new_input_embeds.shape:",new_input_embeds.shape)

    # --- UAOR: 将图像特征与 15 个 robot_obs token 嵌入拼接后，传给第一层 MLP
    try:
        if _robot_obs_features is not None:
            _robot_obs_features = _robot_obs_features.to(device=cur_image_features.device, dtype=cur_image_features.dtype)
            self.model.layers[0].mlp.visual_token = torch.cat((cur_image_features, _robot_obs_features), dim=0)
        else:
            self.model.layers[0].mlp.visual_token = cur_image_features
    except NameError:
        # 极端情况下 cur_image_features 未定义（例如某些无图分支），忽略
        pass
    # --- /UAOR

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def apply_uaor_llama(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaModel.forward = forward
    # llava.model.llava_arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal

    self.model.lm_head = self.lm_head
    self.model.layers[0].mlp.apply_uaor = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(31):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio

def apply_uaor_qwen(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    self.transformer.h[0].mlp.apply_uaor = True
    self.transformer.h[0].mlp.starting_layer = starting_layer
    self.transformer.h[0].mlp.ending_layer = ending_layer
    self.transformer.h[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(31):
        self.transformer.h[layer].mlp.retracing_ratio = retracing_ratio

def apply_uaor_glm(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    self.transformer.encoder.layers[0].mlp.apply_uaor = True
    self.transformer.encoder.layers[0].mlp.starting_layer = starting_layer
    self.transformer.encoder.layers[0].mlp.ending_layer = ending_layer
    self.transformer.encoder.layers[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(31):
        self.transformer.encoder.layers[0].mlp.retracing_ratio = retracing_ratio


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # UAOR Attributes
        self.apply_uaor = False
        self.vision_retracing_layer = 0
        self.visual_token = None
        self.retracing_ratio = 0
        self.entropy_threshold = 1
        self.starting_layer = 0
        self.ending_layer = 0
        self.adpt_sign = 0
        self.adpt_w1 = None
        self.adpt_w2 = None

    def forward(self, x):
        # Standard Gemma MLP logic
        if self.adpt_sign == 0:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        # UAOR logic
        elif self.adpt_sign == 1:
            ffn_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            # Ensure adapter weights are on the correct device/dtype
            if self.adpt_w1.device != x.device:
                self.adpt_w1 = self.adpt_w1.to(x.device).to(x.dtype)
                self.adpt_w2 = self.adpt_w2.to(x.device).to(x.dtype)
                
            adapter_out = torch.matmul(torch.matmul(x, self.adpt_w1.T), self.adpt_w2.T)
            
            # Normalize adapter output to match FFN magnitude scale
            scale_factor = (torch.mean(torch.abs(ffn_out)) / (torch.mean(torch.abs(adapter_out)) + 1e-6))
            norm_adapter_out = scale_factor * adapter_out
            
            return (ffn_out * (1 - self.retracing_ratio) + norm_adapter_out * self.retracing_ratio)

def gemma_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    adarms_cond: Optional[torch.Tensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    """
    adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
        Condition for ADARMS.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    is_prefix = False
    if past_key_values is None:
        is_prefix = True

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # embed positions
    hidden_states = inputs_embeds
    # Convert to bfloat16 if the first layer uses bfloat16
    if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # normalized
    # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
    #hidden_states = hidden_states * normalizer

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # UAOR
    layer = 0
    entropy_list = []
    apply_uaor = self.layers[0].mlp.apply_uaor
    visual_token = self.layers[0].mlp.visual_token
    retracing_ratio = self.layers[0].mlp.retracing_ratio
    entropy_threshold = self.layers[0].mlp.entropy_threshold
    starting_layer = self.layers[0].mlp.starting_layer
    ending_layer = self.layers[0].mlp.ending_layer
    visual_retracing_event = False # to prevent multiple retracing event
    vision_retracing_sign  = False # to decide whether to add visual token in the next layer

    if not is_prefix:
        for layer in range(self.config.num_hidden_layers):
            self.layers[layer].mlp.adpt_sign = 0

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adarms_cond=adarms_cond,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if is_prefix: 
            # UAOR
            norm_hidden_states, _ = self.norm(hidden_states, adarms_cond)
            logits = self.lm_head(norm_hidden_states)
            # print(logits.shape)
            logits = logits[:, -1, :]
            logits = logits.float()
            # print(logits.shape)         # torch.Size([1, 56, 32064])
            
            # n_action_bins = 256
            # pad_to_multiple_of = 64        

            # # logits: [B, 56, 32064]
            # probs = F.softmax(logits, dim=-1)  # 对 vocab 做 softmax

            # 取动作 token 区间概率并求和
            # action_probs_sum = probs[:, :, -(n_action_bins + pad_to_multiple_of):-pad_to_multiple_of].sum(dim=-1)  # [B, 56]
            # action_probs_sum = torch.topk(probs, 256, dim=-1).values.sum(dim=-1)

            # 对 56 个 token 求平均
            # mean_action_prob = action_probs_sum.mean().item()  
            # print(mean_action_prob)      

            # if layer==1:
            #     probs = F.softmax(logits, dim=-1)
            #     max_probs, max_indices = probs.max(dim=-1)  # [batch, 8]
            #     print(max_probs, max_indices)

            # # Calculate the layer entropy
            # action_logits = logits[:, :, - (n_action_bins + pad_to_multiple_of) : -pad_to_multiple_of] 
            # probabilities = F.softmax(action_logits, dim=-1)

            top_k = 256
            top_k_scores, top_k_indices = torch.topk(logits, top_k)
            probabilities = F.softmax(top_k_scores, dim=-1)

            # print(probabilities.shape)
            entropy = torch.sum(-probabilities * torch.log(probabilities+1e-9), dim=-1)/np.log(top_k)
            entropy = entropy.mean().item()

            # formatted_top_k_scores = [f"{score:.3f}" for score in top_k_scores.flatten().tolist()]
            # formatted_top_k_indices = [f"{index:.3f}" for index in top_k_indices.flatten().tolist()]
            # formatted_probabilities = [f"{prob:.3f}" for prob in probabilities.flatten().tolist()]
            formatted_entropy = f"{entropy:.3f}"


            # round n+1
            # vision_retracing_sign is true, meaning that the visual token has been added. Now, clear the adaptation channel, reset the adpt_sign and vision_retracing_sign.
            if vision_retracing_sign == True:

                self.layers[layer].mlp.adpt_sign = 0
                self.layers[layer].mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(visual_token))
                self.layers[layer].mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(visual_token.T))
                # print("\n added visual token with adatption channel at layer ", layer)
                    
                vision_retracing_sign = False


                
            # round n
            # calculate the entropy of the top 10 logits. if the entropy is greater than the threshold, and the visual retracing event is not happening, and the layer is within the range of starting and ending layer, then add the visual token to the next layer with adaptation channel
            # initialize the adaptation channel with the visual token
            if entropy > entropy_threshold and visual_retracing_event == False and layer >= starting_layer-1 and layer <= ending_layer-1:
                
                vision_retracing_sign = True
                visual_retracing_event = False

                self.layers[layer+1].mlp.adpt_sign = 1 # triggers the UAOR adaptation channel in MLP of the next layer
                self.layers[layer+1].mlp.adpt_w1 = torch.nn.Parameter(torch.zeros_like(visual_token))
                self.layers[layer+1].mlp.adpt_w2 = torch.nn.Parameter(torch.zeros_like(visual_token.T))
                self.layers[layer+1].mlp.adpt_w1 += (torch.mean(torch.abs(self.layers[layer+1].mlp.up_proj.weight)) / (torch.mean(torch.abs(visual_token))))  * visual_token
                self.layers[layer+1].mlp.adpt_w2 += (torch.mean(torch.abs(self.layers[layer+1].mlp.down_proj.weight)) / (torch.mean(torch.abs(visual_token))))  * visual_token.T
                
            # print("Extracted Top 10 largest scores:", formatted_top_k_scores)
            # print("Extracted Indices of top 10 largest scores:", formatted_top_k_indices)
            # print("Probabilities of top 10 logits:", formatted_probabilities)
            # print("Entropy of top 10 logits:", formatted_entropy, "\n")
            # entropy_list.append(formatted_entropy)
            entropy_list.append(round(entropy, 4))
            layer += 1

    if entropy_list:
        print(entropy_list)
        # save_path = "./experiments/entropy/10_uaor_0.95_0.05_entropy.txt"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # with open(save_path, "a") as f:
        #     f.write(str(entropy_list) + "\n")


    hidden_states, _ = self.norm(hidden_states, adarms_cond)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def apply_uaor_pi0_pytorch(
    self,
    starting_layer: int,
    ending_layer: int,
    entropy_threshold: float,
    retracing_ratio: float
):
    """
    Applies UAOR to the PI0Pytorch instance.
    Target: pi0_model_instance.paligemma_with_expert.paligemma.language_model
    """
    # import transformers.models.gemma.modeling_gemma as gemma_modeling
    
    # 1. Monkey Patch Classes
    transformers.models.gemma.modeling_gemma.GemmaMLP = GemmaMLP
    transformers.models.gemma.modeling_gemma.GemmaModel.forward = gemma_forward
    
    # 2. Locate the Backbone Model (PaliGemma's LLM)
    # Path based on PI0Pytorch structure:
    backbone = self.paligemma_with_expert.paligemma.language_model
    lm_head = self.paligemma_with_expert.paligemma.lm_head
    
    # 3. Inject LM Head for Entropy Calculation
    backbone.lm_head = lm_head
    
    # 4. Configure UAOR params on Layer 0
    backbone.layers[0].mlp.apply_uaor = True
    backbone.layers[0].mlp.starting_layer = starting_layer
    backbone.layers[0].mlp.ending_layer = ending_layer
    backbone.layers[0].mlp.entropy_threshold = entropy_threshold
    
    for layer in backbone.layers:
        layer.mlp.retracing_ratio = retracing_ratio
        
    print(f"UAOR applied to PI0 Backbone. Layers: {len(backbone.layers)}")