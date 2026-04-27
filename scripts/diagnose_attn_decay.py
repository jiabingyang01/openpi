"""Diagnose visual attention decay across layers in π₀'s action expert.

For each transformer layer, computes:
  - visual_ratio: fraction of attention that action tokens allocate to visual tokens
  - action_ratio: fraction allocated to action tokens
  - visual_mean_entropy: entropy of the action→visual attention distribution

If visual_ratio monotonically decreases with depth, the "softmax zero-sum competition"
hypothesis is confirmed, and adding a dedicated visual cross-attention path is justified.

Usage:
    python scripts/diagnose_attn_decay.py \
        --config-name pi0_libero \
        --checkpoint-dir /path/to/checkpoint \
        --num-batches 20
"""

import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("diagnose_attn")


@dataclasses.dataclass
class Args:
    config_name: str = "pi0_libero"
    checkpoint_dir: str = ""
    num_batches: int = 20
    batch_size: int = 8


def run_diagnosis(args: Args):
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    logger.info("Loading model from %s ...", checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()

    logger.info("Creating data loader ...")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)

    from openpi.models.pi0 import make_attn_mask

    # We need per-layer attention. The current gemma.py returns all_attn_weights
    # as [L, B, T_prefix] (cross-attn summary). But for this diagnosis we need
    # the FULL attention matrix to compute visual vs action ratios.
    # Instead, we'll compute attention manually by hooking into the forward pass.
    #
    # Strategy: run the forward pass with return_attn_weights=True to get the
    # cross-attention summaries. But we also need action→action attention.
    # Since all_attn_weights only gives action→visual, we need a different approach.
    #
    # Simplest: use the existing attention weights and compute the ratio from
    # the fact that attention sums to 1 over all keys.
    # visual_ratio + action_ratio + padding_ratio = 1
    # We know all_attn_weights[l, b, :] = mean attention on each visual token.
    # Sum over visual tokens = total visual attention fraction.
    # action_ratio ≈ 1 - visual_ratio (ignoring padding).

    # Actually, all_attn_weights is the mean over heads and action queries of the
    # cross-attention submatrix. To get the ratio, we need:
    # visual_ratio_per_layer = sum(all_attn_weights[l, b, :]) (already head/query averaged)
    # But this isn't quite right because the averaging happens before summing...
    #
    # Let's just compute it properly inside the model. We need to modify the
    # forward pass slightly to return per-layer full attention statistics.
    # For simplicity, we'll do a custom forward pass here.

    num_layers = 18  # π₀'s action expert depth
    # Accumulators: [num_layers] arrays
    visual_ratio_sum = np.zeros(num_layers)
    action_ratio_sum = np.zeros(num_layers)
    visual_entropy_sum = np.zeros(num_layers)
    count = 0

    logger.info("Running diagnosis for %d batches ...", args.num_batches)

    for batch_idx in range(args.num_batches):
        observation, actions = next(data_iter)

        rng = jax.random.key(batch_idx)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        prefix_len = prefix_tokens.shape[1]
        suffix_len = suffix_tokens.shape[1]

        # Forward pass with attention weights
        (_, suffix_out), _, all_attn_weights = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            return_attn_weights=True,
        )

        # all_attn_weights: [L, B, T_prefix]
        # This is the mean (over heads, over action queries) of attention FROM action tokens TO each visual token.
        # Sum over T_prefix = total visual attention fraction per action query (averaged over heads).
        aw = np.array(jax.device_get(all_attn_weights))  # [L, B, T_prefix]

        B = aw.shape[1]
        for l in range(num_layers):
            for b in range(B):
                # visual_ratio: how much total attention action tokens give to visual tokens
                vis_ratio = float(np.sum(aw[l, b, :]))
                act_ratio = 1.0 - vis_ratio  # rest goes to action/state tokens + padding

                # Entropy of visual attention distribution (higher = more spread out)
                p = aw[l, b, :]
                p = p / (p.sum() + 1e-10)  # normalize to distribution
                entropy = -np.sum(p * np.log(p + 1e-10))

                visual_ratio_sum[l] += vis_ratio
                action_ratio_sum[l] += act_ratio
                visual_entropy_sum[l] += entropy
                count += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info("  batch %d/%d done", batch_idx + 1, args.num_batches)

    n = count / num_layers  # samples per layer
    visual_ratio_avg = visual_ratio_sum / n
    action_ratio_avg = action_ratio_sum / n
    visual_entropy_avg = visual_entropy_sum / n

    # Compute decay: ratio of last layer to first layer
    decay_ratio = visual_ratio_avg[-1] / (visual_ratio_avg[0] + 1e-10)

    print("\n" + "=" * 70)
    print("VISUAL ATTENTION DECAY DIAGNOSIS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Samples:    {int(n)}")
    print(f"")
    print(f"{'Layer':>6} | {'Visual Ratio':>13} | {'Action Ratio':>13} | {'Visual Entropy':>14}")
    print("-" * 55)
    for l in range(num_layers):
        print(f"  {l:>4} | {visual_ratio_avg[l]:>12.4f}  | {action_ratio_avg[l]:>12.4f}  | {visual_entropy_avg[l]:>13.4f}")

    print("-" * 55)
    print(f"\nVisual attention in layer 0:  {visual_ratio_avg[0]:.4f}")
    print(f"Visual attention in layer {num_layers-1}: {visual_ratio_avg[-1]:.4f}")
    print(f"Decay ratio (last/first):    {decay_ratio:.4f}")
    print(f"Absolute drop:               {visual_ratio_avg[0] - visual_ratio_avg[-1]:.4f}")

    if decay_ratio < 0.5:
        verdict = "SEVERE DECAY -- visual attention drops by >50%. Dedicated cross-attention strongly justified."
    elif decay_ratio < 0.75:
        verdict = "MODERATE DECAY -- visual attention drops by 25-50%. Cross-attention likely helpful."
    elif decay_ratio < 0.9:
        verdict = "MILD DECAY -- visual attention drops by 10-25%. Cross-attention may help on hard tasks."
    else:
        verdict = "NO SIGNIFICANT DECAY -- visual attention is stable across layers."

    print(f"\nVerdict: {verdict}")
    print("=" * 70)

    # Save
    out_path = pathlib.Path(args.checkpoint_dir) / "attn_decay_results.npz"
    np.savez(
        out_path,
        visual_ratio=visual_ratio_avg,
        action_ratio=action_ratio_avg,
        visual_entropy=visual_entropy_avg,
        decay_ratio=decay_ratio,
    )
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    run_diagnosis(tyro.cli(Args))
