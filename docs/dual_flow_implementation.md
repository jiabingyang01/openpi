# Dual Flow Matching — Implementation Document

## Overview

非侵入式实现，所有新逻辑通过 `DualFlowConfig.enabled` 开关控制。关闭时模型行为与 baseline 完全一致。

## Files Changed

### New Files
| File | Purpose |
|------|---------|
| `src/openpi/models_pytorch/dual_flow.py` | DualFlowConfig, DualFlowMaskSeqTransform, lambda warmup |

### Modified Files (Minimal Changes)
| File | Change |
|------|--------|
| `src/openpi/models/pi0_config.py` | +1 field: `dual_flow: DualFlowConfig` |
| `src/openpi/models/model.py` | +1 field in Observation: `dual_flow_mask_seq` |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `__init__`: +mask in/out proj; `embed_suffix`: +mask tokens; `forward`: +mask flow loss |
| `src/openpi/training/config.py` | +`LeRobotLiberoDualFlowDataConfig` class |
| `scripts/train_pytorch.py` | Handle `dual_flow_*` keys in loss dict |

## Data Pipeline

### Mask Sequence Loading (`DualFlowMaskSeqTransform`)

```
Input:  episode_index, frame_index, action_horizon
Output: dual_flow_mask_seq: float32[action_horizon, 16, 16]

Logic:
  1. Load episode_{idx:06d}.npz → masks_all [T, 256, 256]
  2. For each step i in [0, action_horizon):
       frame = min(frame_index + i, T - 1)
       mask = masks_all[frame]
       resize to 16x16 (INTER_AREA), binarize > 0.5
  3. Stack → [action_horizon, 16, 16]
```

### Data Config (`LeRobotLiberoDualFlowDataConfig`)

继承 `LeRobotLiberoDataConfig`，插入 `DualFlowMaskSeqTransform`（与 IGCA 的 pattern 相同）。

## Model Architecture Changes

### `__init__` additions (pi0_pytorch.py)

```python
# Dual Flow: mask trajectory projection layers
if self.dual_flow_config.enabled:
    mask_input_dim = 16 * 16  # 256, flattened patch grid
    self.mask_in_proj = nn.Linear(mask_input_dim, action_expert_config.width)
    self.mask_out_proj = nn.Linear(action_expert_config.width, mask_input_dim)
    if not self.pi05:
        self.mask_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.mask_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
```

### `embed_suffix` changes

原始 suffix: `[state(1), action_tokens(H)]`
新 suffix:   `[state(1), action_tokens(H), mask_tokens(H)]`

```python
# After action token embedding...
if self.dual_flow_config.enabled and noisy_masks is not None:
    mask_emb = self.mask_in_proj(noisy_masks)  # [B, H, 256] → [B, H, width]
    # Fuse with same timestep
    time_emb_m = time_emb[:, None, :].expand_as(mask_emb)
    mask_time_emb = torch.cat([mask_emb, time_emb_m], dim=2)
    mask_time_emb = silu(mask_time_mlp_in(mask_time_emb))
    mask_time_emb = mask_time_mlp_out(mask_time_emb)
    
    embs.append(mask_time_emb)
    pad_masks.append(ones [B, H])
    att_masks += [0] * H  # bidirectional with rest of suffix
```

### `forward` changes

```python
# --- Mask flow matching (parallel to action flow matching) ---
if dual_flow_active:
    mask_seq = observation.dual_flow_mask_seq       # [B, H, 16, 16]
    mask_flat = mask_seq.reshape(B, H, 256)         # [B, H, 256]
    
    noise_m = sample_noise(mask_flat.shape)          # [B, H, 256]
    m_t = t * noise_m + (1-t) * mask_flat            # noisy mask trajectory
    u_t_mask = noise_m - mask_flat                   # mask velocity target
    
    # embed_suffix now receives both x_t (noisy action) and m_t (noisy mask)
    suffix_embs, ... = self.embed_suffix(state, x_t, time, noisy_masks=m_t)
    
    # After joint forward:
    # suffix_out shape: [B, 1 + H + H, width]
    action_out = suffix_out[:, 1:1+H, :]             # action tokens
    mask_out   = suffix_out[:, 1+H:1+2*H, :]         # mask tokens
    
    v_t_action = self.action_out_proj(action_out)     # [B, H, 32]
    v_t_mask   = self.mask_out_proj(mask_out)          # [B, H, 256]
    
    action_loss = MSE(u_t, v_t_action, reduction="none")
    mask_loss   = MSE(u_t_mask, v_t_mask, reduction="none").mean()
```

### Attention Mask Pattern

```
Token layout:  [prefix..., state, action_0...action_49, mask_0...mask_49]
att_masks:     [0...0,     1,    1, 0...0(×49),         0...0(×50)      ]

Cumsum:        [0...0,     1,    2, 2...2,               2...2           ]

Result: All suffix tokens (action + mask) have cumsum ≥ 1, so they can:
  ✓ Attend to all prefix tokens (cumsum=0)
  ✓ Attend to all other suffix tokens (same cumsum=2)
  ✗ Prefix cannot attend to suffix
  
→ Action tokens ↔ Mask tokens: BIDIRECTIONAL attention ✓
```

注意：第一个 action token 和第一个 mask token 都需要 att_mask=1 来形成新的 attention boundary（让 mask tokens 和 action tokens 都看不到 state 之前的 tokens）。但实际上由于 state 已经设了 boundary (att_mask=1)，后续 tokens 只需要 `[1] + [0]*(H-1)` for actions 和 `[1] + [0]*(H-1)` for masks。

更新：re-reading 代码，现有逻辑是 `att_masks += [1] + ([0] * (action_horizon - 1))` for action tokens。action_0 的 att_mask=1 表示 "start a new attention group"。为了让 mask tokens 也能看到所有 action tokens（而不是被新 boundary 隔开），mask tokens 的 att_mask 应该全部是 `[0] * action_horizon`（不设新 boundary，延续 action tokens 的 attention group）。

## Inference Changes

### `sample_actions` (pi0_pytorch.py)

在推理时，同样需要联合去噪 action 和 mask：

```python
if dual_flow_active:
    # Initialize both noise trajectories
    noisy_actions = randn(B, H, action_dim)
    noisy_masks = randn(B, H, 256)
    
    for t in timesteps:
        suffix = embed_suffix(state, noisy_actions, t, noisy_masks=noisy_masks)
        ...
        v_action = action_out_proj(action_out)
        v_mask = mask_out_proj(mask_out)
        
        noisy_actions = noisy_actions + dt * v_action
        noisy_masks = noisy_masks + dt * v_mask
    
    return noisy_actions  # mask output discarded (or returned for viz)
```

## Config

```python
@dataclasses.dataclass(frozen=True)
class DualFlowConfig:
    enabled: bool = False
    lambda_mask: float = 0.1          # weight of mask flow loss
    warmup_steps: int = 0             # linear warmup from 0 → 1
    mask_dir: str = "./data/igca_masks"
    patch_grid: int = 16              # 16x16 patch grid
```

## Training Config Example

```python
"libero_dual_flow": TrainConfig(
    model=Pi0Config(
        dual_flow=DualFlowConfig(
            enabled=True,
            lambda_mask=0.1,
            warmup_steps=1000,
        ),
    ),
    data=LeRobotLiberoDualFlowDataConfig(
        dual_flow=DualFlowConfig(enabled=True),
    ),
)
```
