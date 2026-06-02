# Dual Flow Matching: Co-Evolving Action Generation and Object Motion Prediction

## Core Insight

现有方法将 segmentation mask 视为辅助训练信号（如 IGCA 的 contrastive loss），但这种间接监督无法有效影响 action decoder 的行为。我们提出一个范式转变：**将物体运动轨迹建模为与 action 对等的生成目标**，两者在同一个 flow matching 框架内联合去噪、互相引导。

## Motivation

在机器人操作任务中，mask 序列的时序变化本身就是动作的视觉化表达：

```
帧 0:   mask 在桌面左侧     →  物体初始位置
帧 50:  mask 在空中          →  物体被抓起
帧 100: mask 在桌面右侧     →  物体被放下
```

这条 mask 轨迹 = 任务的视觉 plan。而 flow matching 需要的恰恰是从噪声到目标的速度场。如果我们把 mask 轨迹也建模成一个 flow，让 action flow 和 mask flow 在同一个去噪过程中联合生成，两者就可以互相约束、互相引导。

## Method

### 1. Dual Flow Formulation

标准 Pi0 flow matching:
```
x_t = t · ε + (1-t) · a        → 模型预测 v_action = ε - a
```

Dual Flow Matching 加入 mask flow：
```
x_t^a = t · ε_a + (1-t) · a       → 预测 v_action
x_t^m = t · ε_m + (1-t) · m       → 预测 v_mask
```

两个 flow 共享同一个 timestep t，共享同一个 transformer backbone，在 attention 中互相交互。

### 2. Mask Trajectory Representation

- 原始 mask: `[H_action, 16, 16]` (action horizon 步的 patch-grid mask)
- 展平: `[H_action, 256]`
- 线性投影到 `mask_dim`（默认 32，与 action_dim 对齐）
- 得到 mask trajectory: `[H_action, 32]`，与 action `[H_action, 32]` 完全对称

### 3. Architecture (Non-invasive)

原始 suffix 结构:
```
[state(1), action_tokens(50)]  →  total = 51 tokens
```

Dual Flow suffix 结构:
```
[state(1), action_tokens(50), mask_tokens(50)]  →  total = 101 tokens
```

- mask_tokens 与 action_tokens 共享同一个 timestep conditioning
- 各自有独立的 `in_proj` / `out_proj`
- 在 transformer 的 self-attention 中双向交互（bidirectional within suffix）
- mask_tokens 可以看到 action_tokens（知道正在生成什么动作）
- action_tokens 可以看到 mask_tokens（知道物体要怎么移动）

### 4. Training Loss

```
L = L_action + λ_mask · L_mask

L_action = MSE(v_t^action, u_t^action)    # 标准 flow matching loss
L_mask   = MSE(v_t^mask,   u_t^mask)      # mask flow matching loss
```

### 5. Inference

联合去噪 action 和 mask trajectory。取 action 输出执行控制，mask 输出可用于：
- **可视化/可解释性**：模型 "想象" 物体会怎么移动
- **失败检测**：如果预测的 mask trajectory 不合理，说明模型不确定
- 或直接丢弃（核心收益来自训练阶段的联合去噪）

## Why This Works（脑中预演）

**去噪早期 (t ≈ 1)**：
- mask flow 先规划出物体大致运动方向（向左？向上？）
- action flow 据此生成大致运动趋势
- 粗粒度的视觉规划引导粗粒度的动作规划

**去噪后期 (t ≈ 0)**：
- mask flow 精确到物体的具体位置
- action flow 精确到关节角度 / 末端执行器位移
- 精细的空间信息引导精细的动作控制

两个 flow 每一步都通过 attention 互相看到对方 → action 生成天然被物体运动引导。

## Innovation

| 维度 | 现有方法 | Dual Flow Matching |
|------|---------|-------------------|
| Mask 角色 | 辅助 loss target | 与 action 对等的生成目标 |
| 监督方式 | Feature-level contrastive | Generation-level co-denoising |
| 信息流 | 间接（gradient through loss） | 直接（attention in shared transformer） |
| 推理时 | 无影响 / 需要 mask input | 生成 mask trajectory 作为副产品 |
| 范式 | 单目标生成 (action only) | 双目标联合生成 (action + object motion) |

## Key Design Decisions

1. **Mask dim = Action dim (32)**：对称设计，复用相同的 timestep fusion 机制
2. **共享 transformer backbone**：不增加新的 transformer 参数，只增加 in/out projection layers
3. **同一个 timestep t**：保证两个 flow 在去噪过程中同步
4. **Suffix-level attention**：action 和 mask tokens 在 suffix 内双向 attend，无需修改 prefix 处理
