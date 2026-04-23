# Skip-VLA: Surrogate-Gated Inference Skipping for Vision-Language-Action Models

---

## 0. TL;DR

VLA 推理加速的所有现有工作都在优化 "how to run VLA faster"（减步、跳层、剪 token、异步执行）。我们问一个不同的问题：**"do we even need to run VLA at this step?"**

一个 manipulation trajectory 大约 200 步，其中 80%+ 是可预测的惯性运动（自由空间移动、平稳趋近、等待）。这些步用一个 <1M 参数的 MLP predictor 看最近几步 action + proprioception 就能外推，根本不需要跑 3B 的 VLA。

**Skip-VLA** 在每步先用 tiny predictor 预测 action + 估计 confidence。高 confidence → 直接用 predictor 输出（0.1ms），跳过整个 VLA（ViT + VLM + action expert 全跳）。低 confidence → 正常调用 VLA。

与所有现有方法**完全正交**：它减少的是"需要调用 VLA 的步数"，而不是"每次调用 VLA 的延迟"。可以和 SnapFlow / VLA-Cache / DeeR-VLA 任意叠加。

目标 backbone：**π0, π0.5, SmolVLA, OpenVLA-OFT**。

---

## 1. Motivation

### 1.1 核心观察

考虑一个典型的 pick-and-place trajectory（~200 control steps）：

```
Phase 1: 初始趋近   (~60 steps)  → 末端沿直线/弧线移向目标上方
Phase 2: 精细对齐   (~20 steps)  → 调整 orientation, 接近目标
Phase 3: 抓取       (~10 steps)  → gripper close, 可能有力控
Phase 4: 提起+转移  (~60 steps)  → 再次沿平滑轨迹移动
Phase 5: 放置+对齐  (~20 steps)  → 精细放置
Phase 6: 释放+撤退  (~30 steps)  → gripper open, 撤回
```

Phase 1 和 Phase 4 合计 120 步，占 trajectory 的 60%。这些步的 action 高度可预测——robot 本质在做匀速/匀加速运动，不需要场景理解或语言推理。

但当前所有 VLA（π0, SmolVLA, OpenVLA-OFT 等）在**每一步**都跑完整的 ViT + VLM + action expert / AR decoder，无差别地消耗 80-100ms。

### 1.2 物理先验：Robot Action 是平滑的

这不是假设，是物理事实：
- 关节电机有惯性和力矩限制
- 控制器有带宽限制（通常 < 50 Hz）
- 轨迹规划天然产生平滑路径

结果：相邻步 action 的 autocorrelation 极高。在自由空间运动阶段，$a_t \approx a_{t-1} + \Delta$，其中 $\Delta$ 是一个缓变量。一个看过最近几步 action 的 MLP 就能外推。

### 1.3 所有现有工作都在"让 VLA 更快"

| 工作类别 | 代表 | 每步是否仍调用 VLA |
|---|---|---|
| Flow step 压缩 | SnapFlow, OFP, FRMD | ✅ 仍跑 ViT + VLM + action expert |
| Token pruning/caching | VLA-Cache, VLA-Pruner | ✅ 仍跑 ViT + VLM (部分) |
| Layer skipping | DeeR-VLA, MoLe-VLA | ✅ 仍跑 ViT + VLM (部分层) |
| Adaptive computation | AC²-VLA, RD-VLA, GeCO | ✅ 仍跑 router + 部分 backbone |
| Async dual-system | TIDAL, DuoCore-FS, FiS-VLA | ✅ action expert 仍每步跑 |
| Warm-start flow | A2A, Streaming Flow Policy | ✅ 仍每步跑模型 |
| Speculative decoding | Spec-VLA, KERV, HeiSD | ✅ 仍跑 draft + verification |
| **Skip-VLA (Ours)** | | **❌ 高 confidence 步完全不调用 VLA** |

**没有任何一篇工作提出过用 sub-million parameter surrogate 在可预测步上替代整个 VLA forward。**

---

## 2. Method

### 2.1 整体 Pipeline

```
每个 control step t:

┌──────────────────────────────────────────────────────────┐
│  Tiny Predictor P_φ (<500K params, <0.1ms):              │
│    â_t = P_φ(a_{t-1}, ..., a_{t-K}, s_t, Δs_t)          │
│    conf_t = C(â_t, a_{t-1:t-K}, s_t)                    │
└──────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            │ conf_t > τ                 │ conf_t ≤ τ
            │ AND steps_since_VLA < K_max│ OR steps_since_VLA ≥ K_max
            ↓                            ↓
    a_t ← â_t  (SKIP)           a_t ← VLA(o_t, l, s_t)  (FULL)
    cost: 0.1ms                  cost: 80-100ms
            │                            │
            └─────────────┬──────────────┘
                          ↓
                   Execute a_t on robot
```

### 2.2 Tiny Predictor P_φ

**设计原则**：不看图像、不看语言 → 做不了场景理解 → 只能做 trajectory extrapolation。这恰好是我们要的——它只在轨迹可外推的步上工作，"我看不懂场景但我知道接下来该往哪走"。

**输入**（全部 low-dim, 可立即获取）：
```
- action_history: [a_{t-1}, a_{t-2}, ..., a_{t-K}]    # K=5, 7-DoF × 5 = 35 dim
- proprio_state: s_t                                    # 7-DoF joint + gripper = 8 dim
- proprio_delta: [Δs_t, Δs_{t-1}]                      # 速度信号, 16 dim
- gripper_state: g_t                                    # 1 dim (open/close)
总输入维度 ≈ 60
```

**结构**：
```python
class TinyPredictor(nn.Module):
    def __init__(self, input_dim=60, hidden=128, action_dim=7):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)
```

参数量：60×128 + 128×128 + 128×7 ≈ **25K params**。Forward: **< 0.05ms**（CPU 都跑得飞快）。

**不使用图像和语言的原因**：
1. 如果用了图像，就需要跑 ViT → 省不了什么
2. 不看图像 = 天然安全：predictor 的 "无知" 保证了它只在可外推情况下 confident，遇到需要视觉推理的场景自然 confidence 低 → fallback to VLA

### 2.3 Confidence Estimation C

三种方案，从简单到复杂：

**方案 A（zero-shot, training-free, 推荐先试）**：

基于 action trajectory 的局部平滑性。如果最近几步 action 变化是均匀的（近似匀速直线运动），predictor 的外推大概率可靠：

$$
C_A = \exp\left(-\alpha \cdot \frac{\|\Delta^2 a\|_2}{\|\Delta a\|_2 + \epsilon}\right)
$$

其中 $\Delta a_t = a_t - a_{t-1}$，$\Delta^2 a_t = \Delta a_t - \Delta a_{t-1}$（二阶差分/加速度）。

直觉：加速度小 → 匀速运动 → confidence 高。加速度大 → 轨迹在转弯或接触 → confidence 低。

**方案 B（learned gate, 更精确）**：

训一个极小的二分类器 $g_\psi$：
- 输入：action_history, proprio, proprio_delta（和 predictor 一样的输入）
- 输出：$p(\text{skip safe} | \cdot) \in [0,1]$
- Label 来源：从 VLA teacher rollout 收集数据，对每步同时算 predictor output $\hat{a}_t$ 和 VLA output $a_t$，定义 $y^* = \mathbb{1}[\|\hat{a}_t - a_t\|_1 < \delta]$
- 训练：BCE loss，和 predictor 一起训

**方案 C（self-consistency check, 不需要额外训练）**：
- 用 predictor 预测 $\hat{a}_t$
- 计算 $\hat{a}_t$ 和 action history trend 的一致性（是否在 trajectory 的 smooth continuation 上）
- 额外检查 gripper_state 是否在变化（gripper 状态变化 = 关键决策时刻 → confidence 置零）

**推荐策略**：先用方案 A pilot，看 baseline 效果。如果方案 A 就够好（skip rate > 40%, SR drop < 2%），paper 里主推 A 的简洁性；用方案 B 做 ablation 展示可以更好。

### 2.4 Safety Guard

两层安全机制：

1. **Maximum skip count K_max**：连续 skip 不超过 K_max 步（推荐 K_max=8-15）。超过后强制调用 VLA 更新 reference trajectory。

2. **Gripper guard**：如果 predictor 预测的 gripper action 和当前 gripper state 不一致（即 predictor 试图改变 gripper 状态），直接 fallback to VLA。因为 gripper open/close 几乎总是关键决策时刻。

```python
def should_call_vla(conf, steps_since_vla, predicted_gripper, current_gripper, K_max, tau):
    if steps_since_vla >= K_max:
        return True
    if abs(predicted_gripper - current_gripper) > 0.5:  # gripper state change
        return True
    if conf < tau:
        return True
    return False
```

### 2.5 训练

**VLA 本身完全不改、不训。Training-free for VLA.**

只训 Tiny Predictor P_φ（和可选的 gate g_ψ）：

1. 用 target VLA（π0 / SmolVLA）在 LIBERO 上 rollout 成功 trajectories，~200 episodes（几小时 GPU time）
2. 对每步记录 `(action_history, proprio, next_action)`
3. MSE loss 训 predictor；BCE loss 训 gate
4. 训练时间：单张 GPU，< 10 分钟（数据量小、模型小）

**跨任务泛化**：predictor 只做 trajectory extrapolation，不包含任何 task-specific 信息。理论上在 LIBERO-Spatial 上训的 predictor 可以直接用在 LIBERO-Goal 上。这个是 ablation 卖点。

---

## 3. 为什么一定有效

### 3.1 不可能比 baseline 差

Skip-VLA 的最差情况：confidence 永远不够高 → 每步都调用 VLA → 退化成 vanilla VLA。Success rate **严格 ≥ baseline**（假设 guard 设计合理）。

加速效果 = skip rate × (VLA_latency - predictor_latency) ≈ skip rate × VLA_latency。只要 skip rate > 0，就有加速。

### 3.2 物理保证 skip rate 不会太低

Robot action 的 smoothness 是物理决定的，不依赖任何 assumption。在自由空间运动阶段，连续 10-30 步的 action 几乎是线性外推。这部分至少占 trajectory 的 40-60%。

### 3.3 Oracle upper bound 可以先验证

不写任何代码就能估算：用 VLA rollout 数据，post-hoc 对每步做线性外推 $\hat{a}_t = 2a_{t-1} - a_{t-2}$，计算 $\|\hat{a}_t - a_t\|_1$。低于 threshold 的步数 / 总步数 = oracle skip rate。

如果 oracle skip rate < 30%，idea 有问题。如果 > 50%，idea 几乎必成功。

---

## 4. Speedup 分析

### 4.1 单独使用 Skip-VLA

假设：
- VLA 每步延迟: 100ms (π0, 10-step flow matching, A100)
- Predictor 每步延迟: 0.1ms
- Skip rate: 60%

每步平均延迟 = 0.6 × 0.1 + 0.4 × 100 = **40.06ms → 2.5× 加速**

### 4.2 叠加 SnapFlow（flow step 1-step, 约 3× 加速 VLA 本身）

VLA 单步延迟被 SnapFlow 压到 ~35ms：
- 每步平均延迟 = 0.6 × 0.1 + 0.4 × 35 = **14.06ms → 7.1× 加速**

### 4.3 叠加 VLA-Cache

VLA 单步延迟被 VLA-Cache 压到 ~70ms：
- 每步平均延迟 = 0.6 × 0.1 + 0.4 × 70 = **28.06ms → 3.6× 加速**

### 4.4 三者叠加

SnapFlow + VLA-Cache 把 VLA 步压到 ~25ms：
- 每步平均延迟 = 0.6 × 0.1 + 0.4 × 25 = **10.06ms → ~10× 加速**

**关键卖点：Skip-VLA 是 multiplicative 的**，和任何降低单步延迟的方法组合后效果相乘而非相加。

---

## 5. Experiments

### 5.1 Pilot Study (Day 1-2, **go/no-go**)

**必须先跑，不过关则 abort。**

步骤：
1. 在 LIBERO-Spatial / Object / Goal 上用 π0 rollout 100 成功 episodes
2. 记录每步 `(a_t, s_t)`
3. 对每步计算简单线性外推 $\hat{a}_t^{\text{linear}} = 2a_{t-1} - a_{t-2}$
4. 计算 $\text{err}_t = \|a_t - \hat{a}_t^{\text{linear}}\|_1$
5. 画分布图；计算 oracle skip rate：$\text{err}_t < \delta$ 的比例

**Go/no-go**：
- Oracle skip rate ≥ 40% → ✅ 继续
- Oracle skip rate 30-40% → ⚠️ 用 MLP predictor 替换线性外推，可能改善
- Oracle skip rate < 30% → ❌ Abort

### 5.2 Main Experiments

**Benchmarks**: LIBERO 全套 (Spatial / Goal / Object / 10 / Long)

**Backbones**:
- π0（LIBERO fine-tuned checkpoint）
- SmolVLA（LIBERO fine-tuned checkpoint）
- OpenVLA-OFT（如果时间够）

**Metrics**:
- Success rate per task + average
- End-to-end latency (ms/step, A100 wall-clock)
- Skip rate (实际被跳过的步数比例)
- Wall-clock speedup
- VLA call count per episode

### 5.3 Baselines

| Category | Baseline | 说明 |
|---|---|---|
| Vanilla | π0 / SmolVLA | 每步都跑 VLA |
| Token caching | VLA-Cache | Token 级时序复用 |
| Layer skipping | DeeR-VLA | Early exit |
| Flow compression | SnapFlow | 1-step flow |
| Naive skip | Fixed-rate skip | 每 N 步调用一次 VLA，中间线性插值（strawman baseline） |

**Fixed-rate skip 是关键 strawman**：它证明 "固定频率跳过" 不行（因为 critical moments 不可预测），而 Skip-VLA 的 confidence gating 是必要的。

### 5.4 Ablations

1. **Confidence mechanism**: 方案 A (zero-shot) vs B (learned) vs C (self-consistency)
2. **Predictor complexity**: 线性外推 vs 2-layer MLP vs 3-layer MLP
3. **K (action history length)**: K = 3, 5, 8, 10
4. **τ (confidence threshold) sweep**: 画 Pareto frontier (skip rate vs success rate)
5. **K_max (max consecutive skips)**: 5, 8, 15, ∞
6. **Gripper guard ablation**: 有 vs 没有
7. **跨任务泛化**: predictor 在 LIBERO-Spatial 上训，测 LIBERO-Goal / Object / Long
8. **跨 VLA 泛化**: predictor 在 π0 上训，直接用到 SmolVLA

### 5.5 Orthogonality Experiments（核心 table）

| Config | SR | Latency (ms) | Speedup |
|---|---|---|---|
| π0 vanilla | xx | ~100 | 1.0× |
| + SnapFlow | xx | ~35 | 2.9× |
| + VLA-Cache | xx | ~70 | 1.4× |
| + DeeR-VLA | xx | ~60 | 1.7× |
| + Skip-VLA (ours) | xx | ~40 | 2.5× |
| + SnapFlow + Skip-VLA | xx | ~14 | **7.1×** |
| + VLA-Cache + Skip-VLA | xx | ~28 | 3.6× |
| + SnapFlow + VLA-Cache + Skip-VLA | xx | ~10 | **~10×** |

**这个 table 如果做出来，paper 就稳了。** 因为它直接展示 Skip-VLA 和所有现有方法的 multiplicative composability。

### 5.6 Analysis（给 paper 加深度）

1. **Per-phase skip rate breakdown**：可视化一条完整 trajectory，标注每步是 skip 还是 full VLA call。展示 predictor 自动在 "惯性运动" 阶段密集 skip，在 "接触/决策" 阶段 fallback。

2. **Failure case analysis**：predictor 误判 high confidence 但实际应该调 VLA 的 case。分析误判原因（比如环境意外变化、物体滑动）。

3. **Latency distribution**：画每步的 latency histogram，展示 bimodal 分布（0.1ms peak + 100ms peak）。

---

## 6. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Oracle skip rate < 30% | 致命 | Day 1-2 pilot 直接验证；若不过关立刻 abort |
| Predictor 误判导致 drift → 任务失败 | 高 | K_max safety guard + gripper guard；worst case = vanilla VLA |
| 和 AC²-VLA 的 cognition reuse 被认为重合 | 中 | AC²-VLA 每步仍跑 router + 部分 backbone（~40% FLOPs）；我们在 skip 步上跑 0.005% FLOPs。粒度差两个数量级。|
| Reviewer 问"为什么不用更好的 predictor" | 低 | predictor 越强论文越好；MLP 是 minimum viable；可以用 LSTM / Transformer 做 ablation |
| 只在 LIBERO 上验证 | 中 | 加 SIMPLER 或简单 real-robot；LIBERO 仍是社区标准 |
| Skip 后 action 不平滑 | 低 | Predictor 输出本身就是平滑外推；不存在 chunk boundary 问题 |

---

## 7. Paper Structure

```
1. Introduction (1 page)
   - VLA 推理加速的现状：所有方法都在 "让 VLA 更快"
   - 我们的问题："这一步需要跑 VLA 吗？"
   - 核心观察 + 方法概述

2. Related Work (0.75 page)
   - VLA inference acceleration 全景
   - 明确 position：orthogonal to all existing methods

3. Analysis: When Does VLA Computation Matter? (1 page)
   - Oracle skip rate 分析
   - Action autocorrelation / smoothness 的 empirical evidence
   - Per-phase breakdown

4. Method: Skip-VLA (1.5 page)
   - Tiny Predictor
   - Confidence estimation (方案 A/B/C)
   - Safety guard
   - 训练流程

5. Experiments (3.5 page)
   - Main table
   - Orthogonality / composability table（最重要）
   - Ablations
   - Trajectory visualization

6. Conclusion (0.25 page)
```

---

## 8. Timeline (15 days → NeurIPS 2026 May 6)

| Day | Task | Kill criterion |
|---|---|---|
| 1-2 | **Pilot study**: oracle skip rate on LIBERO + π0 | Oracle skip rate < 30% → abort |
| 3 | Implement Tiny Predictor + confidence A (zero-shot) | - |
| 4 | 收集 predictor 训练数据 + 训 predictor & gate | Predictor MSE > naive linear → 调架构 |
| 5-6 | Main benchmark on π0 (LIBERO 全套) | Skip rate < 30% or SR drop > 5% → 调 τ |
| 7 | 迁移到 SmolVLA | - |
| 8 | Baselines (VLA-Cache, DeeR-VLA, SnapFlow) | - |
| 9 | **Orthogonality table**（叠加实验） | - |
| 10 | Ablation table | - |
| 11-14 | 写作 + figures + trajectory visualization | - |
| 15 | Buffer / polish | - |

---

## 9. Scope Control: 不做什么

- ❌ 不做图像输入的 predictor（加图像就没有意义了）
- ❌ 不做 predictor 的 online adaptation / fine-tune（留 future work）
- ❌ 不做 real robot（除非 pilot 特别顺利有余量）
- ❌ 不做超过 2 个 VLA backbone 的对比
- ❌ 不和 TIDAL / DuoCore-FS 做对比（它们解决的是 async execution 问题，不是 skip 问题）

---

## 10. FAQ / Reviewer 预判

**Q: 和 AC²-VLA 的 cognition reuse 有什么区别？**
A: AC²-VLA 的 "reuse" 仍然每步跑 router + 部分 backbone（~40% FLOPs），只是部分 reuse 一些 KV。Skip-VLA 在 skip 步上只跑一个 25K 参数的 MLP，约 VLA 的 0.005% FLOPs。更重要的是：AC²-VLA 无法和 SnapFlow / VLA-Cache 叠加（它修改了 backbone forward），而 Skip-VLA 与所有方法正交。

**Q: 固定频率 skip（每 N 步跑一次 VLA）不就行了？**
A: 固定频率无法适应 task 的 phase 结构。惯性运动阶段可以连续 skip 15 步，接触阶段 1 步都不能 skip。Fixed-rate 是 paper 里的 strawman baseline，我们展示 confidence gating 显著优于它。

**Q: Predictor 够好的话为什么还需要 VLA？**
A: Predictor 只能做轨迹外推，不做视觉理解和语言推理。它在 "往哪走" 明确的阶段好使，在 "该抓哪个、该放哪里" 的决策阶段完全失效（confidence 自然降低 → fallback to VLA）。

**Q: Skip 步的 action 质量会下降吗？**
A: 在 oracle 分析中，skip 步的 predictor action 和 VLA action 的 L1 误差 < δ（δ 就是 skip 的 threshold）。δ 可调，trade-off 由 Pareto frontier 展示。

**Q: Gripper 操作怎么办？**
A: Gripper guard：任何 gripper 状态变化的时刻直接 fallback to VLA。Gripper open/close 是 0/1 决策，不可外推。

**Q: 和 action chunking 有冲突吗？**
A: 不冲突。π0 / SmolVLA 用 action chunking（每次生成 chunk_size=50 个 action）。在 chunk 内，如果某些步可以被 predictor 覆盖，仍然有效。但更准确的 framing 是：Skip-VLA 在 chunk 级别判断是否需要重新调用 VLA 生成新 chunk——如果当前正在执行的 chunk 仍然可靠（confidence 高），延迟 VLA 调用。具体实现需要适配 chunking 逻辑，但原理一致。

---

## 11. Open Questions（开工前确认）

1. π0 / SmolVLA 的 LIBERO fine-tuned checkpoint 你手上有吗？
2. 实验机器：A100 持续 15 天可用？
3. Action chunking 的处理：π0 每次输出 50 步 action chunk 然后执行一部分。Skip-VLA 的 skip 逻辑需要适配到 chunk 级别。两个选项：
   - (a) 在 chunk 内部做 step-level skip（需要修改 execution loop）
   - (b) 在 chunk 边界做 chunk-level skip（更简单：当前 chunk 还能用 → 延迟生成新 chunk）
   - **推荐先做 (b)**，实现简单且 story 清晰
4. Predictor 是每个 VLA backbone 训一个还是通用？建议先 per-backbone，后做 cross-backbone ablation
