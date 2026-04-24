# VGAA: Velocity-Grounded Attention Alignment for Flow-based VLAs

> **一句话总结**：我们发现 flow-based VLA（π₀, π₀.₅）的 action expert 中，cross-attention 对 visual token 的权重分布与 velocity field 对这些 token 的实际依赖程度存在系统性偏差。提出 VGAA——用 velocity Jacobian 范数作为自监督信号，在训练阶段显式对齐 attention 分布，使模型学会"看对地方"。训练完成后推理零开销。

---

## 1. 问题背景与动机

### 1.1 VLA 的注意力利用瓶颈

Flow-based VLA 的典型架构（以 π₀ 为例）：

- VLM 骨干（PaliGemma-3B）编码图像 + 语言指令，输出一组 token $\{o_1, ..., o_N\}$
- Action Expert（~300M Transformer）接收这些 token 和 action latents，通过 self-attention 交互后做 flow matching 去噪生成动作

**核心矛盾**：action token 对 visual token 的 attention（以下称 action→vision attention）是动作生成对视觉信息的唯一读取通道。但这个 attention 仅通过 flow matching MSE loss 隐式训练——梯度从 velocity 预测误差反传到 attention 权重的路径很长（velocity head → action expert 多层 → attention Q/K），attention 收到的学习信号极其间接。

多篇工作独立观察到了这一问题的不同症状：

| 工作 | 观察到的症状 |
|------|------------|
| UAOR (Yang et al., 2026) | 深层 action token 对 observation token 的 attention 权重衰减至 <0.06 |
| FocusVLA (Zhang et al., 2026) | 混合注意力中 action latent 走 shortcut（偏向 action query），视觉 attention 被压制 |
| DeepVision-VLA (Luo et al., 2026) | Grad-CAM 显示深层的视觉贡献弥散到背景区域 |

**但这些工作都没有回答一个前置问题：attention 到底应该对齐到什么目标？**

FocusVLA 的回答是"更聚焦"（TopK 选择），UAOR 的回答是"不确定时补信息"，DeepVision-VLA 的回答是"加外部视觉专家"。这些都是启发式的对策，而非正面定义"什么是好的 attention"。

### 1.2 核心洞察

Flow-based VLA 的架构中存在一个被忽视的内生信号：**velocity Jacobian**。

对 velocity field $v_\theta(x_\tau, \tau, o, \ell)$，其关于第 $i$ 个 observation token 的 Jacobian：

$$J_i = \frac{\partial v_\theta}{\partial o_i} \in \mathbb{R}^{d_a \times d_o}$$

$\|J_i\|_F$ 度量的是：**如果 patch $i$ 被微小扰动，预测的 velocity 会变化多少**。这是一个严格定义的一阶 sensitivity（Taylor 展开的直接推论，无近似）。

**关键观察**：$\|J_i\|_F$ 大 $\Leftrightarrow$ patch $i$ 对动作预测有强因果影响。这个信号：

1. **逐 patch 各异**：不同 patch 有不同的 sensitivity，给出空间分辨率
2. **逐 flow-step 各异**：$J_i$ 依赖 $\tau$ 和 $x_\tau$，不同去噪阶段自动给出不同的 patch 重要性
3. **来自模型自身**：不需要外部模型（DINOv3）、不需要人工标注、不需要额外传感器
4. **对 visual 和 proprio token 统一适用**：$o_i$ 可以是图像 patch 也可以是本体感知 token

**核心 idea**：将 $\|J_i\|_F$ normalize 后作为 attention 的显式对齐目标，在 SFT 训练阶段加一个辅助 loss，让 attention 学会反映 velocity 对各 patch 的真实依赖结构。

### 1.3 为什么 sensitivity ≠ learned attention

这是需要实验验证的核心假设，但有强先验理由：

**原因 1（训练目标间接性）**：flow matching loss $\|v_\theta - u\|^2$ 对 Q/K 投影的梯度路径经过 softmax → value projection → 多层 FFN → velocity head，信号被大幅衰减。attention 收到的梯度是一个高度扭曲后的信号。

**原因 2（VLM 预训练偏置）**：VLM 的 attention 模式是为语言理解预训练的，倾向关注语义显著区域（文字、显眼物体）。但操作任务需要关注几何关键区域（接触面、边缘、夹爪位姿），两者不一致。

**原因 3（信息过载稀释）**：VLM 输出的 visual token 数量很大（如 512 个），softmax 的竞争机制让每个 patch 分到的 attention 非常分散。sensitivity 通常集中在少数操作相关 patch 上，attention 分布的"尖锐度"不够。

**这三个原因都可以通过诊断实验验证**——计算 learned attention 和 sensitivity 的 Spearman 相关，看是否显著低于 1。

---

## 2. 方法

### 2.1 概览

VGAA 在标准 flow matching VLA 训练的基础上，增加一个辅助 loss：

1. **正常 Forward**：给定 $(s, \ell, a_0, \tau, \epsilon)$，计算 $v_\theta(x_\tau, \tau, s, \ell)$ 和 flow matching loss $\mathcal{L}_{\text{flow}}$
2. **Sensitivity 计算**：对 observation tokens $\{o_i\}$，通过一次额外 backward 计算 $s_i = \|\partial v_\theta / \partial o_i\|$（detached，不传梯度）
3. **Attention 提取**：从 action expert 提取 action→observation 的 cross-attention 权重 $W_i$
4. **对齐 Loss**：计算 $\text{KL}(\text{normalize}(s) \| W)$
5. **总 Loss**：$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda \mathcal{L}_{\text{align}}$

### 2.2 Sensitivity 计算

#### 2.2.1 精确定义

给定 action expert 对 observation token $o_i \in \mathbb{R}^{d_o}$ 的 velocity Jacobian：

$$J_i = \frac{\partial v_\theta(x_\tau, \tau, o, \ell)}{\partial o_i} \in \mathbb{R}^{d_a \times d_o}$$

patch $i$ 的 sensitivity 定义为 Frobenius 范数的平方：

$$s_i = \|J_i\|_F^2 = \text{tr}(J_i^T J_i)$$

**精确含义**：对 $o_i$ 施加单位球面上均匀分布的扰动 $\delta_i$（$\|\delta_i\|_2 = 1$），velocity 变化的期望平方范数为：

$$\mathbb{E}_{\delta_i \sim \text{Unif}(\mathbb{S}^{d_o-1})}[\|\Delta v\|^2] = \frac{1}{d_o}\|J_i\|_F^2 + O(\|\delta\|^3)$$

所以 $s_i$ 正比于 patch $i$ 被随机扰动时 velocity 变化的期望幅度。这是一阶 Taylor 展开的直接推论，**不涉及任何近似假设**（除了高阶项被忽略，这在小扰动下严格成立）。

#### 2.2.2 高效计算

直接计算 $J_i \in \mathbb{R}^{d_a \times d_o}$ 代价太高。我们用 Hutchinson 随机估计：

$$\|J_i\|_F^2 = \mathbb{E}_{z \sim \mathcal{N}(0, I_{d_a})}\left[\left\|\frac{\partial (v_\theta^T z)}{\partial o_i}\right\|^2\right]$$

**证明**：

$$\mathbb{E}_z\left[\left\|\frac{\partial (v_\theta^T z)}{\partial o_i}\right\|^2\right] = \mathbb{E}_z\left[\|J_i^T z\|^2\right] = \mathbb{E}_z\left[z^T J_i J_i^T z\right] = \text{tr}(J_i J_i^T) = \|J_i\|_F^2$$

第三步用了 $\mathbb{E}[z z^T] = I$ 的性质。$\square$

每个随机探针 $z$ 需要一次 vector-Jacobian product（标准 autograd backward），取 $K=1$ 或 $K=3$ 即可。

#### 2.2.3 更廉价的 Proxy

如果连 Hutchinson 都嫌贵，可以用 **velocity-projected sensitivity**：

$$\hat{s}_i = \left\|\frac{\partial \|v_\theta\|^2}{\partial o_i}\right\| = \|2 J_i^T v_\theta\|$$

这是 Jacobian 在 velocity 方向上的投影范数，只需一次 backward（对标量 $\|v_\theta\|^2$）。

**偏差分析**：$\hat{s}_i$ 是 $\|J_i\|_F$ 的有偏估计。偏差来源于只看了 $v_\theta$ 方向的 sensitivity，忽略了与 $v_\theta$ 正交方向的变化。但对于 **patch 排序**（我们关心的是相对大小，不是绝对值），只要 $v_\theta$ 方向不与 $J_i$ 的主方向正交，排序就基本保持。这一点需要在诊断实验中验证——比较 cheap proxy 和 Hutchinson(K=5) 的 patch 排序一致性。

#### 2.2.4 关键：Stop-Gradient

$s_i$ 必须 detach，**不允许梯度流过 sensitivity 计算路径**。原因：

如果不 detach，模型可以通过让 $v_\theta$ 对所有 $o_i$ 都不敏感（$J_i \to 0$）来最小化 $\mathcal{L}_{\text{align}}$——sensitivity 全部为零时 $\mathcal{L}_{\text{align}} = 0$。但这同时意味着 $v_\theta$ 不依赖观测，$\mathcal{L}_{\text{flow}}$ 会爆炸。

detach 后，sensitivity 是一个固定的 target，$\mathcal{L}_{\text{align}}$ 的梯度只流经 attention 权重（Q/K 投影），不会破坏 velocity 预测本身。

### 2.3 Attention 提取

#### 2.3.1 对齐哪里的 Attention

在 π₀ 的 action expert 中，action token 和 VLM feature token 在同一序列里做 self-attention。我们提取 **action→observation 子矩阵**——即 query=action tokens、key=observation tokens 的 attention 权重。

设 action expert 第 $\ell$ 层、第 $h$ 头的 attention 矩阵中，action token $j$ 对 observation token $i$ 的权重为 $W_{j,i}^{(\ell,h)}$。我们取 **head 平均 + action token 平均**：

$$\bar{W}_i^{(\ell)} = \frac{1}{H \cdot N_a} \sum_{h=1}^{H} \sum_{j=1}^{N_a} W_{j,i}^{(\ell,h)}$$

**为什么对 head 取平均**：不同 head 可能关注不同类型的信息（object / gripper / context），强制每个 head 都对齐 Jacobian 会杀死 head 多样性。对 head 平均后，约束的是"所有 head 合起来的净效果"，允许 head 间分工。

#### 2.3.2 对齐哪些层

不是所有层都需要对齐。基于 UAOR 的发现（前 2-8 层 entropy 上升、attention 衰减），**选择 attention 衰减最严重的层做对齐**。

具体地，设 $\mathcal{S} \subseteq \{1, ..., L\}$ 为需要对齐的层集合。$\mathcal{S}$ 的选择可以通过预实验确定——在 SFT 完成后，逐层计算 $\rho(\bar{W}^{(\ell)}, s)$，选 $\rho$ 最低的若干层。

或者更简单：**对齐所有层，让 $\lambda$ 控制强度**。实验时对比两种策略。

### 2.4 对齐 Loss

$$\mathcal{L}_{\text{align}} = \frac{1}{|\mathcal{S}|} \sum_{\ell \in \mathcal{S}} D_{\text{KL}}\left(\bar{s} \;\Big\|\; \bar{W}^{(\ell)}\right)$$

其中 $\bar{s}$ 是 temperature-normalized sensitivity 分布：

$$\bar{s}_i = \frac{\exp(s_i / T_s)}{\sum_j \exp(s_j / T_s)}$$

$T_s$ 是温度超参数。$T_s$ 小时 $\bar{s}$ 尖锐（接近 one-hot），$T_s$ 大时 $\bar{s}$ 平缓。

**KL 方向的选择**：$D_{\text{KL}}(\bar{s} \| W)$ 而非 $D_{\text{KL}}(W \| \bar{s})$。前者（forward KL / mean-seeking）惩罚的是 "$W$ 在 $\bar{s}$ 大的地方过小"——即"模型没有看应该看的 patch"。这正是我们要矫正的问题（attention 弥散到背景）。反向 KL 惩罚的是 "$W$ 在 $\bar{s}$ 小的地方过大"——这也有道理但不是主要问题。

> **注意**：$D_{\text{KL}}(\bar{s} \| W) = \sum_i \bar{s}_i \log(\bar{s}_i / W_i)$。当 $W_i \to 0$ 但 $\bar{s}_i > 0$ 时，KL 趋向无穷大。这意味着如果某个高 sensitivity patch 被 attention 完全忽略，loss 会非常大。这是我们想要的行为。

### 2.5 总 Loss

$$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda \cdot \mathcal{L}_{\text{align}}$$

$\lambda$ 是超参数，控制对齐信号的强度。

**$\lambda$ 的安全性**：由于 stop-gradient，$\mathcal{L}_{\text{align}}$ 的梯度只影响 Q/K 投影参数（改变 attention 模式），不影响 V 投影和后续 FFN（不改变 feature 处理方式）。所以即使 $\lambda$ 较大，也不会直接干扰 $\mathcal{L}_{\text{flow}}$ 的 velocity 预测。当然，极端大的 $\lambda$ 可能间接影响（attention 变化太剧烈导致 feature 聚合失衡），需要实验确定合理范围。

### 2.6 Flow-Time 的自动调制

这是 VGAA 最自然的一个性质，**不需要任何额外机制**。

在标准 flow matching 训练中，每个 batch 的 $\tau$ 是随机采样的。对于不同的 $\tau$：

- $\tau$ 接近 1（noise 端）：$x_\tau \approx \epsilon$，$v_\theta$ 需要"从噪声中猜测动作方向"。此时 sensitivity 应该集中在 **语义显著的 patch**（目标物体的大致位置），因为粗方向由这些 patch 决定。
- $\tau$ 接近 0（clean 端）：$x_\tau \approx a_0$，$v_\theta$ 只做微小修正。此时 sensitivity 应该集中在 **几何精细的 patch**（接触面、抓取边缘），因为精确修正由这些 patch 决定。

这意味着在训练过程中，$\mathcal{L}_{\text{align}}$ 在不同 $\tau$ 下提供不同的对齐目标——**模型自动学会在不同去噪阶段关注不同 patch**。这个 coarse-to-fine schedule 不需要手动设计。

**这区分于 FocusVLA 的 TopK**：FocusVLA 对所有 forward pass 使用相同的 K 值，没有 flow-time 适应性。VGAA 的对齐目标在每个 $\tau$ 下都不同。

### 2.7 本体感知 Token 的统一处理

如果 observation tokens $\{o_i\}$ 中包含 proprio tokens（关节角度、末端执行器位姿等），sensitivity $s_i$ 对 proprio token 也自然有定义。在某些 flow step 上，proprio token 的 sensitivity 可能高于 visual token（比如夹爪即将闭合时，gripper 状态对动作的影响 >> 远处视觉 patch）。

$\mathcal{L}_{\text{align}}$ 统一地覆盖所有 observation token，自动在 visual 和 proprio 之间分配 attention。**不需要为 proprio 设计额外机制。**

---

## 3. 理论分析

### 3.1 Sensitivity 的一阶精确性（非近似）

**事实 1（一阶展开）**：对任意可微的 $v_\theta$ 和 observation perturbation $\delta = (\delta_1, ..., \delta_N)$：

$$v_\theta(x_\tau, \tau, o + \delta, \ell) = v_\theta(x_\tau, \tau, o, \ell) + \sum_{i=1}^{N} J_i \delta_i + O(\|\delta\|^2)$$

这是标准的多元 Taylor 展开，一阶项精确。$\|J_i\|_F^2$ 作为 patch $i$ 影响的度量不涉及任何近似（高阶项在 $\|\delta\| \to 0$ 时严格为零）。

### 3.2 Hutchinson 估计的无偏性（严格成立）

**事实 2**：对 $z \sim \mathcal{N}(0, I_{d_a})$：

$$\mathbb{E}\left[\|J_i^T z\|^2\right] = \|J_i\|_F^2$$

证明见 2.2.2 节。有限样本 $K$ 的方差为 $O(1/K)$，$K=3$ 即可得到合理精度。

### 3.3 Attention 在 Flow Matching Loss 下收到的梯度信号

**命题 1（attention 梯度衰减）**：设 action expert 为 $L$ 层 Transformer，第 $\ell$ 层的 attention 权重为 $W^{(\ell)}$，flow matching loss 为 $\mathcal{L}_{\text{flow}} = \|v_\theta - u\|^2$。则：

$$\frac{\partial \mathcal{L}_{\text{flow}}}{\partial W^{(\ell)}} = 2(v_\theta - u)^T \cdot \frac{\partial v_\theta}{\partial W^{(\ell)}}$$

其中 $\partial v_\theta / \partial W^{(\ell)}$ 需要经过第 $\ell+1$ 到第 $L$ 层的 FFN、残差连接和后续 attention 层。每经过一层 FFN，梯度方向会被非线性激活函数的 Jacobian 旋转和缩放。

**推论**：depth 越大（$\ell$ 越小），$\partial \mathcal{L}_{\text{flow}} / \partial W^{(\ell)}$ 的方向与"让 attention 对齐 sensitivity"的理想方向之间的偏差越大。**这就是为什么仅靠 $\mathcal{L}_{\text{flow}}$ 训练出的 attention 不够好**——隐式学习信号在深层严重失真。

**$\mathcal{L}_{\text{align}}$ 的作用**：直接给每一层的 attention 一个显式的、不经过任何其他层传播的对齐目标。梯度路径为 $\mathcal{L}_{\text{align}} \to W^{(\ell)} \to Q^{(\ell)}, K^{(\ell)}$，不经过后续层，信号无衰减。

### 3.4 Sensitivity 与最优 Attention 的关系——以及其局限

我们**不能**严格证明 "sensitivity 正比于最优 attention"。原因：

- "最优 attention" 的定义依赖于具体的优化目标和约束，没有唯一定义
- Attention 权重的作用是加权聚合 value，而 sensitivity 度量的是输入扰动对输出的影响。两者有联系但不等价
- 具体的反例：一个 patch 可能有高 sensitivity（位置偏一点动作就全错）但不需要高 attention（因为位置信息是低维的，低 attention 就能提取）

**我们只做如下 claim**（可通过实验验证的 hypothesis，非定理）：

> **Hypothesis**：在当前 VLA 的主要失败模式——attention 弥散到无关背景 patch——中，sensitivity 和 attention 之间存在系统性偏差（高 sensitivity patch 被分配了低 attention）。VGAA 通过矫正这一偏差来提升性能。

这个 claim 不依赖 "sensitivity = 最优 attention" 的强假设，只依赖 "sensitivity 是比 learned attention 更好的 attention proxy"。后者的验证方式：加了 $\mathcal{L}_{\text{align}}$ 后性能是否提升。

### 3.5 不会导致 $\mathcal{L}_{\text{flow}}$ 退化的条件

**命题 2**：设 $\mathcal{L}_{\text{flow}}(\theta)$ 是标准 flow matching loss，$\mathcal{L}_{\text{align}}(\theta)$ 是对齐 loss（stop-gradient on sensitivity target）。如果 $\mathcal{L}_{\text{align}}$ 的梯度只影响 Q/K 投影参数 $\theta_{QK} \subset \theta$，且 $\lambda$ 足够小使得 $\theta_{QK}$ 的更新不会使 $\mathcal{L}_{\text{flow}}$ 增大超过 $O(\lambda^2)$，则总 loss 的优化轨迹与纯 $\mathcal{L}_{\text{flow}}$ 相比，$\mathcal{L}_{\text{flow}}$ 的值偏差为 $O(\lambda)$。

**直觉**：$\mathcal{L}_{\text{align}}$ 只"建议" attention 看哪里，不限制 attention 看到什么后怎么处理信息。如果 $\mathcal{L}_{\text{align}}$ 的建议是好的（sensitivity 确实指向有用的 patch），$\mathcal{L}_{\text{flow}}$ 不仅不会退化，还可能受益。如果建议是坏的，$\lambda$ 足够小时影响可控。

---

## 4. 与现有方法的关系与区别

### 4.1 对比总结

| 维度 | FocusVLA | DeepVision-VLA | UAOR | VLA-Pruner | **VGAA** |
|------|----------|---------------|------|------------|----------|
| 目标 | 改架构消除 shortcut | 加外部视觉专家 | 推理时补信息 | 推理加速 | **训练时对齐 attention** |
| 核心信号 | Learned attention TopK | 浅层 action→vision attention | Action Entropy | 累计 attention | **Velocity Jacobian** |
| 信号性质 | 相关性（模型学到看哪里） | 相关性 | 不确定性（标量） | 相关性 | **因果性（输出对输入的依赖）** |
| 架构改动 | 大（Cascaded Attn + Focus Attn） | 大（+DINOv3 0.8B） | 无 | 无 | **无** |
| 额外推理开销 | 无（训练完后固定） | 大（DINOv3 forward） | 小（<5%） | 加速 | **零** |
| 额外训练开销 | 标准 | 大 | N/A (training-free) | N/A (training-free) | **+30-50%**（Jacobian 计算） |
| 额外参数 | 有（门控 MLP 等） | 大（DINOv3 + 对齐层） | 无 | 无 | **无** |
| Flow-time 适应性 | 无 | 无 | 无 | 无 | **自动**（Jacobian 随 τ 变化） |

### 4.2 与 UAOR 的关系（正交）

| | UAOR | VGAA |
|---|---|---|
| 阶段 | 推理时 | 训练时 |
| 层级 | FFN（键值记忆） | Attention（Q/K 投影） |
| 机制 | 检测高 entropy → 注入观测到 FFN | 计算 Jacobian → 对齐 attention 分布 |
| 信号类型 | 不确定性（何时补） | Sensitivity（补什么方向） |
| 推理开销 | <5% | 零 |

两者**严格正交**：

- VGAA 解决"训练后 attention 就不看对地方" → 训练时根治
- UAOR 解决"即使 attention 对了，FFN 还是可能丢信息" → 推理时兜底

可以叠加：先用 VGAA 训出更好的 attention，再用 UAOR 在推理时兜底。

### 4.3 与 FocusVLA 的关系

FocusVLA 的核心贡献是 Cascaded Attention——通过架构改动消除 action query 对 visual token 的 attention 抢占。这是一个**结构层面**的修正。

VGAA 不改架构，而是给 attention 一个**更好的学习目标**。两者可以结合：在 FocusVLA 的 Cascaded Attention 架构上，再加 VGAA 的对齐 loss，让 Cascaded Attention 中的 Focus Attention 不再依赖 TopK（一个硬截断的启发式），而是对齐到 Jacobian（一个有明确含义的信号）。

### 4.4 与 Saliency/Grad-CAM 的区别

审稿人可能会问"这和 Grad-CAM 有什么区别"。区别在于：

- **Grad-CAM 是可视化工具**：计算 $\partial \text{output} / \partial \text{feature}$，画热图给人看，不参与训练
- **VGAA 把 sensitivity 作为训练信号**：计算 $\partial v_\theta / \partial o_i$，normalize 后作为 attention 的对齐目标，通过 KL loss 反传梯度修改 Q/K 投影

更关键的区别：Grad-CAM 的经典应用场景是分类网络，类别不变时 saliency 也不变。但 flow VLA 的 velocity 依赖 flow time $\tau$ 和当前噪声水平 $x_\tau$，所以 VGAA 的 sensitivity 是**动态的、flow-time-dependent 的**。这是 VLA 特有的信号结构，不存在于 CV 分类任务中。

---

## 5. 算法伪代码

**Algorithm: VGAA Training**

---

**Input:**
- Flow-based VLA: VLM backbone $f_{\text{vlm}}$（frozen）, Action Expert $v_\theta$
- Training data: $\{(s_i, \ell_i, a_{0,i})\}$
- Hyperparams: $\lambda$（对齐 loss 权重），$T_s$（sensitivity temperature），$K$（Hutchinson 探针数）
- 对齐层集合 $\mathcal{S}$

---

**For each training iteration:**

**▷ Step 1: Standard Flow Matching Forward**

- Sample $\tau \sim U[0, 1]$, $\epsilon \sim \mathcal{N}(0, I)$
- Construct $x_\tau = \tau \cdot a_0 + (1 - \tau) \cdot \epsilon$
- Compute target: $u = a_0 - \epsilon$
- Encode observation: $o = f_{\text{vlm}}(s, \ell)$ （frozen, 含 visual + proprio tokens）
- Forward: $v_\theta(x_\tau, \tau, o) \to \hat{v}$，**同时记录各层 cross-attention 权重** $\{W^{(\ell)}\}_{\ell \in \mathcal{S}}$
- Flow matching loss: $\mathcal{L}_{\text{flow}} = \|\hat{v} - u\|^2$

**▷ Step 2: Sensitivity Computation（detached，不进入计算图）**

```python
with torch.no_grad():
    # Hutchinson estimator (K probes)
    s = torch.zeros(N_obs)  # N_obs = visual + proprio tokens
    for k in range(K):
        z = torch.randn_like(v_hat)
        # 需要对 o enable grad 临时计算，但结果 detach
        o_grad = o.detach().requires_grad_(True)
        v_recompute = action_expert(x_tau, tau, o_grad)
        scalar = (v_recompute * z).sum()
        grad_o = torch.autograd.grad(scalar, o_grad)[0]  # [N_obs, d_o]
        s += grad_o.pow(2).sum(dim=-1)  # [N_obs]
    s = s / K
```

**▷ Step 3: Attention Alignment Loss**

```python
# Normalize sensitivity to probability distribution
s_bar = F.softmax(s / T_s, dim=-1)  # [N_obs]

L_align = 0
for ell in S:
    # W_ell: head-averaged, action-averaged attention on obs tokens
    W_ell = attention_weights[ell]  # [N_obs], already softmaxed
    W_ell = W_ell.clamp(min=1e-8)  # numerical stability
    L_align += F.kl_div(W_ell.log(), s_bar, reduction='batchmean')
L_align = L_align / len(S)
```

**▷ Step 4: Total Loss & Update**

$$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda \cdot \mathcal{L}_{\text{align}}$$

$$\theta \leftarrow \theta - \text{lr} \cdot \nabla_\theta \mathcal{L}$$

---

**Output:** Action expert $v_\theta$ with improved attention alignment. Inference procedure is standard π₀ forward, **zero additional computation**.

---

## 6. 实现细节

### 6.1 架构适配

| VLA 架构 | Action→Obs Attention 位置 | VGAA 适配方式 |
|----------|--------------------------|-------------|
| π₀ / π₀.₅ | Action Expert self-attention 中，action tokens 对 VLM tokens 的子矩阵 | 提取子矩阵 |
| CogACT | Action head 的 cross-attention | 直接使用 |
| OpenVLA-OFT | LLM backbone 中 action tokens 对 vision tokens 的注意力 | 提取子矩阵 |
| GR-3 (MoT) | Action DiT 的 cross-attention | 直接使用 |

VGAA 对所有有 action→observation attention 的架构都适用。

### 6.2 关键超参数

| 超参数 | 含义 | 建议范围 | 默认值 |
|--------|------|---------|--------|
| $\lambda$ | 对齐 loss 权重 | [0.01, 0.5] | 0.1 |
| $T_s$ | Sensitivity temperature | [0.1, 5.0] | 1.0 |
| $K$ | Hutchinson 探针数 | [1, 5] | 1 |
| $\mathcal{S}$ | 对齐层集合 | 全部层 / 深层 / 指定层 | 全部层 |

### 6.3 计算开销分析

每个训练 iteration 的额外开销（相比标准 flow matching SFT）：

| 操作 | 开销 | 说明 |
|------|------|------|
| Sensitivity 计算 | +K×0.3x | K 次 VJP（类 backward），但 no_grad + 仅 action expert |
| Attention 提取 | 可忽略 | 已在 forward 中计算，只需保存 |
| KL 计算 | 可忽略 | 标量运算 |
| **总额外开销 (K=1)** | **~1.3x** | |
| **总额外开销 (K=3)** | **~1.9x** | |

对比：DeepVision-VLA 引入 DINOv3 (0.8B) 做 deep coupling，训练开销 >>2x。

### 6.4 推理开销

**零**。训练完成后，attention 已经被矫正。推理时使用标准 π₀ forward，不计算 sensitivity，不做任何额外操作。

---

## 7. 实验设计

### 7.1 诊断实验（Gate Experiment，决定是否推进）

**最紧急。在投入任何训练之前必须先做。**

**设置**：GigaBrain-R0 或 π₀ 的 SFT checkpoint，LIBERO 200 条成功轨迹。

**Step 1**：对每帧计算两个信号：

- $W_i$：action expert 的 head-averaged action→obs attention（所有层平均）
- $s_i$：$\|\partial v_\theta / \partial o_i\|^2$（Hutchinson K=5）

**Step 2**：计算 Spearman 相关 $\rho(W, s)$。

**判断标准**：

| $\rho$ 范围 | 含义 | 决策 |
|-------------|------|------|
| $\rho < 0.4$ | 严重偏差，attention 和 sensitivity 几乎不相关 | **强推进**，空间巨大 |
| $0.4 \leq \rho < 0.7$ | 中等偏差，attention 大方向对但细节差 | **推进**，空间可观 |
| $0.7 \leq \rho < 0.9$ | 轻微偏差 | 谨慎推进，可能只在难任务上有效 |
| $\rho \geq 0.9$ | 基本对齐 | **放弃**，$\mathcal{L}_{\text{flow}}$ 已经隐式学好了 |

**Step 3**：按 flow time $\tau$ 分 bin 画 $\rho(\tau)$ 曲线。

**预判**：$\rho$ 整体在 0.3-0.6，且随 $\tau \to 0$ 下降。依据：
- UAOR 实验显示深层 attention 衰减到 0.06（背景噪声级），而 sensitivity 不应该这么弥散
- FocusVLA 表明默认 attention 有结构性 shortcut 偏差

**Step 4**：验证 cheap proxy。对 50 帧，同时算 cheap proxy（$\partial \|v\|^2 / \partial o_i$）和 Hutchinson(K=5)，比较 Spearman。如果 > 0.85 则后续训练用 cheap 版。

**Step 5**：空间可视化。挑 5 帧，把 $W_i$ 和 $s_i$ reshape 回 image patch grid 并排画。看 sensitivity 是否比 attention 更聚焦于操作区域。

**代码改动量：~30 行**，跑一次 ~1-2 小时。

### 7.2 主实验

**Base model**：π₀ (PaliGemma 3B + 300M Action Expert)

**训练设置**：标准 LIBERO few-shot SFT（各 sub-suite 50 条 demo），对比：

| 方法 | 描述 |
|------|------|
| SFT baseline | 标准 flow matching 训练 |
| SFT + VGAA (K=1) | 加 $\mathcal{L}_{\text{align}}$，cheap proxy |
| SFT + VGAA (K=3) | 加 $\mathcal{L}_{\text{align}}$，Hutchinson |
| FocusVLA | Cascaded Attention + Focus Attention（最强 attention 方向 baseline） |

**评估基准**：

| Benchmark | 任务数 | 评估重点 |
|-----------|--------|---------|
| LIBERO-Spatial | 10 | 空间推理 |
| LIBERO-Object | 10 | 物体识别 |
| LIBERO-Goal | 10 | 目标导向 |
| LIBERO-Long | 10 | 长序列 |
| SIMPLER | 4 | 精细操作 |
| Real Robot | 4+ | 真实世界迁移 |

### 7.3 消融实验

#### 消融 1：对齐信号来源

| 变体 | 描述 |
|------|------|
| VGAA (Jacobian) | 本文方法 |
| VGAA (attention-self) | 用 learned attention 本身做对齐目标（验证循环性问题） |
| VGAA (random) | 随机 sensitivity 分布 |
| VGAA (uniform) | 均匀分布 |

预期：Jacobian >> attention-self ≈ random ≈ uniform。如果 attention-self 和 Jacobian 接近，说明 learned attention 已经很好了（与 gate experiment 结论一致）。

#### 消融 2：对齐层的选择

| 变体 | 描述 |
|------|------|
| 所有层 | $\mathcal{S} = \{1, ..., L\}$ |
| 仅深层 | $\mathcal{S} = \{L/2+1, ..., L\}$ |
| 仅浅层 | $\mathcal{S} = \{1, ..., L/2\}$ |
| 单最深层 | $\mathcal{S} = \{L\}$ |

预期：深层 >> 浅层（浅层 attention 已经 OK，深层最需要矫正）。

#### 消融 3：$\lambda$ 和 $T_s$ 敏感度

$\lambda \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$
$T_s \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$

#### 消融 4：Cheap proxy vs Hutchinson

| 变体 | 训练开销 | 性能 |
|------|---------|------|
| $\partial \|v\|^2 / \partial o_i$ | ~1.3x | ? |
| Hutchinson K=1 | ~1.3x | ? |
| Hutchinson K=3 | ~1.9x | ? |
| Hutchinson K=5 | ~2.5x | ? |

找到性价比最优的变体。

#### 消融 5：与 UAOR 叠加

| 变体 | 训练 | 推理 |
|------|------|------|
| Baseline | SFT | 标准 |
| UAOR only | SFT | + UAOR |
| VGAA only | SFT + VGAA | 标准 |
| **VGAA + UAOR** | **SFT + VGAA** | **+ UAOR** |

验证两者正交性：如果 VGAA + UAOR > max(VGAA, UAOR)，两者互补。

### 7.4 分析实验

#### 分析 1：训练过程中 $\rho(W, s)$ 的变化曲线

随训练 step 画 $\rho$。预期：baseline 的 $\rho$ 基本不变（$\mathcal{L}_{\text{flow}}$ 对 attention 的隐式信号太弱），VGAA 的 $\rho$ 持续上升。

#### 分析 2：Attention 可视化对比

在相同帧上，对比 baseline 和 VGAA 的 attention 分布热图。预期：VGAA 的 attention 更聚焦于操作相关区域。

#### 分析 3：$s_i(\tau)$ 随 flow time 的变化

固定一帧，画不同 $\tau$ 下 $s_i$ 的空间分布。验证 coarse-to-fine 趋势是否存在。

#### 分析 4：Visual vs Proprio sensitivity 占比

画 $\sum_{i \in \text{visual}} s_i / \sum_i s_i$ 随 $\tau$ 的变化。看 proprio token 在什么 flow step 上 sensitivity 更高。

---

## 8. 潜在风险与缓解

### 8.1 Chicken-and-Egg Problem

**风险**：sensitivity $s_i$ 是用当前模型计算的。如果 attention 已经很差（不看 patch $i$），模型对 patch $i$ 的 sensitivity 本身就很低，不会激励 attention 转向它。

**分析**：在 cross-attention / self-attention 中，softmax 永远不产生精确的 0。所以即使 attention 低，也有非零梯度路径到每个 patch。sensitivity 可能小，但**相对排序**仍有信号。

**缓解**：
1. 初始 attention 来自 SFT，不是随机初始化——SFT 已经学到了大致正确的 attention 模式，只是不够精细
2. $\lambda$ 从小开始 warm up（训练前期少干预，让 $\mathcal{L}_{\text{flow}}$ 先建立基本的 attention 结构）
3. 迭代自我改善：attention 改善 → sensitivity 更准确 → attention 进一步改善

**验证**：如果 gate experiment 中 $\rho > 0.2$（即使是弱相关），说明初始 sensitivity 有足够信号启动迭代。

### 8.2 $\mathcal{L}_{\text{align}}$ 与 $\mathcal{L}_{\text{flow}}$ 梯度冲突

**风险**：$\mathcal{L}_{\text{align}}$ 修改 Q/K 投影，可能间接影响 feature 聚合质量。

**缓解**：
1. stop-gradient on sensitivity target
2. $\lambda$ 足够小，$\mathcal{L}_{\text{flow}}$ 主导
3. 实验中监控 $\mathcal{L}_{\text{flow}}$ 曲线——如果加了 VGAA 后 $\mathcal{L}_{\text{flow}}$ 增大，降低 $\lambda$
4. warm up $\lambda$：前 1k steps $\lambda=0$，线性升到目标值

### 8.3 Sensitivity 噪声

**风险**：Hutchinson K=1 的方差可能很大，导致对齐目标不稳定。

**缓解**：
1. 噪声在 mini-batch 和训练迭代上平均——每次 iteration 采样不同的 $z$，长期统计上是无偏的
2. $T_s$ 温度参数提供 smoothing——即使单帧 sensitivity 有噪声，softmax 温度够高时分布变化缓和
3. 实验对比 K=1 和 K=3 的性能差异——如果接近，K=1 的噪声不影响最终结果

### 8.4 LIBERO 上天花板有限

**风险**：LIBERO 多个 sub-suite 的 SOTA 已经接近 99%（FocusVLA 98.7%），提升空间小。

**缓解**：
1. 重点关注 Long sub-suite（当前最难，94-96% 区间）
2. 在 SIMPLER、RoboTwin Hard 等更难的 benchmark 上测试
3. 真实世界实验——attention 质量在 OOD 场景下差距更大

### 8.5 审稿人可能的挑战

**Q: "sensitivity 不就是 Grad-CAM 吗？"**

A: 计算方式类似（都是输出对输入的梯度），但用途完全不同。Grad-CAM 是离线可视化工具，不参与训练。VGAA 是用 sensitivity 作为 attention 的训练时对齐信号，通过 KL loss 改变模型的 Q/K 投影。更关键的差异：flow VLA 的 sensitivity 是 flow-time-dependent 的（随 τ 变化），这种动态结构不存在于 CV 分类任务中。

**Q: "为什么不直接用 sensitivity 做 attention（推理时 rectification）？"**

A: 可以做（即之前讨论的 SGAR），但有推理开销（需要 backward pass）。VGAA 选择在训练时对齐，推理零开销。两种路线可以对比：训练时对齐 vs 推理时修正。我们预期训练时对齐效果更好，因为模型有机会学习在 sensitivity 信号指导下的最优 feature 利用方式。

**Q: "能保证不伤害主 loss 吗？"**

A: 严格保证需要 $\lambda \to 0$，此时效果也趋向零。实践中，stop-gradient + 小 $\lambda$ + warm up 提供了足够的安全性。我们会在实验中明确展示 $\mathcal{L}_{\text{flow}}$ 的训练曲线，验证无退化。

**Q: "这个 idea 和辅助 loss / attention supervision 的一般框架有什么区别？"**

A: 区别在于**对齐目标的来源**。传统 attention supervision 需要外部标注（人眼注视、SAM 分割掩码等）。VGAA 的对齐目标完全来自模型自身的 velocity Jacobian——自监督的、免费的、且具有 flow-time 动态结构。

---

## 9. 论文叙事结构

### Title

**看对地方：用 Velocity Jacobian 对齐 Flow-based VLA 的视觉注意力**

*Look Where It Matters: Velocity-Grounded Attention Alignment for Flow-based Vision-Language-Action Models*

### Abstract

Flow-based VLA 模型（π₀, π₀.₅）的 action expert 通过 attention 读取视觉信息，但这一 attention 仅靠 flow matching loss 隐式训练，导致注意力弥散到无关背景。我们发现 velocity field 的 Jacobian 范数提供了一个自然的、逐 patch 的因果重要性信号：它度量每个 patch 对动作预测的一阶影响，且随 flow time 自动从粗粒度（语义）转向细粒度（几何）。基于此，我们提出 VGAA——一个轻量辅助 loss，在训练阶段将 attention 对齐到 Jacobian sensitivity 分布。VGAA 不改架构、不加参数、推理零开销。实验验证其在 LIBERO / SIMPLER / 真实机器人上的有效性。

### Introduction

1. Flow-based VLA 的架构回顾：VLM + Action Expert + Flow Matching
2. 注意力瓶颈的证据：UAOR / FocusVLA / DeepVision-VLA 的发现
3. 现有对策的局限：改架构（FocusVLA）或加模块（DeepVision-VLA）成本高，推理时修正（UAOR）无法根治
4. **我们的问题**：能否找到 attention 的"正确目标"？
5. **发现**：Velocity Jacobian 天然提供了一个自监督的、flow-time-adaptive 的 attention 对齐信号
6. 贡献总结

### Method

1. Preliminaries: Flow Matching for VLA, Action Expert Architecture
2. Sensitivity 定义与计算（Hutchinson）
3. Attention 提取与对齐 Loss
4. Flow-time 自动调制
5. Proprio token 的统一处理

### Experiments

1. **诊断实验**：$\rho(W, s)$ 分析（证明偏差存在）
2. **主实验**：LIBERO / SIMPLER
3. **消融**：信号来源、层选择、$\lambda$/$T_s$、K
4. **叠加实验**：VGAA + UAOR
5. **分析**：attention 可视化、$\rho$ 训练曲线、flow-time sensitivity 模式

### Conclusion

VGAA 的核心贡献不是复杂方法，而是**发现了 velocity Jacobian 作为 attention 对齐信号的价值**。这个信号是 flow matching 架构的固有副产物，被整个领域忽略了。VGAA 只是利用这个信号的最简方式——一个辅助 KL loss。

---

## 10. 代码改动量估算

基于 π₀ / GigaBrain-R0 的现有代码：

| 模块 | 改动 | 行数估算 |
|------|------|---------|
| Sensitivity 计算函数 | 新增 | ~20 行 |
| Attention 提取 hook | 新增（register forward hook） | ~15 行 |
| KL alignment loss | 新增 | ~10 行 |
| 总 loss 整合 | 修改 train step | ~5 行 |
| 超参数配置 | config 文件 | ~5 行 |
| **总计** | | **~55 行核心代码** |

不需要修改：模型架构、推理流程、数据加载、环境交互。

---

## 11. 时间线建议

| 阶段 | 内容 | 耗时 |
|------|------|------|
| Week 1 | 诊断实验（$\rho$ 分析 + 可视化） | 2-3 天 |
| Week 1 | 根据 $\rho$ 结果决定是否推进 | — |
| Week 2-3 | VGAA 实现 + LIBERO 主实验 | 2 周 |
| Week 3-4 | 消融 + 分析实验 | 1.5 周 |
| Week 4-5 | SIMPLER / 真实世界 / VGAA+UAOR 叠加 | 1.5 周 |
| Week 5-6 | 论文撰写 | 1.5 周 |
