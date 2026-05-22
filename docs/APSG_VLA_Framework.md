# APSG-VLA: Action-Projected Self-Grounding for Vision-Language-Action Models

> **一句话概括**: 利用 demonstration 中动作标签的 2D 图像投影作为免费、精确、相位跟踪的注意力监督信号，训练时将 action expert 的 cross-attention 拉向"控制相关区域"而非"语义相关区域"，推理时零改动、零开销，plug-and-play 提升多种 VLA 架构的操作精度。

---

## 目录

1. [全景调研：VLA 视觉观测增强 / 注意力设计现有工作分类](#一全景调研)
2. [深度分析：现有工作的根本盲区](#二深度分析现有工作的根本盲区)
3. [核心方法：APSG-VLA](#三核心方法apsg-vla)
4. [数学推导与理论分析](#四数学推导与理论分析)
5. [算法伪代码](#五算法伪代码)
6. [与现有方法的全面对比](#六与现有方法的全面对比)
7. [实验方案设计](#七实验方案设计)
8. [风险分析与缓解策略](#八风险分析与缓解策略)
9. [实现路线图：最小可行验证](#九实现路线图最小可行验证)
10. [创新性声明与贡献总结](#十创新性声明与贡献总结)

---

## 一、全景调研：VLA 视觉观测增强 / 注意力设计现有工作分类

本节基于对 30+ 篇最新论文（2024.06 – 2026.05）的调研，从 **信号来源** 和 **作用机制** 两个维度系统梳理 VLA 视觉观测增强与注意力设计工作。

### 1.1 输入端增强：观测覆盖 / Prompt 注入

通过修改或增强模型的视觉输入来提供额外信息。

| 方法 | 核心策略 | 信号来源 | 推理开销 | 是否需训练 |
|------|---------|---------|:------:|:---:|
| **TraceVLA** (ICLR'25) | 用 CoTracker 追踪机械臂点，把**过去轨迹**画成彩色线叠到图上作为第二张图输入 | 外部 tracker (CoTracker) | 中（每帧跑 CoTracker） | ✓ (finetune) |
| **ATA** (Mar'26) | 注意力引导 + 动作引导：用中间层 attention map 做 observation overlay；用 EE state 构造方向性 RoI | 模型自身 attention + EE state | 低 | ✗ |
| **Explicit Grounding (CoT bbox)** | 在文本 CoT 中先预测 bounding box / 关键点坐标，再输出动作 | 语言推理 | 中（增加 token） | ✓ |

### 1.2 表征对齐 / 蒸馏：向外部 Foundation Model 学习

通过将 VLA 中间层视觉特征对齐到外部 foundation model 的特征来注入特定维度的知识。

| 方法 | 核心策略 | 对齐目标 | 注入知识维度 | 推理开销 | 是否需训练 |
|------|---------|---------|:----------:|:------:|:---:|
| **Spatial Forcing** (ICLR'26) | 中间层 visual token 与 3D foundation model (如 VGGT/DPT) 的几何特征做 cosine alignment | 3D FM | 空间/几何 | **零**（teacher 仅训练时用） | ✓ (finetune) |
| **Dynamic Foresight** (NeurIPS'26 sub) | 中间层 visual token 与 video foundation model 的时序特征做对齐（帧内相似+帧间判别） | Video FM | 时序/动态 | **零** | ✓ |
| **DeepVision-VLA** (Mar'26) | Vision-Language Mixture-of-Transformers: 将 vision foundation 多尺度特征注入 VLA 深层 | Vision FM | 多尺度视觉 | 中（额外 FM forward） | ✓ |

### 1.3 辅助生成 / 重建监督：通过副任务逼注意力 grounding

添加辅助生成任务，间接迫使模型分配正确的视觉注意力。

| 方法 | 核心策略 | 重建目标 | 标签来源 | 推理开销 |
|------|---------|---------|---------|:------:|
| **ReconVLA** (Aug'25) | Diffusion transformer 重建 gaze region 的 tokenized image | 目标物体裁剪图 | GroundingDINO / YOLO 框出 gaze region | **零**（推理时 recon head 可丢弃） |
| **ViDAL** (NeurIPS'26 sub) | Action VAE 重建 action chunk，同时 latent 对齐 future scene dynamics | 未来场景动态 | 未来帧（demo 数据） | **零**（Action VAE 仅训练时用） |
| **ΔVLA** (Mar'26) | 建模 world-knowledge variation（语义/深度/区域变化）相对于当前先验 | 变化量 | 当前/未来帧差 | 低 |

### 1.4 注意力监督 / 正则：用外部信号直接调整 attention 分布

直接监督模型的 cross-attention 分布向特定空间目标收敛。

| 方法 | 核心策略 | 监督信号来源 | 推理开销 | 是否需训练 |
|------|---------|:----------:|:------:|:---:|
| **Gaze-Regularized VLA** (Mar'26) | KL divergence 对齐 transformer attention 到人类注视热图 | **人类 eye-tracking 数据** | **零** | ✓ |
| **GABRIL** (Jul'25) | Gaze mask 正则化 BC agent attention，缓解因果混淆 | **人类 eye-tracking 数据** | **零** | ✓ |
| **AutoFocus-IL** (Nov'25) | 用 VLM 自动生成时序 saliency map → 正则 BC 策略的 attention | **外部 VLM saliency** | **零** | ✓ |
| **GuidedVLA** (May'26) | 手动指定 task-relevant factors → 特化 action decoder 的注意力头 | **人工指定因素** | 低 | ✓ |
| **FocusVLA** (Mar'26) | Modality Cascaded Attention: 强制 language→vision→action 顺序交互 + Focus Attention 抑制无关 token | **架构先验**（无外部信号） | 低 | ✓ |

### 1.5 注意力驱动的 Token Pruning / Caching（效率导向）

通过识别并保留 task-relevant visual token 来加速推理。核心发现：语义显著 ≠ 动作相关。

| 方法 | 核心策略 | 关键发现 | 加速比 |
|------|---------|---------|:------:|
| **VLA-Pruner** (Feb'26) | 双层 token 选择：语义层（prefill attention）+ 动作层（decode attention 时序平滑） | **semantic salience ≠ action relevance; 50% 保留率可提升 SR** | 1.99× |
| **TIES** (Mar'26) | 跨层 token 排名一致性（τ-guided）代替单纯 attention magnitude | **高注意力 token 是任务相关的，甚至会损害策略性能** | 动态 |
| **DTP** (Jan'26) | 检测并剪除"干扰 token" | **pruning distractors 提升成功率** | ~1.3× |
| **BFA++** (Feb'26) | 多视角分层 token pruning：局部视角内 + 全局跨视角 | 视角间重要性动态变化 | 提升 |
| **SP-VLA** (Jun'25) | 空间 token 分类（spatial/semantic）+ 双感知重要性剪枝 | +6% SR | 1.5× |

### 1.6 因果混淆修正：消除虚假视觉依赖

| 方法 | 核心策略 | 干预方式 | 类型 |
|------|---------|---------|------|
| **Causal-ACT** (Jul'25) | Transformer IL 中集成因果图优化 | 数据层：学习忽略无关 observation 成分 | 训练时 |
| **PCD** (May'25) | Policy Contrastive Decoding: 对比原始/扰动观测的输出差异 | 解码层：test-time contrastive decoding | 推理时 |
| **RoCoDA** (Nov'24) | 统一 invariance + equivariance + causality 的反事实数据增强 | 数据层：swap 无关实体 | 训练时 |
| **CAIAC** (ICML'24) | Causal Action Influence Aware Counterfactual Data Augmentation | 数据层：基于 CAI 检测 action-independent 实体并 swap | 训练时 |

### 1.7 视觉-本体感觉模态竞争

| 方法 | 核心策略 | 发现 |
|------|---------|------|
| **GAP** (ICLR'26) | Gradient Adjustment with Phase-guidance: 按 phase 调节本体感觉梯度幅度 | **策略倾向于更快降 loss 的本体感觉信号 → 压制视觉模态学习**；修正后视觉泛化大幅提升 |

### 1.8 深层视觉衰减现象

多篇论文独立发现同一现象：

| 方法 | 发现 |
|------|------|
| **DeepVision-VLA** | 在 vanilla VLA 中，对 task-relevant visual token 的 attention 随深度逐渐减弱 |
| **HuggingFace 系统分析** | 多种 action-generation 范式下，深层对视觉 token 的敏感度逐层递减 |
| **ReconVLA** | 实测当前 VLA 的视觉注意力总是弥散的，不集中在目标区域 |

---

## 二、深度分析：现有工作的根本盲区

### 盲区 1: 所有注意力监督方法都依赖"外部 where-to-look 信号"

当前注意力修正的核心 pipeline 是：**外部信号源 → 定义"该看哪" → 监督 attention → 训好后推理不变**。

问题在于信号源的选择：

| 信号源 | 使用者 | 代价 | 精度 | 是否 phase-tracking |
|-------|--------|------|------|:---:|
| 人类注视 | Gaze-Reg, GABRIL | **极高**（需 eye-tracking 设备+人工采集） | 中（注视点 ≠ 最优注意力） | ✗ |
| 外部检测器 | ReconVLA | 高（需 GroundingDINO/YOLO） | 中（框物体中心 ≠ 接触区域） | ✗（框的是静态物体位置） |
| 外部 VLM | AutoFocus-IL | 中（需额外 VLM forward） | 中（VLM saliency 是语义的） | 部分 |
| 人工指定 | GuidedVLA | 高（需人工定义 task-relevant factors） | 高 | ✗ |
| 架构先验 | FocusVLA | 零 | 不确定（依赖架构设计是否正确） | ✗ |

> **核心盲区**: demonstration 数据中白送着一个比以上所有信号源都更精确、更廉价、更动态的注意力标签——**动作标签本身**——但没有任何一篇论文将其用于注意力监督。

动作标签 a*_t + 当前 EE 位姿 + 相机参数 → FK + 投影 → 图像坐标 (u, v)，这个点精确指向"夹爪下一刻要到的位置"：

- **免费**: 每个 demo 都有动作标签和相机参数
- **精确**: FK + 相机投影是数学精确的（在 sim 中完全精确，real 中误差可控）
- **Phase-tracking**: reach 时指向目标物体、grasp 时指向接触面、transport 时自动漂移到放置目标——全程无需 phase detection
- **Control-relevant**: 指向的是"EE 要到达的位置"（control-relevant sub-region），而非"物体中心"（semantically-relevant region）

### 盲区 2: "语义 grounding" ≠ "控制 grounding"——被反复证实但未被正面解决

VLA-Pruner 和 TIES 从 token pruning 的角度独立发现：**语义显著的 token 和动作相关的 token 是不同的集合**。VLA-Pruner 明确指出：基于语义显著性（prefill attention）做 token 选择会"偏向语义线索，丢掉对动作生成关键的信息，严重损害 VLA 性能。"

GAP 从模态竞争的角度发现了同一个 root cause：策略在训练时倾向于走"最便宜路径"——优先拟合简洁的本体感觉信号，从而压制了对精细视觉证据的学习。

**但没有任何一篇论文从"监督注意力指向 control-relevant 区域而非 semantically-relevant 区域"这个角度来正面解决问题。** 所有注意力监督方法（ReconVLA、Gaze-Reg、AutoFocus）给出的都是 **语义级目标**（物体框 / 注视热点 / VLM saliency），不是 **控制级目标**。

> 动作投影点天然是 control-relevant 的：它不指向"杯子中心"，它指向"夹爪将要闭合的接触面"。这正好填补了"semantic grounding ≠ control grounding"这个被反复确认的 gap。

### 盲区 3: 注意力监督教的是"读题策略"，不是"答案"——与 action loss 不冗余

一个可能的质疑是：动作 loss 已经提供了学习信号，为什么还需要注意力监督？

分析如下：
- **Action loss** 告诉模型"正确答案是什么"（输出层信号），对 attention 权重的梯度需要穿过整个 decoder 才能传回来——极其间接且微弱
- **Attention supervision** 直接告诉模型"该从哪里读取信息来做决策"（attention 层信号），梯度直接作用在 attention 权重上——强且精准

类比：告诉学生"答案是 42"（action loss）vs 告诉学生"关键条件在题目第三行"（attention supervision）。前者帮助记住这道题，后者教会可迁移的读题策略。

这也解释了为什么 ReconVLA、Gaze-Reg 等方法虽然"只是调了 attention"就能获得 5-15% 的成功率提升——因为 VLA 的主要失败模式不是"不会做"，而是"没看对地方"。

### 盲区 4: 现有方法在信号的 phase-tracking 能力上存在根本缺陷

操作任务的注意力需求是**相位依赖**的：

| 阶段 | 应该看哪 | 检测器(ReconVLA) | 人类注视(Gaze-Reg) | 动作投影(本方法) |
|------|---------|:---------:|:----------:|:--------:|
| Reach | 目标物体 | ✓（框物体） | ✓ | ✓（投影指向目标方向） |
| Pre-grasp | 接触面/夹持位 | ✗（框整个物体） | 部分 | **✓（投影精确到接触区域）** |
| Grasp | 接触几何 | ✗ | 部分 | **✓** |
| Transport | 放置目标 | ✗（仍框抓着的物体） | ✓ | **✓（自动切换到目标位置）** |
| Place/Insert | 对齐边缘/插孔 | ✗ | 部分 | **✓** |

**关键差异**: 检测器在整个 episode 中框的是同一个物体（语义不变），而操作的注意力需求是阶段性变化的。动作投影天然跟踪这种变化。

---

## 三、核心方法：APSG-VLA

### 3.1 总体框架

APSG-VLA 是一个 **训练期辅助 loss、推理期零改动** 的 VLA 视觉注意力增强框架。其核心思想是：用动作标签的图像投影作为 cross-attention 的空间监督目标，教会 VLA "在产生动作时从 control-relevant 区域读取视觉信息"。

```
                     Training Pipeline
                     =================

  ┌──────────────┐      ┌─────────────────────────────────┐
  │ Observation  │      │   Action Label a*_{t:t+H}       │
  │   o_t        │      │   (ground-truth action chunk)    │
  └──────┬───────┘      └──────────┬──────────────────────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐      ┌─────────────────────────────────┐
  │ VLA Forward  │      │  FK + Camera Projection         │
  │ (正常流程)   │      │  EE_pose + a* → p* = (u*, v*)   │
  └──────┬───────┘      └──────────┬──────────────────────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐      ┌─────────────────────────────────┐
  │ Cross-Attn   │      │  Gaussian Target M*             │
  │ Weights ā    │      │  centered at p*, width σ        │
  │ (act→vis)    │      │  σ = α·d + σ_min                │
  └──────┬───────┘      └──────────┬──────────────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
              ┌───────────────┐
              │  L_attn =     │
              │  KL(ā ∥ M*)   │
              └───────┬───────┘
                      │
                      ▼
         L_total = L_action + β · L_attn


                     Inference Pipeline
                     ==================

            [ 完全不变，零改动，零开销 ]
            模型正常推理 → attention 权重已被训练塑造
            → 自然更集中在 control-relevant 区域
```

### 3.2 组件 1: Action-to-Image Projection（动作→图像投影）

#### 3.2.1 投影目标的选择

**关键设计决策**: 投影的不是 next-step action delta（太近、可能落在空气中），而是 **K-step look-ahead 的未来 EE 位置**。

对于 **action chunking 架构**（π0、π0.5、GR00T、Diffusion Policy）：
- 投影 chunk 末端位置：$p^* = \text{project}(\text{FK}(s_t + \sum_{i=0}^{H-1} a^*_{t+i}), T_{cam}, K_{cam})$
- 其中 $H$ 为 chunk length（π0.5: H=50）

对于 **非 chunking AR 架构**（OpenVLA、OpenVLA-OFT）：
- 投影 demo 轨迹的 K-step look-ahead：$p^* = \text{project}(\text{FK}(s_{t+K}), T_{cam}, K_{cam})$
- 推荐 $K = 10 \sim 20$（约 1~2 秒前瞻）

#### 3.2.2 投影计算

输入：
- 当前 joint state $s_t$
- Action label chunk $a^*_{t:t+H}$（或轨迹中 $s_{t+K}$）
- Camera extrinsics $T_{cam} \in \mathbb{R}^{4 \times 4}$（robot base → camera frame）
- Camera intrinsics $K_{cam} \in \mathbb{R}^{3 \times 3}$

计算链路：

$$\text{EE}_{world} = \text{FK}(s_t + \sum_{i=0}^{H-1} a^*_{t+i})$$

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \propto K_{cam} \cdot T_{cam} \cdot \begin{bmatrix} \text{EE}_{world} \\ 1 \end{bmatrix}$$

映射到 ViT patch grid 坐标：

$$r^* = \frac{v}{\text{patch\_size}}, \quad c^* = \frac{u}{\text{patch\_size}}$$

**各场景可行性**：

| 场景 | FK | Camera Matrix | 投影精度 |
|------|:---:|:---:|:---:|
| Sim (LIBERO/CALVIN/RoboTwin) | ✓ 精确 | ✓ 精确 | **Exact** |
| Real (Bridge/DROID/ALOHA) | ✓（DH 参数已知） | ✓（已标定） | **高（~3-5 pixel 误差，在 Gaussian σ 容许范围内）** |
| Real（无标定） | ✓ | ✗ | 训练一个 tiny MLP: $f(EE_{pose}) \to (u, v)$，用 gripper 检测 pair 监督，几百样本即可 |

#### 3.2.3 投影点在各操作阶段的表现

| 阶段 | Chunk 末端投影落点 | 作为 attention target 的质量 |
|------|:---:|:---:|
| **Reach（前期）** | 在当前 EE 和目标之间，偏向目标方向 | **中等**：方向正确但不在物体上。但 reach 阶段 VLA 极少失败（动作简单），attention 精度要求低；配合宽 σ Gaussian，尾巴可覆盖目标 |
| **Reach（后期）/ Pre-grasp** | 接近或落在目标物体表面 | **好** |
| **Grasp** | 接触面 / 夹持位 | **很好**：精确到 sub-object 级别 |
| **Transport** | 放置目标附近 | **好**：自动切换到 place target |
| **Place / Insert** | 对齐面 / 插入孔位 | **很好**：最精确，且恰恰是精度最关键的阶段 |

**核心论点**：投影在"不太精确的时候（reach 前期）恰好是精度不太重要的时候，在最精确的时候（grasp/insert）恰好是精度最关键的时候"。这个自然的 phase-precision alignment 是 action 投影信号独有的属性。

### 3.3 组件 2: Adaptive Gaussian Attention Target

在 ViT patch grid（$H_p \times W_p$，如 $16 \times 16 = 256$ tokens）上构造 2D Gaussian 分布：

$$M^*_{i,j} = \exp\left(-\frac{(i - r^*)^2 + (j - c^*)^2}{2\sigma^2}\right)$$

归一化为概率分布：$\hat{M}^* = M^* / \sum_{i,j} M^*_{i,j}$

#### 自适应 σ

σ 与 EE-to-target 距离成正比：

$$\sigma = \alpha \cdot d + \sigma_{\min}$$

其中 $d = \|\text{EE}_{chunk\_end} - \text{EE}_{current}\|_2$（chunk 内总位移量）。

| 阶段 | d（典型值） | σ（推荐 α=3.0, σ_min=1.5） | 效果 |
|------|:---------:|:----:|------|
| Reach | ~30cm | ~5 patches | 宽注意力，覆盖目标物体附近大区域 |
| Pre-grasp | ~5cm | ~2 patches | 中等聚焦 |
| Grasp/Insert | ~1cm | ~1.5 patches | 精确锁定接触区域 |

**设计直觉**：远的时候"大范围扫视"（知道大概在哪），近的时候"精确凝视"（看清接触几何）。和人类操作时的视觉策略一致。

### 3.4 组件 3: Cross-Attention Supervision Loss

#### 提取 cross-attention 分布

**π0 / π0.5 / GR00T（FM-VLA，有独立 action expert）**：

Action expert 的 cross-attention 层中，action queries 对 visual tokens 的 attention weights：
$$A_l \in \mathbb{R}^{N_{heads} \times N_{action\_queries} \times N_{visual\_tokens}}$$

对 heads 和 action queries 取平均 → 空间 attention 分布：
$$\bar{a} = \frac{1}{N_h \cdot N_q} \sum_{h,q} A_{l,h,q,:} \in \mathbb{R}^{N_{vis}}$$

reshape 为 $(H_p, W_p)$ → 归一化为分布 $\hat{a}$。

**OpenVLA / OpenVLA-OFT（AR-VLA，统一 self-attention）**：

从 causal self-attention 中提取 action token → visual token 的子矩阵：
$$A_{act \to vis} \in \mathbb{R}^{N_{heads} \times N_{action\_tokens} \times N_{visual\_tokens}}$$

同样平均、reshape、归一化。

#### Loss 函数

$$\mathcal{L}_{attn} = \text{KL}(\hat{a} \| \hat{M}^*)$$

选 KL 而非 MSE 的理由：KL 作用在概率分布上，不需要 attention 的绝对值和 Gaussian 的绝对值匹配，只约束分布的**形状**。

#### 监督层的选择

基于 DeepVision-VLA 的发现（attention to visual tokens 在深层衰减），推荐监督策略：

- **第 1 个 cross-attention 层**: 提供初始空间 grounding seed
- **最后 1~2 个 cross-attention 层**: 直接对抗深层衰减

中间层不监督——给模型自由度学习中间表征。

#### 总训练 Loss

$$\mathcal{L} = \mathcal{L}_{action} + \beta \cdot \mathcal{L}_{attn}$$

$\beta$ 推荐范围 $[0.01, 0.1]$，初始建议 $\beta = 0.05$。

### 3.5 组件 4: 越界处理与多目标扩展

#### 投影越界

若投影点 $(u, v)$ 落在图像范围外（$u < 0$ 或 $u > W$ 或 $v < 0$ 或 $v > H$）：
- 该时间步 **跳过** $\mathcal{L}_{attn}$，只保留 $\mathcal{L}_{action}$
- 在 LIBERO / CALVIN 标准第三人称相机下，工作区基本在视野内，触发率 < 5%

#### 多目标任务

对于"把 A 放到 B 上"类任务：
- Reach→Grasp 阶段：chunk 末端指向 A → attention 在 A
- Transport→Place 阶段：chunk 末端指向 B → attention 自动切到 B
- 时序上两个目标都被覆盖
- 单一时间步内只看一个目标——是否需要同时看两个？经验上 VLA 的主要失败模式是执行精度不够（grasp/place 时看错位置），不是找不到第二个目标。Gaussian 的宽 σ 尾巴在远距阶段可部分覆盖第二目标

#### 双臂操作

双臂（ALOHA）：两个 EE → 两个投影点 → 构造两个 Gaussian 的 mixture：

$$\hat{M}^* = \frac{1}{2}(\hat{M}^*_{left} + \hat{M}^*_{right})$$

### 3.6 组件 5: 推理不变性论证

**推理时完全不修改模型**。Attention 权重已在训练中被塑造——面对类似观测时，cross-attention 自然更集中在 control-relevant 区域。

这和 ReconVLA 的逻辑完全一致：ReconVLA 训练时加了 diffusion reconstruction head，推理时把 head 丢掉——attention 已经被训好了。我们更简单——连额外的 head 都没有，只有一个 loss。

**为什么泛化到新场景？** attention supervision 在所有训练样本上给出一致的模式（"attend to action target region"），但每个样本里这个区域在图像上的位置不同。网络学到的不是"固定看左上角"，而是 conditional mapping：$(observation, instruction) \to attention\ region$。这个 mapping 的泛化性和 action policy 本身的泛化性同源。

---

## 四、数学推导与理论分析

### 4.1 Attention 如何影响 Action 精度：Positional Encoding 通道

VLA 的 visual tokens 携带两类信息：

$$v_i = f_i + \text{PE}_i$$

其中 $f_i$ 为 patch 的视觉特征（语义/几何），$\text{PE}_i$ 为位置编码（空间坐标）。

当 action expert cross-attend 到 visual tokens 时，attention-weighted sum 为：

$$z_{action} = \sum_i \hat{a}_i \cdot (f_i + \text{PE}_i) = \underbrace{\sum_i \hat{a}_i \cdot f_i}_{\text{特征聚合}} + \underbrace{\sum_i \hat{a}_i \cdot \text{PE}_i}_{\text{位置聚合}}$$

当 attention 集中在 target patch $i^*$ 时：

$$z_{action} \approx f_{i^*} + \text{PE}_{i^*}$$

即 action 表征被 target 的**特征**和**位置**主导。位置信息直接编码了"目标在图像的哪里"，这对产生空间上对齐的动作有直接帮助。

### 4.2 Attention Supervision vs Action Loss 的梯度通路分析

设 action loss 对第 $l$ 层 cross-attention 权重 $A_l$ 的梯度为：

$$\frac{\partial \mathcal{L}_{action}}{\partial A_l} = \frac{\partial \mathcal{L}_{action}}{\partial a} \cdot \frac{\partial a}{\partial z_{L}} \cdot \prod_{k=l+1}^{L} \frac{\partial z_k}{\partial z_{k-1}} \cdot \frac{\partial z_l}{\partial A_l}$$

其中 $L$ 为 decoder 总层数。这个梯度链长度为 $O(L - l)$，且每层的 Jacobian $\partial z_k / \partial z_{k-1}$ 可能引入梯度衰减。

而 attention supervision 的梯度：

$$\frac{\partial \mathcal{L}_{attn}}{\partial A_l} = \frac{\partial \text{KL}(\hat{a}_l \| \hat{M}^*)}{\partial A_l}$$

**直接作用于 $A_l$**，链长为 $O(1)$。信号强度差异至少一个数量级。

这解释了为什么 action loss 本身不足以训好 attention——梯度太间接；也解释了为什么加一个简单的 KL loss 就能产生显著效果——梯度直达。

### 4.3 Control-Relevant vs Semantically-Relevant Grounding

设物体中心在 patch $i_{sem}$，接触面在 patch $i_{ctrl}$（$i_{sem} \neq i_{ctrl}$）。

检测器/VLM 产生的监督：$M^*_{det}$ 以 $i_{sem}$ 为中心 → attention 被拉向物体中心。
动作投影产生的监督：$M^*_{act}$ 以 $i_{ctrl}$ 为中心 → attention 被拉向接触区域。

在 grasp/insert 阶段，$i_{ctrl}$ 处的视觉特征包含接触几何信息（边缘、曲率、对齐），直接决定动作精度。$i_{sem}$ 处的特征偏向语义（物体类别、颜色），对精密动作的帮助较弱。

因此 control-relevant grounding 在精密操作阶段提供更有价值的视觉输入。

### 4.4 与 "直接喂 target position 当 conditioning" 的不等价性论证

一个可能的替代方案是：把投影位置 $(u, v)$ 直接作为 action expert 的额外输入 conditioning。

但这与学习 action 本身等价——action label 编码了"去哪里"，给模型 target position 就是把答案以另一种形式再喂了一遍。训练时冗余，测试时循环（用模型自己的预测作为输入 → circular）。

Attention supervision 不等价于 action loss，因为它教的是 **HOW to read**（从哪些 patch 读信息），不是 **WHAT to output**（正确的动作向量）。它传递的是一种可迁移的 **视觉读取策略**，而非答案。

---

## 五、算法伪代码

```python
# ============================================================
# APSG-VLA: Training Pipeline
# ============================================================
# 输入:
#   policy: 预训练 VLA 模型 (π₀/π₀.5/OpenVLA-OFT/GR00T)
#   dataset: demonstration 数据集 (obs, action, ee_pose, cam_params)
#   β: attention loss 权重 (default 0.05)
#   α, σ_min: Gaussian 宽度参数 (default α=3.0, σ_min=1.5)
#   K_lookahead: AR VLA 的 look-ahead 步数 (default 15)
#   supervised_layers: 被监督的 cross-attention 层索引
# ============================================================

def apsg_train_step(policy, batch, β=0.05, α=3.0, σ_min=1.5):
    obs, actions, ee_poses, cam_params, instructions = batch
    
    # --------------------------------------------------
    # Step 1: 计算 Action-Projected Attention Target
    # --------------------------------------------------
    # 确定投影目标: chunk 末端 or K-step look-ahead
    if policy.uses_action_chunking:
        # 用 action chunk 末端的 EE 位置
        ee_target = forward_kinematics(
            ee_poses + cumsum(actions[:, :chunk_len, :3], dim=1)
        )[:, -1, :]  # [B, 3], chunk 末端 EE position
    else:
        # 用 K-step look-ahead 的 EE 位置 (从 demo 轨迹直接取)
        ee_target = ee_poses_at_t_plus_K  # [B, 3]
    
    # 3D → 2D 投影
    proj_points = project_to_image(ee_target, cam_params)  # [B, 2] (u, v)
    patch_coords = proj_points / patch_size  # [B, 2] (r*, c*)
    
    # 检查越界
    in_bounds = (patch_coords[:, 0] >= 0) & (patch_coords[:, 0] < H_p) & \
                (patch_coords[:, 1] >= 0) & (patch_coords[:, 1] < W_p)  # [B]
    
    # 自适应 σ
    displacement = torch.norm(ee_target - ee_poses[:, :3], dim=-1)  # [B]
    sigma = α * displacement + σ_min  # [B]
    
    # 构造 Gaussian attention target
    grid_r = torch.arange(H_p).float()  # [H_p]
    grid_c = torch.arange(W_p).float()  # [W_p]
    grid_r, grid_c = torch.meshgrid(grid_r, grid_c, indexing='ij')  # [H_p, W_p]
    
    # [B, H_p, W_p]
    M_star = torch.exp(
        -((grid_r - patch_coords[:, 0:1, None])**2 + 
          (grid_c - patch_coords[:, 1:2, None])**2) / 
        (2 * sigma[:, None, None]**2)
    )
    M_star = M_star / M_star.sum(dim=(-2, -1), keepdim=True)  # 归一化
    M_star = M_star.reshape(B, -1)  # [B, N_vis]
    
    # --------------------------------------------------
    # Step 2: 正常 VLA Forward + 提取 Attention Weights
    # --------------------------------------------------
    action_pred, attn_weights_dict = policy.forward_with_attn(
        obs, instructions, return_attn=True
    )
    # attn_weights_dict: {layer_idx: [B, N_heads, N_queries, N_vis]}
    
    # --------------------------------------------------
    # Step 3: 计算 Losses
    # --------------------------------------------------
    # Action loss (标准 BC / flow matching loss)
    L_action = policy.action_loss(action_pred, actions)
    
    # Attention loss (仅在投影有效的样本上)
    L_attn = 0.0
    for l_idx in supervised_layers:
        attn_l = attn_weights_dict[l_idx]  # [B, N_h, N_q, N_vis]
        attn_l = attn_l.mean(dim=(1, 2))  # [B, N_vis]
        attn_l = attn_l / (attn_l.sum(dim=-1, keepdim=True) + 1e-8)
        
        # KL: only for in-bounds samples
        kl = F.kl_div(
            torch.log(attn_l[in_bounds] + 1e-8),
            M_star[in_bounds],
            reduction='batchmean'
        )
        L_attn += kl
    
    L_attn /= len(supervised_layers)
    
    # Total loss
    L_total = L_action + β * L_attn
    
    return L_total


# ============================================================
# APSG-VLA: Inference Pipeline
# ============================================================

def apsg_inference(policy, obs_stream):
    """推理时完全不变——零改动、零额外开销"""
    for obs_t in obs_stream:
        action_t = policy.predict(obs_t)  # 标准 VLA forward
        execute(action_t)
```

---

## 六、与现有方法的全面对比

### 6.1 多维度对比表

| 维度 | ReconVLA | Gaze-Reg | AutoFocus-IL | GABRIL | GuidedVLA | Spatial Forcing | Dynamic Foresight | TraceVLA | **APSG (Ours)** |
|:-----|:--------:|:--------:|:------------:|:------:|:---------:|:--------------:|:-----------------:|:--------:|:--------:|
| **信号来源** | 检测器 (GDINO) | 人类注视 | 外部 VLM | 人类注视 | 人工指定 | 3D FM | Video FM | CoTracker | **Action label + FK + cam** |
| **信号代价** | 高 | 极高 | 中 | 极高 | 高 | 中 | 中 | 中 | **零（数据自带）** |
| **信号精度** | 物体级 | 注视点级 | 语义级 | 注视点级 | 因素级 | 几何级 | 动态级 | 轨迹级 | **Control-relevant 精度** |
| **Phase-tracking** | ✗ | 部分 | 部分 | 部分 | ✗ | ✗ | ✗ | ✓(past) | **✓(future, 自动)** |
| **Semantic vs Control** | Semantic | Semantic | Semantic | Semantic | Manual | 3D | Temporal | Motion | **Control** |
| **推理开销** | 零(丢 head) | 零 | 零 | 零 | 低 | 零 | 零 | 中(CoTracker) | **零** |
| **训练额外模块** | Diffusion head | 无 | 无 | 无 | Head specialization | MLP projector | Dynamic decoder | Finetune | **无（仅 loss）** |
| **架构无关** | 部分 | ✓ | ✓ | ✓(BC only) | 部分 | ✓ | ✓ | 部分 | **✓** |
| **需要额外数据** | ✗ | ✓(eye-tracking) | ✗ | ✓(eye-tracking) | ✗ | ✗ | ✗ | ✗ | **✗** |
| **与 Spatial Forcing / Dynamic Foresight 的关系** | — | — | — | — | — | 正交 | 正交 | — | **正交（可叠加）** |

### 6.2 关键差异化分析

**vs ReconVLA（最直接竞品，同属"训练期注意力监督"范式）**:
- ReconVLA 的信号来自 GroundingDINO/YOLO → 框整个物体（语义级）；APSG 来自动作投影 → 指向接触区域（控制级）
- ReconVLA 需要训练一个额外的 diffusion transformer head；APSG 只加一个 KL loss，零额外参数
- ReconVLA 的 gaze region 是静态的（整个 episode 框同一个物体）；APSG 自动跟踪操作阶段
- ReconVLA 需要在数据预处理阶段跑检测器；APSG 只需 FK + 相机矩阵乘法

**vs Gaze-Regularized VLA / GABRIL（同属"KL 正则 attention"机制）**:
- 机制相同（KL 对齐 attention 到 spatial target）——区别纯在信号源
- Gaze-Reg/GABRIL 需要人类 eye-tracking 数据 → 采集成本极高、不可规模化
- APSG 的信号零成本、精确、自动

**vs TraceVLA（都用了"robot motion trajectory"信息）**:
- TraceVLA: **过去**轨迹 → 画在**图上当输入** → 需要 CoTracker → 改输入、加推理开销
- APSG: **未来**动作投影 → 当 **attention supervision** → 只需 FK+cam → 不改输入、零推理开销
- TraceVLA 教模型"你从哪来"（spatial-temporal awareness）；APSG 教模型"你该看哪读信息"（visual reading strategy）
- 本质不同：input augmentation vs attention supervision

**与 Spatial Forcing / Dynamic Foresight 的互补关系**:
- Spatial Forcing 提升视觉特征的**空间/几何质量**（WHAT features encode）
- Dynamic Foresight 提升视觉特征的**时序/动态质量**（WHAT features encode）
- APSG 提升模型从特征中**读取的位置**（WHERE to read from）
- 三者完全正交，可叠加使用。APSG + Spatial Forcing = better features + reading from the right place

---

## 七、实验方案设计

### 7.1 Baseline 模型选择

| 模型 | 参数量 | Action Head | 注意力结构 | 理由 |
|------|--------|------------|-----------|------|
| **OpenVLA-OFT** | 7B | AR (discrete tokens) | Self-attention (act→vis 子矩阵) | 开源成熟、AR 代表、有 LIBERO 基线 |
| **π₀.5** | 3B | Flow Matching (10-step) | Action expert cross-attention | 当前最强开源 FM-VLA |
| **SmolVLA** | 500M | Flow Matching | Cross-attention | 轻量级快速验证 |

### 7.2 Benchmark 选择

| Benchmark | 任务数 | 评测重点 |
|-----------|:------:|---------|
| **LIBERO-Spatial** | 10 | 空间理解，baseline 成功率高 → 看增量空间 |
| **LIBERO-Object** | 10 | 物体区分，distractor 多 → 看 attention 精度提升 |
| **LIBERO-Goal** | 10 | 目标多样，语义理解 + 精度 |
| **LIBERO-Long** | 10 | 长 horizon 多阶段 → 最能体现 phase-tracking 优势 |
| **CALVIN** | 34 chains | 多任务长序列 → 评估 attention strategy 的跨任务泛化 |
| **RoboTwin** | 多 | 双臂 → 验证多 EE 投影 |
| **Real-world** (if available) | 5+ | 投影精度在有标定噪声时的表现 |

### 7.3 对比方法

| 方法 | 类型 | 对比目的 |
|------|------|---------|
| Vanilla VLA | — | 基线 |
| **ReconVLA** | 检测器驱动的注意力增强 | 直接竞品：同机制不同信号源 |
| **Spatial Forcing** | 3D 表征对齐 | 正交方法，验证叠加效果 |
| **Dynamic Foresight** | 时序表征对齐 | 正交方法，验证叠加效果 |
| **APSG (Ours)** | 动作投影注意力监督 | — |
| **APSG + Spatial Forcing** | 叠加 | 验证"WHERE + WHAT"互补性 |
| **APSG + Dynamic Foresight** | 叠加 | 同上 |

### 7.4 消融实验矩阵

| 编号 | 消融内容 | 验证目标 |
|:---:|---------|---------|
| A1 | 投影目标：chunk 末端 vs next-step delta vs K-step look-ahead (K=5,10,15,20) | 验证"必须用 look-ahead，不能用 next-step"的设计决策 |
| A2 | Gaussian σ：固定(σ=2) vs 固定(σ=4) vs 自适应(α·d+σ_min) | 验证自适应 σ 的必要性 |
| A3 | β 敏感性：β ∈ {0.01, 0.02, 0.05, 0.1, 0.2} | 确定 β 的安全范围 |
| A4 | 监督层选择：仅第1层 / 仅最后层 / 首+尾 / 全部层 | 确定最优监督位置 |
| A5 | 信号源替换：动作投影 vs 物体中心点（GroundingDINO 检测） vs 随机点 | **核心消融**：证明 control-relevant grounding > semantic grounding > random |
| A6 | 在 Spatial Forcing 基础上叠加 APSG vs 单独 Spatial Forcing | 验证 WHERE + WHAT 互补 |
| A7 | 训练数据量：5/10/25/50 demos per task | 数据效率分析 |
| A8 | 单视角 vs 多视角（第三人称 + 腕部） | 多视角下的投影 target 设计 |

### 7.5 评估指标

| 指标 | 说明 |
|------|------|
| **Task Success Rate (SR)** | 标准成功率，主指标 |
| **Phase-Specific SR** | 按操作阶段（reach/grasp/transport/place）分别统计 SR → 看哪个阶段受益最大 |
| **Attention Concentration Ratio** | 投影目标 5×5 patch 范围内的 attention 占比 → 量化 attention 聚焦程度 |
| **Control-Semantic Attention Gap** | 投影点处 attention vs 物体中心处 attention 的比值 → 量化 control vs semantic grounding 差异 |
| **Training Overhead** | 额外训练时间（预期 < 5%，仅 KL 计算 + 投影） |
| **Inference Overhead** | 应为 **0**（推理不变） |

---

## 八、风险分析与缓解策略

### 风险 1: Reach 阶段投影点不在目标物体上

**严重程度**: 中

**分析**: 对于多 chunk 的 reach 阶段，chunk 末端落在 EE 当前位置和目标之间。此时投影点指向"目标方向"但不在物体上。

**为什么不致命**:
1. Reach 阶段 VLA 极少失败——动作简单（直线移动），attention 精度需求低
2. 宽 σ 的 Gaussian 尾巴仍部分覆盖目标物体
3. 方向上的偏移（"看向目标方向"）比完全弥散（"到处乱看"）好得多
4. 使用更长 look-ahead (K=15-20) 可使投影点更接近目标

**缓解措施**:
1. AR VLA 使用 K=15~20 look-ahead（从 demo 轨迹直接取未来 EE 位置）
2. FM VLA 使用 chunk 末端（chunk 越长越接近目标）
3. 在 reach 阶段 σ 自适应地设大（>4 patches），容许宽松 grounding

**实验验证点**: 消融 A1 直接对比 next-step vs chunk-end vs K-step look-ahead

### 风险 2: Attention 调好了但成功率不提升

**严重程度**: 中

**分析**: attention 更集中在 control-relevant 区域不一定直接转化为动作精度提升——如果那个区域的 visual features 本身不够好（缺乏几何/精度信息），看对了地方也做不出精确动作。

**为什么仍然乐观**:
1. ReconVLA、Gaze-Reg、GABRIL 等已经实证证明"仅调 attention 就能提升 5~15% SR"——说明 attention 确实是 bottleneck 之一
2. 即使特征质量不完美，从 control-relevant patch 读到的信息比从 distractor patch 读到的总归更有价值
3. 和 Spatial Forcing/Dynamic Foresight 叠加可同时解决 WHERE + WHAT

**实验验证点**: 如果 APSG 单独增益 < 3%，说明 WHERE 在当前 feature quality 下不是主要 bottleneck → 转向叠加实验

### 风险 3: β 太大压制 action loss，太小没效果

**严重程度**: 低-中

**分析**: KL loss 和 action loss 的 scale 不同，β 需要调节。

**缓解措施**:
1. 推荐从 β=0.05 开始，这在 Gaze-Reg 类方法中是常见量级
2. 消融 A3 系统扫 β
3. 可考虑 loss 自动平衡（如 GradNorm），但初期不建议引入复杂性

### 风险 4: 相机标定误差（real-world）

**严重程度**: 低

**分析**: real 场景的 FK + camera projection 有误差（通常 3~10 pixel）。

**为什么可接受**:
1. Gaussian attention target 的 σ 通常 > 1.5 patches = 21+ pixels（对 224×224, patch_size=14），远大于标定误差
2. ReconVLA 用的 GroundingDINO bbox 本身也有 10-30 pixel 偏差，照样有效
3. 投影误差是随机的（非系统偏差），训练时在多样本上取平均会被"平均掉"

**兜底方案**: 对完全无标定场景，训练 tiny MLP 学习 EE_pose → (u,v) 映射，用 gripper detection 做监督

### 风险 5: 与 ReconVLA 区分度是否足以撑起独立论文

**严重程度**: 中

**分析**: 两者同属"训练期注意力监督"范式，核心区别在信号源。审稿人可能质疑"只是换了个 target"。

**反驳要点**:
1. 信号源的差异不是"换了个 label"——它代表了 "semantic grounding vs control grounding" 这一概念级区分，有独立的理论动机
2. APSG 零额外模块（无 diffusion head）、零额外数据（无检测器标注）、自动 phase-tracking——这些不是信号源换了就自动有的，是因为信号本身的性质不同
3. 消融 A5（action投影 vs 检测器框）直接证明 control grounding > semantic grounding → 如果实验撑住，这就是核心 contribution
4. Spatial Forcing 和 Dynamic Foresight 对 3D FM 和 Video FM 的区分度也主要在 "对齐到什么"而非机制本身——同类结构、不同知识维度 = 独立贡献，已被接受

**最大风险**: 如果消融 A5 显示 action投影 ≈ 检测器框（control grounding ≈ semantic grounding），则核心 claim 不成立 → 需要重新定位论文

---

## 九、实现路线图：最小可行验证

### Phase 0: 核心假设验证（1-2 天，零 GPU）

#### 实验 V1: 投影点可视化验证（3 小时）

**做什么**: 在 LIBERO-Long 的 10 个任务的 demo 轨迹上，逐帧计算 action chunk 末端（或 K-step look-ahead）的 2D 投影点，叠加到图像上生成视频。

**代码**:
```python
for task in libero_long_tasks:
    for demo in task.demos[:3]:
        for t in range(len(demo)):
            ee_target = demo.ee_poses[min(t + K, len(demo) - 1)]
            proj_point = project(ee_target, cam_matrix)
            draw_circle(demo.images[t], proj_point, color='red')
        save_video(f"{task.name}_demo_projection.mp4")
```

**预期结果**: 投影点持续跟踪 task-relevant 区域——reach 时趋向目标物体、grasp 时锁定接触面、transport 时平滑过渡到放置目标。

**通过标准**: 10 个任务中 ≥ 8 个的投影视频中，投影点在 grasp/place 阶段精确落在操作区域（目视判断）

**如果不通过**: 整个 idea 的前提（action 投影 = good attention target）不成立 → 不值得花 GPU，需要换方向

#### 实验 V2: Attention 基线分析（3 小时）

**做什么**: 在 vanilla VLA（OpenVLA-OFT 或 SmolVLA）上跑 LIBERO 的若干 demo，提取 action→visual cross-attention 权重，可视化为热力图叠加在图像上。

**目的**: 确认"vanilla VLA 的 attention 确实是弥散的 / 没有集中在投影点附近"。如果 baseline attention 已经很集中在对的位置，就不需要修了。

**通过标准**: Baseline attention 在投影点 5×5 patch 范围内的占比 < 30%（弥散）

### Phase 1: 单 backbone 验证（1 周）

- 在 OpenVLA-OFT 上实现 APSG
- LIBERO-Goal (10 tasks) 上训练 + 评测
- 消融 A1 (投影目标), A3 (β), A5 (action投影 vs 检测器框 vs 随机)
- 如果 SR 提升 ≥ 3%，进入 Phase 2

### Phase 2: 跨 backbone + 跨 benchmark（1-2 周）

- 适配到 π₀.5 / SmolVLA
- LIBERO 全 4 个 suite + CALVIN 评测
- 消融 A2, A4, A6, A7
- 叠加实验：APSG + Spatial Forcing / Dynamic Foresight

### Phase 3: 深度分析 + 论文撰写（2 周）

- Phase-specific SR 分析（哪个阶段受益最大）
- Attention 可视化对比（vanilla vs APSG vs ReconVLA）
- Control vs Semantic grounding 深度分析
- 数据效率分析（few-shot setting）
- 论文撰写

---

## 十、创新性声明与贡献总结

### 贡献 1: 发现被忽略的免费注意力监督信号（Observation-Level Contribution）

**首次指出** demonstration 数据中的动作标签可通过 FK + 相机投影转化为免费、精确、相位跟踪的视觉注意力监督信号。这个信号的存在对所有使用 demonstration data 的 VLA 方法都是适用的。

### 贡献 2: Control-Relevant vs Semantically-Relevant Grounding 的概念区分（Conceptual Contribution）

**首次将** VLA-Pruner/TIES 反复证实的"语义显著 ≠ 动作相关"这一实证发现，上升为 "control-relevant grounding vs semantically-relevant grounding" 的概念框架，并提供了第一个 control-relevant 注意力监督的具体实现。

### 贡献 3: 最轻量的注意力增强方法（Practical Contribution）

相比现有注意力增强方法，APSG 是目前已知的**最轻量**实现：
- 零额外模型参数（不像 ReconVLA 需要 diffusion head）
- 零额外推理开销（不像 TraceVLA 需要 CoTracker）
- 零额外数据（不像 Gaze-Reg 需要 eye-tracking）
- 零架构改动（不像 FocusVLA 需要重新设计 attention 结构）
- 仅增加一个 KL loss 项，所有计算（FK + 投影 + Gaussian + KL）在 CPU 上可完成

### 贡献 4: 与现有视觉增强方法的正交互补关系（Ecosystem Contribution）

明确论证了 APSG（WHERE to read）与 Spatial Forcing（3D feature quality）/ Dynamic Foresight（temporal feature quality）的正交性，并实验验证叠加效果 → 为 VLA 视觉增强的"分层组合"范式提供证据。

---

### 预期成果

| 指标 | Vanilla | ReconVLA (估) | APSG (保守) | APSG (乐观) | APSG + SF |
|------|:-------:|:--------:|:--------:|:--------:|:--------:|
| LIBERO-Long SR | ~75% | ~82% | **79-83%** | **85%+** | **88%+** |
| LIBERO-Goal SR | ~85% | ~90% | **88-91%** | **92%+** | **93%+** |
| Grasp-phase SR | ~82% | ~87% | **87-90%** | **92%+** | — |
| Insert-phase SR | ~70% | ~77% | **76-82%** | **85%+** | — |
| 训练额外开销 | 0% | ~15% (diffusion head) | **< 3%** | < 3% | ~5% |
| 推理额外开销 | 0% | 0% | **0%** | 0% | 0% |

注: ReconVLA 数据为估计值（基于论文报告的相对增益），非直接可比。精确对比需同 backbone 同 data 复现。

### 目标投稿

- **首选**: NeurIPS 2026 / ICLR 2027
- **备选**: CoRL 2026 / ICRA 2027

---

### 核心卖点（一句话 pitch）

> 现有 VLA 注意力修正方法都在从外部"借"该看哪——借检测器、借人类眼球、借大模型。但最好的注意力标签一直藏在 demonstration 里没人用：**动作标签本身就是最精确的 gaze label**。把它投影到图像上、加一个 KL loss——零参数、零推理开销、零额外数据——VLA 就学会了"看自己要去的地方"。

---

*Document generated: 2026-05-21*
*Based on survey of 30+ VLA visual enhancement / attention design papers (2024.06 – 2026.05)*
