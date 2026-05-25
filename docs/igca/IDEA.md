# IGCA: Instruction-Grounded Contrastive Attention for VLA

## 1. 问题

现有 VLA 模型（Pi0、OpenVLA 等）从整张图像统一提取视觉特征，不区分：
- 指令中提到的目标物体（要操作的）
- 参考物体（提供空间关系的）
- 无关背景（桌面、墙壁等干扰信息）

导致视觉特征被背景噪声稀释，尤其在多物体、相似物体场景下容易产生空间混淆。

## 2. 核心思路

借鉴 MGCAM（CVPR 2018）的对比注意力范式，但做出本质差异：

- **MGCAM**（ReID）：固定的 body vs background 二分，用于行人重识别
- **IGCA**（VLA）：instruction-conditioned 的物体级对比，用于机器人操作

关键创新点：
1. **Instruction-conditioned**：mask 由任务指令动态决定，同一个物体在不同指令下角色不同
2. **面向 action generation**：对比学习的目标是改善 action expert 的输入特征质量
3. **Soft attention + contrastive loss**：不硬约束 attention pattern，而是改善特征空间结构

## 3. 方法详述

### 3.1 Mask 获取

**训练时**：LIBERO 仿真器提供 `SegmentationRenderEnv`，可获取：
- `get_segmentation_of_interest(seg)` → 任务相关物体的二值 mask
- `get_segmentation_instances(seg)` → 每个物体的独立 mask

需要预处理：对所有训练 episode 预计算 segmentation mask 并存储。

**推理时**：使用训练中学到的 attention sub-net 自动预测 mask，不依赖仿真器。
（真机场景可用 SAM2 作为 fallback。）

### 3.2 对比 Attention Sub-net

在 PaliGemma 输出 visual tokens 之后、送入 action expert 之前插入：

1. 从 visual features f 生成 soft attention map：
   - Φ+ = σ(Conv(f))，sigmoid 激活，每个 patch 独立输出 [0,1]
   - Φ- = 1 - Φ+

2. 加权得到两组特征：
   - f_obj = f ⊗ Φ+（物体特征）
   - f_bg  = f ⊗ Φ-（背景特征）

3. 用 GT mask 监督 attention map：
   - L_att = MSE(Φ+, M_target)
   - M_target 是 task-relevant 物体的 mask，resize 到 patch grid 尺寸（16×16）

### 3.3 Region-Level Contrastive Loss

将三组特征 pool 成向量后做对比：
- h_full = GAP(f)          # 全局特征
- h_obj  = GAP(f_obj)      # 物体特征
- h_bg   = GAP(f_bg)       # 背景特征

对比 Loss：
```
L_contrast = ||h_full - h_obj||²
           + max(margin - ||h_full - h_bg||², 0)
```
- 拉近 h_full 和 h_obj：全局特征应主要编码物体信息
- 推远 h_full 和 h_bg：全局特征应排斥背景噪声

### 3.4 总 Loss

```
L = L_flow_matching + α · L_contrast + β · L_att
```
- L_flow_matching：标准 Pi0 flow matching loss，不变
- α = 0.01（对比 loss 权重，参考 MGCAM）
- β = 0.1（attention 监督权重，参考 MGCAM）

### 3.5 推理流程

推理时不需要 GT mask：
1. Attention sub-net 从 visual features 预测 Φ+
2. 用 Φ+ 对 visual features 做 soft modulation（可选）
3. 送入 action expert 做正常的 flow matching inference

推理开销：一个 sigmoid 卷积层 + element-wise 乘法，可忽略。

## 4. 与现有方法对比

| 方法 | 类型 | 约束对象 | 条件 | 问题 |
|------|------|---------|------|------|
| APSG | Attention KL loss | attention 分布 | 固定（EE 投影） | 破坏 multi-head 多样性 |
| Spatial Forcing | 表征对齐 loss | 中间层特征 | 固定（3D 模型） | task-agnostic |
| QDepth-VLA | Depth 预测 loss | VLM 特征 | 固定（全图 depth） | 间接 |
| **IGCA** | **对比 loss** | **特征空间距离** | **instruction-conditioned** | — |

## 5. 预期实验

### Benchmarks
- LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-10

### Baselines
- Pi0 baseline（自己从 pi0_base 训的）
- Pi0 + APSG（已有结果）
- Pi0 官方 checkpoint

### Ablation
- Contrastive loss only vs Attention supervision only vs Both
- Attention map 监督方式：MSE vs BCE
- 推理时是否用 Φ+ modulate visual features
- Loss 权重 α, β 的影响
- Mask 来源：GT mask vs learned attention vs SAM2

### Analysis
- Attention map 可视化（Φ+ 是否正确关注目标物体）
- 特征空间 t-SNE（h_obj vs h_bg 的分离度）
- 失败案例分析
