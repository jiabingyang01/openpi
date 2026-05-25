# IGCA 实现文档

## 非侵入式设计原则

1. 新增文件实现所有 IGCA 逻辑，已有文件最小改动
2. 通过 config 开关控制，disabled 时零开销
3. 复用 APSG 的 pattern：data transform 注入数据 → model.forward() 返回 dict → train loop 组合 loss

## 文件清单

### 新增文件
```
src/openpi/models_pytorch/igca.py              # IGCAConfig + IGCAModule + IGCAMaskTransform
scripts/precompute_libero_masks.py              # 预计算 segmentation masks（从 HDF5 demo 重放）
scripts/run_precompute_masks.sh                 # 8 GPU 并行预计算脚本
scripts/download_libero_demos.py                # 下载 LIBERO HDF5 demo 文件
scripts/visualize_masks.py                      # 可视化 mask 对齐效果
```

### 修改文件（最小改动）
```
src/openpi/models/model.py                      # Observation 加 igca_mask 字段（2 行）
src/openpi/models/pi0_config.py                 # Pi0Config 加 igca 字段（2 行）
src/openpi/models_pytorch/pi0_pytorch.py        # 构造函数 + forward() 中 IGCA loss（~30 行）
src/openpi/training/config.py                   # LeRobotLiberoIGCADataConfig + 训练配置（~60 行）
scripts/train_pytorch.py                        # 处理 IGCA dict return（~15 行）
```

## 数据准备

### Step 1: 下载 LIBERO HDF5 demo 文件
```bash
cd /DATA/disk0/yjb/projects/VLA/openpi
examples/libero/.venv/bin/python scripts/download_libero_demos.py \
    --download-dir ./data/libero_demos
```
输出 `data/libero_demos/{libero_spatial,libero_object,libero_goal,libero_10}/` 各 10 个 HDF5 文件。

### Step 2: 预计算 segmentation masks（8 GPU 并行）
```bash
bash scripts/run_precompute_masks.sh
```
输出 `data/igca_masks/episode_{idx:06d}.npz`，每个包含 `masks: uint8 [T, 256, 256]`。

预计算流程：
1. 从 HDF5 demo 读取 `init_state` + `actions`
2. 通过 action 序列 MSE 匹配 HDF5 demo → LeRobot episode
3. `sim.set_state_from_flattened(init_state)` 恢复初始场景
4. `env.step(actions[t])` 逐帧步进，获取 instance segmentation
5. 根据 `obj_of_interest` 生成二值 mask
6. 128x128 渲染 → resize 到 256x256（匹配 LeRobot 图像处理方式）

### Step 3: 验证 mask 质量
```bash
.venv/bin/python scripts/visualize_masks.py --episode 807
```
输出 `vis/` 下的关键帧图片和视频。

## 数据流

```
LeRobot Dataset (含 episode_index, frame_index)
  ↓ repack_transforms (IGCA 版本保留 episode_index, frame_index)
  ↓ data_transforms:
  │   ├── LiberoInputs (已有)
  │   ├── IGCAMaskTransform (新增) → 加载 mask，写入 data["igca_mask"] [16, 16]
  │   └── DeltaActions (已有，可选)
  ↓ normalize
  ↓ model_transforms
  ↓
data dict → Observation.from_dict() → observation.igca_mask [B, 16, 16]
```

## 模型前向

```
PI0Pytorch.forward(observation, actions)
  ├── _preprocess_observation → images, lang_tokens, state
  ├── embed_prefix → prefix_embs [B, num_vis+L, D]
  │                  (前 num_vis 个是 visual tokens, dim=1152)
  ├── embed_suffix → suffix_embs (action tokens)
  ├── transformer forward → prefix_out, suffix_out
  │
  ├── action_loss = MSE(v_t, u_t)  [标准 flow matching]
  │
  └── [IGCA] 如果 igca.enabled 且训练模式:
      ├── vis_tokens = prefix_embs[:, :num_vis, :]  # [B, 256, 1152]
      ├── igca_module(vis_tokens, observation.igca_mask)
      │   ├── phi_plus = sigmoid(attention_head(vis_tokens))  # [B, 256, 1]
      │   ├── f_obj = vis_tokens * phi_plus
      │   ├── f_bg  = vis_tokens * (1 - phi_plus)
      │   ├── h_full, h_obj, h_bg = GAP(...)  # [B, 1152]
      │   ├── contrast_loss = MSE(h_full, h_obj) + max(margin - dist(h_full, h_bg), 0)
      │   └── att_loss = MSE(phi_plus, gt_mask)
      └── return {action_loss, igca_contrast_loss, igca_att_loss, igca_lambda}
```

## Loss 组合

```
total_loss = action_loss + igca_lambda * (λ_contrast * contrast_loss + λ_att * att_loss)
```

- `igca_lambda`: warmup 系数，从 0 线性增到 1（warmup_steps 步）
- `λ_contrast = 0.01`（对比 loss 权重，参考 MGCAM）
- `λ_att = 0.1`（attention 监督权重，参考 MGCAM）
- `margin = 10.0`（triplet margin）

## 推理

推理时 `igca_mask` 为 None，IGCA loss 不计算，零开销。
模型的 attention_head 已经学会了从 visual tokens 预测 Φ+，但推理时不使用（未来可选用 Φ+ 做 visual token 调制）。

## 训练命令

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py pi0_libero_igca --exp_name=pi0_igca_lc0.01_la0.1_m10_30k
```

## 评测命令

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  RUN_ID="pi0-${suite}-igca_30k" \
  POLICY_CONFIG=pi0_libero_igca \
  POLICY_DIR=./checkpoints/pi0_libero_igca/igca_30k/30000 \
  TASK_SUITE=$suite \
  NUM_GPUS=8 \
  BASE_PORT=9000 \
  NUM_TRIALS=50 \
  bash examples/libero/run_parallel_eval.sh
  pkill -f serve_policy; sleep 5
done
```

## 配置参数

### IGCAConfig
| 参数 | 默认值 | 说明 |
|------|--------|------|
| enabled | False | 是否启用 |
| lambda_contrast | 0.01 | 对比 loss 权重 |
| lambda_att | 0.1 | attention MSE 权重 |
| margin | 10.0 | triplet margin |
| warmup_steps | 1000 | loss warmup 步数 |
| mask_dir | ./data/igca_masks | 预计算 mask 目录 |

### 训练超参（pi0_libero_igca）
| 参数 | 值 |
|------|-----|
| base model | pi0_base (PyTorch) |
| peak_lr | 5e-5 |
| decay_lr | 1e-6 |
| warmup_steps | 1000 |
| num_train_steps | 30000 |
| batch_size | 32 |
| save_interval | 5000 |
