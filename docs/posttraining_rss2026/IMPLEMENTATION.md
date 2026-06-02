# RSS 2026 Post-Training Challenge —— 实现文档（活文档 / Living Doc）

> 与 [GUIDE.md](GUIDE.md) 配套。GUIDE 是"怎么做"，本文是"做到哪了 / 为什么这么做 / 要改哪些代码 / 有什么风险"。
> **每次推进都回来更新本文**（任务勾选、决策、踩坑）。
>
> 状态图例：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成 · `[!]` 阻塞/有风险
> 最近更新：2026-06-02（初版，由代码与官方仓库全量分析生成）

---

## 1. 目标与现状

- **目标**：用本仓库 pi0.5 在 RSS2026 Post-Training Challenge 跑通 训练→推理→提交；先复现 pi05 基线拿到可提交结果（保底），有余力再用 failure/HIL 数据做 post-training 冲榜。
- **现状**：本 fork 基于 upstream Physical-Intelligence/openpi + 自研改进（VGAA/APSG/IGCA/UAOR/DynaActVAE…），**尚未接入本比赛**。已有 ManipArena（pi05 + LeRobot 双臂）完整流程作模板。官方基线 = openpi 另一个 fork，新增 `yam_policy.py` + `DualYamDataConfig` + 3 个 config。
- **核心判断**：本比赛机器人是 **双臂 YAM、14 维关节空间**（≠ ManipArena 的末端位姿）。流程沿用 ManipArena，但 **transform 必须用官方 `yam_policy.py`**。移植量很小（3 步），主要时间花在训练与对接验证。

---

## 2. 关键决策记录（Decision Log）

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | 在哪做？ | **移植官方 YAM 代码进本 fork**，不另起 openpi-baseline | 本 fork 已有 ManipArena 流程、自研改进与本机环境；移植仅 3 步，避免双环境维护 |
| D2 | 模型路径 JAX vs PyTorch | **JAX（`scripts/train.py`）** | 官方基线即 JAX 路径；支持 EMA/LoRA/FSDP、直接加载 `pi05_base`；PyTorch 路径需先转权重且不支持 EMA/LoRA |
| D3 | transform | **官方 `yam_policy.py`（关节空间 + adapt_to_pi）** | 本比赛是 14 维关节；maniparena 是末端位姿，不通用 |
| D4 | 模型超参 | `Pi0Config(pi05=True)` **默认值**（action_horizon=50, discrete_state_input=True, action_dim=32, max_token_len=200） | 逐字对齐官方基线，先求复现再调优 |
| D5 | 训练配方 | `batch_size=32`、`num_train_steps=200_000`、`ema_decay=0.99`（默认）、`use_delta_joint_actions=True`、`adapt_to_pi=True`、`weight_loader=pi05_base` | 官方 challenge config 原样 |
| D6 | 路径字段 | 本 fork 用 `local_root`（非官方的 `local_files_path`），且**支持 list → ConcatDataset** | fork 的 DataConfig/data_loader 用 `local_root`；多任务赛道直接传 3 路径列表 |
| D7 | 部署 | 写 `Pi05Policy(BasePolicy)` 在 `policy_deployment` 内**进程内加载** openpi Policy，obs 透传 | `YamInputs` 推理读 `data["images"]/["state"]/["prompt"]`，与部署线格式几乎一致，无需二跳 |
| D8 | norm_stats 路径/asset_id | 让 `asset_id` 默认 = `repo_id`（官方做法） | 与官方一致；推理时 `create_trained_policy` 从 `<ckpt>/assets/<asset_id>` 读统计 |
| D9 | 自研改进是否上 | **Phase-1 不上**，走 stock pi05 基线 | 自研改进非 post-training 方法、且 pi0/LIBERO 专用，8 天内移植到 pi05+真机风险高（详见 §6） |
| D10 | 优先级 | 基线先行：先拿中间 checkpoint(40–80k) 提交保底，再训满 / 再 DAgger | 8 天窗口紧，必须有保底产物 |

**待定（需向主办/用户确认）** —— 见 §5。

---

## 3. 任务清单（Checklist）

### A. 接入与基线（必须，按序）
- [ ] A1 报名并选定赛道（Single / Multi / 两者）— 截止 5-31 已过，需补报，**阻塞提交**
- [ ] A2 下载数据集（先 3×expert-data）→ `bash scripts/download_rss2026_data.sh`
- [ ] A3 校验一份数据可被 LeRobot 读取（fps/features/tasks）
- [x] A4 复制 `yam_policy.py` 进本 fork（已复制到 `src/openpi/policies/yam_policy.py`）
- [x] A5 在 `config.py` 加 import + `DualYamDataConfig` + 4 个 TrainConfig（已加，用 `local_root`；3 单任务 + `pi05_multitask`）
- [x] A6 导入自检通过：4 个 config 均注册成功（pi05=True, ah=50, adim=32, mtl=200, bs=32, delta/adapt=True）
- [ ] A7 算 norm_stats（已并入 `train_rss2026.sh`，会自动跑；也可单独 `uv run scripts/compute_norm_stats.py <config>`）
- [ ] A8 确认有空闲 GPU；开训 `GPU_LIST="0 1 2" bash scripts/train_rss2026_all_single.sh`（Single-Task 三个）
- [ ] A9 训练健康检查：loss 下降、首个 checkpoint(如 20k/40k) 落盘
- [ ] A10 openpi 原生 serve 自测 checkpoint 能加载出动作

### B. 部署与验证（必须）
- [ ] B1 克隆 `policy_deployment`，openpi venv 补 `websockets/msgpack`
- [x] B2 写部署封装：已建 `deploy_rss2026/pi05_policy.py`（部署时复制/软链到 `policy_deployment/examples/`）
- [ ] B3 `launch.py` 起服务 → `ping.py` 校验 metadata → `smoke_test.py` 校验 actions 2 维
- [ ] B4 `check_in_sim.py --mode replay` 正常
- [ ] B5 `--mode compare` 手臂维 L2/√(N·14) ≲ 0.05；排查夹爪量纲/左右布局
- [ ] B6 分辨率/纵横比与 eval client 对齐（image_shape 决策，§5 风险）

### C. 提交（必须）
- [ ] C1 部署到主办可访问地址（直连或 nginx+TLS），配 api-key/CIDR
- [ ] C2 按官方流程提交服务地址 + key + 赛道
- [ ] C3 确认评测模式（每任务单服务 vs 单服务按 prompt 切任务）

### D. 进阶 post-training（有余力）
- [ ] D1 复制 `merge_lerobot.py`，把 expert + success-and-hil 合并重训（DAgger）
- [ ] D2 探索 failure-data 负样本利用（加权/对比/偏好）— 高风险，慎入
- [ ] D3 评估自研改进（IGCA 最有潜力但需 pi05+真机移植）

---

## 4. 需新增/修改的代码（精确）

> 路径均相对 `/DATA/disk0/yjb/projects/VLA/openpi`。fork 的 `config.py` 用别名 `_transforms`、`pi0_config`、`weight_loaders`、`_optimizer`，与官方 `DualYamDataConfig` 一致，可原样粘贴。

### 4.1 复制 `yam_policy.py`
```bash
cp /tmp/pt-baseline/src/openpi/policies/yam_policy.py \
   /DATA/disk0/yjb/projects/VLA/openpi/src/openpi/policies/yam_policy.py
```
内容要点（无需改）：`YamInputs(action_dim, adapt_to_pi, model_type)`、`YamOutputs(adapt_to_pi)`；`EXPECTED_CAMERAS=(cam_high,cam_left_wrist,cam_right_wrist)`；关节翻转 `[1,-1,1,1,1,1,1, 1,-1,1,1,1,1,1]`；夹爪 linear↔angular；推理读 `data["images"]`(dict CHW)/`data["state"]`(14)/`data["prompt"]`，输出取前 14 维。

### 4.2 `config.py` —— import（在第 27 行后加）
```python
import openpi.policies.yam_policy as yam_policy
```

### 4.3 `config.py` —— `DualYamDataConfig` 类（粘到 `LeRobotManipArenaDataConfig` 附近）
原样从官方粘贴（已确认别名一致）：
```python
@dataclasses.dataclass(frozen=True)
class DualYamDataConfig(DataConfigFactory):
    """Data class for dual-arm yam system."""
    use_delta_joint_actions: bool = True
    default_prompt: str | None = ""
    adapt_to_pi: bool = True

    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "task",
                })
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(action_dim=model_config.action_dim,
                                         adapt_to_pi=self.adapt_to_pi,
                                         model_type=model_config.model_type)],
            outputs=[yam_policy.YamOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```

### 4.4 `config.py` —— 4 个 TrainConfig（加进 `_CONFIGS` 列表）
```python
    # ===== RSS2026 Post-Training Challenge (dual-YAM, 14-dim joint space) =====
    # 单任务赛道：每个任务一个 specialist。model 用 pi05 默认值以逐字对齐官方基线。
    TrainConfig(
        name="pi05_insert-mouse-battery",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="insert-mouse-battery/expert-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_root="/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/insert-mouse-battery/expert-data",
            ),
            use_delta_joint_actions=True,
            adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=32,
        num_workers=64,
        save_interval=20_000,
        keep_period=20_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_seal-water-bottle-cap",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="seal-water-bottle-cap/expert-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_root="/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/seal-water-bottle-cap/expert-data",
            ),
            use_delta_joint_actions=True, adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000, batch_size=32, num_workers=64,
        save_interval=20_000, keep_period=20_000, fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_tower-of-hanoi-game",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="tower-of-hanoi-game/expert-data",
            base_config=DataConfig(
                prompt_from_task=True,
                local_root="/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/tower-of-hanoi-game/expert-data",
            ),
            use_delta_joint_actions=True, adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000, batch_size=32, num_workers=64,
        save_interval=20_000, keep_period=20_000, fsdp_devices=1,
    ),
    # 多任务赛道：一个 generalist 跑 3 个任务（local_root 传 list → ConcatDataset，prompt_from_task 区分任务）
    TrainConfig(
        name="pi05_multitask",
        model=pi0_config.Pi0Config(pi05=True),
        data=DualYamDataConfig(
            repo_id="challenge/multitask",
            assets=AssetsConfig(asset_id="rss2026/multitask"),   # 多源需显式 asset_id（避免按某单一 repo_id）
            base_config=DataConfig(
                prompt_from_task=True,
                local_root=[
                    "/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/insert-mouse-battery/expert-data",
                    "/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/seal-water-bottle-cap/expert-data",
                    "/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset/tower-of-hanoi-game/expert-data",
                ],
            ),
            use_delta_joint_actions=True, adapt_to_pi=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=300_000, batch_size=32, num_workers=64,
        save_interval=30_000, keep_period=30_000, fsdp_devices=1,
    ),
```
> 注意：`pi05_multitask` 用 list `local_root`，fork 的 `data_loader.py` 会对每个 root 建 `LeRobotDataset` 再 `ConcatDataset`，并用 `PromptFromLeRobotTask` 注入各任务 prompt。多源情况下 `asset_id` 必须显式给（否则会取某个 repo_id 当统计 id）。

### 4.5 部署封装 `policy_deployment/examples/pi05_policy.py`
见 [GUIDE.md §7.2](GUIDE.md) 的完整实现。要点：
- `__init__` 用 `create_trained_policy(get_config(config_name), checkpoint_dir)` 进程内加载；`_warmup()` 触发 JAX 编译。
- `metadata`：`control_mode="joints"`, `action_dim=14`, `state_dim=14`, `image_keys=[cam_high,cam_left_wrist,cam_right_wrist]`, `image_shape=[3,180,320]`(待核对), `expects_prompt=True`。
- `infer`：透传 `obs["images"]/["state"]/["prompt"]` 给 `Policy.infer`，返回 `{"actions": (50,14)}`。

### 4.6 便捷脚本（已创建）
- `scripts/download_rss2026_data.sh` — 下载数据（默认 3×expert-data；`all` 全量；支持 TASKS/SUBSETS/DATA_ROOT 覆盖）
- `scripts/train_rss2026.sh <config> [exp]` — 预检 local_root → 算 norm_stats → 训练 + 日志（GPUS/BATCH_SIZE/FSDP_DEVICES/SKIP_NORM/RESUME/WANDB/EXTRA 可调）
- `scripts/train_rss2026_all_single.sh` — 一键并行训 3 个 Single-Task（`GPU_LIST="0 1 2"`，每任务一张卡，后台 + 各自日志）
- 进阶：`cp /tmp/pt-baseline/scripts/merge_lerobot.py scripts/`（DAgger / 多任务合并）

---

## 5. 待确认问题 / 风险（Open Questions & Risks）

| # | 问题/风险 | 影响 | 处理 |
|---|---|---|---|
| Q1 | **报名是否完成、选哪个赛道** | 决定提交资格与训练量 | 向主办确认；建议至少做 Single-Task 三个（基线 60.3，相对易上分） |
| Q2 | **提交对接细节**（地址表单/eval IP/打分 trial 数/评测是真机还是仿真） | 决定 §8 怎么交 | 仅官网/邮件有；代码仓库未给。尽早邮件 `sdzhang23@m.fudan.edu.cn` |
| Q3 | **eval client 发图的分辨率/纵横比** | 错配会掉点 | metadata `image_shape` 先填 `[3,180,320]`；用 sim compare + 真机首测核对；模型内 resize_with_pad 已做 letterbox 兜底 |
| R1 | **GPU 当前被占满**（每卡~70/80GB） | 无法开训 | 训练前确认空闲卡，用 `CUDA_VISIBLE_DEVICES`；与同机其他任务协调 |
| R2 | **200k 步训不完**（8 天窗口） | 拿不到满配方 | `save_interval=20k` 多存点，先提交 40–80k 中间 checkpoint 保底 |
| R3 | **评测可能是"一个服务按 prompt 切任务"** | Single-Task 起 3 个服务 vs 1 个 | 向主办确认；若是后者，多任务 config 同样可用（按 prompt 区分） |
| R4 | **夹爪量纲/14 维布局对齐** | 直接决定能否动 | sim compare 第 6/13 维 + 左右对调是首要排查项 |
| R5 | **JAX 仅单机多卡** | 不能跨机扩 | 单机 8×A100 足够；按 config 切分卡并行 |
| R6 | **prompt 文案**（训练 task 字段 vs 评测 prompt 是否一致） | 影响泛化 | `prompt_from_task=True` 用数据集 task 文本；评测 prompt 以主办为准，必要时设 default_prompt |

---

## 6. 自研改进盘点（Phase-1 结论：先不上）

来自全量代码分析。**核心结论：用户的改进都不是 post-training/RL 方法**，而是 SFT 辅助损失或推理加速，且多为 pi0/LIBERO 专用，非 pi05/真机：

| 改进 | 是什么 | 框架 | 成熟度 | 与本赛相关性 |
|---|---|---|---|---|
| IGCA | 指令对齐对比注意力（需 GT 分割 mask） | PyTorch/pi0 | 有 config+checkpoint+正向结果（LIBERO +2~7pp） | 最有潜力，但需 pi05+真机移植，真机无 GT mask（要 SAM2，未验证） |
| APSG | 动作投影空间引导（需相机内参） | PyTorch/pi0 | 有 checkpoint，结果好坏参半 | 概念契合精密操作，但需逐相机标定，LIBERO 专用 |
| VGAA | 速度梯度注意力对齐 | JAX/pi0 | 已 wire 进 config 但**未跑过** | 未验证 |
| DynaActVAE | 隐动作空间 VLA | PyTorch/pi05 | 唯一基于 pi05+ManipArena 的 config，但**外部 VAE+数据缺失** | 不可运行 |
| UAOR | 不确定性观测回溯（推理时） | PyTorch | 粗糙、被注释、未验证 | 否 |
| SkipVLA/LazyVLA/AdaFlow | 推理加速 | PyTorch/doc | 仅加速，不提精度 | 本赛不奖励延迟，跳过 |
| FSQ tokenizer | FAST 用动作 tokenizer | JAX | upstream，AR 专用 | 与 flow-matching pi05 无关 |

→ Phase-1 走 stock pi05 基线最稳。进阶若做，优先 **DAgger（merge success-and-hil）**（最快见效、最稳），其次才考虑 failure-data 的负样本利用。

---

## 7. 代码地图（关键 file:line）

**pi05 模型**
- `src/openpi/models/pi0.py:70-101` pi05 开关/双专家/adaRMS 初始化；`:155-190` time-MLP vs state-token；`:192-328` flow-matching 损失与采样
- `src/openpi/models/pi0_config.py:21-49` 配置默认（action_dim=32, action_horizon=50, max_token_len→200, pi05/discrete_state_input）
- `src/openpi/models/model.py:39-47` `IMAGE_KEYS=(base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb)`、`IMAGE_RESOLUTION=(224,224)`；`:81-156` Observation/Actions
- `src/openpi/models/tokenizer.py:22-33` pi05 离散 state prompt 格式

**训练/配置**
- `src/openpi/training/config.py:679-770` TrainConfig 全字段；`:69-112` DataConfig；`:120-152` ModelTransformFactory(PI05 分支)；`:180-214` DataConfigFactory；`:1258-1287` `pi05_maniparena_preliminary`（流程模板）
- `scripts/train.py:194-218` 设备/mesh/checkpoint 初始化；CLI=`uv run scripts/train.py <config> --exp_name=... [--overwrite|--resume] [--fsdp_devices=n] [--batch_size=n]`
- `scripts/compute_norm_stats.py:90-135` 快速 parquet 路径（读 state/action 切 14 维，算 mean/std/q01/q99）；写 `assets_dirs/<asset_id>/norm_stats.json`
- `src/openpi/training/checkpoints.py:65-86,145-152` 产物 `params/`(EMA)/`train_state/`/`assets/`；`:20-62` resume
- `src/openpi/training/weight_loaders.py:37-54` `CheckpointWeightLoader`（加载 pi05_base）
- `src/openpi/training/data_loader.py:130-179` `local_root` 单/多源(ConcatDataset)；`:200-219` 无 norm_stats 即报错

**变换/策略/服务**
- `src/openpi/transforms.py:423 pad_to_dim`、`:433 make_bool_mask`、`Normalize/Unnormalize/ResizeImages/TokenizePrompt/PadStatesAndActions`
- `src/openpi/policies/policy_config.py:16-94` `create_trained_policy`（**推理 repack 默认空**，input=[repack?,InjectDefaultPrompt,data_transforms.inputs,Normalize,model_transforms.inputs]）
- `src/openpi/policies/policy.py:67-106` `Policy.infer`（输入 dict → 变换 → sample_actions → 反变换，返回 actions/state）
- `scripts/serve_policy.py` `policy:checkpoint --policy.config=<name> --policy.dir=<step dir>`（host 固定 0.0.0.0，--port 可配）
- `src/openpi/serving/websocket_policy_server.py` openpi 原生 WS（msgpack；首帧发 `policy_metadata`）

**官方仓库（临时克隆）**
- `/tmp/pt-baseline/src/openpi/policies/yam_policy.py`（要复制）；`config.py:280-340 DualYamDataConfig`、`:615-659` 3 个 challenge config
- `/tmp/pt-deploy/server/schema.py`（ServerMetadata/InferenceRequest/InferenceResponse 定义）、`server/policy.py`(BasePolicy)、`examples/{echo,my}_policy.py`、`scripts/{launch,ping,smoke_test}.py`、`sim/check_in_sim.py`（replay/policy/compare；bundle `sim/assets/example_slim.pkl`；模型 `dual_yam_bimanual.xml`）

---

## 8. 时间线建议（8 天，Day0 = 2026-06-02）

| 阶段 | 内容 |
|---|---|
| **Day 0–1** | A1 报名/确认赛道 + 邮件问提交细节(Q2)；A2 下载 3×expert-data；A4–A7 移植代码+算 norm_stats；确认空闲 GPU |
| **Day 1–2** | A8 起训（3 单任务+1 多任务并行）；B1–B2 写部署封装；A10 原生 serve 自测 |
| **Day 2–3** | 出首个中间 checkpoint(20k/40k)；B3–B5 ping/smoke/compare 跑通，修夹爪/布局/分辨率 |
| **Day 3–4** | **C1–C2 提交一个保底版本**（中间 checkpoint）；继续训练 |
| **Day 4–6** | 训练逼近 200k；用更好 checkpoint 覆盖提交；D1 DAgger(merge success-and-hil)重训对比 |
| **Day 6–7** | 选每任务最佳 checkpoint；多轮 sim compare 调 prompt/action_horizon；终版提交 |
| **Day 7–8** | 缓冲：处理评测反馈、稳定性/超时、备份 |

**铁律**：Day3–4 必须有一个能被评测的保底提交，之后都是增量优化。

---

## 9. 变更记录（Changelog）
- 2026-06-02（初版）：完成 openpi fork 与官方两仓库+数据集全量分析；确定移植方案（D1–D10）、代码改动（§4）、风险（§5）、时间线（§8）。
- 2026-06-02（接入落地）：已执行 A4/A5/A6/B2 —— 复制 `src/openpi/policies/yam_policy.py`；`config.py` 加 import + `DualYamDataConfig` + 4 个 TrainConfig（`pi05_insert-mouse-battery`/`pi05_seal-water-bottle-cap`/`pi05_tower-of-hanoi-game`/`pi05_multitask`）；建 `deploy_rss2026/pi05_policy.py`。CPU 导入自检通过。
- 2026-06-02（脚本封装）：新增 `scripts/download_rss2026_data.sh`、`scripts/train_rss2026.sh`、`scripts/train_rss2026_all_single.sh`，均通过 `bash -n` 语法检查 + 预检 local_root 解析验证（单任务 str / 多任务 list 均正确）。**下一步：A1 报名/赛道确认 + `bash scripts/download_rss2026_data.sh` → `GPU_LIST="0 1 2" bash scripts/train_rss2026_all_single.sh`。** 训练前确认 config 里 `local_root` 路径与空闲 GPU。
