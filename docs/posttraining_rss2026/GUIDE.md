# RSS 2026 Post-Training for Robotics —— 用 pi0.5 跑通训练/推理/提交（指导文档）

> 本文是**操作手册（runbook）**：照着一步步做就能用本仓库的 pi0.5 在该比赛上完成
> 数据准备 → 训练 → 推理服务 → 仿真验证 → 提交。
> 配套的**活文档**（决策记录、任务进度、需改的代码、风险）见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
>
> 维护人：jiabingyang01 ·  本机：`/DATA/disk0/yjb/projects/VLA/openpi`（8×A100-80GB）
> 创建：2026-06-02 ·  **⚠️ Phase-1 评测窗口截止 2026-06-10，仅剩 8 天**

---

## 0. 一页速览（TL;DR）

| 维度 | 结论 |
|---|---|
| 比赛 | RSS2026 Workshop & Challenge "Post-Training for Robotics Foundation Models" |
| 机器人 | **双臂 YAM**（Piper/Arx 类），**14 维关节空间**控制（非末端位姿） |
| 任务 | `insert-mouse-battery`、`seal-water-bottle-cap`、`tower-of-hanoi-game` |
| 赛道 | Single-Task（每任务一个 specialist）/ Multi-Task（一个 generalist 跑全部 3 个） |
| 基线模型 | **pi0.5（pi05）**，从 `gs://openpi-assets/checkpoints/pi05_base` 微调 |
| 数据 | HF `Posttraining-RFM-RSS2026/Challenge-phase1-dataset`，LeRobot v2.1，每任务有 `expert-data`/`failure-data`/`success-and-hil-data` |
| 官方代码 | 基线 [openpi-baseline](https://github.com/posttraining-for-robotics/openpi-baseline)（**就是 openpi 的 fork**）+ 部署 [policy_deployment](https://github.com/posttraining-for-robotics/policy_deployment)（WebSocket+msgpack） |
| 提交方式 | 把训练好的策略包成 WebSocket 服务，主办方的 eval client 连进来跑真机/仿真打分 |
| 我们的策略 | **把官方 YAM 适配代码移植进本 fork**（已有 ManipArena 同款流程做模板），先复现 pi05 基线拿到可提交结果，再考虑改进 |

**关键的"四件改动"**（详见 §3、§7）：
1. 复制官方 `yam_policy.py` 到本 fork；
2. 在 `config.py` 加 `DualYamDataConfig` + import + 4 个 `TrainConfig`（3 单任务 + 1 多任务），把官方的 `local_files_path=` 改成本 fork 的 `local_root=`；
3. 写一个 `Pi05Policy(BasePolicy)` 部署封装（obs 几乎可直接透传给 openpi 的 `Policy.infer`）；
4. 用 `sim/check_in_sim.py --mode compare` 验证后再提交。

**端到端命令骨架**（细节见后文）：
```bash
# 1. 下载数据
huggingface-cli download Posttraining-RFM-RSS2026/Challenge-phase1-dataset --repo-type dataset \
    --local-dir /DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset
# 2. 移植配置后，算归一化统计
uv run scripts/compute_norm_stats.py pi05_insert-mouse-battery
# 3. 训练（pi05 从 pi05_base 微调）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_insert-mouse-battery \
    --exp_name=pi05_insert-mouse-battery --overwrite --fsdp_devices=1 --batch_size=32
# 4. 起服务（部署封装）
PYTHONPATH=.:<openpi>/src <openpi>/.venv/bin/python scripts/launch.py \
    --policy examples.pi05_policy:Pi05Policy \
    --policy-kwargs config_name=pi05_insert-mouse-battery \
    --policy-kwargs checkpoint_dir=<ckpt>/40000
# 5. 仿真验证
mjpython sim/check_in_sim.py --mode compare --bundle sim/assets/example_slim.pkl --host 127.0.0.1 --port 8000
```

---

## 1. 比赛背景与规则

- **主办**：复旦/WorldEngine/USC/清华/TRI 等；RSS 2026 Workshop（2026-07-13，悉尼）。
- **两阶段**：Phase-1 所有队伍用同一数据集离线训练后评测；Phase-2 取前 3 名做"迭代 rollout 收集 + post-training"。
- **三任务计分**（每个满分 1.0，按子目标累计）：
  - Insert-Mouse-Battery：放鼠标(+0.4) / 正确极性插电池未到底(+0.4) / 完全插入(+0.2)
  - Tower-of-Hanoi：小环放对侧柱(+0.3) / 大环放中柱(+0.3) / 小环叠大环(+0.4)
  - Seal-Water-Bottle-Cap：插吸管放盖(+0.5) / 拧紧(按圈数 0~+0.5)
- **指标**：Score（进度分）/ SR（成功率）/ Average（最终排名）。评测以 5–10 次 rollout/任务做快速迭代，最终统一 trial 数复评。
- **排行榜参考**：baseline pi05 = Single-Task 60.3 / Multi-Task 52.3；当前榜首 81.1 / 16.7。
- **奖金**：总额 $20k+（两赛道各设奖）。
- **关键日期**：
  - 报名：→ 2026-05-31（**若尚未报名需尽快通过官方 Google 表单补报，并选定赛道**）
  - **Phase-1 评测窗口：现在 → 2026-06-10**
  - 6 月初公布前 3；6-10 ~ 7-05 Phase-2；7-13 Workshop。

> 提交/报名/评测 IP 等"对接细节"不在任何代码仓库里，**必须以官网/主办邮件为准**（联系人 `sdzhang23@m.fudan.edu.cn`）。本指导只覆盖技术侧"如何让策略可被评测"。

---

## 2. 必须吃透的技术事实

### 2.1 pi0.5（pi05）是什么
- 与 pi0 共用 `Pi0` 类（`src/openpi/models/pi0.py`），靠 `Pi0Config.pi05=True` 开关切换；是**连续流匹配（flow-matching）动作专家**，不是 FAST 自回归。
- pi05 相对 pi0 的差异：① 动作专家用 **adaRMSNorm** 注入时间步；② **state 离散化进 prompt**（`discrete_state_input=True`，把归一化后的 state 分 256 桶写成文本 `Task: ..., State: ...; Action:`）；③ `max_token_len=200`；④ 用**分位数归一化**（quantile，q01/q99）。
- 关键默认：`action_dim=32`（pi05_base 就是 32 维，**14 维 state/action 会被零填充到 32**，不要改成 14，否则无法加载基座权重）、`action_horizon=50`、`paligemma=gemma_2b` + `action_expert=gemma_300m`。

### 2.2 YAM 数据格式（与部署/数据集**必须严格对齐**）
- **state `[14]` float32**（关节空间，非末端）：`[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`；**夹爪归一化到 [0,1]**（0=关，1=开）。
- **action `[14]` float32**：GELLO 主臂指令关节，1:1 映射到从臂 14 DOF。
- **3 路相机**（数据集键 → 模型键）：`observation.images.cam_high → base_0_rgb`、`cam_left_wrist → left_wrist_0_rgb`、`cam_right_wrist → right_wrist_0_rgb`；数据集分辨率 **320×180（H=180,W=320）**，RGB，AV1，**60 fps**。
- `adapt_to_pi=True`：把 Yam/Piper 关节空间转到 pi 预训练空间——关节翻转掩码 `[1,-1,1,1,1,1,1, 1,-1,1,1,1,1,1]`（每臂第 2 关节取反），夹爪做 linear↔angular 变换（`yam_policy.py` 里 `_gripper_to_angular`/`_gripper_from_angular`）。
- `use_delta_joint_actions=True`：delta 掩码 `make_bool_mask(6,-1,6,-1)`（每臂 6 关节走 delta，夹爪走绝对值）。

> ⚠️ **EE 空间 ≠ 关节空间**：你之前的 ManipArena 用的是 14 维**末端位姿**（`maniparena_policy.py`）。本比赛是 14 维**关节**，必须用官方 `yam_policy.py`，**不要复用 maniparena 的 transform**。ManipArena 给我们的是**流程模板**，不是 transform。

### 2.3 部署线协议（`policy_deployment`）
- 传输：WebSocket + msgpack（numpy 扩展），图像 **CHW uint8**。
- 握手：连接后服务器先发 `ServerMetadata`（`protocol_version`/`control_mode="joints"`/`action_horizon`/`action_dim`/`state_dim`/`image_keys`/`image_shape=[C,H,W]`/`expects_prompt`）。
- 请求 `InferenceRequest`：`{"images": {cam_high/cam_left_wrist/cam_right_wrist: CHW uint8}, "state": (14,), "prompt": str?}`。
- 响应 `InferenceResponse`：`{"actions": (action_horizon, 14) float32}`（服务器自动补 `server_timing`/`request_id`）。
- **重要对齐**：① 输出夹爪必须是 **[0,1]**，不是 mm，不是关节弧度范围；② 14 维布局 `[0:6]左臂关节(rad) [6]左夹爪[0,1] [7:13]右臂关节 [13]右夹爪`。

> **好消息**：官方 `YamInputs` 推理时直接读 `data["images"]`（相机名→CHW 字典）、`data["state"]`、`data["prompt"]`，这与部署线格式**几乎完全一致**——所以部署封装基本是把 obs 透传给 openpi 的 `Policy.infer`（见 §8）。

---

## 3. 环境准备

本机已有可用 fork 与 venv，**通常无需重装**。校验/补齐：

```bash
cd /DATA/disk0/yjb/projects/VLA/openpi
cat .python-version            # 应为 3.11
ls .venv && which uv           # venv 与 uv 就绪
# 若依赖有缺失（首次或换机）：
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# pi05 推理/训练需要的关键 pin（官方基线）：jax[cuda12]==0.5.3, flax==0.10.2, lerobot(git rev), transformers==4.53.2
```

**GPU 现状（重要）**：`nvidia-smi` 显示 8×A100-80GB 但**当前每卡已占用 ~70GB**（有其他任务在跑）。开训前先确认有空闲卡：
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
# 用 CUDA_VISIBLE_DEVICES 指定空闲卡，避免抢占
```
JAX 训练**仅支持单机多卡**（不支持多进程多机）。pi05 全参微调（3B）+ AdamW + batch=32 单卡 A100-80G 可装下；建议**每个训练任务占 1–2 张空闲卡并行跑**，4 个 config 同时训以抢时间（见 §5.3）。

数据下载目录建议：`/DATA/disk0/yjb/datasets/rss2026/`（disk0 有 6.5TB 空闲；注意 disk1 已不可用，旧 ManipArena 路径全失效）。

---

## 4. Step 1 — 下载数据集

> **便捷脚本**（推荐）：`bash scripts/download_rss2026_data.sh`（默认下 3 任务的 expert-data；`all` 下全部子集）。
> 下面是其等价的手动命令与说明。

```bash
export HF_HOME=/DATA/disk0/yjb/.cache/huggingface     # 可选：把缓存放 disk0
DATA_ROOT=/DATA/disk0/yjb/datasets/rss2026/Challenge-phase1-dataset

# 方案 A：先只下 3 个任务的 expert-data（跑基线最小集，省时省空间）
for t in insert-mouse-battery seal-water-bottle-cap tower-of-hanoi-game; do
  huggingface-cli download Posttraining-RFM-RSS2026/Challenge-phase1-dataset --repo-type dataset \
    --include "$t/expert-data/*" --local-dir "$DATA_ROOT"
done

# 方案 B：全量（含 failure-data / success-and-hil-data，用于后续 post-training/DAgger）
huggingface-cli download Posttraining-RFM-RSS2026/Challenge-phase1-dataset --repo-type dataset \
    --local-dir "$DATA_ROOT"
```

下载后每个任务的目录结构（LeRobot v2.1，parquet + AV1 mp4）：
```
<DATA_ROOT>/<task>/
  expert-data/            # 人类遥操作专家轨迹（主力训练数据）
  failure-data/           # 基线策略失败 rollout（负样本，post-training 用）
  success-and-hil-data/   # 基线成功 + 人类干预(HIL)（post-training 用）
    meta/{info.json, episodes.jsonl, tasks.jsonl, ...}
    data/chunk-000/episode_*.parquet
    videos/chunk-000/<camera>/episode_*.mp4
```
规模参考（episodes / frames / 小时）：

| 任务 | expert-data | failure-data | success-and-hil-data |
|---|---|---|---|
| insert-mouse-battery | 831 / 2.09M / 9.65h | 125 / 0.31M / 1.46h | 164 / 0.69M / 3.20h |
| seal-water-bottle-cap | 379 / 2.04M / 9.43h | 91 / 0.31M / 1.44h | 112 / 0.75M / 3.47h |
| tower-of-hanoi-game | 1004 / 2.14M / 9.92h | 296 / 0.58M / 2.67h | 207 / 0.57M / 2.65h |

> 注意：实际子目录名是 `expert-data`/`failure-data`/`success-and-hil-data`（HIL 并入 success-and-hil-data），与官网文案里的命名略有出入。

校验一份数据能被 LeRobot 读取：
```bash
uv run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
root='$DATA_ROOT/insert-mouse-battery/expert-data'
m=LeRobotDatasetMetadata('insert-mouse-battery/expert-data', root=root)
print('fps',m.fps,'tasks',list(m.tasks)[:3]); print('features',list(m.features)[:8])
"
```

---

## 5. Step 2 — 移植官方 YAM 配置进本 fork + Step 3 训练

### 5.1 移植代码（详细 diff 见 [IMPLEMENTATION.md §4](IMPLEMENTATION.md)）
官方临时克隆在 `/tmp/pt-baseline`（若已清理，重新 `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/posttraining-for-robotics/openpi-baseline /tmp/pt-baseline`）。三步：

```bash
cd /DATA/disk0/yjb/projects/VLA/openpi
# (a) 复制官方 YAM 适配（关节翻转 + 夹爪角度变换 + 相机映射），原样可用
cp /tmp/pt-baseline/src/openpi/policies/yam_policy.py src/openpi/policies/yam_policy.py
```
然后在 `src/openpi/training/config.py` 里加三样东西（见 IMPLEMENTATION.md 的精确代码）：
- import：`import openpi.policies.yam_policy as yam_policy`
- `DualYamDataConfig` 类（从官方原样粘贴，它用的 `_transforms.` 别名与本 fork 一致）
- 4 个 `TrainConfig` 进 `_CONFIGS`（**把官方的 `local_files_path=` 改成本 fork 的 `local_root=`**）

### 5.2 计算归一化统计（每个 config 必做一次）
```bash
uv run scripts/compute_norm_stats.py pi05_insert-mouse-battery
uv run scripts/compute_norm_stats.py pi05_seal-water-bottle-cap
uv run scripts/compute_norm_stats.py pi05_tower-of-hanoi-game
uv run scripts/compute_norm_stats.py pi05_multitask          # 多任务赛道
```
写到 `./assets/<config_name>/<repo_id>/norm_stats.json`（pi05 用分位数，文件含 q01/q99）。**未算统计直接训练会报错。**

### 5.3 训练（pi05 从 pi05_base 全参微调）

> **便捷脚本**（推荐，已封装"预检数据路径 → 算 norm_stats → 训练 + 日志"）：
> ```bash
> # 单个 config（GPUS 指定卡；自动跑 norm_stats）
> GPUS=0 bash scripts/train_rss2026.sh pi05_insert-mouse-battery
> RESUME=1 GPUS=0 bash scripts/train_rss2026.sh pi05_insert-mouse-battery   # 断点续训
> SKIP_NORM=1 GPUS=0,1 BATCH_SIZE=64 bash scripts/train_rss2026.sh pi05_multitask
> # 一键并行训 3 个 Single-Task（各占一张卡）
> GPU_LIST="0 1 2" bash scripts/train_rss2026_all_single.sh
> ```
> 脚本会先校验 `local_root` 数据目录存在，缺失则报错并提示先下载。下面是等价的手动命令。

单任务（每个占用一张空闲卡，可并行起 3 个 + 1 个多任务）：
```bash
# 例：用 0/1/2/3 号卡分别并行训 4 个 config
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_insert-mouse-battery \
    --exp_name=pi05_insert-mouse-battery --overwrite --fsdp_devices=1 --batch_size=32 &

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_seal-water-bottle-cap \
    --exp_name=pi05_seal-water-bottle-cap --overwrite --fsdp_devices=1 --batch_size=32 &

CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_tower-of-hanoi-game \
    --exp_name=pi05_tower-of-hanoi-game --overwrite --fsdp_devices=1 --batch_size=32 &

CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_multitask \
    --exp_name=pi05_multitask --overwrite --fsdp_devices=1 --batch_size=32 &
```
多卡加速单个 config（如用 0,1 两卡，batch 必须能被卡数整除）：
```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_insert-mouse-battery --exp_name=run1 --overwrite \
    --fsdp_devices=1 --batch_size=64       # fsdp_devices=1 即纯数据并行；想分片显存用 --fsdp_devices=2
```
- 输出：`checkpoints/<config>/<exp_name>/<step>/{params,train_state,assets}`；`params/` 是 **EMA(0.99) 权重**（推理就加载它）。
- `--resume` 断点续训（与 `--overwrite` 互斥）；`--wandb_enabled=False` 关 wandb。
- **进度/产物**：official 配方 `num_train_steps=200_000`（≈3 epoch）较久。建议把 `save_interval` 设小（如 20_000）多存中间点，**先拿 40k–80k 的中间 checkpoint 提交保底**，训练继续再用更好的覆盖。
- OOM 时：降 `--batch_size`、设 `--fsdp_devices=2`（分片）、或临时 `--ema_decay=None`（省一份权重）。

监控：
```bash
tail -f logs/*.log            # 若用 train.sh 包装
nvidia-smi -l 5
# wandb 默认开启，project=openpi
```

---

## 6. Step 4 — 本地验证训练是否健康（serve + smoke）

先用 openpi 原生 server 自测策略能加载、能出动作（不依赖部署仓库）：
```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_insert-mouse-battery \
  --policy.dir=./checkpoints/pi05_insert-mouse-battery/pi05_insert-mouse-battery/40000
```
另开终端用本仓库 simple client 发一帧（或直接进 §8 的部署封装 + smoke_test）。**首帧会触发 JAX 编译，耗时数十秒属正常。**

---

## 7. Step 5/6 — 对接 `policy_deployment` + 仿真验证（提交前必做）

### 7.1 准备部署仓库
```bash
git clone https://github.com/posttraining-for-robotics/policy_deployment /DATA/disk0/yjb/projects/VLA/policy_deployment
cd /DATA/disk0/yjb/projects/VLA/policy_deployment
# 服务端运行需要 openpi+jax，用 openpi 的 venv 跑即可；补 websockets/msgpack（若缺）
/DATA/disk0/yjb/projects/VLA/openpi/.venv/bin/python -m pip install -q websockets msgpack
```

### 7.2 写 `Pi05Policy(BasePolicy)` 封装
把下面文件存到 `policy_deployment/examples/pi05_policy.py`（完整版见 [IMPLEMENTATION.md §4.4](IMPLEMENTATION.md)）。核心：`infer()` 把部署 obs 透传给 openpi 的 `Policy.infer`，返回 `(action_horizon,14)`：

```python
import numpy as np
from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

class Pi05Policy(BasePolicy):
    def __init__(self, config_name, checkpoint_dir, default_prompt="",
                 action_horizon=50, image_h=180, image_w=320):
        tc = _config.get_config(config_name)
        self._p = _policy_config.create_trained_policy(tc, checkpoint_dir,
                                                       default_prompt=default_prompt or None)
        self._ah, self._dp = action_horizon, default_prompt
        self._img_shape = [3, image_h, image_w]
        self._warmup()                       # 触发 JAX 编译，避免首帧超时

    def _warmup(self):
        d = {"images": {k: np.zeros((3, self._img_shape[1], self._img_shape[2]), np.uint8)
                        for k in ("cam_high","cam_left_wrist","cam_right_wrist")},
             "state": np.zeros(14, np.float32), "prompt": self._dp or "do the task"}
        try: self._p.infer(d)
        except Exception as e: print("warmup failed:", e)

    @property
    def metadata(self) -> ServerMetadata:
        return {"protocol_version":"1.0","policy_name":"pi05-rss2026","control_mode":"joints",
                "action_horizon":self._ah,"action_dim":14,"state_dim":14,
                "image_keys":["cam_high","cam_left_wrist","cam_right_wrist"],
                "image_shape":self._img_shape,"expects_prompt":True}

    def infer(self, obs) -> InferenceResponse:
        inp = {"images": {k: np.asarray(v) for k,v in obs["images"].items()},
               "state": np.asarray(obs["state"], np.float32)}
        prompt = obs.get("prompt") or self._dp
        if prompt: inp["prompt"] = prompt
        out = self._p.infer(inp)
        return {"actions": np.asarray(out["actions"], np.float32)}   # (action_horizon,14)
```

启动服务（用 openpi venv，PYTHONPATH 同时含部署仓库与 openpi src）：
```bash
cd /DATA/disk0/yjb/projects/VLA/policy_deployment
CKPT=/DATA/disk0/yjb/projects/VLA/openpi/checkpoints/pi05_insert-mouse-battery/pi05_insert-mouse-battery/40000
PYTHONPATH=.:/DATA/disk0/yjb/projects/VLA/openpi/src \
/DATA/disk0/yjb/projects/VLA/openpi/.venv/bin/python scripts/launch.py \
  --policy examples.pi05_policy:Pi05Policy \
  --policy-kwargs config_name=pi05_insert-mouse-battery \
  --policy-kwargs checkpoint_dir=$CKPT \
  --policy-kwargs default_prompt="Insert the battery to the mouse." \
  --port 8000
```

握手与单帧自测：
```bash
PYTHONPATH=. python scripts/ping.py       --host 127.0.0.1 --port 8000   # 校验 metadata
PYTHONPATH=. python scripts/smoke_test.py --host 127.0.0.1 --port 8000   # 校验返回 actions 是 2 维
```

### 7.3 仿真 compare（最关键的提交前闸门）
```bash
# 重放本身正常吗
mjpython sim/check_in_sim.py --mode replay  --bundle sim/assets/example_slim.pkl
# 用你的服务驱动仿真
mjpython sim/check_in_sim.py --mode policy  --bundle sim/assets/example_slim.pkl \
        --host 127.0.0.1 --port 8000 --prompt "Insert the battery to the mouse." --action-horizon 50
# 对比：录制动作 vs 策略预测，看每维 L2 差
mjpython sim/check_in_sim.py --mode compare --bundle sim/assets/example_slim.pkl \
        --host 127.0.0.1 --port 8000 --prompt "Insert the battery to the mouse." --action-horizon 50
```
目标：手臂维度 `L2/sqrt(N*14)` ≲ 0.05。**若第 6/13 维（夹爪）差异巨大 → 夹爪量纲/约定错**；**左右整体对调 → 14 维布局没按"左7右7交错"**。这两类是最常见 bug，compare 模式的报错表会指给你看（`sim/README.md`）。无显示器时用 `python ...(非 mjpython) --output out.mp4` 离屏渲染。

### 7.4 分辨率/纵横比对齐（易踩坑）
- 数据集图像 320×180（16:9），模型内部 `ResizeImages` 用 **resize_with_pad（letterbox）** 缩到 224×224。
- `metadata.image_shape` 是给 eval client 的"该发多大图"提示。**建议填 `[3,180,320]`**（与训练原生分辨率/纵横比一致），让 eval 端发 16:9 帧、模型再 letterbox，训练/推理一致。若 eval client 把真机帧硬缩成你填的尺寸而破坏纵横比，会掉点——**这点需与主办的 eval client 行为核对**（见 IMPLEMENTATION.md 风险项）。

---

## 8. Step 7 — 提交

技术侧"可被评测"=上面 §7 全绿（ping/smoke/compare 通过）。提交动作本身依赖主办流程：

1. 把 `Pi05Policy` 服务部署到主办可访问的地址：本机直连或经 nginx 终止 TLS 暴露 `wss://`（部署仓库带 `deploy/nginx.conf.example`、`deploy/policy-server.service` 模板，`proxy_read_timeout 3600s`）。
2. 安全：设 `POLICY_SERVER_API_KEYS` / `--api-key`，必要时 `--allow-cidr` 放行主办 eval IP。健康检查 `GET /healthz`。
3. 按官网/邮件提供：服务地址 + API key + 选定赛道（Single-Task / Multi-Task）+ 队伍信息。**确切提交表单/eval IP/打分细节以官方为准，代码仓库未给。**
4. Single-Task 赛道：3 个任务各起一个对应 config 的服务（或主办按任务切换 prompt 用同一多 config 流程——需向主办确认评测是"每任务单独服务"还是"一个服务按 prompt 切任务"）。Multi-Task 赛道：用 `pi05_multitask` 一个服务。

---

## 9. 进阶 —— 利用 failure / HIL 数据做 post-training（拿到基线后再做）

比赛主题是 post-training；`failure-data`（负样本）与 `success-and-hil-data`（成功+人工纠正）是加分点。**先确保基线提交可用，再做这些**：
- **DAgger / 数据增广（最快见效、最稳）**：用官方 `merge_lerobot.py` 把 `expert-data` 与 `success-and-hil-data` 合并成更大集，重训。
  ```bash
  cp /tmp/pt-baseline/scripts/merge_lerobot.py scripts/merge_lerobot.py
  uv run scripts/merge_lerobot.py \
    --src_paths <DATA_ROOT>/insert-mouse-battery/expert-data <DATA_ROOT>/insert-mouse-battery/success-and-hil-data \
    --tgt_path  /DATA/disk0/yjb/datasets/rss2026/merged/insert-mouse-battery \
    --repo_id   insert-mouse-battery/merged
  ```
  然后把对应 config 的 `local_root` 指向合并目录，重算 norm_stats 再训。
- **失败样本利用**（更"post-training"，但工作量大、风险高）：把 `failure-data` 当负样本做加权/对比/偏好学习——本 fork **暂无现成 RL/DPO/优势加权代码**（你已有的 VGAA/APSG/IGCA 等是 SFT 辅助损失或推理加速，**不是** post-training 方法，且多为 pi0/LIBERO 专用）。8 天内不建议从零造，优先 DAgger。
- 评估你自研改进的可用性：见 IMPLEMENTATION.md 的"研究改进盘点"——结论是 Phase-1 走 stock pi05 基线最稳。

---

## 10. 故障排查速查

| 症状 | 原因 / 处理 |
|---|---|
| `Normalization stats not found` | 没跑 `compute_norm_stats.py <config>`；或 `local_root`/`asset_id` 路径不对 |
| 训练启动即 OOM | 降 `--batch_size`；`--fsdp_devices=2` 分片；`--ema_decay=None`；确认用的是空闲卡 |
| `make_mesh` 报错 / 设备不整除 | `--fsdp_devices` 必须整除可见 GPU 数；`--batch_size` 必须整除 GPU 数 |
| `TokenizePrompt: Prompt is required` | 推理 obs 没带 prompt 且无 default；给 `--policy-kwargs default_prompt=...` 或客户端每帧带 prompt |
| `Expected images to contain (...)` 报错 | 发来的相机名不在 `cam_high/cam_left_wrist/cam_right_wrist`；检查 obs["images"] 键名 |
| compare 模式夹爪维(6/13)差异大 | 夹爪量纲/约定错——确认输出是 [0,1]，且训练/服务都用 `adapt_to_pi=True` 同一 config |
| compare 左右对调 | 14 维没按"左7+右7"布局 |
| 首帧推理很慢/超时 | JAX 首次编译；已在 `Pi05Policy._warmup` 预热；确保 server 起好再让 eval 连 |
| 加载成了 PyTorch 模型 | `create_trained_policy` 见到 `model.safetensors` 才走 PyTorch；JAX 训练产物只有 `params/`，确认 `--policy.dir` 指向 JAX step 目录 |
| 想复现官方"逐字一致" | model 用 `Pi0Config(pi05=True)` **默认值**（`action_horizon=50`、`discrete_state_input=True`），`use_delta_joint_actions=True`、`adapt_to_pi=True`、`batch_size=32`、`num_train_steps=200_000`、`ema_decay=0.99`（默认） |

---

## 11. 关键文件索引（本 fork）

- 模型：`src/openpi/models/pi0.py`、`pi0_config.py`、`gemma.py`、`tokenizer.py`
- 训练：`src/openpi/training/config.py`（所有 TrainConfig）、`scripts/train.py`、`scripts/compute_norm_stats.py`、`src/openpi/training/{data_loader,checkpoints,weight_loaders}.py`
- 数据/变换：`src/openpi/transforms.py`、`src/openpi/policies/yam_policy.py`（移植后）、`maniparena_policy.py`（EE 空间模板，仅参考）
- 服务：`scripts/serve_policy.py`、`src/openpi/serving/websocket_policy_server.py`、`src/openpi/policies/policy_config.py`（`create_trained_policy`）
- 官方临时克隆：`/tmp/pt-baseline`（baseline）、`/tmp/pt-deploy`（deployment，含 `server/schema.py`、`examples/{echo,my}_policy.py`、`sim/check_in_sim.py`）

更细的 file:line 代码地图见 [IMPLEMENTATION.md §7](IMPLEMENTATION.md)。
