# Skip-VLA Handoff Notes

> Self-contained context dump for continuing Skip-VLA work on another machine.
> Last updated in the session that built the parallel eval + Scheme B predictor/gate.

---

## 1. What is Skip-VLA (short)

Idea doc: `docs/lazy_vla_idea (1).md` (updated version).

Core claim: at each VLA replan boundary, a tiny MLP gate decides whether we can
**skip the VLA forward** and instead fill the next K actions with a tiny
predictor's output. Skip when "boring" (smooth free-space motion), call VLA when
"critical" (contact, gripper transitions, decisions).

  * Predictor `P_φ`: ~28K params, outputs a K-step action chunk
  * Gate `g_ψ`: ~24K params, outputs p(skip-safe) ∈ [0, 1]
  * Safety guards: K_max consecutive skips + gripper flip → force VLA

Design target backbones: pi0, pi0.5 (openpi), later SmolVLA / OpenVLA-OFT.
Primary benchmark: LIBERO (Spatial/Goal/Object/Long/90).

---

## 2. Machine state (as of 2026-04-23)

Working dir: `/DATA/disk1/yjb/projects/VLA/openpi`

  * `.venv/` — server venv (py3.11, torch 2.7, transformers 4.53, openpi).
    Needs `transformers_replace` patch applied (see §5).
  * `examples/libero/.venv/` — client venv (py3.8, torch 1.11+cu113, robosuite,
    libero, openpi_client, mujoco 3.2.3).
  * `.cache/parallel_eval/` — per-GPU torch.compile + triton caches (~4.6 GB
    total). Persists across runs; delete to force recompile.
  * `.cache/skipvla/spatial_P+G_d040.pt` — trained predictor + gate (d=0.40).

Checkpoints:
  * `/DATA/disk1/yjb/.cache/openpi/openpi-assets/pytorch_checkpoints/pi0_libero/`
  * `/DATA/disk1/yjb/.cache/openpi/openpi-assets/pytorch_checkpoints/pi05_libero/`

---

## 3. Files I added (all under `examples/libero/`)

| File | Role |
|---|---|
| `run_parallel_eval.sh` | 8-GPU launcher (1 server + 1 client per GPU, sharded, auto-aggregate) |
| `main_parallel.py` | Sharded client. Adds `--pilot-dump` (traces) and `--skipvla-ckpt` (Scheme B) |
| `aggregate_results.py` | Merges per-worker JSON → `<run>-summary.txt` + combined JSON |
| `pilot_analysis.py` | Oracle skip rate + Scheme A FP rate analysis from pilot traces |
| `skipvla.py` | `TinyPredictor` + `TinyGate` + `SkipVLAController` + training CLI |
| `10_nvidia.json` | Local NVIDIA EGL ICD (system is missing `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`) |

All non-invasive wrt original openpi; `main.py` / `main_ori.py` untouched.

---

## 4. Three standard run flows

```bash
cd /DATA/disk1/yjb/projects/VLA/openpi

# A) Baseline eval (pi0 @ LIBERO-Spatial):
RUN_ID=pi0-spatial-BASELINE \
bash examples/libero/run_parallel_eval.sh

# B) Pilot eval (dumps per-step traces for Scheme B training):
RUN_ID=pi0-spatial-PILOT \
EXTRA_CLIENT_ARGS="--pilot-dump" \
bash examples/libero/run_parallel_eval.sh
# Then analyse:
./examples/libero/.venv/bin/python examples/libero/pilot_analysis.py \
    ./logs/parallel_pi0-spatial-PILOT

# C) Skip-VLA eval (after training a gate):
RUN_ID=pi0-spatial-SKIPVLA-v2-d040-tau080 \
EXTRA_CLIENT_ARGS="--skipvla-ckpt .cache/skipvla/spatial_P+G_d040.pt --skipvla-tau 0.8 --skipvla-max-skips 5" \
bash examples/libero/run_parallel_eval.sh
```

Use `MODEL=pi05` to swap backbone, `TASK_SUITE=libero_goal` etc to swap suite.
`MIRROR_WORKER_0=0` silences the live worker-0 terminal stream for scripted sweeps.

---

## 5. Env setup landmines (fix once per machine)

  1. Two venvs are required — server + client are isolated. See
     `examples/libero/README.md` for canonical commands.
  2. Apply transformers patch:
     ```bash
     cp -r src/openpi/models_pytorch/transformers_replace/* \
           .venv/lib/python3.11/site-packages/transformers/
     ```
     The launcher aborts clearly if server venv is missing deps but does NOT
     detect a missing patch — server crashes at first infer() with
     `ValueError: transformers_replace is not installed correctly`.
  3. EGL ICD: system missing `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
     The launcher points `__EGL_VENDOR_LIBRARY_FILENAMES` at
     `examples/libero/10_nvidia.json`. Permanent fix (requires sudo):
     ```bash
     sudo cp examples/libero/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
     ```
  4. libero writes `~/.libero/config.yaml` on first import with an interactive
     prompt. Launcher pre-creates it; don't panic if first run shows a "Y/N"
     prompt echoed once.
  5. torch.compile cache at `.cache/parallel_eval/` MUST be per-GPU, else 8
     servers race and crash with "could not convert string to float: ''".
     Already handled by launcher.

---

## 6. Metrics (OFT convention)

Reported per run in `<RUN_ID>-summary.txt`. All three numbers are what to
report in paper tables.

| Metric | Meaning |
|---|---|
| **VLA latency** | `client.infer()` wall time per call (ms). Unchanged by Skip-VLA. |
| **MLP latency** | Predictor forward time per call (~0.15 ms). |
| **Effective latency** | Weighted avg across VLA + MLP calls |
| **Throughput (OFT)** | `action_chunk_len / effective_latency` (actions/s) |
| **SR** | successes/episodes — NEVER report throughput without SR |
| **Skip rate** | `n_skip / (n_skip + n_call)` over chunk boundaries |
| **VLA call rate** | fraction of control steps triggering infer() (= 1 / effective replan) |
| **REAL control Hz** | end-to-end `total_steps / total_wall_s` — noisy because failed episodes run to max_steps |

Warning: legacy "throughput = replan_steps / latency" is wrong — it does not
reflect Skip-VLA's benefit because the numerator is fixed. Use effective_latency.

---

## 7. Key results captured so far

### Baseline (pi0 @ LIBERO-Spatial, 500 episodes)
  * `logs/parallel_pi0-libero_spatial-2026_04_22-01_40_50/`
  * SR: **96.00% (480/500)**
  * VLA latency: 72.00 ms  (median 68.68, p95 71.20)
  * Effective latency: 72.00 ms (no skip)
  * Throughput: **694.4 actions/s**
  * Control Hz: 20.10
  * Per-task weakest: task7 (stove) 92%, task4 (top drawer) 92%

### Pilot (oracle analysis)
  * `logs/parallel_pi0-spatial-PILOT/`
  * `pilot_analysis.py` output:
    * Oracle skip rate @ δ=0.05 (per-step L1): 26.30%
    * Oracle skip rate @ δ=0.10: **48.67%** ← idea is viable
    * Oracle skip rate @ δ=0.15: 58.59%
    * Scheme A (smoothness-only gate) FP rate > 5% at any reasonable τ → Scheme B
      required
    * Gripper flip ±3 steps carry **4.82×** L1 error vs elsewhere → gripper guard warranted
    * Per-task oracle skip rate ranges 15.7% (task0) to 36.7% (task7)

### Predictor + Gate training (`spatial_P+G_d040.pt`)
  * δ_chunk = 0.40, K=5, history_k=5
  * 9643 samples from 500 pilot episodes
  * Predictor val MSE 0.014, chunk max-L1: **p50=0.45 (vs linear extrap 0.61)**,
    **p90=0.91 (vs 1.45)**
  * Label pos_frac (using trained predictor as oracle): **41.64%**
  * Gate val @ τ=0.8: skip=9.5%, precision=0.86, TPR=0.20, FPR=2%
  * Gate val @ τ=0.7: skip=18.5%, precision=0.77, FPR=7%
  * Gate val @ τ=0.9: skip=5.1%, precision=0.92, FPR=1%

### Skip-VLA v2 run, τ=0.8, K_max=5
  * `logs/parallel_pi0-spatial-SKIPVLA-v2-d040-tau080/`
  * SR: **96.20%** (+0.20 pp vs baseline — actually improved slightly, noise)
  * Effective latency: 66.62 ms
  * Throughput: **750.6 actions/s (+8.1%)**
  * MLP latency: 0.148 ms
  * Skip rate: 7.80%
  * Interpretation: gate is working, τ=0.8 is conservative sweet spot.

### Earlier failed run (τ=0.8, uncalibrated gate)
  * `logs/parallel_pi0-spatial-SKIPVLA-d040-tau080-Kmax5/` — **keep as negative example**
  * Training used `pos_weight=2.89` which miscalibrated sigmoid → gate skipped
    ~40% (far above 25.7% positive rate) with 50% FP → SR crashed to 92.2%.
  * Root cause + fix documented in this session. Lesson: keep `pos_weight=1.0`
    so τ is interpretable as "model's probability".

---

## 8. Pending work (in priority order)

  1. **τ sweep** (blocked last time by GPU contention from another user's job):
     ```bash
     for tau in 0.5 0.6 0.7 0.9; do
       MIRROR_WORKER_0=0 \
       RUN_ID=pi0-spatial-SKIPVLA-v2-d040-tau$(echo $tau | tr -d '.') \
       EXTRA_CLIENT_ARGS="--skipvla-ckpt .cache/skipvla/spatial_P+G_d040.pt --skipvla-tau $tau --skipvla-max-skips 5" \
       bash examples/libero/run_parallel_eval.sh
     done
     ```
     Combined with the baseline + τ=0.8 already in hand, this gives the first
     Pareto curve for paper figure 1.
  2. Train more gates (different δ):
     ```bash
     for d in 0.30 0.50 0.60; do
       ./examples/libero/.venv/bin/python examples/libero/skipvla.py train \
         --traces-dir ./logs/parallel_pi0-spatial-PILOT \
         --out-ckpt .cache/skipvla/spatial_P+G_d$(echo $d | tr -d '.').pt \
         --K 5 --history-k 5 --delta-chunk $d \
         --predictor-epochs 100 --gate-epochs 80 --pos-weight 1.0
     done
     ```
  3. Upgrade predictor to LSTM / deeper MLP → paper ablation "learned > linear
     extrap". Currently 2-layer 128-wide MLP, val max-L1 p50=0.45. Target <0.35.
  4. Run pi0.5 baseline + Skip-VLA on same suite → cross-backbone generalization.
  5. Stack with SnapFlow (needs external code) → main orthogonality table,
     multiplicative speedup story.

---

## 9. Theoretical ceiling (be honest in paper)

Per-step composition on A100:

```
Baseline per step = env_step (~35 ms) + VLA_frac × VLA_latency (72 ms)
                  = 35 + 0.2 × 72 = 49.4 ms  →  20.1 Hz
```

Even with **100% skip** (hypothetical), floor is 35 ms/step = 28.5 Hz →
**1.41× ceiling** from Skip-VLA alone on this setup. To break through, need to
stack with per-call acceleration (SnapFlow / VLA-Cache) — this is the
multiplicative story in paper §4.4.

With SnapFlow compressing VLA to ~25 ms:
```
SnapFlow alone: ~36 Hz (+80%)
SnapFlow + Skip-VLA @ 20% skip: 35 + 0.8 × 25 = 55 ms → 58 Hz (+188%)
```

---

## 10. Failure modes to watch

  1. **Over-skipping consecutive** → gate's input history becomes off-distribution
     (it was trained on VLA-produced history). Guard: `K_max=5`. Paper should
     ablate this.
  2. **Gate false-positives during contact approach** → smooth trajectory right
     up to collision, gate fires skip, action drifts past target. Mitigation:
     gate uses predictor's own `step_var / grip_std / max_delta` as features
     now, so it partly self-gates. Also gripper guard.
  3. **Label mismatch** — if training uses δ=0.40 but deployment is under a
     tighter SR constraint, skip coverage is labeled but not actually safe.
     Re-train with smaller δ.
  4. **Warmup latency spike**: first VLA call per episode ~110 ms (torch.compile
     autotune is cached but Python overhead). Dropped from stats in main_parallel.

---

## 11. Quick debug cheatsheet

  * Run hanged: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
    If someone else owns all memory → wait.
  * OOM on startup: some leftover process. `ps -ef | grep -E "serve_policy|main_parallel" | grep -v grep | awk '{print $2}' | xargs -r kill -9`
  * EGL error: confirm `__EGL_VENDOR_LIBRARY_FILENAMES` points at an ICD JSON
    containing `libEGL_nvidia.so.0`.
  * "transformers_replace not installed": re-run the `cp -r` line from §5.
  * Suddenly every eval gets SR ≈ 20%: check gate ckpt was trained with
    `pos_weight=1.0`; re-train if older.

---

## 12. Directory tree at handoff time

```
/DATA/disk1/yjb/projects/VLA/openpi/
├── docs/
│   ├── lazy_vla_idea (1).md       # canonical idea doc
│   └── skipvla_handoff.md         # THIS FILE
├── examples/libero/
│   ├── main.py                    # upstream eval client (untouched)
│   ├── main_parallel.py           # sharded client + Skip-VLA hook (NEW)
│   ├── aggregate_results.py       # per-subtask summary (NEW)
│   ├── pilot_analysis.py          # go/no-go + FP analysis (NEW)
│   ├── skipvla.py                 # predictor + gate + trainer (NEW)
│   ├── run_parallel_eval.sh       # 8-GPU launcher (NEW)
│   ├── 10_nvidia.json             # EGL ICD workaround (NEW)
│   └── .venv/                     # py3.8 client env
├── .venv/                         # py3.11 server env (has transformers patch)
├── .cache/
│   ├── parallel_eval/             # torch.compile caches
│   └── skipvla/
│       └── spatial_P+G_d040.pt    # trained Scheme B (P_φ + g_ψ)
└── logs/
    ├── parallel_pi0-libero_spatial-2026_04_22-01_40_50/   # baseline
    ├── parallel_pi0-spatial-PILOT/                        # pilot (+traces)
    └── parallel_pi0-spatial-SKIPVLA-v2-d040-tau080/       # τ=0.8 result
```

---

## 13. Minimum to resume on a fresh machine

Assuming you cloned the repo to `$ROOT` and have the pi0 checkpoint and both
venvs already built:

```bash
cd $ROOT
# 1. transformers patch
cp -r src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/

# 2. Sanity: baseline + τ=0.8 Skip-VLA (should match §7 numbers within ±1 pp)
RUN_ID=sanity-baseline bash examples/libero/run_parallel_eval.sh
RUN_ID=sanity-skipvla-tau08 \
  EXTRA_CLIENT_ARGS="--skipvla-ckpt .cache/skipvla/spatial_P+G_d040.pt --skipvla-tau 0.8 --skipvla-max-skips 5" \
  bash examples/libero/run_parallel_eval.sh

# 3. Resume: run τ sweep per §8 step 1.
```

If gates aren't present: re-run pilot + re-train per §8 step 2.
