"""Skip-VLA Scheme B: learned chunk-boundary skip gate.

At each VLA replan boundary, a tiny MLP gate decides whether we can skip the VLA
forward and instead fill the action plan with K steps of rolling linear extrapolation.

  * Predictor: parameter-free rolling linear extrapolation of the last two actions
    (gripper dim is held constant, not extrapolated).
  * Gate g_ψ: 2-layer MLP binary classifier trained on pilot traces with
    label y=1 iff the rolled linear extrap chunk's max-over-K L1 error vs the
    actual VLA chunk falls below δ_chunk.

This module is self-contained: CLI trains, and an importable SkipVLAController
plugs into main_parallel.py with a single branch.

CLI:
    python examples/libero/skipvla.py train \\
        --traces-dir logs/parallel_<PILOT_RUN_ID> \\
        --out-ckpt .cache/skipvla/spatial_gate.pt
"""

import argparse
import glob
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Predictor (parameter-free).
# ---------------------------------------------------------------------

def linear_extrap_chunk(last_two: np.ndarray, K: int, hold_dim: Optional[int] = -1) -> np.ndarray:
    """Roll linear extrapolation K steps forward from last_two = [a[t-2], a[t-1]].

    Args:
        last_two: (2, action_dim) array of the two most recent actions.
        K: how many steps to predict.
        hold_dim: dim to hold constant (default -1 = gripper). None to disable.
    Returns: (K, action_dim).
    """
    assert last_two.shape[0] == 2
    a_prev, a_cur = last_two[0], last_two[1]
    out = np.empty((K, a_cur.shape[0]), dtype=a_cur.dtype)
    for k in range(K):
        a_next = 2.0 * a_cur - a_prev
        if hold_dim is not None:
            a_next = a_next.copy()
            a_next[hold_dim] = a_cur[hold_dim]
        out[k] = a_next
        a_prev, a_cur = a_cur, a_next
    return out


# ---------------------------------------------------------------------
# Models: Predictor P_φ + Gate g_ψ.
# ---------------------------------------------------------------------

@dataclass
class GateConfig:
    history_k: int = 5
    action_dim: int = 7
    proprio_dim: int = 8
    hidden: int = 128
    predict_k: int = 5       # how many future actions the predictor outputs (= replan_steps)
    delta_norms: bool = True

    @property
    def input_dim(self) -> int:
        base = self.history_k * self.action_dim + 2 * self.proprio_dim
        if self.delta_norms:
            base += 2
        return base


class TinyPredictor(nn.Module):
    """P_φ: (action_history, proprio, proprio_delta) → (predict_k, action_dim) chunk.
    Gripper dim (last) is predicted directly (not held) — MSE loss will naturally
    pull it toward the teacher's decision (stay vs flip)."""
    def __init__(self, cfg: GateConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.predict_k * cfg.action_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        return self.net(x).view(B, self.cfg.predict_k, self.cfg.action_dim)


class TinyGate(nn.Module):
    """g_ψ: binary classifier. Takes the same features as the predictor PLUS the predictor's
    own output norm-stats (predicted-chunk variance, predicted-gripper-stability) so the
    gate can learn 'trust the predictor if its output looks smooth/consistent'. Shared MLP
    trunk kept tiny."""
    def __init__(self, cfg: GateConfig, use_predictor_features: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_pred_feats = use_predictor_features
        extra = 3 if use_predictor_features else 0   # [pred_chunk_step_variance, pred_gripper_std, pred_max_delta]
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim + extra, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, x, pred_chunk=None):
        if self.use_pred_feats:
            assert pred_chunk is not None, "gate requires predictor output"
            # three scalar features summarising how "confident-looking" the predicted chunk is.
            step_var = pred_chunk.diff(dim=1).pow(2).mean(dim=(1, 2)).unsqueeze(-1)  # (B,1)
            grip_std = pred_chunk[:, :, -1].std(dim=1).unsqueeze(-1)                  # (B,1)
            max_delta = pred_chunk.diff(dim=1).abs().amax(dim=(1, 2)).unsqueeze(-1)   # (B,1)
            x = torch.cat([x, step_var, grip_std, max_delta], dim=-1)
        return self.net(x).squeeze(-1)


def make_feature(action_history: np.ndarray, proprio_t: np.ndarray,
                 proprio_delta: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """Flat feature vector for the gate. action_history shape (K, action_dim)."""
    parts = [action_history.flatten().astype(np.float32),
             proprio_t.astype(np.float32),
             proprio_delta.astype(np.float32)]
    if cfg.delta_norms:
        if action_history.shape[0] >= 2:
            delta_a = action_history[1:] - action_history[:-1]
            delta_a_norm = np.linalg.norm(delta_a[-1], ord=2)
        else:
            delta_a_norm = 0.0
        if action_history.shape[0] >= 3:
            delta_a2 = action_history[1:] - action_history[:-1]
            delta2_a = delta_a2[1:] - delta_a2[:-1]
            delta2_a_norm = np.linalg.norm(delta2_a[-1], ord=2)
        else:
            delta2_a_norm = 0.0
        parts.append(np.array([delta_a_norm, delta2_a_norm], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------
# Dataset builder (from pilot traces).
# ---------------------------------------------------------------------

def build_dataset_from_traces(traces_dir: str, cfg: GateConfig, success_only: bool = True):
    """Extract chunk-level (feature, target_chunk) pairs at each VLA boundary.

    Returns:
      X: (N, input_dim)            gate/predictor input features
      Y_chunk: (N, K, action_dim)  the ACTUAL VLA chunk that was executed (teacher for predictor)
      task_ids: (N,)
    """
    K = cfg.predict_k
    trace_files = sorted(glob.glob(os.path.join(traces_dir, "*traces-worker*.pkl")))
    if not trace_files:
        raise FileNotFoundError(f"no *-traces-worker*.pkl files in {traces_dir}")

    X_list, Y_list, tid_list = [], [], []
    n_ep = 0

    for tf in trace_files:
        with open(tf, "rb") as f:
            episodes = pickle.load(f)
        for ep in episodes:
            n_ep += 1
            if success_only and not ep["success"]:
                continue
            actions = ep["actions"]
            proprio = ep["proprio"]
            is_vla = ep["is_vla"]
            T = actions.shape[0]
            if T < cfg.history_k + K + 1:
                continue
            for t in np.where(is_vla)[0]:
                if t < cfg.history_k or t + K > T or t < 1:
                    continue
                hist = actions[t - cfg.history_k:t]
                prop_t = proprio[t]
                prop_dt = proprio[t] - proprio[t - 1]
                x = make_feature(hist, prop_t, prop_dt, cfg)
                y_chunk = actions[t:t + K].astype(np.float32)
                X_list.append(x)
                Y_list.append(y_chunk)
                tid_list.append(int(ep["task_id"]))

    X = np.stack(X_list).astype(np.float32)
    Y_chunk = np.stack(Y_list).astype(np.float32)
    task_ids = np.array(tid_list, dtype=np.int32)
    meta = {
        "n_samples": int(len(X)),
        "n_episodes": n_ep,
        "K": K,
        "cfg": cfg.__dict__,
    }
    return X, Y_chunk, task_ids, meta


def train_predictor_and_gate(X: np.ndarray, Y_chunk: np.ndarray, cfg: GateConfig,
                              delta_chunk: float = 0.40,
                              predictor_epochs: int = 100, gate_epochs: int = 100,
                              lr: float = 1e-3, batch_size: int = 512,
                              val_frac: float = 0.2, seed: int = 0, device: str = "cpu",
                              pos_weight: float = 1.0):
    """Two-stage training:
       Stage 1: train predictor P_φ with chunk-MSE against actual VLA chunks.
       Stage 2: freeze P_φ; compute binary labels y = 1[max_L1(P_φ(x), Y_chunk) < δ];
                train gate g_ψ on those labels (using predictor's chunk as extra features).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y_chunk)

    X_tr, X_va = X_t[train_idx].to(device), X_t[val_idx].to(device)
    Y_tr, Y_va = Y_t[train_idx].to(device), Y_t[val_idx].to(device)

    # ---- Stage 1: predictor
    predictor = TinyPredictor(cfg).to(device)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-6)
    print(f"[Stage 1] Train predictor P_φ  (N_train={len(X_tr)}, N_val={len(X_va)})")
    for ep in range(predictor_epochs):
        predictor.train()
        perm = torch.randperm(len(X_tr))
        tot = 0.0
        for b in range(0, len(perm), batch_size):
            bi = perm[b:b + batch_size]
            pred = predictor(X_tr[bi])
            loss = F.mse_loss(pred, Y_tr[bi])
            opt_p.zero_grad(); loss.backward(); opt_p.step()
            tot += loss.item() * len(bi)
        if ep == 0 or (ep + 1) % 20 == 0 or ep == predictor_epochs - 1:
            predictor.eval()
            with torch.no_grad():
                pv = predictor(X_va)
                mse = F.mse_loss(pv, Y_va).item()
                # chunk-level max-L1 distribution vs VLA teacher (diagnostic)
                maxl1 = (pv - Y_va).abs().sum(-1).amax(-1)
                # baseline: linear extrap error distribution for comparison
                last_two_idx = slice(cfg.history_k - 2, cfg.history_k)  # not quite — compute below
            print(f"  pred ep{ep:3d}  train_mse={tot/len(X_tr):.5f}  val_mse={mse:.5f}  "
                  f"val_maxL1 p50={maxl1.median().item():.3f}  p90={maxl1.quantile(0.9).item():.3f}")

    # ---- Stage 2: labels from predictor's actual errors
    predictor.eval()
    with torch.no_grad():
        pred_all = predictor(X_t.to(device)).cpu()
    # label: y = 1 iff max over K steps of L1(pred - actual) < delta_chunk
    errs = (pred_all - Y_t).abs().sum(-1).amax(-1)  # (N,)
    y_label = (errs < delta_chunk).float()
    pos_frac = y_label.mean().item()
    print(f"[Labels] δ_chunk={delta_chunk}: positive fraction = {pos_frac*100:.2f}%  "
          f"(predictor-based labels; oracle upper bound on skip coverage)")
    y_tr, y_va = y_label[train_idx].to(device), y_label[val_idx].to(device)

    # ---- Stage 3: gate
    gate = TinyGate(cfg, use_predictor_features=True).to(device)
    opt_g = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-6)
    pw = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    print(f"[Stage 2] Train gate g_ψ  (pos_frac={pos_frac:.3f}, pos_weight={pos_weight})")
    for ep in range(gate_epochs):
        gate.train()
        perm = torch.randperm(len(X_tr))
        tot = 0.0
        for b in range(0, len(perm), batch_size):
            bi = perm[b:b + batch_size]
            with torch.no_grad():
                pred = predictor(X_tr[bi])
            logits = gate(X_tr[bi], pred_chunk=pred)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr[bi], pos_weight=pw)
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            tot += loss.item() * len(bi)

        if ep == 0 or (ep + 1) % 20 == 0 or ep == gate_epochs - 1:
            gate.eval()
            with torch.no_grad():
                pv = predictor(X_va)
                probs = torch.sigmoid(gate(X_va, pred_chunk=pv))
            msg = f"  gate ep{ep:3d}  loss={tot/len(X_tr):.4f}"
            for tau in [0.5, 0.7, 0.8, 0.9]:
                pred_pos = (probs > tau).float()
                tp = ((pred_pos == 1) & (y_va == 1)).float().sum().item()
                fp = ((pred_pos == 1) & (y_va == 0)).float().sum().item()
                pos_n = y_va.sum().item()
                neg_n = len(y_va) - pos_n
                msg += (f"   τ={tau}:skip={pred_pos.mean().item()*100:4.1f}%"
                        f" prec={tp/max(tp+fp,1):.2f}"
                        f" tpr={tp/max(pos_n,1):.2f}"
                        f" fpr={fp/max(neg_n,1):.2f}")
            print(msg)

    return predictor, gate, pos_frac
    """Train TinyGate. pos_weight=1.0 (default) keeps sigmoid output CALIBRATED
    so τ=0.5 means "model thinks P(skip-safe) > 0.5". Higher pos_weight up-weights the
    positive class in the loss which biases the output toward positive at inference,
    which is BAD when you then pick τ by probability threshold. Only use >1 if
    positives are <5% and the gate can't learn at all with pos_weight=1."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = TinyGate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    pos = y_t.sum().item()
    pw = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    X_train = X_t[train_idx].to(device)
    y_train = y_t[train_idx].to(device)
    X_val = X_t[val_idx].to(device)
    y_val = y_t[val_idx].to(device)

    print(f"  N_train={len(X_train)}  N_val={len(X_val)}  pos_frac={pos/N:.3f}  pos_weight={pos_weight:.2f} (calibrated={pos_weight==1.0})")

    def _tau_eval(taus=(0.3, 0.5, 0.7, 0.8, 0.9)):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val))
        out = {}
        for tau in taus:
            pred = (probs > tau).float()
            tp = ((pred == 1) & (y_val == 1)).float().sum().item()
            fp = ((pred == 1) & (y_val == 0)).float().sum().item()
            tn = ((pred == 0) & (y_val == 0)).float().sum().item()
            pos_n = y_val.sum().item()
            neg_n = len(y_val) - pos_n
            out[tau] = {
                "gated_rate": (pred == 1).float().mean().item(),
                "tpr": tp / max(pos_n, 1),
                "fpr": fp / max(neg_n, 1),
                "precision": tp / max(tp + fp, 1),
            }
        return out

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for b in range(0, len(perm), batch_size):
            bi = perm[b:b + batch_size]
            logits = model(X_train[bi])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[bi], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(bi)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            taus = _tau_eval()
            line = f"  ep{epoch:3d}  loss={total_loss/len(X_train):.4f}"
            for tau in [0.5, 0.7, 0.8, 0.9]:
                r = taus[tau]
                line += f"   τ={tau}:skip={r['gated_rate']*100:4.1f}% prec={r['precision']:.2f} tpr={r['tpr']:.2f} fpr={r['fpr']:.2f}"
            print(line)

    return model


# ---------------------------------------------------------------------
# Inference-time controller.
# ---------------------------------------------------------------------

class SkipVLAController:
    """Chunk-boundary gate. Usage in the main loop:

        skipvla = SkipVLAController(ckpt_path, tau=0.7)
        for episode: skipvla.reset(); prev_proprio = None
            for step:
                if action_plan empty:
                    proprio = ...
                    if skipvla.should_skip(proprio, prev_proprio):
                        action_plan.extend(skipvla.predict_chunk(K))
                        skipvla.on_skip()
                    else:
                        action_chunk = client.infer(...)
                        skipvla.on_call()
                    prev_proprio = proprio.copy()
                action = action_plan.popleft()
                skipvla.record(action)
    """

    def __init__(self, ckpt_path: str, tau: float = 0.7, max_skips: int = 15,
                 device: str = "cpu"):
        sd = torch.load(ckpt_path, map_location=device)
        cfg = GateConfig(**sd["cfg"])
        predictor = TinyPredictor(cfg)
        gate = TinyGate(cfg, use_predictor_features=True)
        predictor.load_state_dict(sd["predictor_state_dict"])
        gate.load_state_dict(sd["gate_state_dict"])
        predictor.eval().to(device)
        gate.eval().to(device)
        self.cfg = cfg
        self.predictor = predictor
        self.gate = gate
        self.tau = tau
        self.max_skips = max_skips
        self.device = device
        self.history = []
        self.consecutive_skips = 0
        self.n_call = 0
        self.n_skip = 0
        self.ckpt_meta = sd.get("meta", {})
        self._last_pred_chunk = None  # cache the predictor output from the most recent should_skip() call

    def reset(self):
        self.history = []
        self.consecutive_skips = 0

    def record(self, action):
        """Append an executed action to the history buffer (call every step)."""
        a = np.asarray(action, dtype=np.float32).copy()
        self.history.append(a)
        if len(self.history) > self.cfg.history_k:
            self.history = self.history[-self.cfg.history_k:]

    def should_skip(self, proprio_t: np.ndarray, proprio_prev) -> bool:
        """Run predictor, then gate on (features, predicted_chunk). Caches the predicted
        chunk so the caller can fetch it via predict_chunk() without re-running P_φ."""
        self._last_pred_chunk = None
        if len(self.history) < self.cfg.history_k:
            return False
        if proprio_prev is None:
            return False
        if self.consecutive_skips >= self.max_skips:
            return False
        hist = np.stack(self.history[-self.cfg.history_k:])
        # Gripper guard: last two actions' gripper sign flipped → don't skip.
        if hist.shape[0] >= 2 and np.sign(hist[-1, -1]) != np.sign(hist[-2, -1]):
            return False
        prop_delta = (proprio_t - proprio_prev).astype(np.float32)
        x = make_feature(hist, proprio_t.astype(np.float32), prop_delta, self.cfg)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.predictor(x_t)                       # (1, K, D)
            logit = self.gate(x_t, pred_chunk=pred)
            p = torch.sigmoid(logit).item()
        self._last_pred_chunk = pred.squeeze(0).cpu().numpy()  # (K, D)
        return p > self.tau

    def predict_chunk(self, K: int) -> np.ndarray:
        """Return the chunk produced by the predictor during the last should_skip() call.
        K is ignored if it matches cfg.predict_k; if K > cfg.predict_k we fall back to
        extending with rolling linear extrap from the last predicted actions."""
        assert self._last_pred_chunk is not None, "call should_skip() first"
        chunk = self._last_pred_chunk.astype(np.float32)
        if K <= chunk.shape[0]:
            return chunk[:K]
        # Extend by rolling linear extrap from the last two predicted actions.
        tail = linear_extrap_chunk(chunk[-2:], K=K - chunk.shape[0])
        return np.concatenate([chunk, tail], axis=0)

    def on_skip(self):
        self.consecutive_skips += 1
        self.n_skip += 1

    def on_call(self):
        self.consecutive_skips = 0
        self.n_call += 1

    def skip_rate(self) -> float:
        total = self.n_skip + self.n_call
        return self.n_skip / total if total > 0 else 0.0


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def _cli_train(args):
    sample_pkl = sorted(glob.glob(os.path.join(args.traces_dir, "*traces-worker*.pkl")))
    if not sample_pkl:
        raise FileNotFoundError(f"no traces in {args.traces_dir}")
    with open(sample_pkl[0], "rb") as f:
        sample = pickle.load(f)
    action_dim = sample[0]["actions"].shape[1]
    proprio_dim = sample[0]["proprio"].shape[1]

    cfg = GateConfig(history_k=args.history_k, action_dim=action_dim,
                     proprio_dim=proprio_dim, predict_k=args.K)
    print(f"Cfg: history_k={cfg.history_k} action_dim={cfg.action_dim} "
          f"proprio_dim={cfg.proprio_dim} predict_k={cfg.predict_k} "
          f"hidden={cfg.hidden} input_dim={cfg.input_dim}")

    X, Y_chunk, task_ids, meta = build_dataset_from_traces(
        args.traces_dir, cfg, success_only=args.success_only,
    )
    print(f"Dataset: N={meta['n_samples']}  episodes={meta['n_episodes']}  "
          f"X{X.shape}  Y{Y_chunk.shape}")

    predictor, gate, pos_frac = train_predictor_and_gate(
        X, Y_chunk, cfg,
        delta_chunk=args.delta_chunk,
        predictor_epochs=args.predictor_epochs,
        gate_epochs=args.gate_epochs,
        lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, pos_weight=args.pos_weight,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)) or ".", exist_ok=True)
    torch.save({
        "predictor_state_dict": predictor.state_dict(),
        "gate_state_dict": gate.state_dict(),
        "cfg": cfg.__dict__,
        "meta": {**meta, "pos_frac": pos_frac, "delta_chunk": args.delta_chunk},
    }, args.out_ckpt)
    print(f"Saved predictor+gate → {args.out_ckpt}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="build dataset from pilot traces and train gate")
    pt.add_argument("--traces-dir", required=True)
    pt.add_argument("--out-ckpt", required=True)
    pt.add_argument("--K", type=int, default=5, help="predict-k; should match replan_steps in deployment")
    pt.add_argument("--history-k", type=int, default=5)
    pt.add_argument("--delta-chunk", type=float, default=0.20,
                    help="label threshold: y=1 iff max-step L1 err(predictor, VLA) < δ")
    pt.add_argument("--predictor-epochs", type=int, default=100)
    pt.add_argument("--gate-epochs", type=int, default=100)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--batch-size", type=int, default=512)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--success-only", action="store_true", default=True)
    pt.add_argument("--pos-weight", type=float, default=1.0,
                    help="BCE positive class weight; keep at 1.0 for CALIBRATED sigmoid output.")
    pt.set_defaults(func=_cli_train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
