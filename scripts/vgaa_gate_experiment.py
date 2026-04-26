"""VGAA Gate Experiment: Diagnose whether attention-sensitivity misalignment exists.

Computes Spearman correlation rho(W, s) between:
  - W: action->observation cross-attention weights (head-averaged, layer-averaged)
  - s: velocity Jacobian sensitivity (Hutchinson K=5)

on a trained pi0 checkpoint using LIBERO data.

Decision criteria:
  rho < 0.4  -> strong misalignment, VGAA has large room to improve
  0.4-0.7    -> moderate misalignment, VGAA should help
  0.7-0.9    -> mild misalignment, VGAA may help on hard tasks only
  rho >= 0.9 -> already aligned, VGAA won't help

Usage:
    python scripts/vgaa_gate_experiment.py \
        --config-name pi0_libero \
        --checkpoint-dir /path/to/checkpoint \
        --num-batches 50
"""

import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("vgaa_gate")


@dataclasses.dataclass
class Args:
    config_name: str = "pi0_libero"
    checkpoint_dir: str = ""
    num_batches: int = 50
    hutchinson_k: int = 5
    batch_size: int = 4


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1D arrays."""
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho)


def run_gate_experiment(args: Args):
    config = _config.get_config(args.config_name)
    # Override batch size for diagnostics (smaller to fit in memory).
    config = dataclasses.replace(config, batch_size=args.batch_size)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    logger.info("Loading model from %s ...", checkpoint_dir)
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()

    logger.info("Creating data loader ...")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)

    # ---- Per-frame diagnostic ----
    all_rhos = []           # per-sample Spearman correlations
    all_rhos_by_tau = {}    # binned by flow time

    from openpi.models.pi0 import make_attn_mask, posemb_sincos  # noqa: E402

    logger.info("Running gate experiment for %d batches (K=%d) ...", args.num_batches, args.hutchinson_k)

    for batch_idx in range(args.num_batches):
        observation, actions = next(data_iter)

        rng = jax.random.key(batch_idx)
        preprocess_rng, noise_rng, time_rng, vgaa_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Embed prefix and suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass WITH attention weights
        (prefix_out, suffix_out), _, all_attn_weights = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            return_attn_weights=True,
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])

        # ---- Compute sensitivity (Hutchinson, K probes) ----
        def v_from_prefix(pt):
            (_, suf), _, _ = model.PaliGemma.llm(
                [pt, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(suf[:, -model.action_horizon:])

        sensitivity = jnp.zeros(prefix_tokens.shape[:2])  # [B, T_prefix]
        for k in range(args.hutchinson_k):
            probe_rng = jax.random.fold_in(vgaa_rng, k)
            v_val, vjp_fn = jax.vjp(v_from_prefix, prefix_tokens)
            z = jax.random.normal(probe_rng, v_val.shape, dtype=v_val.dtype)
            (grad_prefix,) = vjp_fn(z)
            sensitivity = sensitivity + jnp.sum(grad_prefix.astype(jnp.float32) ** 2, axis=-1)
        sensitivity = sensitivity / args.hutchinson_k  # [B, T_prefix]

        # ---- Extract attention distribution ----
        # all_attn_weights: [L, B, T_prefix] from scan stacking
        attn_dist = jnp.mean(all_attn_weights, axis=0)  # average over layers -> [B, T_prefix]

        # ---- Compute per-sample Spearman correlation ----
        sensitivity_np = np.array(jax.device_get(sensitivity))
        attn_np = np.array(jax.device_get(attn_dist))
        time_np = np.array(jax.device_get(time))

        B = sensitivity_np.shape[0]
        for b in range(B):
            s_vec = sensitivity_np[b]
            w_vec = attn_np[b]
            # Skip if either is constant (no variance)
            if np.std(s_vec) < 1e-12 or np.std(w_vec) < 1e-12:
                continue
            rho = spearman_rank_corr(s_vec, w_vec)
            if np.isnan(rho):
                continue
            all_rhos.append(rho)

            # Bin by flow time tau
            tau = float(time_np[b])
            if tau > 0.7:
                tau_bin = "high(0.7-1.0)"
            elif tau > 0.3:
                tau_bin = "mid(0.3-0.7)"
            else:
                tau_bin = "low(0.0-0.3)"
            all_rhos_by_tau.setdefault(tau_bin, []).append(rho)

        if (batch_idx + 1) % 10 == 0:
            current_mean = np.mean(all_rhos) if all_rhos else 0
            logger.info(
                "  batch %d/%d: %d samples so far, mean rho = %.4f",
                batch_idx + 1, args.num_batches, len(all_rhos), current_mean,
            )

    # ---- Report ----
    all_rhos = np.array(all_rhos)
    mean_rho = np.mean(all_rhos)
    std_rho = np.std(all_rhos)
    median_rho = np.median(all_rhos)

    print("\n" + "=" * 60)
    print("VGAA GATE EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Checkpoint:   {args.checkpoint_dir}")
    print(f"Samples:      {len(all_rhos)}")
    print(f"Hutchinson K: {args.hutchinson_k}")
    print(f"")
    print(f"Spearman rho(attention, sensitivity):")
    print(f"  Mean:   {mean_rho:.4f} +/- {std_rho:.4f}")
    print(f"  Median: {median_rho:.4f}")
    print(f"  Min:    {np.min(all_rhos):.4f}")
    print(f"  Max:    {np.max(all_rhos):.4f}")
    print()
    print("By flow time tau:")
    for tau_bin in sorted(all_rhos_by_tau.keys()):
        vals = np.array(all_rhos_by_tau[tau_bin])
        print(f"  {tau_bin}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")
    print()

    # Decision
    if mean_rho < 0.4:
        decision = "STRONG PROCEED -- severe misalignment, VGAA has large room"
    elif mean_rho < 0.7:
        decision = "PROCEED -- moderate misalignment, VGAA should help"
    elif mean_rho < 0.9:
        decision = "CAUTIOUS -- mild misalignment, may only help on hard tasks"
    else:
        decision = "ABORT -- attention already well-aligned, VGAA won't help"

    print(f"Decision: {decision}")
    print("=" * 60)

    # Save raw data
    out_path = pathlib.Path(args.checkpoint_dir) / "vgaa_gate_results.npz"
    np.savez(
        out_path,
        all_rhos=all_rhos,
        mean_rho=mean_rho,
        std_rho=std_rho,
        median_rho=median_rho,
    )
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    run_gate_experiment(tyro.cli(Args))
