"""VGAA: Velocity-Grounded Attention Alignment for Flow-based VLAs.

This module implements the VGAA auxiliary loss that aligns the action→observation
cross-attention distribution with the velocity Jacobian sensitivity distribution.

Key components:
  - VGAAConfig: Configuration dataclass for VGAA hyperparameters
  - compute_sensitivity: Computes per-token sensitivity via Hutchinson estimator
  - compute_alignment_loss: Computes KL(sensitivity || attention) alignment loss
"""

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class VGAAConfig:
    """Configuration for VGAA (Velocity-Grounded Attention Alignment).

    When enabled=False (default), no VGAA computation is performed and the
    training/inference pipeline is identical to the baseline.
    """

    enabled: bool = False
    # Weight of the alignment loss: L_total = L_flow + lambda_align * L_align
    lambda_align: float = 0.1
    # Temperature for softmax normalization of sensitivity distribution.
    # Lower = sharper (more concentrated on top patches), higher = smoother.
    temperature: float = 1.0
    # Number of Hutchinson random probes for sensitivity estimation.
    # K=1 is cheapest (~1.3x overhead), K=3 is more accurate (~1.9x overhead).
    num_probes: int = 1
    # Which transformer layers to align. None = all layers.
    # Use list of indices, e.g., [12,13,14,15,16,17] for last 6 layers of an 18-layer model.
    layer_indices: tuple[int, ...] | None = None
    # Lambda warmup steps: linearly ramp lambda from 0 to lambda_align over this many steps.
    # Prevents VGAA from interfering with early training when flow matching hasn't converged yet.
    warmup_steps: int = 0
    # Whether to use cheap proxy (grad of ||v||^2) instead of Hutchinson estimator.
    # Faster but biased (only captures sensitivity along v direction).
    use_cheap_proxy: bool = False


def compute_sensitivity_hutchinson(
    forward_fn,
    prefix_tokens: jnp.ndarray,
    rng: jax.Array,
    num_probes: int = 1,
) -> jnp.ndarray:
    """Compute per-token sensitivity using Hutchinson estimator.

    Uses the identity: ||J_i||_F^2 = E_z[||J_i^T z||^2] where z ~ N(0, I).
    Each probe requires one VJP (backward pass) through forward_fn.

    Args:
        forward_fn: Callable mapping prefix_tokens -> v_t [B, H, D_action].
            Must be a pure function of prefix_tokens with all other args closed over.
        prefix_tokens: Observation token embeddings [B, T_prefix, D].
        rng: PRNG key for sampling random probes.
        num_probes: Number of Hutchinson probes (K). More = less variance.

    Returns:
        sensitivity: Per-token sensitivity [B, T_prefix], stop-gradiented.
    """
    sensitivity = jnp.zeros(prefix_tokens.shape[:2])  # [B, T_prefix]

    for k in range(num_probes):
        probe_rng = jax.random.fold_in(rng, k)

        # Compute v_t and VJP function in one call
        v_t, vjp_fn = jax.vjp(forward_fn, prefix_tokens)

        # Sample random probe direction z ~ N(0, I) matching v_t shape
        z = jax.random.normal(probe_rng, v_t.shape, dtype=v_t.dtype)

        # Compute J^T z via VJP: grad of (v_t . z) w.r.t. prefix_tokens
        (grad_prefix,) = vjp_fn(z)  # [B, T_prefix, D]

        # Per-token sensitivity: ||J_i^T z||^2 = sum_d (grad[b,i,d])^2
        sensitivity = sensitivity + jnp.sum(
            grad_prefix.astype(jnp.float32) ** 2, axis=-1
        )  # [B, T_prefix]

    sensitivity = sensitivity / num_probes
    return jax.lax.stop_gradient(sensitivity)


def compute_sensitivity_cheap(
    forward_fn,
    prefix_tokens: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-token sensitivity using cheap proxy: grad of ||v||^2.

    This computes hat{s}_i = ||d(||v||^2)/d(o_i)||, which is ||2 J_i^T v||.
    Only requires one backward pass (computing grad of a scalar).
    Biased: only captures sensitivity along the velocity direction.

    Args:
        forward_fn: Callable mapping prefix_tokens -> v_t [B, H, D_action].
        prefix_tokens: Observation token embeddings [B, T_prefix, D].

    Returns:
        sensitivity: Per-token sensitivity [B, T_prefix], stop-gradiented.
    """

    def scalar_fn(pt):
        v_t = forward_fn(pt)
        return jnp.sum(v_t.astype(jnp.float32) ** 2)

    grad_prefix = jax.grad(scalar_fn)(prefix_tokens)  # [B, T_prefix, D]
    sensitivity = jnp.sum(grad_prefix.astype(jnp.float32) ** 2, axis=-1)  # [B, T_prefix]
    return jax.lax.stop_gradient(sensitivity)


def compute_alignment_loss(
    all_attn_weights: jnp.ndarray,
    sensitivity: jnp.ndarray,
    temperature: float = 1.0,
    layer_indices: Sequence[int] | None = None,
) -> jnp.ndarray:
    """Compute KL(sensitivity || attention) alignment loss.

    Uses forward KL which penalizes "model didn't look at important patches"
    (attention too low where sensitivity is high).

    Args:
        all_attn_weights: Cross-attention summaries from all layers [L, B, T_prefix].
            Each entry is the head-averaged, action-query-averaged attention weight
            that action tokens place on each observation token.
        sensitivity: Per-token sensitivity [B, T_prefix].
        temperature: Temperature for softmax normalization of sensitivity.
        layer_indices: Which layers to include. None = all layers.

    Returns:
        align_loss: Scalar alignment loss (mean over batch).
    """
    # Select layers
    if layer_indices is not None:
        attn_weights = all_attn_weights[jnp.array(layer_indices)]  # [|S|, B, T_prefix]
    else:
        attn_weights = all_attn_weights  # [L, B, T_prefix]

    # Normalize sensitivity to probability distribution via temperature softmax
    s_logits = sensitivity / temperature  # [B, T_prefix]
    s_dist = jax.nn.softmax(s_logits, axis=-1)  # [B, T_prefix]

    # Compute per-layer KL divergence: KL(s_dist || attn_dist)
    # Clamp attention weights for numerical stability (softmax never produces exact 0,
    # but float16 rounding could)
    attn_clamped = jnp.maximum(attn_weights, 1e-8)  # [|S|, B, T_prefix]

    # Normalize attention weights to valid distributions (they should already sum to ~1
    # since they come from softmax, but averaging over heads/queries may break this)
    attn_dist = attn_clamped / jnp.sum(attn_clamped, axis=-1, keepdims=True)  # [|S|, B, T_prefix]

    # KL(s || attn) = sum_i s_i * log(s_i / attn_i)
    # = sum_i s_i * (log(s_i) - log(attn_i))
    kl_per_layer = jnp.sum(
        s_dist[None] * (jnp.log(s_dist[None] + 1e-8) - jnp.log(attn_dist + 1e-8)),
        axis=-1,
    )  # [|S|, B]

    # Average over layers and batch
    return jnp.mean(kl_per_layer)


def get_vgaa_lambda(config: VGAAConfig, step: int | jnp.ndarray) -> jnp.ndarray:
    """Get effective lambda with optional warmup."""
    if config.warmup_steps <= 0:
        return jnp.float32(config.lambda_align)
    progress = jnp.clip(step / config.warmup_steps, 0.0, 1.0)
    return jnp.float32(config.lambda_align * progress)
