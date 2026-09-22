"""Differentiable Fourier interpolation of H(k).

Convention (normative):

    H(k_f) = sum_R w_R H_R exp(+i 2 pi k_f . R)

``k`` is always fractional reciprocal coordinates with shape
``(..., 3)``; leading batch dimensions are preserved. Pure JAX,
vectorized over R; no Python loop over k.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel


def _phase_matrix(model: WannierModel, k: Array) -> Array:
    """Phases exp(+i 2 pi k.R) with shape (..., n_R)."""
    k = jnp.asarray(k)
    if k.shape[-1] != 3:
        raise ValueError(f"k must have last dimension 3, got shape {k.shape}")
    # (..., 1, 3) @ (1, n_R, 3) -> (..., n_R)
    kr = jnp.sum(k[..., None, :] * model.R[None], axis=-1)
    phase = jnp.exp(1j * 2.0 * jnp.pi * kr)
    return phase * model.weights


def hamiltonian(model: WannierModel, k: Array) -> Array:
    """Differentiable Fourier interpolation of the Hamiltonian.

    Parameters:
        model: canonical :class:`WannierModel`.
        k: fractional reciprocal k, shape (..., 3).

    Returns:
        Complex H(k) with shape (..., n_orb, n_orb), eV.

    Differentiable in ``k`` and in all model array fields. JIT/vmap
    compatible. Raises on incompatible k shape.
    """
    wphase = _phase_matrix(model, k)  # (..., n_R)
    # einsum over R: leading dims preserved.
    return jnp.einsum("...R,Rmn->...mn", wphase, model.H_R)
