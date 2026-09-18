"""Smooth Gaussian density of states (WX-023).

Reference method: Gaussian broadening.

    DOS(E) = (1 / (sqrt(2 pi) sigma)) sum_k w_k sum_n
             exp(-(E - eps_nk)^2 / (2 sigma^2))

DOS is a BZ average per unit energy: the integrated DOS over a wide
energy window equals the number of bands. Units eV^-1.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel
from wannierx.kpoints.integration import integrate_bands
from wannierx.kpoints.mesh import KMesh
from wannierx.linalg.eigh import eigh


def dos(
    model: WannierModel,
    mesh: KMesh,
    energies: Array,
    broadening: float,
    method: str = "gaussian",
) -> Array:
    """Smooth broadened density of states.

    Parameters:
        model: canonical model.
        mesh: KMesh with normalized BZ-average weights.
        energies: (n_E,) energies, eV.
        broadening: Gaussian sigma, eV.
        method: currently only ``"gaussian"`` (the reference method).

    Returns:
        DOS with shape (n_E,), eV^-1 per unit cell (BZ average).

    Differentiable in energies, broadening, and all model parameters.
    """
    if method != "gaussian":
        raise ValueError(f"unsupported DOS method: {method!r}")
    eig = eigh(model, mesh.points)  # (nk, norb)
    eps = eig.energies
    E = jnp.asarray(energies)  # (nE,)
    sigma = float(broadening)
    if sigma <= 0:
        raise ValueError("broadening must be > 0 for a smooth DOS")
    kernel = jnp.exp(-((E[:, None, None] - eps[None, :, :]) ** 2) / (2.0 * sigma**2))
    dos_k = jnp.sum(kernel, axis=-1) / (jnp.sqrt(2.0 * jnp.pi) * sigma)  # (nE, nk)
    # weighted BZ reduction (weights sum to 1)
    return dos_k @ mesh.weights
