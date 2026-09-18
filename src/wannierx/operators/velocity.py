"""Velocity operator and eigenbasis velocity matrix (WX-015).

v_alpha = (1/hbar) dH/dk_alpha, from the analytic cartesian k-derivative
(no finite differences). Physical output units: m/s.

Conversions:
    dH/dk [eV Angstrom] / hbar [eV s] * (1 Angstrom = 1e-10 m) -> m/s.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from wannierx.constants import EV_ANG_OVER_HBAR_TO_M_S
from wannierx.core.model import WannierModel
from wannierx.fourier.derivatives import dH_dk
from wannierx.linalg.eigh import eigh


def velocity_operator(model: WannierModel, k: Array) -> Array:
    """Cartesian velocity operator in the Wannier basis.

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).

    Returns:
        (..., 3, n_orb, n_orb) complex Hermitian matrices, m/s.

    Failure behavior: cartesian velocity requires lattice information,
    which is always present in :class:`WannierModel`; incompatible k
    shapes raise.
    """
    return dH_dk(model, k) * EV_ANG_OVER_HBAR_TO_M_S


def velocity_matrix(model: WannierModel, k: Array) -> Array:
    """Cartesian velocity operator in the eigenbasis of H(k).

    Returns:
        (..., 3, n_orb, n_orb), V_ab = <u_a| v |u_b>, m/s. Diagonal
        elements are the band group velocities.

    Differentiable in smooth regimes; combined with eigh derivatives
    away from degeneracies.
    """
    eig = eigh(model, k)
    v_op = velocity_operator(model, k)  # (..., 3, norb, norb)
    U = eig.vectors
    Ud = jnp.swapaxes(jnp.conjugate(U), -1, -2)
    # einsum keeps velocity index explicit
    return jnp.einsum("...mi,...aij,...in->...amn", Ud, v_op, U)
