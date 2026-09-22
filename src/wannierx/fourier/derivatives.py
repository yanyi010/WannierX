"""Analytic cartesian k-derivatives of the interpolated Hamiltonian.

Conventions: the public API takes
fractional reciprocal k (shape (..., 3)); derivatives are with respect to
*cartesian* k (Angstrom^-1).

    H(k)                  = sum_R w_R H_R e^{+i 2 pi k_f . R}
    dH/dk_cart_a          = i sum_R R_a^cart w_R H_R e^{+i 2 pi k_f . R},
                            R^cart = R @ A
    d2H/dk_cart_a dk_cart_b = - sum_R R_a^cart R_b^cart w_R H_R e^{+i 2 pi k_f . R}

These identities already include the cartesian conversion: with
k_cart = k_f B and B = 2 pi A^{-T} the chain rule gives
dk_cart/dk_f = B, so a fractional phase derivative supplies R (times 2 pi),
and multiplying by A^{-T}/(1) via R^cart/(2 pi)*2 pi = R^cart keeps the
result in cartesian units. Derivatives are exact analytic expressions; no
finite differences are used.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel
from wannierx.fourier.transform import _phase_matrix

_TWO_PI = 2.0 * jnp.pi


def _R_cart(model: WannierModel) -> Array:
    """Cartesian R vectors, (n_R, 3), Angstrom."""
    return model.lattice.frac_to_cart_real(model.R.astype(model.lattice.direct.dtype))


def dH_dk(model: WannierModel, k: Array) -> Array:
    """Analytic first cartesian k-derivative of H.

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).

    Returns:
        Shape (..., 3, n_orb, n_orb), units eV Angstrom. Hermitian for a
        Hermitian model. Differentiable, JIT/vmap safe.

    Derivation: dH/dk_cart = (dH/dk_f) @ B^{-1}; with
    dH/dk_f = i 2 pi sum_R R w H_R e^{i 2 pi k_f.R} and B^{-1} = A/(2 pi),
    one obtains i sum_R (R @ A) w H_R e^{i 2 pi k_f.R}.
    """
    wphase = _phase_matrix(model, k)  # (..., n_R)
    Rc = _R_cart(model)  # (n_R, 3)
    return 1j * jnp.einsum("...R,Ra,Rmn->...amn", wphase, Rc, model.H_R)


def d2H_dk2(model: WannierModel, k: Array) -> Array:
    """Analytic second cartesian k-derivative tensor of H.

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).

    Returns:
        Shape (..., 3, 3, n_orb, n_orb), units eV Angstrom^2, symmetric
        in the derivative indices up to floating-point roundoff.
    """
    wphase = _phase_matrix(model, k)
    Rc = _R_cart(model)
    RR = Rc[:, :, None] * Rc[:, None, :]  # (n_R, 3, 3)
    return -jnp.einsum("...R,Rab,Rmn->...abmn", wphase, RR, model.H_R)
