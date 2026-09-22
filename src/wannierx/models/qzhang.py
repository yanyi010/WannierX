"""Qi-Wu-Zhang (QWZ) model.

Convention (documented here, locked by tests):

    H(k) = sin(kx) sigma_x + sin(ky) sigma_y + (u + cos kx + cos ky) sigma_z

with lattice constants a1 = a2 = 1 Angstrom embedded in 3D (z non-periodic).
On a 2D mesh with fractional coordinates the physical momenta are
k_alpha = 2 pi k_f_alpha (a = 1).

Phase regimes (with the above sign convention, pinned by
:func:`wannierx.chern_number`):

    u < -2                : C = 0
    -2 < u < 0            : C = -1
     0 < u < 2            : C = +1
     u > 2                : C = 0

(Both nontrivial signs are locked by a test on one regime; the full
mapping is recorded in this module's tests.)
"""

from __future__ import annotations

import jax.numpy as jnp

from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel

# Pauli matrices
_SX = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
_SY = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)
_SZ = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)


def qiwuzhang(u: float) -> WannierModel:
    """Build the 2D QWZ Chern-insulator model.

    Parameters:
        u: Dirac mass parameter, eV-scale (dimensionless hopping units are
            normalized to 1 eV).

    Returns:
        Two-band model, x and y periodic.
    """
    lattice = Lattice(jnp.eye(3))
    # R points: (0,0), (±1,0), (0,±1)
    R = jnp.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    )
    # H_mn(R) = <0m|H|Rn>; from H(k) = sum_R H(R) e^{ik.R} and the target
    # H(k) above:
    #   sin(kx) sx    -> H(+x) = sx/(2i), H(-x) = -sx/(2i)
    #   (u + cos kx + cos ky) sz -> H(0) = u sz, H(±x) += sz/2, H(±y) = sz/2
    #   sin(ky) sy    -> H(+y) = sy/(2i), H(-y) = -sy/(2i)
    H0 = u * _SZ
    Hpx = _SX / (2.0j) + _SZ / 2.0
    Hmx = -_SX / (2.0j) + _SZ / 2.0
    Hpy = _SY / (2.0j) + _SZ / 2.0
    Hmy = -_SY / (2.0j) + _SZ / 2.0
    H_R = jnp.stack([H0, Hpx, Hmx, Hpy, Hmy])
    weights = jnp.ones(5)
    centers = jnp.zeros((2, 3))
    return WannierModel(lattice, R, H_R, weights, (True, True, False), centers)
