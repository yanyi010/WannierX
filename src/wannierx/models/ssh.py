"""Su-Schrieffer-Heeger (SSH) model.

Convention:

    H(k) = [[0, t1 + t2 e^{-i k a}], [t1 + t2 e^{i k a}, 0]]
    E_pm(k) = +- sqrt(t1^2 + t2^2 + 2 t1 t2 cos(k a))

with a the lattice constant. In fractional reciprocal k_f (kx a = 2 pi k_f)
this gives the intercell hopping at R = +a carrying the e^{+i k a} phase
in the lower-left (row A, col B) element, per the project convention
H_mn(R) = <0m|H|Rn> and the +i k.R Fourier sign.

Topological regime: |t2| > |t1| (intercell dominated).
"""

from __future__ import annotations

import jax.numpy as jnp

from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel


def ssh(t1: float, t2: float, lattice_constant: float = 1.0) -> WannierModel:
    """Build the SSH chain.

    Parameters:
        t1: intracell hopping A0-B0, eV.
        t2: intercell hopping A0-B_{+1}, eV.
        lattice_constant: cell length, Angstrom.

    Returns:
        Two-orbital model (orbitals 0=A, 1=B) with periodic x direction.
    """
    lattice = Lattice(jnp.diag(jnp.asarray([lattice_constant, 1.0, 1.0])))
    # R list: 0, +1, -1 along x
    # Derivation (locked by tests):
    #   H(k)_mn = sum_R H_mn(R) e^{+ik.R}, H_mn(R) = <0m|H|Rn>.
    #   Target matrix H(k)[0,1] = t1 + t2 e^{-ika} requires
    #   H_01(0) = t1, H_01(-a) = t2, H_01(+a) = 0;
    #   Hermiticity then gives H_10(+a) = t2.
    R = jnp.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    H_R = jnp.array(
        [
            [[0.0, t1], [t1, 0.0]],      # H(0)
            [[0.0, 0.0], [t2, 0.0]],      # H(+a): [1,0] = t2
            [[0.0, t2], [0.0, 0.0]],      # H(-a): [0,1] = t2
        ],
        dtype=jnp.complex128,
    )
    weights = jnp.ones(3)
    centers = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return WannierModel(lattice, R, H_R, weights, (True, False, False), centers)
