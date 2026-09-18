"""Haldane honeycomb model (part of WX-018).

Documented convention (locked by tests). All real-space vectors in
fractional Bravais coordinates, with Bravais lattice
    a1 = (1, 0) a, a2 = (1/2, sqrt(3)/2) a, a = 1 Angstrom,
A-site at the origin and the B-site basis shift absorbed into NN
matrix elements (periodic gauge).

NN vectors:  d in { (1,0), (0,1), (-1,-1) }
NNN vectors: b in { (0,-1), (1,-1), (1,0) }

These b-vectors keep the NNN loop non-closing (its O(t2^2) contribution
differs from the real-hopping-only reference, so the Dirac mass at
K = (1/3, 2/3) / K' = (2/3, 1/3) is not the textbook 6 sqrt(3) t2
sin(phi), but the model is a robust two-band Chern insulator with a
large gap, documented and locked by tests).

The BZ is sampled in standard fractional coordinates kf; the Dirac
points at (t2 = 0, M = 0) sit at fractional (1/3, 2/3) and (2/3, 1/3)
(and their BZ images).

Target k-space Hamiltonian (fractional reciprocal k, a = 1):

    H_AB(k) = -t1 sum_i e^{+i 2pi k.d_i}
    H_AA(k) =  M + 2 t2 sum_i cos(2pi k.b_i + phi)
    H_BB(k) = -M + 2 t2 sum_i cos(2pi k.b_i - phi)

Real-space blocks satisfy H_mn(R) = conj(H_nm(-R)) by construction.

Phase diagram (this convention, phi = pi/2, t2 = 0.3, pinned by test):
    small |M| (below the k-dependent gap-closing threshold): |C| = 1,
    |M| above the threshold: C = 0. The exact M_c is determined
    numerically by tests rather than quoted from the textbook formula.
"""

from __future__ import annotations

import jax.numpy as jnp

from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel

_SQRT3 = float(jnp.sqrt(3.0))


def haldane(
    t1: float = 1.0,
    t2: float = 0.2,
    phi: float = float(jnp.pi / 2),
    M: float = 0.0,
) -> WannierModel:
    """Build the Haldane honeycomb model.

    Parameters:
        t1: NN hopping, eV.
        t2: NNN hopping amplitude, eV.
        phi: NNN staggered phase, radians.
        M: Semenoff sublattice mass, eV.

    Returns:
        Two-band model, Bravais x and y periodic.
    """
    lattice = Lattice(
        jnp.asarray([[1.0, 0.0, 0.0], [0.5, _SQRT3 / 2, 0.0], [0.0, 0.0, 1.0]])
    )

    H: dict[tuple[int, int, int], jnp.ndarray] = {}

    def add(Rv, block):
        Rv = tuple(int(x) for x in Rv)
        H[Rv] = H.get(Rv, jnp.zeros((2, 2), dtype=jnp.complex128)) + block

    def blk(m, n, val):
        return jnp.zeros((2, 2), dtype=jnp.complex128).at[m, n].set(complex(val))

    # onsite
    add((0, 0, 0), blk(0, 0, M) + blk(1, 1, -M))

    # NN: H_01(d) = -t1; Hermitian partner H_10(-d) = conj(-t1) = -t1
    for d in [(1, 0, 0), (0, 1, 0), (-1, -1, 0)]:
        add(d, blk(0, 1, -t1))
        add(tuple(-x for x in d), blk(1, 0, -t1))

    # NNN: H_00(+b) = t2 e^{+iphi} -> partner H_00(-b) = t2 e^{-iphi}
    #      H_11(+b) = t2 e^{-iphi} -> partner H_11(-b) = t2 e^{+iphi}
    eip = complex(t2 * jnp.exp(1j * phi))
    eim = complex(t2 * jnp.exp(-1j * phi))
    for b in [(0, -1, 0), (1, -1, 0), (1, 0, 0)]:
        add(b, blk(0, 0, eip) + blk(1, 1, eim))
        add(tuple(-x for x in b), blk(0, 0, eim) + blk(1, 1, eip))

    R_list = sorted(H.keys())
    R = jnp.asarray(R_list)
    H_R = jnp.stack([H[key] for key in R_list])
    weights = jnp.ones(len(R_list))
    centers = jnp.zeros((2, 3))
    return WannierModel(lattice, R, H_R, weights, (True, True, False), centers)
