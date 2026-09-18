"""Graphene nearest-neighbor tight-binding toy model (part of WX-018).

Same structural conventions as :mod:`wannierx.models.haldane` with only
the NN term: H_AB(k) = -t sum_i e^{+i 2pi k.d_i}, d in {(1,0),(0,1),(-1,-1)},
Bravais a1 = (1,0), a2 = (1/2, sqrt(3)/2), a = 1 Angstrom.

Known analytic content used by tests: the two bands touch (Dirac points)
at fractional reciprocal K = (1/3, 2/3) and K' = (2/3, 1/3) with this
NN convention, and also at their BZ-equivalent images (e.g. (1/3, 0))
because d3 = -d1 - d2 makes K and (1/3, 0) equivalent under this real
lattice. Zero-gap model: topology/AHC evaluation must avoid the
gap-closing points.
"""

from __future__ import annotations

import jax.numpy as jnp

from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel


def graphene(t: float = 2.7) -> WannierModel:
    """Build NN graphene.

    Parameters:
        t: NN hopping magnitude, eV (positive).

    Returns:
        Two-band model, Bravais x and y periodic.
    """
    lattice = Lattice(
        jnp.asarray(
            [[1.0, 0.0, 0.0], [0.5, float(jnp.sqrt(3.0)) / 2, 0.0], [0.0, 0.0, 1.0]]
        )
    )
    R_list: list[tuple[int, int, int]] = []
    H_list = []

    def blk(m, n, val):
        H = jnp.zeros((2, 2), dtype=jnp.complex128)
        return H.at[m, n].set(complex(val))

    R_list.append((0, 0, 0))
    H_list.append(jnp.zeros((2, 2), dtype=jnp.complex128))
    for d in [(1, 0, 0), (0, 1, 0), (-1, -1, 0)]:
        R_list.append(d)
        H_list.append(blk(0, 1, -t))
        R_list.append(tuple(-x for x in d))
        H_list.append(blk(1, 0, -t))

    R = jnp.asarray(R_list)
    H_R = jnp.stack(H_list)
    weights = jnp.ones(len(R_list))
    centers = jnp.zeros((2, 3))
    return WannierModel(lattice, R, H_R, weights, (True, True, False), centers)
