"""Band/subspace projectors (WX-017).

P_S = sum_{n in S} |u_n><u_n|, with the canonical column-eigenvector
convention U[..., :, n] = |u_n>. Projector formulations are preferred
for degenerate manifolds and gauge-covariant observables.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array


def projector(vectors: Array, bands: Sequence[int] | Array) -> Array:
    """Subspace projector from eigenvectors.

    Parameters:
        vectors: (..., n_orb, n_orb) with columns |u_n>.
        bands: band indices selecting the subspace. May be a Python
            sequence or integer array of any shape (flattened).

    Returns:
        Complex (..., n_orb, n_orb) projector P = U_S U_S^dagger.

    Properties (tested): Hermitian, idempotent, trace == len(bands),
    invariant under unitary rotations within the selected subspace.
    Differentiable w.r.t. ``vectors``.
    """
    U = jnp.asarray(vectors)
    idx = jnp.asarray(bands).reshape(-1)
    U_s = U[..., :, idx]  # (..., n_orb, n_sel)
    return U_s @ jnp.swapaxes(jnp.conjugate(U_s), -1, -2)
