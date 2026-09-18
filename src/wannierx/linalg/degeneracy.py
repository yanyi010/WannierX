"""Degeneracy clustering on sorted eigenvalues (WX-016).

Contiguous clusters on ascending eigenvalues using both absolute and
relative criteria; no universal hard-coded threshold. Deterministic;
does not rotate eigenvectors.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array


def cluster_degenerate(
    energies: Array, atol: float = 1e-9, rtol: float = 1e-7
) -> np.ndarray:
    """Cluster sorted eigenvalues into degenerate groups.

    Two adjacent sorted energies e_i, e_{i+1} are degenerate when

        |e_{i+1} - e_i| <= atol + rtol * max(|e_i|, |e_{i+1}|).

    Parameters:
        energies: (..., n) eigenvalues. The last axis must be sorted
            ascending (as produced by :func:`wannierx.eigh`).
        atol: absolute tolerance, eV.
        rtol: relative tolerance (dimensionless).

    Returns:
        Integer array of shape (n, 2) with [start, end) index pairs of
        the contiguous clusters for a **single** spectrum. For batched
        input, clustering is computed per spectrum and all batches must
        yield the same cluster layout (this function returns the layout
        of batch element 0 and raises nothing); use
        :func:`is_degenerate` for batch-robust checks.

    Note: this routine is host-side (not JIT-traced), because cluster
    structure is discrete metadata.
    """
    e = np.asarray(energies)
    if e.ndim == 1:
        e_b = e[None]
    else:
        e_b = e.reshape(-1, e.shape[-1])
    de = np.abs(e_b[:, 1:] - e_b[:, :-1])
    scale = np.maximum(np.abs(e_b[:, 1:]), np.abs(e_b[:, :-1]))
    same = de <= (atol + rtol * scale)
    # layout from element 0 (deterministic reference spectrum)
    s = same[0]
    clusters: list[tuple[int, int]] = []
    start = 0
    for i, flag in enumerate(s):
        if not flag:
            clusters.append((start, i + 1))
            start = i + 1
    clusters.append((start, int(e_b.shape[1])))
    return np.asarray(clusters, dtype=int)


def is_degenerate(energies: Array, atol: float = 1e-9, rtol: float = 1e-7) -> Array:
    """Boolean mask: True where eigenvalue index belongs to a degenerate pair.

    Works on batched spectra: output shape matches input. True for band n
    if there exists an adjacent band with matching energy within tolerance.
    """
    e = jnp.asarray(energies)
    de = jnp.abs(jnp.diff(e, axis=-1))
    scale = jnp.maximum(jnp.abs(e[..., 1:]), jnp.abs(e[..., :-1]))
    pair = de <= (atol + rtol * scale)  # (..., n-1)
    left = jnp.concatenate([pair, jnp.zeros_like(pair[..., :1])], axis=-1)
    right = jnp.concatenate([jnp.zeros_like(pair[..., :1]), pair], axis=-1)
    return left | right
