"""Reusable chunked BZ reductions.

Centralizes mesh-weight handling and k-chunking. Weights satisfy sum w_k = 1
(BZ average). Chunking never changes numerical semantics. Low-level kernels
accept any batch; chunk loops live only here (host-side scheduler).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel
from wannierx.kpoints.mesh import KMesh


def integrate(
    model: WannierModel,
    mesh: KMesh,
    kernel: Callable[[WannierModel, Array], Array],
    chunk_size: int | None = None,
) -> Array:
    """Weighted BZ reduction of a per-k kernel.

    Computes sum_k w_k kernel(model, k_chunk), where ``kernel`` maps a
    batch of fractional k-points (Nc, 3) to (Nc, ...). Leading batch
    dims are the chunk's k-batch.

    Parameters:
        model: canonical model.
        mesh: KMesh with normalized weights.
        kernel: pure JAX function (k_chunk) -> (Nk_chunk, ...).
        chunk_size: optional max k per chunk; None evaluates all at once.

    Returns:
        Weighted reduction with kernel's trailing shape (no k axis).

    Numerical semantics are chunk-size independent. The per-chunk kernel
    is jitted here; callers must keep ``kernel`` pure (no I/O, no RNG).
    """
    k = mesh.points
    w = mesh.weights
    kern = jax.jit(kernel)
    if chunk_size is None or chunk_size >= mesh.n_k:
        v = kern(model, k)
        return jnp.tensordot(w, v, axes=(0, 0))

    outs = []
    for i in range(0, mesh.n_k, chunk_size):
        kc = k[i : i + chunk_size]
        wc = w[i : i + chunk_size]
        v = kern(model, kc)
        outs.append(jnp.tensordot(wc, v, axes=(0, 0)))
    out = outs[0]
    for o in outs[1:]:
        out = out + o
    return out


def integrate_bands(
    model: WannierModel,
    mesh: KMesh,
    kernel: Callable[[WannierModel, Array], Array],
    chunk_size: int | None = None,
) -> Array:
    """BZ reduction where ``kernel`` returns (Nk, n_bands[, ...]).

    The band axis is preserved; only the k axis is reduced. Semantics and
    chunking rules match :func:`integrate`.
    """
    k = mesh.points
    w = mesh.weights
    kern = jax.jit(kernel)
    if chunk_size is None or chunk_size >= mesh.n_k:
        v = kern(model, k)
        return jnp.einsum("k,k...->...", w, v)

    out = None
    for i in range(0, mesh.n_k, chunk_size):
        kc = k[i : i + chunk_size]
        wc = w[i : i + chunk_size]
        v = kern(model, kc)
        acc = jnp.einsum("k,k...->...", wc, v)
        out = acc if out is None else out + acc
    return out
