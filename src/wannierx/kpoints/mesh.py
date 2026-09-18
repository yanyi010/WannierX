"""Canonical k-mesh representation and uniform reciprocal grids (WX-021).

Points are fractional reciprocal coordinates. Weights satisfy
sum_k w_k = 1 and represent a BZ average (spec section 9). Geometry
(Jacobian) factors live only in the integration layer.

Grid convention (documented, deterministic): for a direction d with
n_d > 1 the coordinates are

    k_d(i) = (i - floor(n_d/2) + shift_d) / n_d,   i = 0, ..., n_d - 1

giving the standard Monkhorst-Pack set (off-Gamma for even n_d,
Gamma-including for odd n_d when shift=0). A direction with n_d == 1
holds the single point k_d = 0 (non-sampled / disabled direction).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class KMesh:
    """Uniform reciprocal mesh.

    Attributes:
        points: (n_k, 3) fractional reciprocal k-points.
        weights: (n_k,) real weights summing to 1 (BZ average).
        shape: (n1, n2, n3) grid shape; 1 disables that direction.

    Immutable JAX pytree.
    """

    points: Array
    weights: Array
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        p = jnp.asarray(self.points)
        w = jnp.asarray(self.weights)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"mesh points must be (n_k, 3), got {p.shape}")
        if w.shape != (p.shape[0],):
            raise ValueError(f"weights must be ({p.shape[0]},), got {w.shape}")
        object.__setattr__(self, "points", p)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))

    @property
    def n_k(self) -> int:
        """Number of k-points."""
        return int(self.points.shape[0])


def _flat(m: KMesh):
    return (m.points, m.weights), m.shape


def _unflat(shape, children):
    return KMesh(children[0], children[1], shape)


jax.tree_util.register_pytree_node(KMesh, _flat, _unflat)


def monkhorst_pack(
    shape: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> KMesh:
    """Uniform Monkhorst-Pack mesh in fractional reciprocal coordinates.

    Parameters:
        shape: (n1, n2, n3); n_d == 1 disables that direction (the single
            point k_d = 0).
        shift: per-direction shift in units of the grid spacing.

    Returns:
        :class:`KMesh` with uniform weights 1/(n1 n2 n3) each.
        Points are ordered with the first axis slowest (C order).
    """
    n = tuple(int(s) for s in shape)
    if len(n) != 3 or any(s < 1 for s in n):
        raise ValueError(f"mesh shape must be three positive ints, got {shape}")
    sh = tuple(float(s) for s in shift)
    axes = []
    for nd, sd in zip(n, sh, strict=True):
        if nd == 1:
            axes.append(jnp.asarray([0.0]))
        else:
            i = jnp.arange(nd)
            axes.append((i - nd // 2 + sd) / nd)
    g1, g2, g3 = jnp.meshgrid(*axes, indexing="ij")
    points = jnp.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)
    n_k = n[0] * n[1] * n[2]
    weights = jnp.full(n_k, 1.0 / n_k)
    return KMesh(points=points, weights=weights, shape=n)
