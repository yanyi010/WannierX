"""Canonical k-mesh representation and uniform reciprocal grids.

Points are fractional reciprocal coordinates. Weights satisfy
sum_k w_k = 1 and represent a BZ average. Geometry
(Jacobian) factors live only in the integration layer.

Two canonical uniform grids are provided per direction with n_d > 1:

- ``centering="mp"`` (standard Monkhorst-Pack,
  Monkhorst & Pack, PRB 13, 5188 (1976)):
  k_d(i) = (2*i - n_d + 1) / (2*n_d), i = 0, ..., n_d - 1,
  which is off-Gamma for even n_d and Gamma-including for odd n_d.
- ``centering="gamma"``: k_d(i) = (i - n_d // 2) / n_d,
  which always includes Gamma.

Both grids coincide for odd n_d. An optional per-direction shift
(in units of the grid spacing 1/n_d) offsets either grid.
A direction with n_d == 1 holds the single point k_d = 0
(non-sampled / disabled direction).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    n = tuple(int(s) for s in shape)
    if len(n) != 3 or any(s < 1 for s in n):
        raise ValueError(f"mesh shape must be three positive ints, got {shape}")
    n1, n2, n3 = n
    return n1, n2, n3


def _validate_shift(shift: Sequence[float]) -> tuple[float, float, float]:
    sh = tuple(float(s) for s in shift)
    if len(sh) != 3:
        raise ValueError(f"shift must have three components, got {shift}")
    s1, s2, s3 = sh
    return s1, s2, s3


def _axis(n_d: int, shift_d: float, centering: Literal["gamma", "mp"]) -> Array:
    if n_d == 1:
        return jnp.asarray([0.0])
    i = jnp.arange(n_d)
    if centering == "gamma":
        coords = (i - n_d // 2 + shift_d) / n_d
    else:
        coords = (2 * i - n_d + 1) / (2 * n_d) + shift_d / n_d
    return coords


def _build_mesh(
    n: tuple[int, int, int],
    sh: tuple[float, float, float],
    centering: Literal["gamma", "mp"],
) -> KMesh:
    axes = [_axis(nd, sd, centering) for nd, sd in zip(n, sh, strict=True)]
    g1, g2, g3 = jnp.meshgrid(*axes, indexing="ij")
    points = jnp.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)
    n_k = n[0] * n[1] * n[2]
    weights = jnp.full(n_k, 1.0 / n_k)
    return KMesh(points=points, weights=weights, shape=n)


def gamma_centered_mesh(
    shape: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> KMesh:
    """Uniform Gamma-centered mesh in fractional reciprocal coordinates.

    Parameters:
        shape: (n1, n2, n3); n_d == 1 disables that direction (the single
            point k_d = 0).
        shift: per-direction shift in units of the grid spacing 1/n_d.

    Returns:
        :class:`KMesh` with uniform weights 1/(n1 n2 n3) each.
        Points are ordered with the first axis slowest (C order).
        k_d(i) = (i - n_d//2 + shift_d) / n_d, always including
        Gamma when shift = 0.
    """
    n = _validate_shape(shape)
    sh = _validate_shift(shift)
    return _build_mesh(n, sh, "gamma")


def monkhorst_pack_mesh(
    shape: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> KMesh:
    """Uniform standard Monkhorst-Pack mesh in fractional reciprocal
    coordinates.

    Implements the standard MP grid (Monkhorst & Pack, PRB 13, 5188
    (1976)): k_d(i) = (2*i - n_d + 1) / (2*n_d) plus an optional
    per-direction shift in units of the grid spacing 1/n_d. This grid is
    off-Gamma for even n_d and Gamma-including for odd n_d (shift = 0).

    Parameters:
        shape: (n1, n2, n3); n_d == 1 disables that direction (the single
            point k_d = 0).
        shift: per-direction shift in units of the grid spacing 1/n_d.

    Returns:
        :class:`KMesh` with uniform weights 1/(n1 n2 n3) each.
        Points are ordered with the first axis slowest (C order).
    """
    n = _validate_shape(shape)
    sh = _validate_shift(shift)
    return _build_mesh(n, sh, "mp")


Centering = Literal["gamma", "mp"]


def uniform_mesh(
    shape: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    centering: Centering = "mp",
) -> KMesh:
    """Uniform reciprocal mesh with selectable centering convention.

    Parameters:
        shape: (n1, n2, n3); n_d == 1 disables that direction (the single
            point k_d = 0).
        shift: per-direction shift in units of the grid spacing 1/n_d.
        centering: ``"mp"`` for the standard Monkhorst-Pack grid
            (off-Gamma for even n_d), ``"gamma"`` for a Gamma-centered
            grid. Both coincide for odd n_d with shift = 0.

    Returns:
        :class:`KMesh` with uniform weights 1/(n1 n2 n3) each.
        Points are ordered with the first axis slowest (C order).

    Raises:
        ValueError: if ``centering`` is not ``"gamma"`` or ``"mp"``.
    """
    if centering not in ("gamma", "mp"):
        raise ValueError(f"centering must be 'gamma' or 'mp', got {centering!r}")
    n = _validate_shape(shape)
    sh = _validate_shift(shift)
    return _build_mesh(n, sh, centering)


def monkhorst_pack(
    shape: Sequence[int],
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> KMesh:
    """Uniform standard Monkhorst-Pack mesh (compatibility alias).

    Alias of :func:`monkhorst_pack_mesh`: k_d(i) = (2*i - n_d + 1)/(2*n_d)
    plus an optional per-direction shift in units of the grid spacing
    1/n_d, i.e. off-Gamma for even n_d and Gamma-including for odd n_d
    (shift = 0).

    Parameters:
        shape: (n1, n2, n3); n_d == 1 disables that direction (the single
            point k_d = 0).
        shift: per-direction shift in units of the grid spacing 1/n_d.

    Returns:
        :class:`KMesh` with uniform weights 1/(n1 n2 n3) each.
        Points are ordered with the first axis slowest (C order).
    """
    return monkhorst_pack_mesh(shape, shift=shift)
