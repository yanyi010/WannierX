"""Canonical direct/reciprocal lattice.

Conventions:

- Direct lattice ``A`` has lattice vectors as **rows**, units Angstrom.
- ``r_cart = r_frac @ A``.
- Reciprocal lattice ``B = 2*pi * A^{-T}``, so ``A B^T = 2*pi I``.
- ``k_cart = k_frac @ B``.

All coordinate transforms accept arrays with arbitrary leading batch
dimensions; the last axis has length 3. The object is immutable, a JAX
pytree, and contains no file-format knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from wannierx.core.exceptions import LatticeError


def _as_float3x3(A: Any) -> Array:
    a = jnp.asarray(A)
    if a.shape != (3, 3):
        raise LatticeError(f"direct lattice must have shape (3, 3), got {a.shape}")
    return a


@dataclass(frozen=True)
class Lattice:
    """Direct lattice with derived reciprocal lattice.

    Attributes:
        direct: (3, 3) direct lattice, vectors as rows, Angstrom.
    """

    direct: Array

    def __post_init__(self) -> None:
        A = _as_float3x3(self.direct)
        object.__setattr__(self, "direct", A)
        # Concrete (non-traced) construction validates singularity; traced
        # pytree unflattens skip this host check (shape contract is static).
        if not isinstance(A, jax.core.Tracer):
            det = float(jnp.linalg.det(A))
            if not np.isfinite(det) or abs(det) < 1e-12:
                raise LatticeError(
                    f"direct lattice is singular or ill-conditioned (det={det!r})"
                )

    # -- reciprocal ---------------------------------------------------
    @property
    def reciprocal(self) -> Array:
        """Reciprocal lattice ``B = 2*pi*A^{-T}`` (rows), Angstrom^-1."""
        return 2.0 * jnp.pi * jnp.linalg.inv(self.direct).T

    # -- real-space transforms ---------------------------------------
    def frac_to_cart_real(self, r_frac: Array) -> Array:
        """``r_cart = r_frac @ A``. Input shape (..., 3) -> (..., 3)."""
        return jnp.asarray(r_frac) @ self.direct

    def cart_to_frac_real(self, r_cart: Array) -> Array:
        """Inverse of :meth:`frac_to_cart_real`."""
        return jnp.asarray(r_cart) @ jnp.linalg.inv(self.direct)

    # -- reciprocal-space transforms ---------------------------------
    def frac_to_cart_k(self, k_frac: Array) -> Array:
        """``k_cart = k_frac @ B``. Input shape (..., 3) -> (..., 3), Angstrom^-1."""
        return jnp.asarray(k_frac) @ self.reciprocal

    def cart_to_frac_k(self, k_cart: Array) -> Array:
        """Inverse of :meth:`frac_to_cart_k`."""
        return jnp.asarray(k_cart) @ jnp.linalg.inv(self.reciprocal)

    # -- geometry -----------------------------------------------------
    @property
    def cell_volume(self) -> Array:
        """Direct-cell volume |det(A)|, Angstrom^3."""
        return jnp.abs(jnp.linalg.det(self.direct))

    @property
    def bz_volume(self) -> Array:
        """Brillouin-zone volume |det(B)|, Angstrom^-3."""
        return jnp.abs(jnp.linalg.det(self.reciprocal))


def _lattice_flatten(lat: Lattice):
    return (lat.direct,), None


def _lattice_unflatten(_, children):
    return Lattice(children[0])


jax.tree_util.register_pytree_node(Lattice, _lattice_flatten, _lattice_unflatten)
