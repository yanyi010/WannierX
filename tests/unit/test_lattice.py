"""WX-002: Lattice tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wannierx.core.exceptions import LatticeError
from wannierx.core.lattice import Lattice

ATOL = 1e-14


def test_cubic_orthogonal() -> None:
    lat = Lattice(jnp.eye(3) * 4.0)
    assert lat.direct.shape == (3, 3)
    B = np.asarray(lat.reciprocal)
    np.testing.assert_allclose(B, np.eye(3) * (2 * np.pi / 4.0), atol=ATOL)
    # A B^T = 2 pi I
    np.testing.assert_allclose(
        np.asarray(lat.direct) @ B.T, 2 * np.pi * np.eye(3), atol=ATOL
    )


def test_nonorthogonal_roundtrip() -> None:
    A = jnp.asarray([[1.0, 0.2, 0.0], [0.0, 2.0, 0.1], [0.1, 0.0, 3.0]])
    lat = Lattice(A)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(7, 3)))
    # round trips
    np.testing.assert_allclose(
        np.asarray(lat.cart_to_frac_real(lat.frac_to_cart_real(r))),
        np.asarray(r),
        atol=ATOL,
    )
    k = jnp.asarray(rng.normal(size=(7, 3)))
    np.testing.assert_allclose(
        np.asarray(lat.cart_to_frac_k(lat.frac_to_cart_k(k))), np.asarray(k), atol=1e-13
    )
    # reciprocal identity on a non-orthogonal cell
    B = np.asarray(lat.reciprocal)
    np.testing.assert_allclose(
        np.asarray(lat.direct) @ B.T, 2 * np.pi * np.eye(3), atol=1e-13
    )


def test_batch_shapes() -> None:
    lat = Lattice(jnp.eye(3))
    r = jnp.zeros((2, 3, 4, 3))
    assert lat.frac_to_cart_real(r).shape == (2, 3, 4, 3)
    assert lat.frac_to_cart_k(r).shape == (2, 3, 4, 3)


def test_singular_rejected() -> None:
    with pytest.raises(LatticeError):
        Lattice(jnp.zeros((3, 3)))
    with pytest.raises(LatticeError):
        Lattice(jnp.asarray([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]]))


def test_pytree_and_jit() -> None:
    lat = Lattice(jnp.eye(3) * 2.0)

    @jax.jit
    def f(lat_in: Lattice, k) -> jnp.ndarray:
        return lat_in.frac_to_cart_k(k)

    out = f(lat, jnp.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(jnp.array([0.5, 0.0, 0.0])) @ np.asarray(lat.reciprocal), atol=1e-14
    )


def test_volumes() -> None:
    lat = Lattice(jnp.eye(3) * 2.0)
    np.testing.assert_allclose(float(lat.cell_volume), 8.0, atol=ATOL)
    np.testing.assert_allclose(float(lat.bz_volume), (2 * np.pi / 2.0) ** 3, rtol=1e-13)
