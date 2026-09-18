"""WX-003: WannierModel tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wannierx.core.exceptions import ModelError
from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel
from wannierx.models.chain import chain


def _toy() -> WannierModel:
    return chain(t=-1.0)


def test_construction_and_shape_rejection() -> None:
    m = _toy()
    assert m.n_orb == 1 and m.n_R == 3
    lat = Lattice(jnp.eye(3))
    with pytest.raises(ModelError):
        WannierModel(lat, jnp.zeros((2, 3)), jnp.zeros((3, 1, 1)), jnp.ones(3), (True,) * 3)
    with pytest.raises(ModelError):
        WannierModel(lat, jnp.zeros((3, 3)), jnp.zeros((3, 2, 3)), jnp.ones(3), (True,) * 3)
    with pytest.raises(ModelError):
        WannierModel(lat, jnp.zeros((3, 3)), jnp.zeros((3, 2, 2)), jnp.ones(2), (True,) * 3)
    with pytest.raises(ModelError):
        WannierModel(lat, jnp.zeros((1, 3)), jnp.zeros((1, 1, 1)), jnp.ones(1),
                     (True,) * 3, centers=jnp.zeros((2, 3)))


def test_pytree_roundtrip() -> None:
    m = _toy()
    leaves, treedef = jax.tree_util.tree_flatten(m)
    m2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert m2.periodic == m.periodic
    np.testing.assert_array_equal(np.asarray(m2.R), np.asarray(m.R))
    np.testing.assert_array_equal(np.asarray(m2.H_R), np.asarray(m.H_R))


def test_jit_function_accepts_model() -> None:
    m = _toy()

    @jax.jit
    def trace_norm(mm: WannierModel) -> jnp.ndarray:
        return jnp.trace(mm.H_R, axis1=-2, axis2=-1).sum()

    out = trace_norm(m)
    assert np.isfinite(np.asarray(out))


def test_complex128_preserved() -> None:
    assert jnp.asarray(_toy().H_R).dtype == jnp.complex128
    assert jnp.asarray(_toy().lattice.direct).dtype == jnp.float64
