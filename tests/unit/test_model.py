"""WannierModel tests."""

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


def _clone_value(m: WannierModel, **kw) -> WannierModel:
    return WannierModel(
        lattice=kw.get("lattice", m.lattice),
        R=kw.get("R", m.R),
        H_R=kw.get("H_R", m.H_R),
        weights=kw.get("weights", m.weights),
        periodic=kw.get("periodic", m.periodic),
        centers=kw.get("centers", m.centers),
    )


def test_construction_rejects_non_integer_R() -> None:
    m = _toy()
    R = m.R.astype(jnp.float64).at[0, 0].set(0.5)
    with pytest.raises(ModelError, match="integer"):
        _clone_value(m, R=R)


def test_construction_accepts_integer_valued_float_R() -> None:
    m = _toy()
    R = m.R.astype(jnp.float64)  # integer-valued floats are fine
    m2 = _clone_value(m, R=R)
    np.testing.assert_array_equal(np.asarray(m2.R), np.asarray(m.R))


def test_construction_rejects_duplicate_R() -> None:
    m = _toy()
    R = m.R.at[2].set(m.R[0])  # duplicate (-1,0,0)
    with pytest.raises(ModelError, match="unique"):
        _clone_value(m, R=R)


def test_construction_rejects_non_finite_H_R() -> None:
    m = _toy()
    for bad in (jnp.nan, jnp.inf):
        H_R = m.H_R.at[0, 0, 0].set(bad)
        with pytest.raises(ModelError, match=r"H_R.*finite"):
            _clone_value(m, H_R=H_R)


def test_construction_rejects_non_finite_weights() -> None:
    m = _toy()
    for bad in (jnp.nan, jnp.inf):
        weights = m.weights.at[0].set(bad)
        with pytest.raises(ModelError, match=r"weights.*finite"):
            _clone_value(m, weights=weights)


def test_construction_rejects_zero_weights() -> None:
    m = _toy()
    weights = m.weights.at[0].set(0.0)
    with pytest.raises(ModelError, match=r"weights.*nonzero"):
        _clone_value(m, weights=weights)


def test_construction_rejects_non_finite_lattice() -> None:
    m = _toy()
    for bad in (jnp.nan, jnp.inf):
        # bypass Lattice's own host check so the *model-level* finite
        # check is what fires (pytree unflatten of a traced lattice
        # also skips Lattice.__post_init__ checks)
        lat = Lattice.__new__(Lattice)
        object.__setattr__(lat, "direct", jnp.asarray(m.lattice.direct).at[0, 0].set(bad))
        with pytest.raises(ModelError, match=r"lattice.*finite"):
            _clone_value(m, lattice=lat)
