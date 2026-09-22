"""Staging guards: host validation vs jit/vmap/checkify staging.

Locks the v0.1.1 fix for staging detection: checkify re-traces
transformed bodies with ArrayImpl leaves that np.asarray *can* convert
while JAX operations on them stay traced, so tracer-type-based and
np.asarray-based guards misclassify them. All guards route through
wannierx.core.staging.is_staged, which probes via a JAX reduction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import wannierx as wx
from wannierx.core.staging import is_staged
from wannierx.models.graphene import graphene


def test_is_staged_concrete_and_traced() -> None:
    assert not is_staged(jnp.asarray([1.0, 2.0]))
    assert not is_staged(jnp.zeros((3, 3), dtype=jnp.complex128))
    # wrap in jnp.asarray: is_staged returns a host bool, jit needs an
    # array output; the staged input is detected during tracing
    assert bool(jax.jit(lambda x: jnp.asarray(is_staged(x)))(jnp.asarray([1.0])))
    out = jax.vmap(lambda x: jnp.asarray(is_staged(x)))(jnp.asarray([[1.0], [2.0]]))
    assert np.all(np.asarray(out))


def test_is_staged_checkify_retraced_leaves() -> None:
    # checkify re-traces the body with ArrayImpl leaves for which
    # np.asarray succeeds; the reduction probe must still detect them
    results: list[bool] = []

    def probe(x):
        results.append(is_staged(x))
        return x

    checkify.checkify(jax.vmap(probe))(jnp.zeros((2, 3)))
    # outer (concrete) call + inner re-trace: the re-traced leaf is staged
    assert True in results


def test_lattice_and_model_pytree_under_checkify_vmap() -> None:
    """Regression: model arguments crashed Lattice validation under
    checkify(vmap(..., in_axes=(None, 0))) before the fix."""
    m = graphene()
    ks = jnp.asarray([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]])
    f = jax.vmap(wx.hamiltonian, in_axes=(None, 0))
    err, out = checkify.checkify(f)(m, ks)
    err.throw()
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(
        np.asarray(out[0]),
        np.asarray(wx.hamiltonian(m, ks[0])),
        atol=1e-14,
    )


def test_model_validation_still_fires_eagerly() -> None:
    """Staging guards must not weaken concrete-path validation."""
    m = graphene()
    with pytest.raises(Exception, match="finite"):
        wx.WannierModel(
            m.lattice,
            m.R,
            m.H_R.at[0, 0, 0].set(jnp.nan),
            m.weights,
            m.periodic,
            m.centers,
        )
    with pytest.raises(Exception, match="unique"):
        R = m.R.at[2].set(m.R[0])
        wx.WannierModel(m.lattice, R, m.H_R, m.weights, m.periodic, m.centers)
    with pytest.raises(Exception, match="singular"):
        wx.Lattice(jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_jit_and_vmap_accept_model() -> None:
    m = graphene()
    k = jnp.asarray([0.1, 0.2, 0.0])
    np.testing.assert_allclose(
        np.asarray(jax.jit(wx.hamiltonian)(m, k)),
        np.asarray(wx.hamiltonian(m, k)),
        atol=1e-14,
    )
    ks = jnp.stack([k, k])
    np.testing.assert_allclose(
        np.asarray(jax.vmap(wx.hamiltonian, in_axes=(None, 0))(m, ks))[0],
        np.asarray(wx.hamiltonian(m, k)),
        atol=1e-14,
    )
