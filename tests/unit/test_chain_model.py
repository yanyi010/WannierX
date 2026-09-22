"""1D chain fixture tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from wannierx.models.chain import chain


def test_exact_data() -> None:
    m = chain(t=-1.5, onsite=0.25, lattice_constant=2.5)
    np.testing.assert_array_equal(
        np.asarray(m.R), np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    )
    np.testing.assert_allclose(
        np.asarray(m.H_R).ravel(), [-1.5, 0.25, -1.5], atol=0.0
    )
    np.testing.assert_array_equal(np.asarray(m.weights), np.ones(3))
    assert m.periodic == (True, False, False)
    np.testing.assert_allclose(
        np.asarray(m.lattice.direct), np.diag([2.5, 1.0, 1.0]), atol=0.0
    )


def test_analytic_dispersion() -> None:
    import wannierx as wx

    eps0, t = 0.3, -1.25
    m = chain(t=t, onsite=eps0, lattice_constant=1.0)
    kf = jnp.linspace(-0.5, 0.5, 21)
    kpts = jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], axis=-1)
    E = wx.eigh(m, kpts).energies[:, 0]
    expect = eps0 + 2 * t * jnp.cos(2 * jnp.pi * kf)
    np.testing.assert_allclose(np.asarray(E), np.asarray(expect), atol=1e-14)
