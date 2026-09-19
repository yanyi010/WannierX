"""WX-024: chunked BZ integrator tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.kpoints.integration import integrate, integrate_bands
from wannierx.kpoints.mesh import monkhorst_pack
from wannierx.models.chain import chain
from wannierx.models.qzhang import qiwuzhang

ATOL = 1e-13


def test_constant_integrand() -> None:
    m = chain(t=-1.0)
    mesh = monkhorst_pack((8, 4, 1))
    out = integrate(m, mesh, lambda md, k: jnp.ones((k.shape[0], 3)))
    np.testing.assert_allclose(np.asarray(out), np.ones(3), atol=ATOL)


def test_polynomial_toy() -> None:
    # integral of cos(2pi kx) over BZ = 0 exactly on this uniform grid
    m = chain(t=-1.0)
    mesh = monkhorst_pack((32, 1, 1))
    out = integrate(m, mesh, lambda md, k: jnp.cos(2 * jnp.pi * k[:, 0]))
    assert abs(float(out)) < 1e-14


def test_chunk_size_invariance() -> None:
    m = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((32, 32, 1))

    def kern(md, kc):
        return jnp.sum(wx.eigh(md, kc).energies, axis=-1)

    full = integrate(m, mesh, kern)
    c7 = integrate(m, mesh, kern, chunk_size=7)
    c64 = integrate(m, mesh, kern, chunk_size=64)
    np.testing.assert_allclose(np.asarray(full), np.asarray(c7), atol=1e-14)
    np.testing.assert_allclose(np.asarray(full), np.asarray(c64), atol=1e-14)


def test_integrate_bands_preserves_band_axis() -> None:
    m = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((16, 16, 1))

    def kern(md, kc):
        return wx.eigh(md, kc).energies  # (nk, norb)

    out = integrate_bands(m, mesh, kern)
    assert out.shape == (m.n_orb,)
    # sum of band BZ averages: each band individually BZ-averages to 0
    # (sin/cos are zero-mean), hence the total is also 0.
    np.testing.assert_allclose(float(jnp.sum(out)), 0.0, atol=1e-13)


def test_chunked_jit_kernel() -> None:
    m = chain(t=-1.0)
    mesh = monkhorst_pack((64, 1, 1))
    kern = jax.jit(lambda md, kc: jnp.sum(wx.eigh(md, kc).energies, axis=-1))
    a = integrate(m, mesh, kern)
    b = integrate(m, mesh, kern, chunk_size=13)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-15)
