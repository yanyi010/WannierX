"""Analytic k-derivative tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.models.chain import chain
from wannierx.models.qzhang import qiwuzhang

ATOL = 1e-12


def _chain_k(nx: int = 9) -> jnp.ndarray:
    kf = jnp.linspace(-0.4, 0.4, nx)
    return jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], axis=-1)


def test_chain_dH_dk_analytic() -> None:
    t, eps0 = -1.25, 0.4
    m = chain(t=t, onsite=eps0)
    k = _chain_k()
    dH = wx.dH_dk(m, k)  # (..., 3, 1, 1)
    # E(k) = eps0 + 2 t cos(2 pi kf) is scalar; dH/dKx must equal its derivative
    # dE/dKx = -2 t * sin(2 pi kf) * 2 pi * a=1 ... careful: cartesian K=2pi kf /a, a=1
    # dE/dKx = dE/dkf * dkf/dKx = (-4 pi t sin(2pi kf)) * (1/2pi) = -2 t sin(2 pi kf)
    expect = -2 * t * jnp.sin(2 * jnp.pi * k[:, 0])
    np.testing.assert_allclose(np.asarray(dH[:, 0, 0, 0]).real, np.asarray(expect), atol=ATOL)
    np.testing.assert_allclose(np.asarray(dH[:, 1:]), 0.0, atol=ATOL)


def test_chain_d2H_dk2_analytic_and_symmetry() -> None:
    t, eps0 = -1.25, 0.4
    m = chain(t=t, onsite=eps0)
    k = _chain_k(5)
    d2 = wx.d2H_dk2(m, k)  # (..., 3, 3, 1, 1)
    # d2E/dKx^2 = -2 t * (2 pi cos(2pi kf)) * (1/2pi) ... = -4 pi^? , exact:
    # dE/dKx = -2 t sin(2 pi kf); d2E/dKx^2 = d/dKx[-2 t sin(Kx a)] = -2 t a cos(Kx a) * a = -2 t cos(2 pi kf)
    expect = -2 * t * jnp.cos(2 * jnp.pi * k[:, 0])
    np.testing.assert_allclose(np.asarray(d2[:, 0, 0, 0, 0]).real, np.asarray(expect), atol=ATOL)
    # symmetric in (a, b)
    np.testing.assert_allclose(
        np.asarray(d2), np.asarray(jnp.swapaxes(d2, -4, -3)), atol=ATOL
    )
    np.testing.assert_allclose(np.asarray(d2[:, 1:, :, 0, 0]), 0.0, atol=ATOL)


def test_dH_dk_multi_step_finite_difference() -> None:
    """Analytic matches central finite differences with O(h^2) convergence."""
    m = qiwuzhang(u=-1.0)
    k0 = np.array([0.31, 0.22, 0.0])
    lat = np.asarray(m.lattice.direct)
    B = 2 * np.pi * np.linalg.inv(lat).T  # k cart row-vec Jacobian

    def dH_fd(h: float, comp: int) -> np.ndarray:
        # step along cartesian component `comp`
        dk_cart = np.zeros(3)
        dk_cart[comp] = h
        dk_frac = dk_cart @ np.linalg.inv(B)
        Hp = np.asarray(wx.hamiltonian(m, jnp.asarray(k0 + dk_frac)))[0]
        Hm = np.asarray(wx.hamiltonian(m, jnp.asarray(k0 - dk_frac)))[0]
        return (Hp - Hm) / (2 * h)

    dH = np.asarray(wx.dH_dk(m, jnp.array([k0])))[0]
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        for comp in (0, 1):
            errs.append(np.abs(dH[comp] - dH_fd(h, comp)).max())
    # error must decrease, showing the expected truncation/roundoff behavior
    assert errs[2] < errs[0] and errs[3] < errs[1]
    assert max(errs) < 1e-6


def test_dH_dk_hermitian_jit_batch() -> None:
    m = qiwuzhang(u=-1.0)
    k = jnp.asarray(np.random.default_rng(0).normal(size=(6, 3)) * 0.3)
    dH = wx.dH_dk(m, k)
    dHd = jnp.swapaxes(jnp.conjugate(dH), -1, -2)
    np.testing.assert_allclose(np.asarray(dH), np.asarray(dHd), atol=ATOL)
    fj = jax.jit(lambda kk: wx.dH_dk(m, kk))
    np.testing.assert_allclose(np.asarray(fj(k)), np.asarray(dH), atol=ATOL)


def test_d2H_dk2_fd_of_dH_dk() -> None:
    m = qiwuzhang(u=-1.0)
    k0 = np.array([0.31, 0.22, 0.0])
    B = 2 * np.pi * np.linalg.inv(np.asarray(m.lattice.direct)).T

    def d2_fd(h: float) -> np.ndarray:
        dk = np.zeros(3)
        dk[0] = h
        dkp = (k0 + dk @ np.linalg.inv(B))
        dkm = (k0 - dk @ np.linalg.inv(B))
        dp = np.asarray(wx.dH_dk(m, jnp.asarray(dkp)))[0, 0]
        dm = np.asarray(wx.dH_dk(m, jnp.asarray(dkm)))[0, 0]
        return (dp - dm) / (2 * h)

    d2 = np.asarray(wx.d2H_dk2(m, jnp.array([k0])))[0, 0, 0]
    assert np.abs(d2 - d2_fd(1e-4)).max() < 1e-8
