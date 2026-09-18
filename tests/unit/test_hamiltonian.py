"""WX-006: H(k) tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.models.chain import chain

ATOL = 1e-14


def _kpts(nx: int = 7) -> jnp.ndarray:
    kf = jnp.linspace(-0.5, 0.5, nx)
    return jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], axis=-1)


def test_analytic_chain() -> None:
    eps0, t = 0.3, -1.25
    m = chain(t=t, onsite=eps0)
    k = _kpts()
    H = wx.hamiltonian(m, k)[:, 0, 0]
    expect = eps0 + 2 * t * jnp.cos(2 * jnp.pi * k[:, 0])
    np.testing.assert_allclose(np.asarray(H).real, np.asarray(expect), atol=ATOL)


def test_periodic_under_integer_G() -> None:
    m = chain(t=-1.0)
    k = _kpts(5)
    G = jnp.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(wx.hamiltonian(m, k)),
        np.asarray(wx.hamiltonian(m, k + G)),
        atol=ATOL,
    )


def test_hermitian_output() -> None:
    m = chain(t=-1.0)
    H = wx.hamiltonian(m, _kpts(5))
    np.testing.assert_allclose(
        np.asarray(H), np.asarray(jnp.swapaxes(jnp.conjugate(H), -1, -2)), atol=ATOL
    )


def test_scalar_vs_batch() -> None:
    m = chain(t=-1.0)
    k_batch = _kpts(6)
    Hb = np.asarray(wx.hamiltonian(m, k_batch))
    for i in range(k_batch.shape[0]):
        Hi = np.asarray(wx.hamiltonian(m, k_batch[i : i + 1]))[0]
        np.testing.assert_allclose(Hb[i], Hi, atol=ATOL)


def test_jit_and_vmap() -> None:
    m = chain(t=-1.0)
    f = jax.jit(lambda k: wx.hamiltonian(m, k))
    H = f(_kpts(4))
    assert H.shape == (4, 1, 1)
    g = jax.vmap(lambda kk: wx.hamiltonian(m, kk[None])[0, 0, 0])
    np.testing.assert_allclose(np.asarray(g(_kpts(4))), np.asarray(H)[:, 0, 0], atol=ATOL)


def test_batch_dims_preserved() -> None:
    m = chain(t=-1.0)
    k = jnp.zeros((2, 3, 3))
    assert wx.hamiltonian(m, k).shape == (2, 3, 1, 1)
