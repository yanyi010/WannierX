"""H(k) tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel
from wannierx.models.chain import chain
from wannierx.models.qzhang import qiwuzhang

ATOL = 1e-14


def _random_hermitian_model(seed: int, n_R: int = 6, n_orb: int = 3) -> WannierModel:
    """Random Hermitian model: (R, -R) blocks are conjugate pairs."""
    rng = np.random.default_rng(seed)
    R = rng.integers(-2, 3, size=(n_R, 3))
    R = np.concatenate([R, -R, [[0, 0, 0]]], axis=0)
    # construction contract: R vectors must be unique (random draws can
    # collide with each other, their negatives, or the origin)
    R = np.unique(R, axis=0)
    n = len(R)
    neg_index = {tuple(-r): i for i, r in enumerate(R)}
    blocks = rng.normal(size=(n, n_orb, n_orb)) + 1j * rng.normal(size=(n, n_orb, n_orb))
    for i, r in enumerate(R):
        j = neg_index.get(tuple(r))
        if j is None:
            raise AssertionError("fixture bug: R set not closed under negation")
        if j == i:  # origin: self-partner, symmetrize
            blocks[i] = blocks[i] + blocks[i].conj().T
        elif j > i:
            blocks[j] = blocks[i].conj().T
    return WannierModel(
        Lattice(jnp.eye(3)),
        jnp.asarray(R),
        jnp.asarray(blocks),
        jnp.ones(len(R)),
        (True, True, True),
    )


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


def test_shape_contract() -> None:
    """Locked contract: k:(3,)->H:(n,n); k:(Nk,3)->H:(Nk,n,n); k:(B,Nk,3)->H:(B,Nk,n,n)."""
    m = qiwuzhang(u=-1.0)  # n_orb = 2
    n = m.n_orb
    assert wx.hamiltonian(m, jnp.zeros(3)).shape == (n, n)
    assert wx.hamiltonian(m, jnp.zeros((7, 3))).shape == (7, n, n)
    assert wx.hamiltonian(m, jnp.zeros((2, 5, 3))).shape == (2, 5, n, n)


def test_scalar_k_matches_batch_row() -> None:
    m = qiwuzhang(u=-1.0)
    k0 = jnp.array([0.31, -0.17, 0.05])
    H_single = np.asarray(wx.hamiltonian(m, k0))
    H_batch = np.asarray(wx.hamiltonian(m, k0[None]))
    assert H_single.shape == H_batch.shape[1:]
    np.testing.assert_allclose(H_single, H_batch[0], atol=ATOL)


def test_property_periodicity_random_hermitian() -> None:
    """Property: for random Hermitian H_R, H(k+G) == H(k) for integer G."""
    rng = np.random.default_rng(123)
    for seed in (11, 22, 33):
        m = _random_hermitian_model(seed)
        k = jnp.asarray(rng.normal(size=(6, 3)))
        for G in ([1, 0, 0], [0, -2, 1], [3, -1, 2]):
            G_arr = jnp.asarray(G, dtype=k.dtype)
            np.testing.assert_allclose(
                np.asarray(wx.hamiltonian(m, k)),
                np.asarray(wx.hamiltonian(m, k + G_arr)),
                atol=1e-12,
            )
            # single-k point too
            np.testing.assert_allclose(
                np.asarray(wx.hamiltonian(m, k[0])),
                np.asarray(wx.hamiltonian(m, k[0] + G_arr)),
                atol=1e-12,
            )
