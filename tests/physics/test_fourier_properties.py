"""WX-007: Fourier property/regression suite."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel
from wannierx.models.chain import chain
from wannierx.models.ssh import ssh

ATOL = 1e-13


def _random_hermitian_model(seed: int, n_R: int = 7, n_orb: int = 3) -> WannierModel:
    rng = np.random.default_rng(seed)
    lattice = Lattice(jnp.eye(3))
    R = rng.integers(-2, 3, size=(n_R, 3))
    # enforce -R partner for hermiticity
    R = np.concatenate([R, -R, [[0, 0, 0]]], axis=0)
    blocks = []
    for i in range(len(R)):
        if tuple(R[i]) == (0, 0, 0):
            h = rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb))
            h = h + h.conj().T
        else:
            h = rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb))
        blocks.append(h)
    blocks = np.asarray(blocks)
    # make pair (R, -R) Hermitian-conjugate
    n0 = len(blocks) - 1
    for i in range(n_R):
        blocks[n_R + i] = blocks[i].conj().T
    H_R = jnp.asarray(blocks)
    return WannierModel(lattice, jnp.asarray(R), H_R, jnp.ones(len(R)), (True, True, True))


def test_periodicity_random_hermitian() -> None:
    m = _random_hermitian_model(1)
    rng = np.random.default_rng(2)
    k = jnp.asarray(rng.normal(size=(9, 3)))
    for G in ([1, 0, 0], [0, -1, 2], [3, 1, -1]):
        np.testing.assert_allclose(
            np.asarray(wx.hamiltonian(m, k)),
            np.asarray(wx.hamiltonian(m, k + jnp.asarray(G, dtype=float))),
            atol=ATOL,
        )


def test_hermiticity_random_model() -> None:
    m = _random_hermitian_model(3)
    rep = wx.validate_hermiticity(m)
    assert rep.is_hermitian, rep
    k = jnp.asarray(np.random.default_rng(4).normal(size=(5, 3)))
    H = wx.hamiltonian(m, k)
    np.testing.assert_allclose(
        np.asarray(H), np.asarray(jnp.swapaxes(jnp.conjugate(H), -1, -2)), atol=ATOL
    )


def test_basis_invariance() -> None:
    """k-independent unitary basis change leaves eigenvalues invariant."""
    m = _random_hermitian_model(5, n_R=5, n_orb=3)
    rng = np.random.default_rng(6)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Q, _ = jnp.linalg.qr(jnp.asarray(a))
    # rotate all H_R: H_R -> Q^dagger H_R Q (k-independent similarity)
    Qd = jnp.conjugate(Q).T
    H_R2 = jnp.einsum("ij,Rjk,kl->Ril", Qd, m.H_R, Q)
    m2 = WannierModel(m.lattice, m.R, H_R2, m.weights, m.periodic)
    k = jnp.asarray(rng.normal(size=(4, 3)))
    E1 = wx.eigh(m, k).energies
    E2 = wx.eigh(m2, k).energies
    np.testing.assert_allclose(np.asarray(E1), np.asarray(E2), atol=1e-12)


def test_ssh_periodicity_hermiticity() -> None:
    m = ssh(t1=0.8, t2=1.2)
    assert wx.validate_hermiticity(m).is_hermitian
    k = jnp.array([[0.17, 0.0, 0.0]])
    for G in ([1, 0, 0], [2, 0, 0]):
        np.testing.assert_allclose(
            np.asarray(wx.hamiltonian(m, k)),
            np.asarray(wx.hamiltonian(m, k + jnp.asarray(G, dtype=float))),
            atol=ATOL,
        )
