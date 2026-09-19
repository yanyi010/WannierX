"""WX-014..017: eigh, velocity, degeneracy clustering, projectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.constants import EV_ANG_OVER_HBAR_TO_M_S
from wannierx.linalg.degeneracy import cluster_degenerate, is_degenerate
from wannierx.models.chain import chain
from wannierx.models.qzhang import qiwuzhang
from wannierx.models.ssh import ssh

ATOL = 1e-12


def test_eigh_reconstruction_orthonormal() -> None:
    m = qiwuzhang(u=-1.0)
    k = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)) * 0.2)
    eig = wx.eigh(m, k)
    U, E = eig.vectors, eig.energies
    H = wx.hamiltonian(m, k)
    Ud = jnp.swapaxes(jnp.conjugate(U), -1, -2)
    rec = U @ (E[..., None] * jnp.eye(m.n_orb)) @ Ud
    np.testing.assert_allclose(np.asarray(rec), np.asarray(H), atol=ATOL)
    np.testing.assert_allclose(
        np.asarray(Ud @ U), np.broadcast_to(jnp.eye(m.n_orb), Ud.shape), atol=ATOL
    )
    # ascending
    assert jnp.all(E[..., 1:] >= E[..., :-1] - ATOL)


def test_eigh_ssh_analytic() -> None:
    m = ssh(t1=0.8, t2=1.2)
    kf = jnp.linspace(-0.4, 0.4, 9)
    k = jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], -1)
    E = wx.eigh(m, k).energies
    Ea = jnp.sqrt(0.8**2 + 1.2**2 + 2 * 0.8 * 1.2 * jnp.cos(2 * jnp.pi * kf))
    np.testing.assert_allclose(np.asarray(E), np.stack([-Ea, Ea], -1), atol=ATOL)
    np.testing.assert_allclose(np.asarray(E[:, 0]), -np.asarray(E[:, 1]), atol=ATOL)


def test_eigh_jit_batch() -> None:
    m = chain(t=-1.0)
    k = jnp.asarray([[0.1, 0, 0], [0.2, 0, 0]])
    f = jax.jit(lambda kk: wx.eigh(m, kk).energies)
    np.testing.assert_allclose(np.asarray(f(k)), np.asarray(wx.eigh(m, k).energies), atol=ATOL)


def test_velocity_chain_group_velocity() -> None:
    m = chain(t=-1.0)  # E(K) = -2 cos(K), K cartesian
    k = jnp.array([[0.25, 0.0, 0.0]])
    vm = np.asarray(wx.velocity_matrix(m, k))
    expect = 2.0 * np.sin(2 * np.pi * 0.25) * EV_ANG_OVER_HBAR_TO_M_S
    np.testing.assert_allclose(vm[0, 0, 0, 0].real, expect, rtol=1e-13)


def test_velocity_operator_hermitian_and_eigenbasis() -> None:
    m = qiwuzhang(u=-1.0)
    k = jnp.array([[0.2, 0.3, 0.0]])
    v_op = np.asarray(wx.velocity_operator(m, k))
    np.testing.assert_allclose(
        v_op, np.swapaxes(v_op.conj(), -1, -2), atol=1e-13
    )
    # eigenbasis: v_nn = <u_n|v|u_n> equals real diagonal of velocity_matrix
    vm = np.asarray(wx.velocity_matrix(m, k))
    np.testing.assert_allclose(
        vm[..., 0, 0, 0], vm[..., 0, 0, 0].real + 0j, atol=1e-13
    )


def test_cluster_degenerate() -> None:
    e = np.array([0.0, 0.0, 1.0, 2.0, 2.0 + 1e-11, 5.0])
    cl = cluster_degenerate(e, atol=1e-9)
    np.testing.assert_array_equal(cl, np.array([[0, 2], [2, 3], [3, 5], [5, 6]]))
    # separated levels
    cl2 = cluster_degenerate(np.array([0.0, 1.0, 2.0]), atol=1e-9)
    np.testing.assert_array_equal(cl2, np.array([[0, 1], [1, 2], [2, 3]]))
    # scale dependence
    cl3 = cluster_degenerate(np.array([1.0, 1.0 + 1e-8]), atol=1e-9, rtol=1e-6)
    np.testing.assert_array_equal(cl3, np.array([[0, 2]]))


def test_is_degenerate_batch() -> None:
    e = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]])
    d = np.asarray(is_degenerate(e))
    np.testing.assert_array_equal(d[0], [True, True, False])
    np.testing.assert_array_equal(d[1], [False, False, False])


def test_projector_properties() -> None:
    m = qiwuzhang(u=-1.0)
    k = jnp.array([[0.2, 0.31, 0.0]])
    eig = wx.eigh(m, k)
    P = np.asarray(wx.projector(eig.vectors, [0]))[0]
    # Hermitian
    np.testing.assert_allclose(P, P.conj().T, atol=ATOL)
    # idempotent
    np.testing.assert_allclose(P @ P, P, atol=ATOL)
    # trace == subspace dim
    assert abs(np.trace(P) - 1.0) < 1e-14


def test_projector_subspace_rotation_invariance() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(a[:2, :2])  # unitary within selected 1,2 subspace
    Ufull = np.eye(4, dtype=complex)
    Ufull[:2, :2] = Q
    P1 = wx.projector(jnp.asarray(Ufull), [0, 1])
    P2 = wx.projector(jnp.eye(4, dtype=jnp.complex128), [0, 1])
    np.testing.assert_allclose(np.asarray(P1), np.asarray(P2), atol=ATOL)
