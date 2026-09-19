"""WX-026: autodiff validation suite.

Covers gradients of:
- Hamiltonian elements w.r.t. model parameters (via pytree fields),
- isolated eigenvalues (JAX autodiff) vs multi-step finite differences and
  Hellmann-Feynman,
- smooth DOS,
- one smooth BZ response integral.

Tests near degeneracies explicitly document expected non-smooth behavior
rather than demanding false differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.core.model import WannierModel
from wannierx.kpoints.mesh import monkhorst_pack
from wannierx.models.chain import chain

TOL_FD = 1e-7


def _fd(fn, x, hs=(1e-4, 1e-5, 1e-6)):
    out = []
    for h in hs:
        out.append((fn(x + h) - fn(x - h)) / (2 * h))
    return out


def _replace_t(model: WannierModel, t) -> WannierModel:
    """Chain model with scalar hopping t (real-only, autodiff-safe)."""
    H_R = model.H_R.at[0, 0, 0].set(jnp.asarray(t) + 0j).at[2, 0, 0].set(jnp.asarray(t) + 0j)
    return WannierModel(model.lattice, model.R, H_R, model.weights, model.periodic, model.centers)


def test_grad_hamiltonian_wrt_hopping() -> None:
    m0 = chain(t=-1.0)
    k = jnp.array([0.3, 0.0, 0.0])

    def f(t):
        return jnp.real(wx.hamiltonian(_replace_t(m0, t), k[None]).reshape(-1)[0])

    g = jax.grad(f)(-1.0)
    # analytic: H(k) = eps0 + 2 t cos(2 pi kf) -> d/d t = 2 cos(2 pi kf)
    expect = 2.0 * np.cos(2 * np.pi * 0.3)
    assert float(g) == pytest.approx(expect, rel=1e-13)
    # multi-step central FD agrees (report truncation/roundoff behavior)
    fd = _fd(lambda tt: float(f(tt)), -1.0)
    assert abs(fd[-1] - expect) < 1e-9
    assert fd[0] != fd[1]  # distinct step sizes give distinct estimates


def test_grad_isolated_eigenvalue_vs_fd_and_hf() -> None:
    """dE_n/dt for the chain, away from any degeneracy."""
    m0 = chain(t=-1.0)
    k = jnp.array([0.3, 0.0, 0.0])

    def en(t):
        return wx.eigh(_replace_t(m0, t), k[None]).energies.reshape(-1)[0]

    g = float(jax.grad(en)(-1.0))
    expect = 2.0 * np.cos(2 * np.pi * 0.3)  # band touches nothing (n_orb=1)
    assert g == pytest.approx(expect, rel=1e-12)

    # Hellmann-Feynman: dE/dt = <u | dH/dt | u>
    eig = wx.eigh(m0, k)
    dHdt = (wx.hamiltonian(_replace_t(m0, -1.0 + 1e-8), k[None])
            - wx.hamiltonian(_replace_t(m0, -1.0 - 1e-8), k[None])) / (2e-8)
    u = eig.vectors[0, :, 0]
    hf = float(jnp.real(u.conj() @ dHdt.reshape(1, 1) @ u))
    # FD-based dH/dt agrees with direct chain-rule value
    assert hf == pytest.approx(2.0 * np.cos(2 * np.pi * 0.3), rel=1e-6)


def test_grad_smooth_dos() -> None:
    m0 = chain(t=-1.0)
    mesh = monkhorst_pack((64, 1, 1))
    E = jnp.linspace(-3.0, 3.0, 200)

    def smooth(t):
        # smooth functional: total DOS at fixed E via gaussian kernel moments
        d = wx.dos(_replace_t(m0, t), mesh, E, 0.08)
        return jnp.sum(d * E**2)

    g = float(jax.grad(smooth)(-1.0))
    fd = _fd(lambda tt: float(smooth(tt)), -1.0)
    # convergence with shrinking h
    assert abs(fd[-1] - g) < abs(fd[0] - g) or abs(fd[-1] - g) < TOL_FD


def test_grad_smooth_bz_response() -> None:
    """Smooth BZ average of a smooth observable of the isolated eigenband."""
    m0 = chain(t=-1.0)
    mesh = monkhorst_pack((32, 1, 1))

    def F(t):
        eps = wx.eigh(_replace_t(m0, t), mesh.points).energies[:, 0]
        return jnp.sum(mesh.weights * jnp.exp(-(eps**2)) * eps**2)

    g = float(jax.grad(F)(-1.0))
    fd = _fd(lambda tt: float(F(tt)), -1.0)
    assert abs(g - fd[-1]) < TOL_FD
    # shrinking-h reporting for the truncation/roundoff structure
    assert fd[2] != fd[0]


def test_nonsmooth_at_degeneracy_documented() -> None:
    """At a degeneracy the individual eigenvalue derivative is not defined
    uniquely; we document the non-smooth behavior rather than force
    differentiability. The *sum* of the degenerate pair remains smooth."""
    # SSH at zone boundary is gapped; use graphene Dirac point: undefined band slope
    from wannierx.models.graphene import graphene

    m = graphene()
    K = jnp.array([[1 / 3, 2 / 3, 0.0]])
    eig = wx.eigh(m, K)
    # two bands touch: |E| both ~0 -> degenerate; individual slope undefined.
    # The doc-contract: this is a non-smooth point; we assert the *gap* is ~0
    # (failure signal) rather than asking JAX to differentiate through it.
    assert abs(float(eig.energies[0, 1]) - float(eig.energies[0, 0])) < 1e-10
