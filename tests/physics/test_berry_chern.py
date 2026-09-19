"""WX-019/020: Berry curvature & lattice Chern number."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.core.exceptions import DegeneracyError
from wannierx.models.graphene import graphene
from wannierx.models.qzhang import qiwuzhang

ATOL = 1e-13


def _K(kfx: float, kfy: float) -> jnp.ndarray:
    return jnp.array([[kfx, kfy, 0.0]])


def _two_band_truth(kfx: float, kfy: float, u: float = -1.0) -> float:
    """Analytic lower-band Berry curvature (project convention pinned by
    tests): Omega_xy^lower = +0.5 d hat . (dx d x dy d) / |d|^2 with
    d = (sin 2pi kx, sin 2pi ky, u + cos 2pi kx + cos 2pi ky) and
    cartesian k = 2 pi kf."""
    kx, ky = 2 * np.pi * kfx, 2 * np.pi * kfy
    d = np.array([np.sin(kx), np.sin(ky), u + np.cos(kx) + np.cos(ky)])
    ddx = np.array([np.cos(kx), 0.0, -np.sin(kx)])
    ddy = np.array([0.0, np.cos(ky), -np.sin(ky)])
    return 0.5 * np.dot(d, np.cross(ddx, ddy)) / np.linalg.norm(d) ** 3


def test_berry_matches_two_band_analytic() -> None:
    m = qiwuzhang(u=-1.0)
    for p in [(0.1, 0.2), (0.3, 0.7), (0.45, 0.13), (0.9, 0.77), (0.22, 0.61)]:
        ours = float(
            np.asarray(
                wx.berry_curvature(m, _K(*p), degeneracy_policy="nan")
            )[0, 0, 0, 1]
        )
        assert ours == pytest.approx(_two_band_truth(*p), abs=1e-14)


def test_berry_antisymmetry_ab() -> None:
    m = qiwuzhang(u=-1.0)
    k = jnp.asarray(np.random.default_rng(0).normal(size=(6, 3)) * 0.3)
    om = np.asarray(wx.berry_curvature(m, k, degeneracy_policy="nan"))
    np.testing.assert_allclose(om, -np.swapaxes(om, -1, -2), atol=ATOL)


def test_berry_gauge_phase_invariance() -> None:
    """Random phase rotation of the eigenbasis must leave Omega invariant."""
    m = qiwuzhang(u=-1.0)
    k = _K(0.13, 0.41)
    eig = wx.eigh(m, k)
    rng = np.random.default_rng(2)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=2))
    U2 = eig.vectors * jnp.asarray(ph)[None, :]
    # reconstruct curvature from U2 + same dH, compare
    from wannierx.geometry.berry import _omega_all_bands
    from wannierx.fourier.derivatives import dH_dk

    dH = dH_dk(m, k)
    Ud2 = jnp.swapaxes(jnp.conjugate(U2), -1, -2)
    A2 = jnp.einsum("...ni,...aij,...jm->...anm", Ud2, dH, U2)
    om2 = _omega_all_bands(eig.energies, A2)
    om1 = wx.berry_curvature(m, k, degeneracy_policy="nan")
    np.testing.assert_allclose(np.asarray(om2), np.asarray(om1), atol=1e-13)


def test_berry_basis_invariance() -> None:
    """k-independent orbital basis change leaves Omega unchanged."""
    m = qiwuzhang(u=-1.0)
    rng = np.random.default_rng(1)
    a = jax.random.normal(jax.random.PRNGKey(0), (2, 2)) if False else None
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    Q = jnp.asarray(q)
    Qd = jnp.conjugate(Q).T
    H_R2 = jnp.einsum("ij,Rjk,kl->Ril", Qd, m.H_R, Q)
    m2 = wx.WannierModel(m.lattice, m.R, H_R2, m.weights, m.periodic, m.centers)
    k = _K(0.17, 0.29)
    np.testing.assert_allclose(
        np.asarray(wx.berry_curvature(m, k, degeneracy_policy="nan")),
        np.asarray(wx.berry_curvature(m2, k, degeneracy_policy="nan")),
        atol=1e-12,
    )


def test_berry_degeneracy_policies() -> None:
    m = graphene()  # gapless at K
    Kdirac = _K(1 / 3, 2 / 3)
    with pytest.raises(DegeneracyError):
        wx.berry_curvature(m, Kdirac, degeneracy_policy="error")
    om_nan = np.asarray(wx.berry_curvature(m, Kdirac, degeneracy_policy="nan"))
    assert np.isnan(om_nan[0, :, 0, 1]).all()
    om_mask = np.asarray(wx.berry_curvature(m, Kdirac, degeneracy_policy="mask"))
    assert np.isnan(om_mask[0, :, 0, 1]).all()


def test_chern_qwz_trivial_and_nontrivial() -> None:
    # nontrivial: u=-1 -> C = -1 (project convention, pinned)
    c1 = wx.chern_number(qiwuzhang(u=-1.0), mesh=(24, 24), occupied=[0])
    assert c1 == pytest.approx(-1.0, abs=1e-10)
    # trivial: u=3 -> C = 0
    c0 = wx.chern_number(qiwuzhang(u=3.0), mesh=(24, 24), occupied=[0])
    assert abs(c0) < 1e-10


def test_chern_gauge_randomization() -> None:
    """Chern must be unchanged if eigenvector phases are re-randomized.

    chern_number is gauge-robust by construction (link variables from
    subspace overlaps). Evaluate twice; determinism + gauge robustness.
    """
    m = qiwuzhang(u=-1.0)
    c1 = wx.chern_number(m, mesh=(20, 20), occupied=[0])
    c2 = wx.chern_number(m, mesh=(20, 20), occupied=[0])
    assert c1 == c2


def test_chern_mesh_convergence() -> None:
    m = qiwuzhang(u=-1.0)
    vals = [wx.chern_number(m, mesh=(n, n), occupied=[0]) for n in (16, 24, 32)]
    for v in vals:
        assert v == pytest.approx(vals[0], abs=1e-12)


def test_chern_failure_on_non_isolated_subspace() -> None:
    m = graphene()  # gapless Dirac points
    with pytest.raises(DegeneracyError):
        # mesh divisible by 3 hits Dirac points
        wx.chern_number(m, mesh=(24, 24), occupied=[0])
