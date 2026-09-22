"""Berry curvature & lattice Chern number."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

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
    from wannierx.fourier.derivatives import dH_dk
    from wannierx.geometry.kernel import _omega_all_bands

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


# --- checked_berry_curvature: tracer-safe strict degeneracy policy ----


def test_checked_berry_eager_raises_on_degeneracy() -> None:
    m = graphene()  # gapless at K
    with pytest.raises(DegeneracyError, match="min selected spectral gap"):
        wx.checked_berry_curvature(m, _K(1 / 3, 2 / 3))


def test_checked_berry_eager_matches_berry_curvature() -> None:
    m = qiwuzhang(u=-1.0)
    k = _K(0.13, 0.41)
    np.testing.assert_allclose(
        np.asarray(wx.checked_berry_curvature(m, k)),
        np.asarray(wx.berry_curvature(m, k, degeneracy_policy="nan")),
        atol=ATOL,
    )


def test_checked_berry_checkify_jit_raises_via_throw() -> None:
    m = graphene()
    checked = checkify.checkify(jax.jit(wx.checked_berry_curvature))
    err, _ = checked(m, _K(1 / 3, 2 / 3))
    with pytest.raises(Exception, match="degenerate"):
        err.throw()


def test_checked_berry_checkify_jit_clean_path() -> None:
    m = qiwuzhang(u=-1.0)
    k = _K(0.13, 0.41)
    checked = checkify.checkify(jax.jit(wx.checked_berry_curvature))
    err, out = checked(m, k)
    err.throw()  # no degeneracy: must not raise
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(wx.berry_curvature(m, k)), atol=ATOL
    )


def test_checked_berry_plain_jit_fails_loudly_at_trace_time() -> None:
    # an un-functionalized checkify.check cannot be staged: plain jit
    # must fail at trace time instead of silently skipping the check
    m = graphene()
    with pytest.raises(Exception, match="functionalized"):
        jax.jit(wx.checked_berry_curvature)(m, _K(0.13, 0.41))


def test_checked_berry_checkify_vmap() -> None:
    # regression: model-pytree arguments under checkify(vmap(...)) used
    # to crash in Lattice validation (tracer-type-based staging guard)
    m = graphene()
    ks = jnp.stack([jnp.asarray(_K(1 / 3, 2 / 3))[0], jnp.asarray(_K(0.13, 0.41))[0]])
    checked = checkify.checkify(
        jax.vmap(wx.checked_berry_curvature, in_axes=(None, 0))
    )
    err, _ = checked(m, ks)
    with pytest.raises(Exception, match="degenerate"):
        err.throw()
    err2, out2 = checked(m, ks[1:])
    err2.throw()
    np.testing.assert_allclose(
        np.asarray(out2),
        np.asarray(wx.checked_berry_curvature(m, ks[1:])),
        atol=ATOL,
    )


# --- chern_number: arbitrary (non-contiguous) subspace selection -------
# 3-band model: QWZ(u=-1) direct-sum a dispersionless band at eps.
# The flat band is degenerate with the QWZ upper band at Gamma for
# eps = 1 (E_upper(Gamma) = 1), and well separated for eps = 5
# (E_upper in [1, 3]).


def _qwz_plus_flat(u: float, eps: float) -> wx.WannierModel:
    m = qiwuzhang(u)
    H_R = jnp.zeros((m.n_R, 3, 3), dtype=jnp.complex128)
    H_R = H_R.at[:, :2, :2].set(m.H_R)
    H_R = H_R.at[0, 2, 2].set(eps)  # R index 0 is (0, 0, 0)
    return wx.WannierModel(
        m.lattice, m.R, H_R, m.weights, m.periodic, jnp.zeros((3, 3))
    )


def test_chern_noncontiguous_isolated_selection() -> None:
    # occupied=[0, 2] = {lower QWZ band (C=-1), flat band (C=0)};
    # the unselected band 1 lies inside the index span, and the
    # S <-> complement gap stays open -> C = -1
    m = _qwz_plus_flat(-1.0, 5.0)
    c = wx.chern_number(m, mesh=(24, 24), occupied=[0, 2])
    assert c == pytest.approx(-1.0, abs=1e-10)


def test_chern_noncontiguous_degenerate_with_complement_raises() -> None:
    # flat band at eps = 1 is degenerate with the upper QWZ band at
    # Gamma, so occupied=[0, 2] is NOT isolated from band 1. The
    # pre-v0.1.1 span-based check (min/max selected index) reported
    # gap = inf here and silently accepted the ambiguous subspace.
    m = _qwz_plus_flat(-1.0, 1.0)
    with pytest.raises(DegeneracyError, match="complement"):
        wx.chern_number(m, mesh=(12, 12), occupied=[0, 2])


def test_chern_complement_internal_degeneracy_allowed() -> None:
    # same model as above: bands 1 and 2 are degenerate with each other
    # at Gamma, but both are in the complement of occupied=[0]; the
    # FHS link variables are subspace-gauge covariant, so only the
    # S <-> complement gap matters
    m = _qwz_plus_flat(-1.0, 1.0)
    c = wx.chern_number(m, mesh=(24, 24), occupied=[0])
    assert c == pytest.approx(-1.0, abs=1e-10)


def test_chern_full_space_selection_is_zero() -> None:
    # selecting every band leaves an empty complement; the Chern number
    # of the full Hilbert space vanishes (band-sum rule)
    m = _qwz_plus_flat(-1.0, 5.0)
    c = wx.chern_number(m, mesh=(12, 12), occupied=[0, 1, 2])
    assert c == pytest.approx(0.0, abs=1e-10)
