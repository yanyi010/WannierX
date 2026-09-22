"""Anomalous Hall conductivity."""

from __future__ import annotations

import numpy as np
import pytest

import wannierx as wx
from wannierx.constants import E2_OVER_H_SI
from wannierx.core.exceptions import DegeneracyError, DimensionalityError
from wannierx.kpoints.mesh import gamma_centered_mesh, monkhorst_pack
from wannierx.models.haldane import haldane
from wannierx.models.qzhang import qiwuzhang

TOL = 1e-8


def test_ahc_topological_2d_quantized() -> None:
    m = qiwuzhang(u=-1.0)
    s = wx.ahc(m, monkhorst_pack((96, 96, 1)), mu=0.0)
    sxy = float(s[2] / E2_OVER_H_SI)
    # 2D native sheet conductance; occupied lower band, C = -1 (locked)
    assert sxy == pytest.approx(-1.0, abs=TOL)


def test_ahc_trivial_zero() -> None:
    m = qiwuzhang(u=3.0)
    s = wx.ahc(m, monkhorst_pack((96, 96, 1)), mu=0.0)
    assert abs(float(s[2] / E2_OVER_H_SI)) < TOL


def test_ahc_chemical_potential_dependence() -> None:
    m = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((24, 24, 1))
    # mu inside the gap (~empty valence fill) -> quantized;
    # mu far below all bands -> zero
    s_gap = float(wx.ahc(m, monkhorst_pack((48, 48, 1)), mu=0.0)[2] / E2_OVER_H_SI)
    s_empty = float(wx.ahc(m, mesh, mu=100.0)[2] / E2_OVER_H_SI)
    assert abs(s_gap + 1.0) < 0.05
    assert abs(s_empty) < 1e-12


def test_ahc_temperature_smoke() -> None:
    m = qiwuzhang(u=-1.0)
    s0 = wx.ahc(m, monkhorst_pack((24, 24, 1)), mu=0.0, temperature=0.0)
    sT = wx.ahc(m, monkhorst_pack((24, 24, 1)), mu=0.0, temperature=10.0)
    assert np.allclose(np.asarray(s0), np.asarray(sT), rtol=0.05)


def test_ahc_chunk_invariance() -> None:
    m = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((32, 32, 1))
    a = wx.ahc(m, mesh, mu=0.0)
    b = wx.ahc(m, mesh, mu=0.0, chunk_size=17)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-15)


def test_ahc_missing_sampling_dimension_raises() -> None:
    m = qiwuzhang(u=-1.0)
    with pytest.raises(DimensionalityError):
        wx.ahc(m, monkhorst_pack((32, 1, 1)), mu=0.0)


def test_ahc_2d_thickness_converts_units() -> None:
    m = qiwuzhang(u=-1.0)
    sheet = float(wx.ahc(m, monkhorst_pack((48, 48, 1)), mu=0.0)[2])
    per_m = float(wx.ahc(m, monkhorst_pack((48, 48, 1)), mu=0.0, thickness=5.0)[2])
    assert per_m == pytest.approx(sheet / 5e-10, rel=1e-13)


def test_ahc_haldane_topological_regime() -> None:
    s = wx.ahc(haldane(t2=0.3, M=0.0), monkhorst_pack((48, 48, 1)), mu=0.0)
    assert abs(abs(float(s[2] / E2_OVER_H_SI)) - 1.0) < 0.05


# --- degeneracy contract (v0.1.1) -------------------------------------
# QWZ with u = -2 is gapless exactly at Gamma (d = 0 there, E = 0 = E),
# and a Gamma-centered mesh always contains Gamma. Before the v0.1.1
# hotfix, ahc() called the raw curvature kernel directly: the occupied
# Dirac degeneracy produced inf/NaN silently instead of an error.


def test_ahc_occupied_degeneracy_raises() -> None:
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))  # includes Gamma
    with pytest.raises(DegeneracyError, match="degeneracy"):
        wx.ahc(m, mesh, mu=0.0)  # mu at the degenerate E = 0 -> f = 1/2


def test_ahc_degeneracy_error_is_chunk_invariant() -> None:
    # the error is raised on the concrete reduced result, so internal
    # JIT chunking must not change the failure semantics
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))
    with pytest.raises(DegeneracyError):
        wx.ahc(m, mesh, mu=0.0, chunk_size=5)


def test_ahc_unoccupied_degeneracy_masked_exactly() -> None:
    # same gapless model, mu below the degeneracy: the Dirac point has
    # f = 0 exactly (T = 0), so its ill-defined curvature contributes
    # exactly zero and must neither raise nor contaminate the result
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))
    s = np.asarray(wx.ahc(m, mesh, mu=-1.5))
    assert np.isfinite(s).all()


def test_ahc_fully_unoccupied_is_exactly_zero() -> None:
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))
    s = np.asarray(wx.ahc(m, mesh, mu=-100.0))
    assert np.all(s == 0.0)


def test_ahc_degeneracy_nan_policy() -> None:
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))
    s = np.asarray(wx.ahc(m, mesh, mu=0.0, degeneracy_policy="nan"))
    assert np.isnan(s).all()


def test_ahc_degeneracy_mask_policy_finite() -> None:
    # documented approximation: degenerate occupied bands are dropped
    m = qiwuzhang(u=-2.0)
    mesh = gamma_centered_mesh((8, 8, 1))
    s = np.asarray(wx.ahc(m, mesh, mu=0.0, degeneracy_policy="mask"))
    assert np.isfinite(s).all()


def test_ahc_rejects_unknown_policy_and_nonfinite_mu() -> None:
    m = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((8, 8, 1))
    with pytest.raises(ValueError, match="degeneracy_policy"):
        wx.ahc(m, mesh, mu=0.0, degeneracy_policy="ignore")
    with pytest.raises(ValueError, match="finite"):
        wx.ahc(m, mesh, mu=float("nan"))
