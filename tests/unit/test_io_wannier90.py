"""WX-008/009/010: Wannier90 I/O tests."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.core.exceptions import ParseError, WannierXError
from wannierx.io.wannier90.hr import parse_hr
from wannierx.io.wannier90.win import parse_win_lattice

FIX = Path("tests/reference/fixtures")


def test_parse_hr_valid() -> None:
    raw = parse_hr(FIX / "toy_hr.dat")
    assert raw.num_wann == 2 and raw.n_R == 3
    np.testing.assert_array_equal(raw.degeneracy, np.ones(3))
    np.testing.assert_array_equal(raw.R, np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))
    # check <0,1|H|0,2> = t1
    assert raw.H_R[0, 0, 1] == pytest.approx(0.8)
    assert raw.H_R[1, 1, 0] == pytest.approx(1.2)  # <0,B|H|A_{+1}>? layout by (m,n)
    assert raw.H_R.dtype == np.complex128


def test_parse_hr_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad_hr.dat"
    bad.write_text("header\n2\n3\n1 1\n")  # degeneracy too short + truncated blocks
    with pytest.raises(ParseError):
        parse_hr(bad)


def test_parse_win_lattice() -> None:
    lat = parse_win_lattice(FIX / "toy.win")
    np.testing.assert_allclose(np.asarray(lat.direct), np.diag([4.0, 8.0, 10.0]))


def test_parse_win_bohr(tmp_path: Path) -> None:
    p = tmp_path / "bohr.win"
    p.write_text("begin unit_cell_cart\nbohr\n 2 0 0\n 0 2 0\n 0 0 2\nend unit_cell_cart\n")
    lat = parse_win_lattice(p)
    np.testing.assert_allclose(
        np.asarray(lat.direct), np.eye(3) * 2 * 0.529177210903, rtol=1e-14
    )


def test_parse_win_malformed(tmp_path: Path) -> None:
    p = tmp_path / "x.win"
    p.write_text("begin unit_cell_cart\nang\n 1 0\nend unit_cell_cart\n")
    with pytest.raises(ParseError):
        parse_win_lattice(p)
    with pytest.raises(ParseError):
        parse_win_lattice(tmp_path / "nope.win")


def test_load_wannier90_auto_weights_hermiticity() -> None:
    out = wx.load_wannier90(str(FIX / "toy"), mode="auto")
    assert out.mode == "legacy_hr"
    m = out.model
    assert m.n_orb == 2 and m.n_R == 3
    np.testing.assert_array_equal(np.asarray(m.weights), np.ones(3))
    assert wx.validate_hermiticity(m).is_hermitian
    np.testing.assert_allclose(np.asarray(m.lattice.direct), np.diag([4.0, 8.0, 10.0]))


def test_load_wannier90_missing_lattice(tmp_path: Path) -> None:
    # copy hr without win
    (tmp_path / "only_hr.dat").write_text((FIX / "toy_hr.dat").read_text())
    with pytest.raises(ParseError):
        wx.load_wannier90(str(tmp_path / "only"))
    # works if lattice passed explicitly
    out = wx.load_wannier90(
        str(tmp_path / "only"), lattice=wx.Lattice(jnp.eye(3))
    )
    assert out.model.n_orb == 2


def test_load_wannier90_exact_mode_unsupported() -> None:
    with pytest.raises(WannierXError):
        wx.load_wannier90(str(FIX / "toy"), mode="expanded_exact")
