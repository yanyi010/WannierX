"""Wannier90 I/O tests."""

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


def _write_hr_line(r: tuple[int, int, int], m: int, n: int, re: float, im: float) -> str:
    return f" {r[0]:4d} {r[1]:4d} {r[2]:4d} {m:4d} {n:4d} {re:14.8f} {im:14.8f}\n"


def _hr_header(num_wann: int = 2, n_R: int = 1, degen: int = 1) -> str:
    return f"header\n{num_wann}\n{n_R}\n{degen}\n"


def test_parse_hr_duplicate_entry_raises(tmp_path: Path) -> None:
    # correct total line count, but (m, n)=(1, 1) appears twice and
    # (2, 2) never appears in the R block
    text = _hr_header()
    text += _write_hr_line((0, 0, 0), 1, 1, 0.0, 0.0)
    text += _write_hr_line((0, 0, 0), 1, 2, 0.8, 0.0)
    text += _write_hr_line((0, 0, 0), 2, 1, 0.8, 0.0)
    text += _write_hr_line((0, 0, 0), 1, 1, 5.0, 0.0)  # duplicate (1,1)
    p = tmp_path / "dup_hr.dat"
    p.write_text(text)
    with pytest.raises(ParseError, match="duplicate"):
        parse_hr(p)


def test_parse_hr_missing_entry_raises(tmp_path: Path) -> None:
    # correct total line count, but (m, n)=(2, 2) is replaced by a
    # repetition of (1, 2); without duplicate detection the missing
    # entry would silently keep np.empty garbage
    text = _hr_header()
    text += _write_hr_line((0, 0, 0), 1, 1, 0.0, 0.0)
    text += _write_hr_line((0, 0, 0), 1, 2, 0.8, 0.0)
    text += _write_hr_line((0, 0, 0), 1, 2, 0.9, 0.0)  # wrong entry
    text += _write_hr_line((0, 0, 0), 2, 1, 0.8, 0.0)
    p = tmp_path / "miss_hr.dat"
    p.write_text(text)
    with pytest.raises(ParseError, match="duplicate"):
        parse_hr(p)
    # missing-only variant: fewer lines than num_wann^2 in the block
    text2 = _hr_header()
    text2 += _write_hr_line((0, 0, 0), 1, 1, 0.0, 0.0)
    text2 += _write_hr_line((0, 0, 0), 1, 2, 0.8, 0.0)
    text2 += _write_hr_line((0, 0, 0), 2, 1, 0.8, 0.0)
    p2 = tmp_path / "miss2_hr.dat"
    p2.write_text(text2)
    with pytest.raises(ParseError, match="missing"):
        parse_hr(p2)


def test_parse_hr_nonpositive_degeneracy_raises(tmp_path: Path) -> None:
    text = _hr_header(degen=0)
    for m in (1, 2):
        for n in (1, 2):
            text += _write_hr_line((0, 0, 0), m, n, 0.0, 0.0)
    p = tmp_path / "degen0_hr.dat"
    p.write_text(text)
    with pytest.raises(ParseError, match="positive"):
        parse_hr(p)
    p2 = tmp_path / "degenneg_hr.dat"
    p2.write_text(_hr_header(degen=-2))
    with pytest.raises(ParseError, match="positive"):
        parse_hr(p2)


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
