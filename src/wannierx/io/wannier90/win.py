"""Minimal ``.win`` parser extracting the direct lattice (WX-009).

Not a general ``.win`` parser: only the ``unit_cell_cart`` block is
decoded. Supported units: ``ang``/``angstrom`` (native) and ``bohr``
(converted with 1 bohr = 0.529177210903 Angstrom, CODATA), because the
canonical internal length unit is Angstrom.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from wannierx.core.exceptions import ParseError
from wannierx.core.lattice import Lattice

_BOHR_TO_ANG = 0.529177210903

_BLOCK_RE = re.compile(
    r"begin\s+unit_cell_cart\s*(.*?)\s*end\s+unit_cell_cart", re.S | re.I
)
_COMMENT_RE = re.compile(r"[#!].*")


def _strip_comments(text: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", ln) for ln in text.splitlines())


def parse_win_lattice(path: str | Path) -> Lattice:
    """Parse the direct lattice from a Wannier90 ``.win`` file.

    Parameters:
        path: filesystem path to the ``.win`` file.

    Returns:
        Canonical :class:`Lattice` (rows = lattice vectors, Angstrom).

    Failure behavior: raises :class:`ParseError` for a missing/malformed
    block or unsupported units; :class:`LatticeError` for a singular cell.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ParseError(f"cannot read {p}: {exc}") from exc
    text = _strip_comments(text)
    m = _BLOCK_RE.search(text)
    if not m:
        raise ParseError(f"{p}: no unit_cell_cart block found")
    body = m.group(1).strip().splitlines()
    if not body:
        raise ParseError(f"{p}: empty unit_cell_cart block")

    unit = "ang"
    rows: list[list[float]] = []
    for ln in body:
        toks = ln.split()
        if not toks:
            continue
        if all(_is_float(t) for t in toks) and len(toks) == 3:
            rows.append([float(t) for t in toks])
        elif len(rows) == 0 and len(toks) == 1 and not _is_float(toks[0]):
            unit = toks[0].lower()
        else:
            raise ParseError(f"{p}: malformed unit_cell_cart line: {ln!r}")

    if len(rows) != 3:
        raise ParseError(f"{p}: unit_cell_cart must contain 3 lattice rows, got {len(rows)}")

    A = np.array(rows, dtype=np.float64)
    if unit in ("bohr",):
        A = A * _BOHR_TO_ANG
    elif unit not in ("ang", "angstrom"):
        raise ParseError(f"{p}: unsupported lattice unit {unit!r} (support: ang, bohr)")
    return Lattice(A)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
