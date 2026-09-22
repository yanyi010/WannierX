"""Low-level parser for Wannier90 ``*_hr.dat``.

Decodes the conventional file layout into a host-side intermediate
representation. Applies no physics beyond format decoding — weights,
R vectors, and matrix elements are preserved verbatim.

Layout (Wannier90 user guide):
    line 1:      header/comment (free text)
    line 2:      num_wann      (int)
    line 3:      n_R (nrpts)   (int)
    then:        degeneracy list, 15 ints per line, len == n_R
    then:        n_R blocks of num_wann^2 lines, each
                     i j k  m  n  Re(H)  Im(H)
                 with the R vector constant within each block in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wannierx.core.exceptions import ParseError


@dataclass(frozen=True)
class HrRaw:
    """Raw decoded content of a Wannier90 ``_hr.dat`` file.

    Attributes:
        num_wann: number of Wannier functions.
        n_R: number of R points.
        degeneracy: (n_R,) int weights ndegen(R).
        R: (n_R, 3) int lattice vectors.
        H_R: (n_R, num_wann, num_wann) complex128 matrix elements, eV.
    """

    num_wann: int
    n_R: int
    degeneracy: NDArray[np.int64]
    R: NDArray[np.int64]
    H_R: NDArray[np.complex128]


def parse_hr(path: str | Path) -> HrRaw:
    """Parse a Wannier90 ``*_hr.dat`` file.

    Parameters:
        path: filesystem path.

    Returns:
        :class:`HrRaw` with verbatim decoded arrays (float64/complex128).

    Failure behavior: raises :class:`ParseError` on malformed header,
    degeneracy length mismatch, non-positive degeneracy, or
    incomplete/inconsistent matrix blocks (including duplicate or
    missing (m, n) entries within an R block).
    """
    p = Path(path)
    try:
        lines = p.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"cannot read {p}: {exc}") from exc
    if len(lines) < 3:
        raise ParseError(f"{p}: too few lines for a valid _hr.dat file")

    def _int(s: str, what: str) -> int:
        try:
            return int(s.strip())
        except ValueError as exc:
            raise ParseError(f"{p}: malformed {what}: {s!r}") from exc

    # header = lines[0]
    num_wann = _int(lines[1], "num_wann")
    n_R = _int(lines[2], "n_R (nrpts)")
    if num_wann < 1 or n_R < 1:
        raise ParseError(f"{p}: num_wann and n_R must be positive")

    # degeneracy list: 15 per line
    degen: list[int] = []
    idx = 3
    n_degen_lines = (n_R + 14) // 15
    for _ in range(n_degen_lines):
        if idx >= len(lines):
            raise ParseError(f"{p}: truncated degeneracy list")
        for tok in lines[idx].split():
            degen.append(_int(tok, "degeneracy"))
        idx += 1
    if len(degen) != n_R:
        raise ParseError(
            f"{p}: degeneracy list has {len(degen)} entries, expected {n_R}"
        )
    for i, d in enumerate(degen):
        if d <= 0:
            raise ParseError(
                f"{p}: degeneracy entry {i} must be positive, got {d}"
            )

    # matrix blocks
    n_entries = n_R * num_wann * num_wann
    expected_end = idx + n_entries
    if len(lines) < expected_end:
        raise ParseError(
            f"{p}: incomplete matrix data ({len(lines) - idx} of {n_entries} entries)"
        )

    # np.zeros (not np.empty): any entry a malformed file fails to fill
    # would otherwise survive as uninitialized garbage; duplicate/missing
    # detection below guarantees every (m, n) is written exactly once.
    R = np.zeros((n_R, 3), dtype=np.int64)
    H_R = np.zeros((n_R, num_wann, num_wann), dtype=np.complex128)
    read = 0
    for r in range(n_R):
        R_first = None
        seen = np.zeros((num_wann, num_wann), dtype=bool)
        for _m in range(num_wann):
            for _n in range(num_wann):
                parts = lines[idx].split()
                idx += 1
                if len(parts) < 7:
                    raise ParseError(
                        f"{p}: malformed matrix entry line: {lines[idx - 1]!r}"
                    )
                try:
                    ri, rj, rk = int(parts[0]), int(parts[1]), int(parts[2])
                    m_ = int(parts[3])
                    n_ = int(parts[4])
                    re, im = float(parts[5]), float(parts[6])
                except ValueError as exc:
                    raise ParseError(
                        f"{p}: non-numeric matrix entry: {lines[idx - 1]!r}"
                    ) from exc
                if not (1 <= m_ <= num_wann and 1 <= n_ <= num_wann):
                    raise ParseError(
                        f"{p}: orbital index out of range: m={m_}, n={n_}"
                    )
                m, n = m_ - 1, n_ - 1
                if R_first is None:
                    R_first = (ri, rj, rk)
                elif (ri, rj, rk) != R_first:
                    raise ParseError(
                        f"{p}: inconsistent R within block {r}: "
                        f"{(ri, rj, rk)} != {R_first}"
                    )
                if seen[m, n]:
                    raise ParseError(
                        f"{p}: duplicate (m, n)=({m_}, {n_}) entry in R block "
                        f"{r} (R={R_first})"
                    )
                seen[m, n] = True
                # m is the row index of <0m|H|Rn>
                H_R[r, m, n] = re + 1j * im
                read += 1
        assert R_first is not None
        if not seen.all():
            missing_mn = [
                (int(mm) + 1, int(nn) + 1)
                for mm, nn in zip(*np.nonzero(~seen), strict=True)
            ]
            raise ParseError(
                f"{p}: R block {r} (R={R_first}) is missing "
                f"{len(missing_mn)} orbital entr{'y' if len(missing_mn) == 1 else 'ies'} "
                f"({len(seen) ** 2 - int(seen.sum())} of {num_wann ** 2} expected), "
                f"first missing (m, n)={missing_mn[0]}"
            )
        R[r] = R_first

    return HrRaw(
        num_wann=num_wann,
        n_R=n_R,
        degeneracy=np.asarray(degen, dtype=np.int64),
        R=R,
        H_R=H_R,
    )
