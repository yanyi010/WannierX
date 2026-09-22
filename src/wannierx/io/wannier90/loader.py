"""Convert parsed Wannier90 input into a canonical model."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from wannierx.core.exceptions import ParseError, WannierXError
from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel
from wannierx.io.wannier90.hr import parse_hr
from wannierx.io.wannier90.win import parse_win_lattice


class LoadedWannier90:
    """Result of :func:`load_wannier90`.

    Attributes:
        model: canonical :class:`WannierModel`.
        mode: interpolation mode actually used (``legacy_hr`` or
            ``expanded_exact``).
    """

    def __init__(self, model: WannierModel, mode: str):
        self.model = model
        self.mode = mode

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LoadedWannier90(mode={self.mode!r}, model={self.model!r})"


def load_wannier90(
    seedname: str | Path,
    mode: str = "auto",
    lattice: Lattice | None = None,
) -> LoadedWannier90:
    """Load a Wannier90 real-space Hamiltonian into a canonical model.

    Parameters:
        seedname: prefix used to locate ``<seedname>_hr.dat`` and
            ``<seedname>.win``.
        mode: one of ``legacy_hr``, ``expanded_exact``, ``auto``.
            ``legacy_hr`` uses w_R = 1 / ndegen(R) (documented legacy
            interpolation; not exact Wigner-Seitz). ``auto`` currently
            resolves to ``legacy_hr`` and reports the choice.
        lattice: optional explicit lattice; when None the ``.win`` file
            must exist. Missing lattice raises :class:`ParseError`.

    Returns:
        :class:`LoadedWannier90` with model and selected mode.

    Constraints: exact Wigner-Seitz interpolation is NOT claimed when
    replica semantics are unavailable.
    """
    if mode == "auto":
        selected = "legacy_hr"
    elif mode in ("legacy_hr", "expanded_exact"):
        selected = mode
    else:
        raise ValueError(f"unsupported mode: {mode!r}")

    seed = str(seedname)
    hr_path = seed + "_hr.dat"
    raw = parse_hr(hr_path)

    if lattice is None:
        win_path = seed + ".win"
        if not Path(win_path).exists():
            raise ParseError(
                f"no lattice supplied and {win_path!r} not found; "
                "pass lattice= explicitly or provide the .win file"
            )
        lattice = parse_win_lattice(win_path)

    if selected == "legacy_hr":
        weights = 1.0 / raw.degeneracy.astype(float)
    else:  # expanded_exact
        raise WannierXError(
            "expanded_exact mode requires input already expanded with exact "
            "replica semantics; conventional _hr.dat does not provide this. "
            "Use legacy_hr."
        )

    model = WannierModel(
        lattice=lattice,
        R=jnp.asarray(raw.R),
        H_R=jnp.asarray(raw.H_R),
        weights=jnp.asarray(weights),
        periodic=(True, True, True),
        centers=None,
    )
    return LoadedWannier90(model=model, mode=selected)
