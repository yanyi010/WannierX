"""Real-space Hermiticity diagnostics (WX-004).

Checks H_mn(R) = conj(H_nm(-R)) without modifying the model. Missing
-R partners are diagnosed explicitly. Never symmetrizes silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel


@dataclass(frozen=True)
class HermiticityReport:
    """Result of :func:`validate_hermiticity`.

    Attributes:
        is_hermitian: True when all related pairs match within ``atol`` and
            no -R partner is missing.
        max_absolute_error: max |H_mn(R) - conj(H_nm(-R))| over related pairs
            (NaN when no related pairs exist).
        max_relative_error: absolute error normalized by max(|H|) of the pair,
            0 when both are zero (NaN when no related pairs exist).
        worst_R: integer R vector of the worst-matching pair, or None.
        worst_indices: (m, n) orbital index of the worst pair, or None.
        missing_partners: R vectors (n_missing, 3) whose -R partner is absent.
        atol: tolerance used for ``is_hermitian``.
    """

    is_hermitian: bool
    max_absolute_error: float
    max_relative_error: float
    worst_R: Array | None
    worst_indices: tuple[int, int] | None
    missing_partners: Array = field(default_factory=lambda: jnp.zeros((0, 3), dtype=int))
    atol: float = 1e-12


def _r_key(r: Array) -> tuple[int, int, int]:
    return (int(r[0]), int(r[1]), int(r[2]))


def validate_hermiticity(model: WannierModel, atol: float = 1e-12) -> HermiticityReport:
    """Validate H_mn(R) = conj(H_nm(-R)) for a model.

    Parameters:
        model: canonical Wannier model.
        atol: absolute tolerance used for the boolean verdict. Errors are
            reported regardless of the verdict.

    Returns:
        :class:`HermiticityReport`.

    Units/conventions: host-side diagnostic; not JIT-traced; eV.
    Failure behavior: never raises for a well-formed model; the report's
        ``is_hermitian`` and ``missing_partners`` fields carry the diagnosis.
    """
    R_list = [tuple(int(x) for x in row) for row in model.R]
    index = {key: i for i, key in enumerate(R_list)}
    missing: list[tuple[int, int, int]] = []
    max_abs = -jnp.inf
    max_rel = 0.0
    worst_key: tuple[int, int, int] | None = None
    worst_idx: tuple[int, int] | None = None

    H = model.H_R
    n = model.n_orb

    for i, key in enumerate(R_list):
        neg = (-key[0], -key[1], -key[2])
        if neg not in index:
            missing.append(key)
            continue
        j = index[neg]
        # error matrix: H(R) - conj(H(-R))^T
        err = H[i] - jnp.conjugate(H[j].T)
        # relative reference per (m,n)
        ref = jnp.maximum(jnp.abs(H[i]), jnp.abs(H[j].T))
        abs_err = jnp.abs(err)
        a = float(jnp.max(abs_err))
        if a > float(max_abs):
            max_abs = jnp.asarray(a)
            flat = int(jnp.argmax(abs_err))
            worst_idx = (flat // n, flat % n)
            worst_key = key
        rel = jnp.where(ref > 0, abs_err / jnp.where(ref > 0, ref, 1.0), 0.0)
        max_rel = max(max_rel, float(jnp.max(rel)))

    no_pairs = not R_list or all(
        _r_key(jnp.asarray(k)) and (-k[0], -k[1], -k[2]) not in index for k in []
    )
    any_related = any(((-k[0], -k[1], -k[2]) in index) for k in R_list)
    if not any_related:
        max_abs_out = float("nan")
        max_rel_out = float("nan")
        is_herm = len(missing) == 0
    else:
        max_abs_out = float(max_abs)
        max_rel_out = max_rel
        is_herm = (max_abs_out <= atol) and (len(missing) == 0)

    return HermiticityReport(
        is_hermitian=is_herm,
        max_absolute_error=max_abs_out,
        max_relative_error=max_rel_out,
        worst_R=None if worst_key is None else jnp.asarray(worst_key, dtype=int),
        worst_indices=worst_idx,
        missing_partners=jnp.asarray(missing, dtype=int).reshape(-1, 3),
        atol=atol,
    )
