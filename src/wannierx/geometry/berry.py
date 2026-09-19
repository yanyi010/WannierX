"""Isolated-band Berry curvature (WX-019).

Convention (PHYSICS_CONVENTIONS.md, section 8), units Angstrom^2:

    Omega_n^{ab} = -2 Im sum_{m != n}
        <n| d_a H |m> <m| d_b H |n> / (eps_n - eps_m)^2

No hidden epsilon. Allowed degeneracy policies: ``error``, ``nan``,
``mask``. The returned tensor has leading batch dims of k, then
(n_sel, 3, 3) with the last two being cartesian derivative indices.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import Array

from wannierx.core.exceptions import DegeneracyError
from wannierx.core.model import WannierModel
from wannierx.fourier.derivatives import dH_dk
from wannierx.linalg.degeneracy import is_degenerate
from wannierx.linalg.eigh import eigh


def _omega_all_bands(
    E: Array, A: Array
) -> Array:
    """Full curvature for every band.

    Parameters:
        E: (..., norb) energies.
        A: (..., 3, norb, norb) eigenbasis derivative matrices
            A[a, n, m] = <u_n| d_a H |u_m>.

    Returns:
        (..., norb, 3, 3) real Berry curvature, Angstrom^2.
    """
    norb = E.shape[-1]
    gap = E[..., :, None] - E[..., None, :]  # (..., n, m)
    offdiag = 1.0 - jnp.eye(norb, dtype=gap.dtype)  # (n, m)
    inv_den = offdiag / jnp.where(offdiag.astype(bool), gap**2, 1.0)  # (..., n, m)
    # Omega_n^{ab} = -2 Im sum_{m != n} A[a, n, m] * A[b, m, n] / (E_n - E_m)^2
    # Explicit broadcast (avoids any index-order ambiguity):
    Anm = A                                # (..., a, n, m)
    Bmn = jnp.swapaxes(A, -1, -2)          # Bmn[..., b, n, m] = A[..., b, m, n]
    num = Anm[..., :, None, :, :] * Bmn[..., None, :, :, :]  # (..., a, b, n, m)
    weighted = num * inv_den[..., None, None, :, :]
    omega = -2.0 * jnp.imag(jnp.sum(weighted, axis=-1))  # (..., a, b, n)
    return jnp.moveaxis(omega, -1, -3)  # (..., n, a, b)


def berry_curvature(
    model: WannierModel,
    k: Array,
    bands: Sequence[int] | None = None,
    degeneracy_policy: str = "error",
    degeneracy_atol: float = 1e-9,
    degeneracy_rtol: float = 1e-7,
) -> Array:
    """Interband-derivative Berry curvature for isolated bands.

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).
        bands: band indices to compute; None = all bands.
        degeneracy_policy: ``"error"`` raises :class:`DegeneracyError` for
            a selected degenerate band; ``"nan"``/``"mask"`` return NaN
            for those bands.
        degeneracy_atol, degeneracy_rtol: degeneracy detection
            tolerances (gap <= atol + rtol*scale flags degeneracy).

    Returns:
        Real tensor, shape (..., n_sel, 3, 3), Angstrom^2, antisymmetric
        in the last two indices.

    Failure behavior:
        degenerate selected bands follow ``degeneracy_policy``; they
        never silently become zero. Under ``"error"`` and JIT tracing the
        host-side check is skipped (see :func:`is_degenerate` policy);
        no denominator regularization is ever applied.
    """
    if degeneracy_policy not in ("error", "nan", "mask"):
        raise ValueError(f"unknown degeneracy_policy: {degeneracy_policy!r}")

    k = jnp.asarray(k)
    eig = eigh(model, k)
    U = eig.vectors
    Ud = jnp.swapaxes(jnp.conjugate(U), -1, -2)
    dH = dH_dk(model, k)  # (..., 3, norb, norb), eV Angstrom
    A = jnp.einsum("...ni,...aij,...jm->...anm", Ud, dH, U)

    E = eig.energies
    n_orb = E.shape[-1]
    sel = jnp.arange(n_orb) if bands is None else jnp.asarray(bands).reshape(-1)
    omega_all = _omega_all_bands(E, A)  # (..., norb, 3, 3)
    out = omega_all[..., sel, :, :]

    deg = is_degenerate(E, atol=degeneracy_atol, rtol=degeneracy_rtol)
    deg_sel = deg[..., sel]  # (..., n_sel) bool

    if degeneracy_policy == "error":
        # Host-side strict check; skipped silently only when values are tracers
        # (i.e. inside jit/vmap), which is documented behavior.
        try:
            has_deg = bool(np.asarray(deg_sel).any())
        except Exception:
            has_deg = False  # traced: cannot host-check
        if has_deg:
            raise DegeneracyError(
                "berry_curvature requested for a band degenerate within the "
                "given tolerances; isolate a nondegenerate band or use a "
                "subspace/projector formulation"
            )
        return out

    mask = jnp.broadcast_to(deg_sel[..., None, None], out.shape)
    return jnp.where(mask, jnp.nan, out)
