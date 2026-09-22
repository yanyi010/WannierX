"""Isolated-band Berry curvature, Hamiltonian-only (tight-binding) form.

Units Angstrom^2:

    Omega_n^{ab} = -2 Im sum_{m != n}
        <u_n| d_a H |u_m> <u_m| d_b H |u_n> / (eps_n - eps_m)^2

Physics scope: this is the *Hamiltonian-only* (tight-binding) interband
Kubo curvature, computed from H(k) and its analytic k-derivatives
alone. It is exact for tight-binding models, and it is the correct
Hamiltonian contribution for Wannier-interpolated Hamiltonians. The
full ab-initio Wannier Berry curvature additionally requires
position-operator matrix elements <0m|r|Rn> (the D-D, D-Abar and
Abar-Omega terms of Wannier90's postw90); that operator layer is
planned for WannierX v0.3+. Until then, results obtained from
Wannier90 ``hr.dat`` input are tight-binding-quality geometry and must
not be reported as general ab-initio Wannier curvature.

No hidden epsilon. Allowed degeneracy policies: ``error``, ``nan``,
``mask``. The returned tensor has leading batch dims of k, then
(n_sel, 3, 3) with the last two being cartesian derivative indices.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import checkify

from wannierx.core.exceptions import DegeneracyError
from wannierx.core.model import WannierModel
from wannierx.core.staging import is_staged
from wannierx.geometry.kernel import BandGeometry, band_geometry


def _sel_index(bands: Sequence[int] | None, n_orb: int) -> Array:
    """Band-selection indices; None selects all bands."""
    if bands is None:
        return jnp.arange(n_orb)
    return jnp.asarray(bands).reshape(-1)


def _select_bands(geom: BandGeometry, bands: Sequence[int] | None) -> tuple[Array, Array]:
    """Selected curvature slices and degeneracy mask from a geometry."""
    sel = _sel_index(bands, geom.energies.shape[-1])
    out = geom.omega[..., sel, :, :]
    deg_sel = geom.degenerate[..., sel]
    return out, deg_sel


def berry_curvature(
    model: WannierModel,
    k: Array,
    bands: Sequence[int] | None = None,
    degeneracy_policy: str = "error",
    degeneracy_atol: float = 1e-9,
    degeneracy_rtol: float = 1e-7,
) -> Array:
    """Hamiltonian-only (tight-binding) Berry curvature for isolated bands.

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).
        bands: band indices to compute; None = all bands.
        degeneracy_policy: ``"error"`` raises :class:`DegeneracyError`
            for a selected degenerate band; ``"nan"``/``"mask"`` return
            NaN for those bands (``"mask"`` is currently an alias of
            ``"nan"``; it will carry the validity-mask semantics of the
            planned ``GeometryResult`` API).
        degeneracy_atol, degeneracy_rtol: degeneracy detection
            tolerances (gap <= atol + rtol*scale flags degeneracy).

    Returns:
        Real tensor, shape (..., n_sel, 3, 3), Angstrom^2, antisymmetric
        in the last two indices.

    Failure behavior:
        degenerate selected bands follow ``degeneracy_policy``; they
        never silently become zero. Under ``"error"`` and JIT/vmap
        tracing the host-side check cannot concretize and is skipped
        (the raw kernel then returns its natural inf/NaN at exact
        degeneracies); for tracer-safe strict errors use
        :func:`checked_berry_curvature`. No denominator regularization
        is ever applied.
    """
    if degeneracy_policy not in ("error", "nan", "mask"):
        raise ValueError(f"unknown degeneracy_policy: {degeneracy_policy!r}")

    geom = band_geometry(
        model, k, degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol
    )
    out, deg_sel = _select_bands(geom, bands)

    if degeneracy_policy == "error":
        # Host-side strict check; runs whenever the energies are
        # concrete (any staging mode defers to checked_berry_curvature,
        # whose checkify-functionalized check is tracer-safe).
        if not is_staged(deg_sel) and bool(np.asarray(deg_sel).any()):
            sel = _sel_index(bands, geom.energies.shape[-1])
            min_gap = float(jnp.min(geom.min_gap[..., sel]))
            raise DegeneracyError(
                "berry_curvature requested for a band degenerate within "
                "the given tolerances (min selected spectral gap "
                f"{min_gap:.3e} eV); isolate a nondegenerate band or use "
                "a subspace/projector formulation"
            )
        return out

    mask = jnp.broadcast_to(deg_sel[..., None, None], out.shape)
    return jnp.where(mask, jnp.nan, out)


def checked_berry_curvature(
    model: WannierModel,
    k: Array,
    bands: Sequence[int] | None = None,
    degeneracy_atol: float = 1e-9,
    degeneracy_rtol: float = 1e-7,
) -> Array:
    """Strict degeneracy-checked variant of :func:`berry_curvature`.

    Same computation and units as
    ``berry_curvature(..., degeneracy_policy="error")``, but the
    degeneracy check is functionalized with ``checkify.check`` and
    therefore also fires under ``jax.jit``/``jax.vmap``/``lax.scan``:

        checked = checkify.checkify(jax.jit(checked_berry_curvature))
        err, omega = checked(model, k)
        err.throw()  # raises if a selected band was degenerate

    Failure behavior:
        * eager (concrete k): raises :class:`DegeneracyError` directly,
          including the minimum selected spectral gap in the message;
        * staged with ``checkify.checkify``: the error is returned in
          the :class:`checkify.Error` and raised by ``err.throw()``;
        * staged with a plain ``jax.jit`` (no checkify): fails loudly
          at trace time, because an un-functionalized ``check`` cannot
          be staged -- the check is never silently skipped.
        degenerate selected bands never silently become zero; no
        denominator regularization is ever applied.
    """
    geom = band_geometry(
        model, k, degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol
    )
    out, deg_sel = _select_bands(geom, bands)

    # Eager path: concrete check with full diagnostics.
    if not is_staged(deg_sel) and bool(np.asarray(deg_sel).any()):
        sel = _sel_index(bands, geom.energies.shape[-1])
        min_gap = float(jnp.min(geom.min_gap[..., sel]))
        raise DegeneracyError(
            "checked_berry_curvature: selected band degenerate within "
            f"tolerances (min selected spectral gap {min_gap:.3e} eV); "
            "isolate a nondegenerate band or use a subspace/projector "
            "formulation"
        )

    # Traced path: functionalized check, re-raised by err.throw() when
    # the caller wraps this function in checkify.checkify.
    checkify.check(
        ~jnp.any(deg_sel),
        "checked_berry_curvature: selected band degenerate within "
        "tolerances; isolate a nondegenerate band or use a "
        "subspace/projector formulation",
    )
    return out
