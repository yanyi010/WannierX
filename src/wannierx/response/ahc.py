"""Intrinsic anomalous Hall conductivity, Hamiltonian-only formulation.

3D contract:

    sigma_{a b} = -(e^2/hbar) sum_n integral_BZ d^3k/(2pi)^3 f_nk Omega_n^{ab}
    Output: S/m.

Physics scope: this is the *Hamiltonian-only* (tight-binding) Kubo
AHC, i.e. the BZ integral of the per-band curvature produced by the
shared geometry kernel (:func:`wannierx.band_geometry`). It is exact
for tight-binding models. The full ab-initio Wannier AHC of Wannier90's
postw90 additionally contains the D-D and D-Abar terms built from
position-operator matrix elements <0m|r|Rn>; that operator layer is
planned for WannierX v0.3+ (roadmap: reproduce the Wannier90 Fe
tutorial AHC decomposition). Until then, AHC from Wannier90 ``hr.dat``
input is tight-binding quality.

Derivation used here (centralized geometry):

  Omega is computed w.r.t. *cartesian* k (Angstrom^2). Writing the BZ
  integral in the fractional coordinates of the mesh,
      d^3k_cart = |det B| d^3k_frac,
      BZ average = (sum_k w_k f Omega) -> integral_BZ d^3k_frac f Omega,
  so, with bz_vol = |det B| in Angstrom^-3,

      sigma_{ab} [S/m]
        = -(e^2/hbar)[S] * (bz_vol/(2pi)^3)[Angstrom^-3]
          * (BZ average of f*Omega)[Angstrom^2] * (1e10 Angstrom^-1 -> m^-1)
        = -(e^2/hbar)[S] * (bz_vol/(2pi)^3) * avg * 1e10.

For a strictly 2D model (exactly one mesh axis of length 1 in directions
i, j sampled) the native quantity is the sheet conductance

      sigma_sheet^{ab} [S]
        = -(e^2/hbar)[S] * (|B_i x B_j|/(2pi)^2)[Angstrom^-2]
          * avg[Angstrom^2] * 1  -> but Angstrom^-2 * Angstrom^2 = 1 (dimensionless)

  and ``thickness`` (Angstrom) optionally converts sheet S -> S/m by
  dividing by thickness in meters. A 2D model is never silently reported
  as a 3D conductivity.

Degeneracy contract (v0.1.1): the per-band curvature is ill-defined
inside a degenerate cluster (the 1/(eps_n - eps_m)^2 denominators). The
BZ kernel routes through the shared geometry kernel, and:

* a degenerate band with f == 0 contributes exactly zero and is masked
  exactly (no approximation: 0 times anything is 0);
* a degenerate band with f > 0 follows ``degeneracy_policy``:
  ``"error"`` (default) raises :class:`DegeneracyError` after the BZ
  reduction -- the check runs on the concrete reduced result, so it
  fires identically with and without the internal JIT chunking;
  ``"nan"`` propagates NaN into the returned tensor; ``"mask"`` drops
  those bands' contribution (a documented approximation).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from wannierx.constants import E2_OVER_HBAR_SI
from wannierx.core.exceptions import DegeneracyError, DimensionalityError
from wannierx.core.model import WannierModel
from wannierx.geometry.kernel import band_geometry
from wannierx.kpoints.integration import integrate_bands
from wannierx.kpoints.mesh import KMesh
from wannierx.response.occupations import fermi_dirac

# the three antisymmetric (a, b) pairs: (y,z)->x, (z,x)->y, (x,y)->z
_PAIRS = ((1, 2), (2, 0), (0, 1))

_ANG_PER_M = 1.0e10  # 1 / Angstrom in m^-1 (Angstrom^-1 = 1e10 m^-1)


def ahc(
    model: WannierModel,
    mesh: KMesh,
    mu: float,
    temperature: float = 0.0,
    thickness: float | None = None,
    chunk_size: int | None = None,
    degeneracy_policy: str = "error",
    degeneracy_atol: float = 1e-9,
    degeneracy_rtol: float = 1e-7,
) -> Array:
    """Intrinsic anomalous Hall conductivity (Hamiltonian-only form).

    Parameters:
        model: canonical model.
        mesh: uniform KMesh with normalized weights. For a strictly 2D
            model exactly one sampled direction must have length 1.
        mu: chemical potential, eV (must be finite).
        temperature: K (T = 0 supported explicitly).
        thickness: for 2D models only, a physical layer thickness in
            Angstrom; when given converts the native sheet conductance
            (S) to S/m. Never supply for a 3D model.
        chunk_size: optional mesh chunk size (semantically inert).
        degeneracy_policy: policy for degenerate bands with f > 0; see
            the module docstring for the exact contract.
        degeneracy_atol, degeneracy_rtol: degeneracy detection
            tolerances of the geometry kernel.

    Returns:
        3-component array (sigma_yz, sigma_zx, sigma_xy):
          * 3D model: S/m.
          * 2D model, ``thickness=None``: native sheet conductance, S.
          * 2D model, ``thickness`` given: S/m.

    Failure behavior:
        3D mesh on fewer than 3 sampled directions raises
        :class:`DimensionalityError`; requesting S/m from a 2D model
        without thickness is allowed only via the explicit returned
        sheet conductance (documented above). An occupied (f > 0)
        degeneracy follows ``degeneracy_policy`` (default:
        :class:`DegeneracyError`, raised after the reduction so the
        semantics are identical with and without JIT chunking).
    """
    if degeneracy_policy not in ("error", "nan", "mask"):
        raise ValueError(f"unknown degeneracy_policy: {degeneracy_policy!r}")
    if isinstance(mu, (int, float)) and not np.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu}")
    if isinstance(temperature, (int, float)):
        if not np.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {temperature}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

    sampled = [int(s) for s in mesh.shape if s > 1]
    is_2d = len(sampled) == 2
    is_3d = len(sampled) == 3
    if not (is_2d or is_3d):
        raise DimensionalityError(f"AHC requires a 2D or 3D mesh, got shape {mesh.shape}")
    if is_3d and thickness is not None:
        raise DimensionalityError("thickness is a 2D-only convention; model is 3D")

    def kern(m: WannierModel, kchunk: Array) -> Array:
        geom = band_geometry(
            m, kchunk, degeneracy_atol=degeneracy_atol, degeneracy_rtol=degeneracy_rtol
        )
        f = fermi_dirac(geom.energies, mu, temperature)  # (nkc, norb)
        om = jnp.stack([geom.omega[..., a, b] for a, b in _PAIRS], axis=-1)  # (nkc, norb, 3)
        # a degeneracy matters only where it carries occupation weight
        relevant = geom.degenerate & (f > 0.0)  # (nkc, norb)
        if degeneracy_policy == "mask":
            om = jnp.where(relevant[..., None], 0.0, om)
        else:
            # "error" / "nan": poison now; "error" raises after the
            # reduction (on the concrete result), so the failure
            # semantics do not depend on the internal JIT chunking.
            om = jnp.where(relevant[..., None], jnp.nan, om)
        # f == 0 bands contribute exactly zero; this also shields the
        # reduction from the raw inf/NaN of unoccupied degenerate
        # clusters (exact masking, not an approximation).
        return jnp.where((f > 0.0)[..., None], f[..., None] * om, 0.0)  # (nkc, norb, 3)

    per_band = integrate_bands(model, mesh, kern, chunk_size=chunk_size)  # (norb, 3)

    if degeneracy_policy == "error" and not bool(jnp.all(jnp.isfinite(per_band))):
        raise DegeneracyError(
            "ahc: the per-band Berry curvature is ill-defined at an "
            "occupied (f > 0) degeneracy on the mesh (e.g. a gap closing "
            "at the Fermi level, or a symmetry-enforced degeneracy in an "
            "occupied band). Options: move mu/temperature away from the "
            "degeneracy, or pass degeneracy_policy='nan' to propagate "
            "NaN, or degeneracy_policy='mask' to drop the degenerate "
            "bands' contribution (approximation)"
        )

    avg = jnp.sum(per_band, axis=0)  # (3,) Angstrom^2 * BZ(fractional) average

    lat = model.lattice
    if is_3d:
        bz_vol = float(lat.bz_volume)  # Angstrom^-3
        jacobian = bz_vol / (2.0 * np.pi) ** 3  # Angstrom^-3
        # Angstrom^2 * Angstrom^-3 = Angstrom^-1 -> *1e10 = m^-1; e^2/hbar gives S
        return -E2_OVER_HBAR_SI * avg * jacobian * _ANG_PER_M  # S/m

    i, j = [d for d, s in enumerate(mesh.shape) if s > 1]
    area_B2 = float(np.linalg.norm(np.cross(np.array(lat.reciprocal[i]), np.array(lat.reciprocal[j]))))
    jacobian2 = area_B2 / (2.0 * np.pi) ** 2  # Angstrom^-2
    # Angstrom^2 * Angstrom^-2 = 1 -> sheet conductance in S
    sigma_sheet = -E2_OVER_HBAR_SI * avg * jacobian2  # S
    if thickness is not None:
        t = float(thickness)
        if t <= 0:
            raise ValueError("thickness must be positive")
        return sigma_sheet / (t * 1.0e-10)  # S/m
    return sigma_sheet
