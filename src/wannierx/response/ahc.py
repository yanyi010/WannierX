"""Intrinsic anomalous Hall conductivity.

3D contract:

    sigma_{a b} = -(e^2/hbar) sum_n integral_BZ d^3k/(2pi)^3 f_nk Omega_n^{ab}
    Output: S/m.

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
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from wannierx.constants import E2_OVER_HBAR_SI
from wannierx.core.exceptions import DimensionalityError
from wannierx.core.model import WannierModel
from wannierx.fourier.derivatives import dH_dk
from wannierx.geometry.berry import _omega_all_bands
from wannierx.kpoints.integration import integrate_bands
from wannierx.kpoints.mesh import KMesh
from wannierx.linalg.eigh import eigh
from wannierx.response.occupations import fermi_dirac

# the three antisymmetric (a, b) pairs: (y,z)->x, (z,x)->y, (x,y)->z
_PAIRS = ((1, 2), (2, 0), (0, 1))

_ANG_PER_M = 1.0e10  # 1 / Angstrom in m^-1 (Angstrom^-1 = 1e10 m^-1)


def _occupied_omega_per_k(model: WannierModel, k: Array, mu: float, temperature: float):
    eig = eigh(model, k)
    U = eig.vectors
    Ud = jnp.swapaxes(jnp.conjugate(U), -1, -2)
    dH = dH_dk(model, k)
    A = jnp.einsum("...ni,...aij,...jm->...anm", Ud, dH, U)
    omega = _omega_all_bands(eig.energies, A)  # (nk, norb, 3, 3) Angstrom^2
    f = fermi_dirac(eig.energies, mu, temperature)  # (nk, norb)
    return omega, f


def ahc(
    model: WannierModel,
    mesh: KMesh,
    mu: float,
    temperature: float = 0.0,
    thickness: float | None = None,
    chunk_size: int | None = None,
) -> Array:
    """Intrinsic anomalous Hall conductivity.

    Parameters:
        model: canonical model.
        mesh: uniform KMesh with normalized weights. For a strictly 2D
            model exactly one sampled direction must have length 1.
        mu: chemical potential, eV.
        temperature: K (T = 0 supported explicitly).
        thickness: for 2D models only, a physical layer thickness in
            Angstrom; when given converts the native sheet conductance
            (S) to S/m. Never supply for a 3D model.
        chunk_size: optional mesh chunk size (semantically inert).

    Returns:
        3-component array (sigma_yz, sigma_zx, sigma_xy):
          * 3D model: S/m.
          * 2D model, ``thickness=None``: native sheet conductance, S.
          * 2D model, ``thickness`` given: S/m.

    Failure behavior:
        3D mesh on fewer than 3 sampled directions raises
        :class:`DimensionalityError`; requesting S/m from a 2D model
        without thickness is allowed only via the explicit returned
        sheet conductance (documented above). Occupied-boundary
        degeneracies propagate from the Berry-curvature layer.
    """
    sampled = [int(s) for s in mesh.shape if s > 1]
    is_2d = len(sampled) == 2
    is_3d = len(sampled) == 3
    if not (is_2d or is_3d):
        raise DimensionalityError(f"AHC requires a 2D or 3D mesh, got shape {mesh.shape}")
    if is_3d and thickness is not None:
        raise DimensionalityError("thickness is a 2D-only convention; model is 3D")

    def kern(m: WannierModel, kchunk: Array) -> Array:
        omega, f = _occupied_omega_per_k(m, kchunk, mu, temperature)
        om = jnp.stack([omega[..., a, b] for a, b in _PAIRS], axis=-1)  # (nkc, norb, 3)
        return f[..., None] * om  # (nkc, norb, 3)

    per_band = integrate_bands(model, mesh, kern, chunk_size=chunk_size)  # (norb, 3)
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
