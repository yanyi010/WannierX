"""Gauge-robust lattice Chern number (Fukui-Hatsugai-Suzuki).

Uses link variables from subspace overlaps rather than derivatives of
arbitrary eigenvector phases:

    U_mu(k) = det[ U^dagger(k) U(k+mu) ] / |det[ U^dagger(k) U(k+mu) ]|
    F_12(k) = ln[ U_1(k) U_2(k+1) U_1(k+2)^{-1} U_2(k)^{-1} ]
    C = (1 / 2 pi i) sum_k F_12(k)

The algorithm requires an isolated selected subspace; the isolation is
checked on the given mesh and failure raises loudly (no silent zero).

Reference: Fukui, Hatsugai, Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from wannierx.core.exceptions import DegeneracyError
from wannierx.core.model import WannierModel
from wannierx.linalg.eigh import eigh


def _subspace_vectors(model: WannierModel, k: Array, occupied: Array) -> Array:
    eig = eigh(model, k)
    return eig.vectors[..., :, occupied]  # (nk, norb, n_occ)


def chern_number(
    model: WannierModel,
    mesh: tuple[int, int],
    occupied,
    isolation_atol: float = 1e-8,
    isolation_rtol: float = 1e-5,
) -> float:
    """Lattice Chern number of the ``occupied`` subspace over a 2D mesh.

    Parameters:
        model: canonical model; only the first two reciprocal axes are
            sampled (k3 fixed at 0). Convention: fractional reciprocal
            coordinates on a uniform periodic (N1, N2) grid on [0, 1)^2.
        mesh: (N1, N2) grid dimensions (each >= 2).
        occupied: band indices forming the selected subspace; any
            subset of bands, contiguous or not (e.g. ``range(n_occ)``
            or ``[0, 2]``). Internal degeneracy *within* the selected
            subspace (or within its complement) is allowed -- the FHS
            link variables are subspace-gauge covariant -- as long as
            the gap between the selected subspace and its complement
            stays open on the mesh.
        isolation_atol, isolation_rtol: tolerances for the selected
            subspace to complement direct gap,
            min over k, i in S, j not in S of |eps_i - eps_j|.

    Returns:
        Chern number as a float; converges to an integer.

    Failure behavior: raises :class:`DegeneracyError` when the selected
        subspace is not isolated from its complement anywhere on the
        mesh (the S <-> complement gap closes), or on a vanishing
        subspace overlap. No regularization is applied.
    """
    n1, n2 = int(mesh[0]), int(mesh[1])
    if n1 < 2 or n2 < 2:
        raise ValueError("mesh dimensions must be >= 2 in both sampled directions")
    occ = np.sort(np.unique(np.asarray(occupied).reshape(-1).astype(int)))
    n_occ = occ.size
    if n_occ == 0:
        raise ValueError("occupied subspace must be non-empty")
    n_orb = model.n_orb
    if np.any(occ < 0) or np.any(occ >= n_orb):
        raise ValueError(f"occupied indices out of range for n_orb={n_orb}")

    # periodic grid on [0, 1) x [0, 1) x {0}, row-major (n1, n2)
    g1 = np.arange(n1) / n1
    g2 = np.arange(n2) / n2
    K = np.stack(np.meshgrid(g1, g2, [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
    k = jnp.asarray(K)

    eig = eigh(model, k)
    E = np.asarray(eig.energies)
    U = _subspace_vectors(model, k, jnp.asarray(occ))  # (nk, norb, n_occ)
    U_np = np.asarray(U)

    # --- isolation check (host-side, strict) ---
    # Direct gap between the selected subspace S and its complement:
    #   min over k on the mesh, i in S, j not in S of |E_i - E_j|.
    # This is the exact condition the FHS algorithm needs; a span-based
    # check (min/max selected index) silently misses non-contiguous
    # selections whose unselected interior bands are degenerate with a
    # selected band. Degeneracies internal to S or to the complement
    # are allowed (subspace-gauge covariant algorithm).
    in_s = np.zeros(n_orb, dtype=bool)
    in_s[occ] = True
    E_s = E[:, in_s]  # (nk, n_occ)
    E_c = E[:, ~in_s]  # (nk, n_orb - n_occ)
    if E_c.shape[1] > 0:
        gap_floor = float(np.min(np.abs(E_s[:, :, None] - E_c[:, None, :])))
        scale = max(float(np.max(np.abs(E))), 1.0)
        if gap_floor <= isolation_atol + isolation_rtol * scale:
            raise DegeneracyError(
                f"selected subspace is not isolated from its complement on "
                f"the mesh (min S<->complement gap {gap_floor}); refine the "
                f"mesh or choose an isolated subspace"
            )
    # else: the full band space is selected; the complement is empty,
    # so there is no subspace boundary that could close.

    # --- FHS link variables ---
    U3 = U_np.reshape(n1, n2, *U_np.shape[1:])  # (n1, n2, norb, n_occ)

    def shift(a, d1, d2):
        return np.roll(np.roll(a, -d1, axis=0), -d2, axis=1)

    def overlap_arg(V, W):
        # det(V^dagger W), normalized to a U(1) link
        M = np.einsum("abni,abnj->abij", V.conj(), W)
        d = np.linalg.det(M)
        mag = np.abs(d)
        if np.any(mag == 0):
            raise DegeneracyError("vanishing subspace overlap encountered on the mesh")
        return d / mag

    u1 = overlap_arg(U3, shift(U3, 1, 0))  # (n1, n2)
    u2 = overlap_arg(U3, shift(U3, 0, 1))

    F = np.log(u1 * shift(u2, 1, 0) / (shift(u1, 0, 1) * u2))
    chern = np.sum(F) / (2j * np.pi)
    return float(np.real(chern))
