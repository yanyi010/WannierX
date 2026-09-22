"""Shared band-geometry kernel: eigensystem -> spectral gaps -> curvature.

Single source of truth for the per-band Hamiltonian-only Berry
curvature. The selected-band APIs (:func:`wannierx.berry_curvature`,
:func:`wannierx.checked_berry_curvature`) and the BZ response API
(:func:`wannierx.ahc`) all consume :func:`band_geometry`, so curvature
values and degeneracy diagnostics can no longer diverge between the
layers. (v0.1.1 hotfix: AHC previously called the raw curvature routine
directly, bypassing the degeneracy policy that the Berry layer
documented.)

Physics contract: the curvature computed here is the
*Hamiltonian-only* (tight-binding) interband Kubo form

    Omega_n^{ab} = -2 Im sum_{m != n}
        <u_n| d_a H |u_m> <u_m| d_b H |u_n> / (eps_n - eps_m)^2,

valid for isolated bands of H(k). For ab-initio Wannier interpolation
the Berry curvature of the electronic states additionally requires
position-operator matrix elements <0m|r|Rn> (the D-D, D-Abar and
Abar-Omega terms of Wannier90's postw90); WannierX gains that operator
layer from v0.3 on. Until then, geometry computed from Wannier90
``hr.dat`` input is tight-binding quality and is documented as such at
every consuming API.

Kernel semantics: never raises on degeneracies and never regularizes
denominators. It returns the raw per-band curvature together with a
degenerate-band mask and the per-band minimum spectral gap; policy
decisions (``error`` / ``nan`` / ``mask``) live with the consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel
from wannierx.fourier.derivatives import dH_dk
from wannierx.linalg.degeneracy import is_degenerate
from wannierx.linalg.eigh import eigh


@dataclass(frozen=True)
class BandGeometry:
    """Per-band geometry of H(k) at a batch of k-points.

    Attributes:
        energies: (..., n_orb) eigenvalues, ascending, eV.
        omega: (..., n_orb, 3, 3) per-band Berry curvature,
            Angstrom^2, antisymmetric in the last two (cartesian
            derivative) indices. Ill-defined (NaN/inf or huge) inside
            degenerate clusters; see ``degenerate``.
        degenerate: (..., n_orb) bool; True where the band belongs to a
            pair of adjacent sorted energies degenerate within the
            kernel tolerances.
        min_gap: (..., n_orb) minimum |eps_n - eps_m| over all other
            bands m, eV; +inf for a single-band model. Spectral
            diagnostic; never used to regularize anything.
    """

    energies: Array
    omega: Array
    degenerate: Array
    min_gap: Array


def _flat(g: BandGeometry):
    return (g.energies, g.omega, g.degenerate, g.min_gap), None


def _unflat(_, children):
    return BandGeometry(*children)


jax.tree_util.register_pytree_node(BandGeometry, _flat, _unflat)


def _omega_all_bands(E: Array, A: Array) -> Array:
    """Raw per-band curvature for every band (no degeneracy handling).

    Parameters:
        E: (..., norb) eigenvalues, ascending.
        A: (..., 3, norb, norb) eigenbasis derivative matrices,
            A[..., a, n, m] = <u_n| d_a H |u_m>.

    Returns:
        (..., norb, 3, 3) real Berry curvature, Angstrom^2. Exactly
        degenerate input bands produce inf/NaN entries; callers decide
        the policy (this kernel never regularizes).
    """
    norb = E.shape[-1]
    gap = E[..., :, None] - E[..., None, :]  # (..., n, m)
    offdiag = 1.0 - jnp.eye(norb, dtype=gap.dtype)  # (n, m)
    inv_den = offdiag / jnp.where(offdiag.astype(bool), gap**2, 1.0)  # (..., n, m)
    # Omega_n^{ab} = -2 Im sum_{m != n} A[a, n, m] A[b, m, n] / (E_n - E_m)^2.
    # A[b, m, n] is the last-two-axes swap of A; inv_den is symmetric
    # in (n, m) because gap^2 is.
    A_swapped = jnp.swapaxes(A, -1, -2)  # (..., a, m, n)
    weighted = jnp.einsum("...anm,...bnm,...nm->...abn", A, A_swapped, inv_den)
    omega = -2.0 * jnp.imag(weighted)  # (..., a, b, n)
    return jnp.moveaxis(omega, -1, -3)  # (..., n, a, b)


def band_geometry(
    model: WannierModel,
    k: Array,
    degeneracy_atol: float = 1e-9,
    degeneracy_rtol: float = 1e-7,
) -> BandGeometry:
    """Eigensystem, per-band curvature and degeneracy diagnostics of H(k).

    Parameters:
        model: canonical Wannier model.
        k: fractional reciprocal k, shape (..., 3).
        degeneracy_atol, degeneracy_rtol: degeneracy detection
            tolerances (gap <= atol + rtol*scale flags degeneracy).

    Returns:
        :class:`BandGeometry`; all fields carry the leading batch dims
        of ``k``.

    Failure behavior: never raises on degeneracies; the ``degenerate``
        mask and ``min_gap`` diagnostics carry the diagnosis. Pure JAX:
        jit/vmap/grad-safe.
    """
    k = jnp.asarray(k)
    eig = eigh(model, k)
    E = eig.energies
    U = eig.vectors
    Ud = jnp.swapaxes(jnp.conjugate(U), -1, -2)
    dH = dH_dk(model, k)  # (..., 3, norb, norb), eV Angstrom
    A = jnp.einsum("...ni,...aij,...jm->...anm", Ud, dH, U)

    omega = _omega_all_bands(E, A)  # (..., norb, 3, 3)
    degenerate = is_degenerate(E, atol=degeneracy_atol, rtol=degeneracy_rtol)

    # per-band minimum |E_n - E_m| over m != n (diagonal excluded)
    gap_abs = jnp.abs(E[..., :, None] - E[..., None, :])  # (..., n, m)
    not_self = ~jnp.eye(E.shape[-1], dtype=bool)
    min_gap = jnp.min(jnp.where(not_self, gap_abs, jnp.inf), axis=-1)  # (..., n)

    return BandGeometry(energies=E, omega=omega, degenerate=degenerate, min_gap=min_gap)
