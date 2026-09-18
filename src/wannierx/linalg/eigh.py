"""Hermitian eigensystem (WX-014).

Canonical eigenvector convention: ``U[..., :, n] = |u_n>`` (columns),
eigenvalues ascending. Uses ``jax.numpy.linalg.eigh``; batch semantics
preserve all leading dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from wannierx.core.model import WannierModel
from wannierx.fourier.transform import hamiltonian


@dataclass(frozen=True)
class EigenSystem:
    """Eigen-decomposition of H(k) at a set of k-points.

    Attributes:
        energies: (..., n_orb) real eigenvalues, ascending, eV.
        vectors:  (..., n_orb, n_orb) complex, columns are the eigenvectors.

    Reconstruction satisfies H = U diag(E) U^dagger. Differentiable:
    JAX differentiates through :func:`jax.numpy.linalg.eigh`; the
    eigenvector convention must not be changed internally.
    """

    energies: Array
    vectors: Array


def _flat(e: EigenSystem):
    return (e.energies, e.vectors), None


def _unflat(_, children):
    return EigenSystem(*children)


jax.tree_util.register_pytree_node(EigenSystem, _flat, _unflat)


def eigh(model: WannierModel, k: Array) -> EigenSystem:
    """Solve the Hermitian eigensystem of H(k).

    Parameters:
        model: canonical model.
        k: fractional reciprocal k, shape (..., 3).

    Returns:
        :class:`EigenSystem` with energies (..., n_orb) ascending and
        vectors (..., n_orb, n_orb) with columns as eigenvectors.

    Failure behavior: non-finite input propagates through the
    eigensolver; ill-defined individual eigenvectors at exact
    degeneracies are not made unique by this routine.
    """
    H = hamiltonian(model, k)
    w, v = jnp.linalg.eigh(H)
    return EigenSystem(energies=w, vectors=v)
