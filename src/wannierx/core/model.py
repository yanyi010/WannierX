"""Canonical real-space Wannier Hamiltonian container (WX-003).

Physics convention (PHYSICS_CONVENTIONS.md, section 2):

    H(k_f) = sum_R w_R H_R exp(+i 2 pi k_f . R)

with ``H_R[..., m, n] = <0m|H|Rn>`` (eV), ``R`` integer lattice vectors in
fractional direct coordinates, and ``weights`` the per-R degeneracy
weights ``w_R``.

The model is immutable, a JAX pytree, performs no filesystem I/O, and
carries no source-format metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from wannierx.core.exceptions import ModelError
from wannierx.core.lattice import Lattice


@dataclass(frozen=True)
class WannierModel:
    """Immutable canonical Wannier/tight-binding model.

    Attributes:
        lattice:  (3, 3) :class:`Lattice`, Angstrom.
        R:        (n_R, 3) integer fractional direct lattice vectors.
        H_R:      (n_R, n_orb, n_orb) complex Hamiltonian blocks, eV.
        weights:  (n_R,) real degeneracy weights w_R.
        periodic: which of the three lattice directions are periodic.
        centers:  optional (n_orb, 3) Wannier centers in fractional
            direct coordinates, or None if unknown.

    Shapes follow the project specification exactly. All arrays are
    validated for shape consistency at construction.
    """

    lattice: Lattice
    R: Array
    H_R: Array
    weights: Array
    periodic: tuple[bool, bool, bool]
    centers: Array | None = None

    # pytree aux data (static, non-array fields)
    @staticmethod
    def _static_fields() -> tuple:
        return ("periodic",)

    def __post_init__(self) -> None:
        R = jnp.asarray(self.R)
        H_R = jnp.asarray(self.H_R)
        weights = jnp.asarray(self.weights)
        centers = None if self.centers is None else jnp.asarray(self.centers)

        if R.ndim != 2 or R.shape[1] != 3:
            raise ModelError(f"R must have shape (n_R, 3), got {R.shape}")
        n_R = R.shape[0]
        if H_R.ndim != 3 or H_R.shape[0] != n_R:
            raise ModelError(
                f"H_R must have shape (n_R, n_orb, n_orb) with same n_R={n_R}, got {H_R.shape}"
            )
        if H_R.shape[1] != H_R.shape[2]:
            raise ModelError(f"H_R orbital axes must be square, got {H_R.shape}")
        if weights.shape != (n_R,):
            raise ModelError(f"weights must have shape ({n_R},), got {weights.shape}")
        if centers is not None:
            if centers.shape != (H_R.shape[1], 3):
                raise ModelError(
                    f"centers must have shape (n_orb, 3)=({H_R.shape[1]}, 3), got {centers.shape}"
                )
        if len(tuple(self.periodic)) != 3:
            raise ModelError(f"periodic must have length 3, got {self.periodic}")

        object.__setattr__(self, "R", R)
        object.__setattr__(self, "H_R", H_R)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "periodic", tuple(bool(b) for b in self.periodic))
        object.__setattr__(self, "centers", centers)

    @property
    def n_R(self) -> int:
        """Number of real-space lattice vectors."""
        return int(self.R.shape[0])

    @property
    def n_orb(self) -> int:
        """Number of orbitals / Wannier functions."""
        return int(self.H_R.shape[1])


# JAX pytree registration: arrays are children; lattice is a nested pytree;
# ``periodic`` is static metadata.
def _model_flatten(model: WannierModel):
    children = (model.lattice.direct, model.R, model.H_R, model.weights, model.centers)
    aux = model.periodic
    return children, aux


def _model_unflatten(aux, children):
    lattice = Lattice(children[0])
    return WannierModel(
        lattice=lattice,
        R=children[1],
        H_R=children[2],
        weights=children[3],
        periodic=aux,
        centers=children[4],
    )


jax.tree_util.register_pytree_node(WannierModel, _model_flatten, _model_unflatten)
