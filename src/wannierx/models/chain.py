"""Canonical one-orbital 1D nearest-neighbor chain.

Permanent regression fixture:

    E(k_f) = onsite + 2 t cos(2 pi k_f)

Embedded in the 3D representation with only x periodicity:
lattice = diag(a, 1, 1), R in {(-1,0,0), (0,0,0), (+1,0,0)},
H_R[..., 0, 0] = t, onsite, t and weights all 1.

All quantities use the canonical package units (energy eV, length
Angstrom).
"""

from __future__ import annotations

import jax.numpy as jnp

from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel


def chain(t: float, onsite: float = 0.0, lattice_constant: float = 1.0) -> WannierModel:
    """Build the 1D nearest-neighbor chain model.

    Parameters:
        t: nearest-neighbor hopping, eV.
        onsite: on-site energy epsilon_0, eV.
        lattice_constant: 1D lattice constant a, Angstrom.

    Returns:
        :class:`WannierModel` with dispersion E(k_f) = onsite + 2 t cos(2 pi k_f).
    """
    A = jnp.diag(jnp.asarray([lattice_constant, 1.0, 1.0]))
    lattice = Lattice(A)
    R = jnp.asarray([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    H_R = jnp.asarray([[[complex(t)]], [[complex(onsite)]], [[complex(t)]]])
    weights = jnp.ones(3)
    return WannierModel(
        lattice=lattice,
        R=R,
        H_R=H_R,
        weights=weights,
        periodic=(True, False, False),
        centers=jnp.zeros((1, 3)),
    )
