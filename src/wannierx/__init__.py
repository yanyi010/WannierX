"""WannierX public API.

Importing this package never mutates global JAX configuration.
Enable x64 explicitly (``wx.config.enable_x64()``)
or via ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

from wannierx import config, constants, models
from wannierx.core.lattice import Lattice
from wannierx.core.model import WannierModel
from wannierx.core.validation import HermiticityReport, validate_hermiticity
from wannierx.fourier.derivatives import d2H_dk2, dH_dk
from wannierx.fourier.transform import hamiltonian
from wannierx.geometry.berry import berry_curvature
from wannierx.geometry.topology import chern_number
from wannierx.io.wannier90.loader import LoadedWannier90, load_wannier90
from wannierx.kpoints import mesh as kmesh
from wannierx.kpoints.integration import integrate, integrate_bands
from wannierx.kpoints.mesh import (
    KMesh,
    gamma_centered_mesh,
    monkhorst_pack,
    monkhorst_pack_mesh,
    uniform_mesh,
)
from wannierx.linalg.degeneracy import cluster_degenerate, is_degenerate
from wannierx.linalg.eigh import EigenSystem, eigh
from wannierx.linalg.projectors import projector
from wannierx.operators.velocity import velocity_matrix, velocity_operator
from wannierx.response.ahc import ahc
from wannierx.response.dos import dos
from wannierx.response.occupations import fermi_dirac

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022  (grouped by subsystem, not alphabetically)
    "__version__",
    "config",
    "constants",
    "models",
    # core
    "Lattice",
    "WannierModel",
    "HermiticityReport",
    "validate_hermiticity",
    # fourier
    "hamiltonian",
    "dH_dk",
    "d2H_dk2",
    # linalg
    "EigenSystem",
    "eigh",
    "cluster_degenerate",
    "is_degenerate",
    "projector",
    # geometry
    "berry_curvature",
    "chern_number",
    # kpoints
    "kmesh",
    "KMesh",
    "gamma_centered_mesh",
    "monkhorst_pack",
    "monkhorst_pack_mesh",
    "uniform_mesh",
    "integrate",
    "integrate_bands",
    # io
    "load_wannier90",
    "LoadedWannier90",
    # operators
    "velocity_operator",
    "velocity_matrix",
    # response
    "fermi_dirac",
    "dos",
    "ahc",
]
