"""WannierX-specific exceptions.

Error philosophy (spec, section 13): fail loudly on ambiguous physics.
These are the specific exception types used across the library.
"""

from __future__ import annotations


class WannierXError(Exception):
    """Base class for all WannierX errors."""


class LatticeError(WannierXError):
    """Invalid or singular lattice."""


class ModelError(WannierXError):
    """Invalid WannierModel data (shape/consistency)."""


class ParseError(WannierXError):
    """Malformed input file."""


class NonHermitianError(WannierXError):
    """Non-Hermitian input passed to a Hermitian-only routine."""


class DegeneracyError(WannierXError):
    """Quantity undefined at an unresolved degeneracy."""


class PeriodicityError(WannierXError):
    """Operation requires periodicity that the model does not have."""


class DimensionalityError(WannierXError):
    """Operation is undefined for the model's dimensionality."""


class UnitConventionError(WannierXError):
    """A unit/thickness convention was required but not supplied."""
