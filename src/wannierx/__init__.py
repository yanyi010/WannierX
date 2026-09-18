"""WannierX public API.

Importing this package never mutates global JAX configuration
(spec, section 4). Enable x64 explicitly (``wx.config.enable_x64()``)
or via ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

from wannierx import config, constants, models

__version__ = "0.1.0"

__all__ = ["__version__", "config", "constants", "models"]
