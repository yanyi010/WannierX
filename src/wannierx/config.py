"""Precision and runtime configuration diagnostics.

Importing :mod:`wannierx` MUST NOT mutate global JAX configuration
(spec, section 4). This module only provides *read-only* runtime
diagnostics and explicit helpers the caller may invoke.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def x64_enabled() -> bool:
    """Return True when JAX 64-bit precision is enabled.

    Reads the current JAX configuration without modifying it.
    """
    return bool(jax.config.jax_enable_x64)


def reference_float_dtype():
    """Return the effective default floating dtype of a JAX array."""
    return jnp.asarray(1.0).dtype


def diagnostics() -> dict[str, object]:
    """Return runtime diagnostics useful for reference-test reports.

    Includes x64 status, devices, and effective dtypes. Purely
    informational; has no side effects.
    """
    return {
        "jax_version": jax.__version__,
        "x64_enabled": x64_enabled(),
        "devices": [str(d) for d in jax.devices()],
        "float_dtype": str(jnp.asarray(1.0).dtype),
        "complex_dtype": str(jnp.asarray(1.0 + 0.0j).dtype),
    }


def enable_x64() -> None:
    """Explicitly enable JAX x64 mode for the calling process.

    Provided as a convenience for scripts/demos. Library import never
    calls this automatically.
    """
    jax.config.update("jax_enable_x64", True)
