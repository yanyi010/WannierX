"""Shared pytest configuration.

Reference scientific correctness is defined in CPU x64 mode.
These tests MUST run with JAX_ENABLE_X64=1; the
session fixture enforces it rather than silently passing in x32.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_x64() -> None:
    if not jax.config.jax_enable_x64:
        pytest.exit(
            "WannierX reference tests require JAX_ENABLE_X64=1; "
            "refusing to run in x32.",
            returncode=2,
        )
