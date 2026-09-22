"""Smoke test: import works in a clean environment and does not
mutate global JAX precision configuration."""

from __future__ import annotations

import subprocess
import sys


def test_import_and_no_global_mutation() -> None:
    code = (
        "import jax\n"
        "before = jax.config.jax_enable_x64\n"
        "import wannierx\n"
        "after = jax.config.jax_enable_x64\n"
        "assert before == after, (before, after)\n"
        "print('ok')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ok" in out.stdout
