"""WX-011: trusted Wannier90 interpolation fixture regression."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import wannierx as wx

FIX = Path("tests/reference/fixtures")


def test_reproduce_reference_eigenvalues() -> None:
    m = wx.load_wannier90(str(FIX / "toy"), mode="legacy_hr").model
    E_ref = np.load(FIX / "toy_eigenvalues.npy")
    kf = np.linspace(-0.5, 0.5, 9)
    K = jnp.stack([jnp.asarray(kf), jnp.zeros(9), jnp.zeros(9)], axis=-1)
    E = np.asarray(wx.eigh(m, K).energies)
    np.testing.assert_allclose(E, E_ref, atol=1e-12)
