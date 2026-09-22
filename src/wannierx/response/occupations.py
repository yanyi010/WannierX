"""Numerically stable Fermi-Dirac occupations.

f(E, mu, T) = 1 / (exp((E - mu)/(kB T)) + 1)

Energy and mu in eV, temperature K. T = 0 uses the explicit step
convention f = theta(mu - E) with f(mu) = 1/2 (standard convention).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from wannierx.constants import KB_EV_PER_K


def fermi_dirac(energy: Array, mu: float, temperature: float) -> Array:
    """Fermi-Dirac occupation.

    Parameters:
        energy: (...,) energies, eV.
        mu: chemical potential, eV.
        temperature: temperature, K. Must be >= 0. T = 0 gives the explicit
            step function with f(E = mu) = 1/2.

    Returns:
        Occupations in [0, 1], same shape as ``energy``.

    Stability: computed with clipped exponent argument so it is free of
    overflow for any finite input. Differentiable at any positive
    temperature (and at E != mu for T = 0).
    """
    T = float(temperature)
    if T < 0:
        raise ValueError(f"temperature must be >= 0, got {T}")
    e = jnp.asarray(energy)
    kT = KB_EV_PER_K * T
    if T == 0.0:
        # explicit step convention; f(mu) = 1/2
        return jnp.where(e < mu, 1.0, jnp.where(e > mu, 0.0, 0.5))
    x = (e - mu) / kT
    xc = jnp.clip(x, -500.0, 500.0)  # exp underflow at ~745 for float64
    return 1.0 / (jnp.exp(xc) + 1.0)
