"""Numerically stable Fermi-Dirac occupations.

f(E, mu, T) = 1 / (exp((E - mu)/(kB T)) + 1) = sigmoid((mu - E)/(kB T))

Energy and mu in eV, temperature K. T = 0 uses the explicit step
convention f = theta(mu - E) with f(mu) = 1/2 (standard convention).
"""

from __future__ import annotations

import jax
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

    Stability: computed as ``sigmoid((mu - E) / (kB T))``, which is free
    of overflow for any finite input. Differentiable at any positive
    temperature (and at E != mu for T = 0). ``temperature`` stays a
    traced array (no ``float()`` concretization), so it works as a
    dynamic scalar under ``jax.jit`` and in ``jax.grad``.
    """
    e = jnp.asarray(energy)
    # static validation only (traced temperatures cannot be concretized
    # here; negative traced T falls back to the T = 0 branch).
    if isinstance(temperature, (int, float)) and temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    # T = 0 branch selected dynamically (traced-safe); the mu < E
    # comparison broadcasts over the energy shape.
    kT = KB_EV_PER_K * jnp.asarray(temperature)
    finite_T = jnp.asarray(temperature) > 0.0

    # T > 0: numerically stable sigmoid form (no manual clip/exp).
    x = (mu - e) / jnp.where(finite_T, kT, 1.0)  # guard kT == 0 division
    f_finite = jax.nn.sigmoid(x)

    # T = 0: explicit step convention; f(mu) = 1/2
    f_zero = jnp.where(e < mu, 1.0, jnp.where(e > mu, 0.0, 0.5))

    return jnp.where(finite_T, f_finite, f_zero)
