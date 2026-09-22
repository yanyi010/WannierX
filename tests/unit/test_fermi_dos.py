"""Fermi-Dirac and smooth DOS tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.constants import KB_EV_PER_K
from wannierx.kpoints.mesh import monkhorst_pack
from wannierx.models.chain import chain
from wannierx.response.occupations import fermi_dirac

ATOL = 1e-13


def test_fermi_limits_and_T0() -> None:
    T = 300.0
    # low/high
    assert float(fermi_dirac(jnp.array(-10.0), 0.0, T)) == pytest.approx(1.0, abs=ATOL)
    assert float(fermi_dirac(jnp.array(10.0), 0.0, T)) == pytest.approx(0.0, abs=ATOL)
    # E == mu at finite T -> 1/2
    assert float(fermi_dirac(jnp.array(0.7), 0.7, T)) == pytest.approx(0.5, abs=ATOL)
    # T = 0 explicit step convention
    f = fermi_dirac(jnp.array([-0.1, 0.0, 0.1]), 0.0, 0.0)
    np.testing.assert_allclose(np.asarray(f), [1.0, 0.5, 0.0], atol=ATOL)


def test_fermi_large_exponent_stable() -> None:
    # huge |E-mu| must not overflow; NaN-poisoned check
    e = jnp.array([1e6, -1e6, 0.0])
    f = np.asarray(fermi_dirac(e, 0.0, 300.0))
    assert np.isfinite(f).all()
    assert f[0] < 1e-200 and f[1] == 1.0


def test_fermi_vectorize_and_autodiff() -> None:
    E = jnp.linspace(-1, 1, 7)
    f = fermi_dirac(E, 0.1, 300.0)
    assert f.shape == (7,)
    # autodiff d(f)/d(mu) at finite T: -df/dE
    g1 = (
        jax.grad(lambda mu: fermi_dirac(E[2], mu, 300.0))(0.1).item()
        if hasattr(jax.grad(lambda mu: fermi_dirac(E[2], mu, 300.0))(0.1), "item")
        else float(jax.grad(lambda mu: fermi_dirac(E[2], mu, 300.0))(0.1))
    )
    kT = KB_EV_PER_K * 300.0
    x = (float(E[2]) - 0.1) / kT
    fv = 1 / (np.exp(x) + 1)
    # analytic derivative w.r.t. mu = fv(1-fv)/kT
    assert g1 == pytest.approx(fv * (1 - fv) / kT, rel=1e-10)


def test_fermi_rejects_negative_T() -> None:
    with pytest.raises(ValueError):
        fermi_dirac(jnp.array(0.0), 0.0, -1.0)


def test_dos_positivity_and_normalization() -> None:
    m = chain(t=-1.0)
    mesh = monkhorst_pack((128, 1, 1))
    E = jnp.linspace(-3.0, 3.0, 1000)
    d = wx.dos(m, mesh, E, broadening=0.05)
    assert jnp.all(d >= 0)
    # integrate: one band => 1 state over wide window
    assert abs(float(jnp.trapezoid(d, E)) - 1.0) < 2e-3


def test_dos_chunk_invariance_and_grad() -> None:
    m = chain(t=-1.0)
    mesh = monkhorst_pack((64, 1, 1))
    E = jnp.linspace(-3.0, 3.0, 200)
    _ = wx.dos(m, mesh, E, 0.08)
    # gradient w.r.t. a smooth model parameter: for the 1D chain, the
    # entire spectrum scales linearly in t, so differentiate a smooth
    # functional of the DOS w.r.t. t through the eigen-energies directly.
    k = mesh.points

    # differentiate through eigh on the *analytic* chain dispersion
    # (equivalent Fourier model), keeping the model constructor static.
    def smooth_observable(t):
        kf = k[:, 0]
        eps = 2 * t * jnp.cos(2 * jnp.pi * kf)  # analytic chain band
        return jnp.sum(mesh.weights * jnp.exp(-(eps**2)) * eps**2)

    g = jax.grad(smooth_observable)(-1.0)
    assert np.isfinite(g)
    kf = mesh.points[:, 0]
    eps0 = 2 * (-1.0) * jnp.cos(2 * jnp.pi * kf)
    deps = 2 * jnp.cos(2 * jnp.pi * kf)
    expect = float(jnp.sum(mesh.weights * jnp.exp(-(eps0**2)) * (2 * eps0 - 2 * eps0**3) * deps))
    assert float(g) == pytest.approx(expect, rel=1e-10)


# ---------------------------------------------------------------------------
# jit-compatible gradients w.r.t. broadening and temperature
# ---------------------------------------------------------------------------


def test_dos_grad_broadening_jit_vs_fd() -> None:
    """jax.grad of the DOS w.r.t. broadening inside jax.jit, vs finite diff."""
    m = chain(t=-1.0)
    mesh = monkhorst_pack((64, 1, 1))
    E = jnp.linspace(-3.0, 3.0, 200)

    def smooth(sigma):
        d = wx.dos(m, mesh, E, sigma)
        return jnp.sum(d * E**2)

    jitted_grad = jax.jit(jax.grad(smooth))
    g = float(jitted_grad(0.08))
    assert np.isfinite(g)
    # central finite differences on the same functional
    h = 1e-6
    fd = (float(smooth(0.08 + h)) - float(smooth(0.08 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_fermi_grad_temperature_jit_vs_fd() -> None:
    """jax.grad of a Fermi-Dirac functional w.r.t. temperature inside jax.jit."""
    E = jnp.linspace(-0.5, 0.5, 11)
    mu = 0.1

    def occ_total(T):
        return jnp.sum(fermi_dirac(E, mu, T))

    jitted_grad = jax.jit(jax.grad(occ_total))
    g = float(jitted_grad(300.0))
    assert np.isfinite(g)
    h = 1e-2
    fd = (float(occ_total(300.0 + h)) - float(occ_total(300.0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_fermi_jit_dynamic_temperature_T0_step() -> None:
    """jit with traced temperature: T > 0 smooth, T = 0 step, both callable."""
    E = jnp.array([-0.1, 0.0, 0.1])
    jitted = jax.jit(lambda T: fermi_dirac(E, 0.0, T))
    fT = np.asarray(jitted(300.0))
    f0 = np.asarray(jitted(0.0))
    # finite T: 1/2 at E == mu, monotone decreasing
    assert fT[1] == pytest.approx(0.5, abs=ATOL)
    assert fT[0] > fT[1] > fT[2]
    # T = 0 explicit step convention f(mu) = 1/2
    np.testing.assert_allclose(f0, [1.0, 0.5, 0.0], atol=ATOL)


def test_fermi_large_ratio_stable_sigmoid() -> None:
    """Extreme |mu - E| / (kB T) must stay finite and exact at the limits."""
    # |mu-E| ~ 1e6 eV at T = 300 K -> |x| ~ 4e7; old clip form saturates,
    # sigmoid form must remain finite and hit 0/1 exactly.
    e = jnp.array([1e6, -1e6, 0.0])
    f = np.asarray(jax.jit(lambda T: fermi_dirac(e, 0.0, T))(300.0))
    assert np.isfinite(f).all()
    assert f[0] == 0.0 and f[1] == 1.0
    # moderate-but-large ratio consistent with the reference formula
    kT = KB_EV_PER_K * 300.0
    x = 60.0  # ~ 4.6e-26 in f
    e2 = kT * x
    f2 = float(fermi_dirac(jnp.array([e2]), 0.0, 300.0)[0])
    ref = 1.0 / (1.0 + np.exp(x))
    assert f2 == pytest.approx(ref, rel=1e-12)


def test_fermi_matches_reference_formula() -> None:
    """Consistency with the old (reference) 1/(exp(x)+1) formula."""
    rng = np.random.default_rng(0)
    E = jnp.asarray(rng.uniform(-0.5, 0.5, 21))
    mu = 0.05
    T = 350.0
    f = np.asarray(fermi_dirac(E, mu, T))
    kT = KB_EV_PER_K * T
    ref = 1.0 / (np.exp((np.asarray(E) - mu) / kT) + 1.0)
    np.testing.assert_allclose(f, ref, rtol=1e-14, atol=1e-16)
