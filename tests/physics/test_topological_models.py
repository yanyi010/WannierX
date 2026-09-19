"""WX-018: SSH, Haldane, QWZ, graphene fixture tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.models.graphene import graphene
from wannierx.models.haldane import haldane
from wannierx.models.qzhang import qiwuzhang
from wannierx.models.ssh import ssh

ATOL = 1e-12


def _k2(kx: float, ky: float) -> jnp.ndarray:
    return jnp.array([[kx, ky, 0.0]])


def test_ssh_analytic_spectrum_and_regimes() -> None:
    m = ssh(t1=0.8, t2=1.2)
    kf = jnp.linspace(-0.5, 0.5, 17)
    k = jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], -1)
    E = np.asarray(wx.eigh(m, k).energies)
    Ea = np.sqrt(0.8**2 + 1.2**2 + 2 * 0.8 * 1.2 * np.cos(2 * np.pi * np.asarray(kf)))
    np.testing.assert_allclose(E, np.stack([-Ea, Ea], -1), atol=ATOL)
    # trivial when |t1|>|t2|: gap at BZ edge kf=0.5 = 2|t1-t2|
    edge = np.abs(np.asarray(wx.eigh(m, _k2(0.5, 0.0)).energies)).min()
    assert edge == pytest.approx(abs(0.8 - 1.2), abs=ATOL)


def test_qwz_analytic_spectrum() -> None:
    m = qiwuzhang(u=-1.0)
    kfx, kfy = 0.21, 0.33
    d = np.array(
        [
            np.sin(2 * np.pi * kfx),
            np.sin(2 * np.pi * kfy),
            -1 + np.cos(2 * np.pi * kfx) + np.cos(2 * np.pi * kfy),
        ]
    )
    E = np.asarray(wx.eigh(m, _k2(kfx, kfy)).energies).ravel()
    np.testing.assert_allclose(E, [-np.linalg.norm(d), np.linalg.norm(d)], atol=ATOL)


def test_qwz_chern_phases() -> None:
    # locked by chern_number tests
    assert wx.chern_number(qiwuzhang(u=-1.0), mesh=(24, 24), occupied=[0]) == pytest.approx(-1.0, abs=1e-9)
    assert wx.chern_number(qiwuzhang(u=3.0), mesh=(24, 24), occupied=[0]) == pytest.approx(0.0, abs=1e-9)


def test_haldane_chern_regimes() -> None:
    # phi = pi/2, t2 = 0.3: topological at M = 0, trivial at large |M|
    c_top = wx.chern_number(haldane(t2=0.3, M=0.0), mesh=(24, 24), occupied=[0])
    assert abs(abs(c_top) - 1.0) < 1e-9
    c_triv = wx.chern_number(haldane(t2=0.3, M=2.0), mesh=(24, 24), occupied=[0])
    assert abs(c_triv) < 1e-9


def test_graphene_dirac_and_hermiticity() -> None:
    m = graphene()
    assert wx.validate_hermiticity(m).is_hermitian
    # gapless at Dirac point (1/3, 2/3) of this convention
    E = np.asarray(wx.eigh(m, _k2(1 / 3, 2 / 3)).energies).ravel()
    np.testing.assert_allclose(E, [0.0, 0.0], atol=1e-12)


def test_models_hermiticity_and_periodicity() -> None:
    for mod in (
        ssh(0.8, 1.2),
        qiwuzhang(u=-1.0),
        haldane(t2=0.3, M=0.0),
        graphene(),
    ):
        assert wx.validate_hermiticity(mod).is_hermitian
        k = jnp.array([[0.13, 0.29, 0.0]])
        for G in ([1, 0, 0], [0, 1, 0], [1, 1, 0]):
            np.testing.assert_allclose(
                np.asarray(wx.hamiltonian(mod, k)),
                np.asarray(wx.hamiltonian(mod, k + jnp.asarray(G, dtype=float))),
                atol=ATOL,
            )
