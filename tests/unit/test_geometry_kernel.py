"""Geometry kernel contract: shapes, diagnostics, pytree behavior."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.models.graphene import graphene
from wannierx.models.qzhang import qiwuzhang

ATOL = 1e-13


def test_band_geometry_shape_contract() -> None:
    """k: (3,) and (Nk, 3) batch the leading dims of every field."""
    m = qiwuzhang(u=-1.0)
    g0 = wx.band_geometry(m, jnp.array([0.13, 0.29, 0.0]))
    assert g0.energies.shape == (2,)
    assert g0.omega.shape == (2, 3, 3)
    assert g0.degenerate.shape == (2,)
    assert g0.min_gap.shape == (2,)

    g1 = wx.band_geometry(m, jnp.array([[0.13, 0.29, 0.0], [0.4, 0.4, 0.0]]))
    assert g1.energies.shape == (2, 2)
    assert g1.omega.shape == (2, 2, 3, 3)
    assert g1.degenerate.shape == (2, 2)
    assert g1.min_gap.shape == (2, 2)


def test_band_geometry_min_gap_is_spectral_gap() -> None:
    """Two-band model: min_gap[n] = |E_0 - E_1| for both bands."""
    m = qiwuzhang(u=-1.0)
    k = jnp.array([[0.13, 0.29, 0.0], [0.4, 0.1, 0.0]])
    g = wx.band_geometry(m, k)
    E = np.asarray(g.energies)
    gap = np.abs(E[:, 1] - E[:, 0])
    np.testing.assert_allclose(np.asarray(g.min_gap)[:, 0], gap, atol=ATOL)
    np.testing.assert_allclose(np.asarray(g.min_gap)[:, 1], gap, atol=ATOL)
    assert not np.asarray(g.degenerate).any()


def test_band_geometry_flags_dirac_degeneracy() -> None:
    m = graphene()  # gapless at K
    g = wx.band_geometry(m, jnp.array([[1 / 3, 2 / 3, 0.0]]))
    assert np.asarray(g.degenerate).all()
    assert float(np.asarray(g.min_gap)[0, 0]) < 1e-9


def test_band_geometry_single_band_model() -> None:
    """One orbital: no other band, min_gap = +inf, curvature = 0."""
    lattice = wx.Lattice(jnp.eye(3))
    R = jnp.asarray([[0, 0, 0]])
    H_R = jnp.asarray([[[2.0]]], dtype=jnp.complex128)
    m = wx.WannierModel(lattice, R, H_R, jnp.ones(1), (True, True, False))
    g = wx.band_geometry(m, jnp.array([0.2, 0.3, 0.0]))
    assert g.energies.shape == (1,)
    assert np.all(np.asarray(g.min_gap) == np.inf)
    assert not np.asarray(g.degenerate).any()
    np.testing.assert_array_equal(np.asarray(g.omega), 0.0)


def test_band_geometry_is_pytree() -> None:
    """BandGeometry round-trips through jit and tree operations."""
    m = qiwuzhang(u=-1.0)
    k = jnp.array([0.13, 0.29, 0.0])
    g_eager = wx.band_geometry(m, k)
    g_jit = jax.jit(wx.band_geometry)(m, k)
    np.testing.assert_allclose(
        np.asarray(g_jit.omega), np.asarray(g_eager.omega), atol=ATOL
    )
    leaves = jax.tree_util.tree_leaves(g_eager)
    assert len(leaves) == 4
    # tree_map composes over the dataclass fields
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, g_eager)
    np.testing.assert_allclose(
        np.asarray(doubled.omega), 2 * np.asarray(g_eager.omega), atol=ATOL
    )


def test_band_geometry_matches_berry_curvature() -> None:
    """berry_curvature is exactly the kernel's omega restricted to bands."""
    m = qiwuzhang(u=-1.0)
    k = jnp.array([[0.13, 0.29, 0.0], [0.4, 0.1, 0.0]])
    g = wx.band_geometry(m, k)
    om = np.asarray(wx.berry_curvature(m, k, degeneracy_policy="nan"))
    np.testing.assert_allclose(om, np.asarray(g.omega), atol=ATOL)


def test_band_geometry_single_k_no_spurious_batch() -> None:
    """A k of shape (3,) yields un-batched fields (shape-contract lock)."""
    m = qiwuzhang(u=-1.0)
    g = wx.band_geometry(m, jnp.array([0.13, 0.29, 0.0]))
    assert g.omega.ndim == 3  # (n_orb, 3, 3), not (1, n_orb, 3, 3)
