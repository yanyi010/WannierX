"""KMesh / Monkhorst-Pack tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from wannierx.kpoints.mesh import (
    gamma_centered_mesh,
    monkhorst_pack,
    monkhorst_pack_mesh,
    uniform_mesh,
)

ATOL = 1e-15


def test_point_count_and_weights_normalize() -> None:
    for n in ((4, 4, 1), (3, 5, 1), (2, 2, 2), (8, 8, 8)):
        mesh = monkhorst_pack(n)
        assert mesh.n_k == n[0] * n[1] * n[2]
        assert abs(float(jnp.sum(mesh.weights)) - 1.0) < ATOL
        assert mesh.shape == n


def test_ordering_deterministic() -> None:
    m1 = monkhorst_pack((4, 3, 1))
    m2 = monkhorst_pack((4, 3, 1))
    np.testing.assert_array_equal(np.asarray(m1.points), np.asarray(m2.points))


def test_mp_even_grid_regression() -> None:
    """N=4 standard MP grid is exactly [-3/8, -1/8, 1/8, 3/8] (off-Gamma)."""
    mesh = monkhorst_pack_mesh((4, 4, 4))
    first_axis = np.asarray(mesh.points)[:, 0]
    np.testing.assert_allclose(
        np.unique(first_axis),
        np.array([-3 / 8, -1 / 8, 1 / 8, 3 / 8]),
        atol=ATOL,
    )


def test_mp_even_grid_first_direction_exact() -> None:
    """First direction of monkhorst_pack_mesh((4,4,4)) equals the MP
    coordinates [-3/8, -1/8, 1/8, 3/8] exactly."""
    mesh = monkhorst_pack_mesh((4, 4, 4))
    pts = np.asarray(mesh.points)
    # C ordering: first axis slowest, so every n2*n3 = 16 consecutive
    # points share the same first-direction coordinate.
    np.testing.assert_array_equal(
        jnp.array(pts[::16, 0]),
        jnp.array([-3 / 8, -1 / 8, 1 / 8, 3 / 8]),
    )


def test_mp_odd_grid_regression() -> None:
    """N=3 standard MP grid is [-1/3, 0, 1/3] (Gamma-including)."""
    mesh = monkhorst_pack_mesh((3, 3, 3))
    first_axis = np.asarray(mesh.points)[:, 0]
    np.testing.assert_allclose(
        np.unique(first_axis),
        np.array([-1 / 3, 0.0, 1 / 3]),
        atol=ATOL,
    )


def test_gamma_centered_even_grid_regression() -> None:
    """Gamma-centered N=4 grid is [-1/2, -1/4, 0, 1/4] (includes Gamma)."""
    mesh = gamma_centered_mesh((4, 1, 1))
    np.testing.assert_allclose(
        np.asarray(mesh.points)[:, 0],
        np.array([-1 / 2, -1 / 4, 0.0, 1 / 4]),
        atol=ATOL,
    )


def test_gamma_centered_odd_grid_regression() -> None:
    """Gamma-centered N=3 grid is [-1/3, 0, 1/3]."""
    mesh = gamma_centered_mesh((3, 1, 1))
    np.testing.assert_allclose(
        np.asarray(mesh.points)[:, 0],
        np.array([-1 / 3, 0.0, 1 / 3]),
        atol=ATOL,
    )


def test_gamma_centered_includes_gamma_even() -> None:
    mesh = gamma_centered_mesh((4, 1, 1))
    assert np.any(np.isclose(np.asarray(mesh.points)[:, 0], 0.0, atol=ATOL))


def test_mp_off_gamma_even() -> None:
    mesh = monkhorst_pack_mesh((4, 1, 1))
    assert not np.any(np.isclose(np.asarray(mesh.points)[:, 0], 0.0, atol=ATOL))


def test_mp_gamma_included_odd() -> None:
    mesh = monkhorst_pack_mesh((3, 1, 1))
    assert np.any(np.isclose(np.asarray(mesh.points)[:, 0], 0.0, atol=ATOL))


def test_gamma_centered_odd_includes_gamma() -> None:
    mesh = gamma_centered_mesh((3, 1, 1))
    assert np.any(np.isclose(np.asarray(mesh.points)[:, 0], 0.0, atol=ATOL))


def test_mp_and_gamma_coincide_odd() -> None:
    a = np.asarray(monkhorst_pack_mesh((5, 5, 5)).points)
    b = np.asarray(gamma_centered_mesh((5, 5, 5)).points)
    np.testing.assert_allclose(a, b, atol=ATOL)


def test_shift() -> None:
    """Shift moves every point along each shifted direction by
    shift_d * (1/n_d) for MP centering."""
    base = np.asarray(monkhorst_pack_mesh((4, 4, 1)).points)
    shifted = np.asarray(monkhorst_pack_mesh((4, 4, 1), shift=(1.0, 0.0, 0.0)).points)
    np.testing.assert_allclose(shifted[:, 0], base[:, 0] + 1 / 4, atol=ATOL)
    np.testing.assert_allclose(shifted[:, 1], base[:, 1], atol=ATOL)
    np.testing.assert_allclose(shifted[:, 2], base[:, 2], atol=ATOL)
    # half shift on even MP grid: (2i-3)/8 + 1/8 = [-1/4, 0, 1/4, 1/2]
    half = np.asarray(monkhorst_pack_mesh((4, 1, 1), shift=(0.5, 0.0, 0.0)).points)
    np.testing.assert_allclose(half[:, 0], np.array([-1 / 4, 0.0, 1 / 4, 1 / 2]), atol=ATOL)


def test_monkhorst_pack_alias_matches_mesh() -> None:
    a = monkhorst_pack((6, 2, 1))
    b = monkhorst_pack_mesh((6, 2, 1))
    np.testing.assert_array_equal(np.asarray(a.points), np.asarray(b.points))


def test_uniform_mesh_centering_dispatch() -> None:
    mp = np.asarray(uniform_mesh((4, 4, 1), centering="mp").points)
    gm = np.asarray(uniform_mesh((4, 4, 1), centering="gamma").points)
    np.testing.assert_array_equal(mp, np.asarray(monkhorst_pack_mesh((4, 4, 1)).points))
    np.testing.assert_array_equal(gm, np.asarray(gamma_centered_mesh((4, 4, 1)).points))


def test_uniform_mesh_bad_centering_rejected() -> None:
    with pytest.raises(ValueError):
        uniform_mesh((4, 4, 1), centering="shifted")  # type: ignore[arg-type]


def test_disabled_direction() -> None:
    mesh = monkhorst_pack((4, 1, 1))
    assert np.allclose(np.asarray(mesh.points)[:, 1], 0.0)
    assert np.allclose(np.asarray(mesh.points)[:, 2], 0.0)


def test_bad_shape_rejected() -> None:
    with pytest.raises(ValueError):
        monkhorst_pack((0, 4, 4))


# ---------------------------------------------------------------------------
# Property-based tests (deterministic pseudo-random, no external deps)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(20260922)  # reserved for future stochastic props


def _in_first_bz(pts: np.ndarray) -> np.ndarray:
    return np.all((pts >= -0.5 - 1e-12) & (pts < 0.5 - 1e-9))


@pytest.mark.parametrize("centering", ["gamma", "mp"])
@pytest.mark.parametrize("seed", range(8))
def test_mesh_properties_random(centering: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = tuple(int(x) for x in rng.integers(1, 9, size=3))
    mesh = uniform_mesh(n, centering=centering)  # type: ignore[arg-type]
    pts = np.asarray(mesh.points)
    # point count
    assert mesh.n_k == n[0] * n[1] * n[2]
    assert pts.shape == (mesh.n_k, 3)
    # weights
    assert abs(float(jnp.sum(mesh.weights)) - 1.0) < ATOL
    # distinctness within the fundamental [0,1) representation
    folded = np.mod(pts, 1.0)
    folded_rounded = np.round(folded, 9)
    unique = np.unique(folded_rounded, axis=0)
    assert unique.shape[0] == mesh.n_k, "mesh points must be distinct mod G"
    # equivalent interval: every coordinate lies in [-0.5, 0.5)
    assert _in_first_bz(pts), "all coordinates must lie in [-0.5, 0.5)"
    # uniform spacing along each sampled direction
    for d in range(3):
        if n[d] > 1:
            coords = np.unique(pts[:, d])
            assert coords.shape[0] == n[d]
            np.testing.assert_allclose(np.diff(coords), 1.0 / n[d], atol=1e-12)


@pytest.mark.parametrize("centering", ["gamma", "mp"])
def test_mesh_properties_large_even(centering: str) -> None:
    n = (8, 6, 2)
    mesh = uniform_mesh(n, centering=centering)  # type: ignore[arg-type]
    pts = np.asarray(mesh.points)
    assert mesh.n_k == 8 * 6 * 2
    folded = np.round(np.mod(pts, 1.0), 9)
    assert np.unique(folded, axis=0).shape[0] == mesh.n_k
    assert _in_first_bz(pts)


@pytest.mark.parametrize("centering", ["gamma", "mp"])
def test_mesh_properties_odd(centering: str) -> None:
    n = (5, 3, 7)
    mesh = uniform_mesh(n, centering=centering)  # type: ignore[arg-type]
    pts = np.asarray(mesh.points)
    assert mesh.n_k == 5 * 3 * 7
    folded = np.round(np.mod(pts, 1.0), 9)
    assert np.unique(folded, axis=0).shape[0] == mesh.n_k
    assert _in_first_bz(pts)
    # odd grids always include Gamma for both centerings (shift = 0)
    assert np.any(np.all(np.isclose(pts, 0.0, atol=ATOL), axis=1))
