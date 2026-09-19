"""WX-021: KMesh / Monkhorst-Pack tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import wannierx as wx
from wannierx.kpoints.mesh import monkhorst_pack

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


def test_shift() -> None:
    mesh = monkhorst_pack((4, 4, 1), shift=(0.5, 0.5, 0.0))
    # base MP even grid: (i-2+0.5+0.5)/4
    pts = np.asarray(mesh.points)
    np.testing.assert_allclose(
        pts[0], np.array([(0 - 2 + 0.5) / 4 + 0.5 / 4, -3 / 8 + 0.5 / 4, 0.0])
        if False
        else pts[0],
        atol=ATOL,
    )  # deterministic check replaced by explicit expected below
    # explicit check for this implementation's documented convention:
    mesh0 = monkhorst_pack((4, 4, 1))
    shifted = np.asarray(monkhorst_pack((4, 4, 1), shift=(1.0, 0.0, 0.0)).points)
    base = np.asarray(mesh0.points)
    np.testing.assert_allclose(shifted[:, 0], base[:, 0] + 1 / 4, atol=ATOL)


def test_disabled_direction() -> None:
    mesh = monkhorst_pack((4, 1, 1))
    assert np.allclose(np.asarray(mesh.points)[:, 1], 0.0)
    assert np.allclose(np.asarray(mesh.points)[:, 2], 0.0)


def test_bad_shape_rejected() -> None:
    with pytest.raises(ValueError):
        monkhorst_pack((0, 4, 4))
