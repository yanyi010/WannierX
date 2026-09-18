"""WX-004: Hermiticity validator tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from wannierx.core.model import WannierModel
from wannierx.core.validation import validate_hermiticity
from wannierx.models.chain import chain


def _clone(m: WannierModel, **kw) -> WannierModel:
    return WannierModel(
        lattice=kw.get("lattice", m.lattice),
        R=kw.get("R", m.R),
        H_R=kw.get("H_R", m.H_R),
        weights=kw.get("weights", m.weights),
        periodic=kw.get("periodic", m.periodic),
        centers=kw.get("centers", m.centers),
    )


def test_exact_hermitian() -> None:
    rep = validate_hermiticity(chain(t=-1.0))
    assert rep.is_hermitian
    assert rep.max_absolute_error == 0.0
    assert rep.missing_partners.shape == (0, 3)


def test_controlled_perturbation_detected() -> None:
    m = chain(t=-1.0)
    H_R = m.H_R.at[0, 0, 0].add(1e-6)  # break one hopping partner only
    rep = validate_hermiticity(_clone(m, H_R=H_R))
    assert not rep.is_hermitian
    # perturbation magnitude is recovered down to x64 roundoff of the
    # underlying values (~1e-16 relative), not tighter
    assert abs(rep.max_absolute_error - 1e-6) < 1e-16
    assert rep.worst_R is not None


def test_missing_partner_diagnosed() -> None:
    m = chain(t=-1.0)
    # drop the R=-1 block -> R=+1 has no partner
    keep = jnp.asarray([1, 2])
    rep = validate_hermiticity(
        _clone(m, R=m.R[keep], H_R=m.H_R[keep], weights=m.weights[keep])
    )
    assert not rep.is_hermitian
    assert rep.missing_partners.shape[0] == 1
    np.testing.assert_array_equal(np.asarray(rep.missing_partners[0]), np.array([1, 0, 0]))


def test_onsite_imaginary_contamination() -> None:
    m = chain(t=-1.0)
    H_R = m.H_R.at[1, 0, 0].add(1j * 0.5)  # onsite must be real
    rep = validate_hermiticity(_clone(m, H_R=H_R))
    assert not rep.is_hermitian
    assert abs(rep.max_absolute_error - 1.0) < 1e-15  # |eps - conj(eps)| = 2*Im
