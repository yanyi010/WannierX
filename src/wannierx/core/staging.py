"""Host-side staging detection for JAX transformations.

Guards for host-side validation of pytree leaves must detect *any*
staging mode, and the probe must follow the same evaluation path as the
guarded validation itself:

* ``jit``/``vmap`` stage ``jax.core.Tracer`` instances (np.asarray and
  float()/bool() both raise), but
* ``checkify`` re-traces transformed bodies with ``ArrayImpl`` leaves
  for which ``np.asarray`` *succeeds* while JAX operations on them
  yield traced results (float()/bool() raise
  ``ConcretizationTypeError``).

So neither ``isinstance(x, jax.core.Tracer)`` nor an ``np.asarray``
probe is reliable. The reliable probe runs a trivial JAX reduction and
attempts the boolean concretization -- exactly the pattern every
host-side validation in this package uses.
"""

from __future__ import annotations

import jax
import jax.errors
import jax.numpy as jnp
from jax import Array


def is_staged(x: Array) -> bool:
    """True when JAX operations on ``x`` cannot be evaluated host-side
    because a transformation (jit, vmap, grad, checkify, ...) staged it.

    Use this to skip host-side value validation of pytree leaves during
    staging. Never use ``isinstance(x, jax.core.Tracer)`` or
    ``np.asarray`` success/failure as the probe: both misclassify
    checkify-staged ``ArrayImpl`` leaves.
    """
    try:
        bool(jnp.all(jnp.equal(x, x)))
    except jax.errors.ConcretizationTypeError:
        return True
    return False
