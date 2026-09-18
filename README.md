# WannierX

Differentiable, accelerator-native electronic-structure and response engine for
real-space Wannier/tight-binding Hamiltonians, built on JAX.

Canonical pipeline:

```
H_mn(R) -> H(k) -> {eps_n(k), |u_nk>} -> operators -> geometric and transport observables
```

## Conventions

The normative physics conventions (Fourier sign `exp(+ik.R)`, fractional
reciprocal k-coordinates by default, B = 2*pi*A^{-T}, eigenvectors as columns,
units) are defined by the project specification and are locked by the test
suite. Reference precision is float64/complex128; importing `wannierx` never
mutates global JAX configuration — tests run with `JAX_ENABLE_X64=1`.

## Install

```bash
pip install -e .
```

## Test

```bash
JAX_ENABLE_X64=1 pytest tests/unit tests/physics tests/autodiff tests/reference
```

## Quick example

```python
import wannierx as wx

model = wx.models.chain(t=-1.0)
import jax.numpy as jnp
k = jnp.array([0.25, 0.0, 0.0])  # fractional reciprocal
Hk = wx.hamiltonian(model, k)
eig = wx.eigh(model, k)
print(eig.energies)  # -2*cos(2*pi*0.25) == 0
```

## Repository layout

```
src/wannierx/
  core/        lattice, model, hermiticity validation
  io/wannier90/  _hr.dat / .win parsers, loader
  fourier/     H(k) and analytic k-derivatives
  linalg/      eigh, degeneracy clustering, projectors
  operators/   velocity
  geometry/    Berry curvature, Chern number
  kpoints/     meshes, weights, chunked BZ integration
  response/    Fermi-Dirac, DOS, AHC
  models/      chain, SSH, graphene, Haldane, Qi-Wu-Zhang
  utils/       chunking helpers
tests/
  unit/  physics/  reference/  autodiff/  performance/
benchmarks/  examples/
```
