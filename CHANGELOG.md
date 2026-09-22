# Changelog

All notable changes to WannierX are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows semver (in the 0.x series: patch = correctness fixes, minor =
breaking or feature releases).

## [0.1.1] - 2026-09-22

Correctness hotfix over 0.1.0. No new physics capability; contracts and
failure semantics are now enforced where 0.1.0 documented but did not
implement them.

### Fixed

- **k-mesh convention (behavior change)**: `monkhorst_pack()` even-mesh
  grids now use the standard Monkhorst-Pack set
  `k_d(i) = (2 i - n_d + 1) / (2 n_d)` (off-Gamma for even `n_d`,
  Gamma-including for odd). v0.1.0 produced Gamma-including grids for
  even `n_d`, contradicting its own docstring. **Results on even meshes
  differ from v0.1.0 — that was the bug.**
- Single k of shape `(3,)` no longer produces a spurious leading batch
  axis in `H(k)` and all derived quantities (`k: (3,) -> H: (n, n)`;
  `k: (Nk, 3) -> H: (Nk, n, n)`).
- DOS `broadening` and `fermi_dirac` `temperature` are no longer
  concretized with `float()`; both are jit-dynamic and differentiable
  as documented. `fermi_dirac` uses the numerically stable sigmoid
  form with a traced-safe `T = 0` branch.
- `validate_hermiticity()` checks the effective hopping
  `T_R = w_R H_R` against `T_{-R}^dagger` — the exact real-space
  condition for Hermitian `H(k)`. v0.1.0 checked the bare `H_R`, which
  wrongly passed for asymmetric weights `w_R != w_{-R}`.
- `hr.dat` parser rejects duplicate and missing `(m, n)` entries within
  an R block (missing entries previously retained uninitialized memory)
  and non-positive degeneracy counts.
- `chern_number()` isolation uses the exact selected-subspace to
  complement direct gap
  `min over k, i in S, j not in S of |eps_i - eps_j|`. The previous
  min/max-index span check silently accepted non-contiguous selections
  whose unselected interior bands were degenerate with a selected band.
  Arbitrary (non-contiguous) band subsets are now supported; internal
  degeneracy within the selected subspace or its complement is allowed.
- AHC no longer bypasses the Berry-layer degeneracy policy: a
  degenerate band with occupation `f > 0` raises `DegeneracyError` by
  default (`degeneracy_policy="error"`), or propagates NaN (`"nan"`),
  or drops the contribution (`"mask"`, documented approximation).
  Degenerate clusters with `f == 0` are masked exactly (no
  approximation). The error semantics are identical with and without
  internal JIT chunking.
- Pytree model arguments no longer crash under
  `checkify(vmap(..., in_axes=(None, ...)))`: staging detection is now
  a concretization probe (`wannierx.core.staging.is_staged`) instead of
  `isinstance(x, jax.core.Tracer)`; checkify re-traces transformed
  bodies with `ArrayImpl` leaves that the isinstance/np.asarray checks
  misclassify.
- Model construction validates `R` (integer, unique), finiteness of
  `H_R` / `weights` / lattice, and nonzero weights (host-side;
  skipped under staging).

### Added

- `gamma_centered_mesh()`, `monkhorst_pack_mesh()`,
  `uniform_mesh(centering="gamma" | "mp")` — explicit grid conventions
  (locked by even/odd regressions); `monkhorst_pack()` kept as an alias
  of `monkhorst_pack_mesh`.
- `checked_berry_curvature()` — strict degeneracy-checked Berry
  curvature whose error is functionalized with `checkify.check` and
  fires under `jax.jit` / `jax.vmap` via `checkify.checkify`; a plain
  `jax.jit` fails loudly at trace time (the check is never silently
  skipped).
- `band_geometry()` / `BandGeometry` — shared per-band geometry kernel
  (energies, curvature, degeneracy mask, minimum spectral gap)
  consumed by `berry_curvature`, `checked_berry_curvature` and `ahc`.
- `ahc(..., degeneracy_policy=..., degeneracy_atol=..., degeneracy_rtol=...)`.
- `LICENSE` (MIT), included in wheel and sdist metadata.

### Changed

- `berry_curvature()` and `ahc()` docstrings state the
  Hamiltonian-only (tight-binding) physics scope explicitly: full
  ab-initio Wannier Berry curvature / AHC additionally requires
  position-operator matrix elements (the D-D, D-Abar, Abar-Omega terms
  of Wannier90's postw90); the operator layer is planned for v0.3+.
- mypy `python_version = "3.12"` so JAX's PEP 695 stubs parse; the
  runtime floor remains Python 3.11 (guarded by ruff `target-version`).

### Tests

- 150 -> 179: AHC degeneracy matrix (raise, chunk-invariance, exact
  unoccupied masking, nan/mask policies), checked Berry curvature under
  eager / checkify+jit / checkify+vmap / loud plain-jit trace failure,
  Chern non-contiguous selections (isolated, degenerate-with-complement,
  complement-internal degeneracy, full space), geometry-kernel shape /
  diagnostic / pytree contracts, staging-guard matrix.

## [0.1.0] - 2026-09-22

First public release. See the
[v0.1.0 release notes](https://github.com/yanyi010/WannierX/releases/tag/v0.1.0).
