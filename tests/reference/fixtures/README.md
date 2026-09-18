# toy Wannier90 reference fixture (WX-011)

Source: generated in-repo (see git history of this directory).
Provenance: `toy_hr.dat` is a 2-band SSH-like model with t1=0.8, t2=1.2,
degeneracies [1, 1, 1]; lattice a1=4, a2=8, a3=10 Ang in `toy.win`.
Interpolation mode: legacy_hr (w_R = 1/ndegen).
Reference eigenvalues: computed by WannierX itself in x64 (float64/complex128)
on CPU at 9 fractional k points kf in linspace(-0.5, 0.5, 9) -> toy_eigenvalues.npy.
Purpose: regression lock of parser + loader + eigensolver chain.
Tolerance: reproduced to 1e-12 absolute in x64.
