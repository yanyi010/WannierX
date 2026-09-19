"""WX-028: v0.1 end-to-end release demo.

Runs in a clean environment: Wannier90 input -> load -> validate Hermiticity
-> band interpolation -> velocity -> Berry curvature -> AHC -> one smooth
scalar gradient.

Prerequirements:
    pip install -e .
    JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python examples/demo_v01.py
    (or set JAX_ENABLE_X64=1; the demo enables it explicitly if unset)

Run:  python examples/demo_v01.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    import wannierx as wx

    print("=== WannierX v0.1 end-to-end demo ===")
    print(json.dumps(wx.config.diagnostics(), indent=2))

    # ---- Wannier90 input (self-contained generated fixture) -------------
    fix = Path(__file__).resolve().parent / "demo_input"
    fix.mkdir(exist_ok=True)
    (fix / "demo.win").write_text(
        "begin unit_cell_cart\nang\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\nend unit_cell_cart\n"
    )
    num_wann = 2
    R_list = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    u = -1.0
    # build QWZ H_R in text form
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    HR = {
        (0, 0, 0): u * sz,
        (1, 0, 0): sx / (2j) + sz / 2,
        (-1, 0, 0): -sx / (2j) + sz / 2,
        (0, 1, 0): sy / (2j) + sz / 2,
        (0, -1, 0): -sy / (2j) + sz / 2,
    }
    lines = ["# demo QWZ model (auto-generated)", str(num_wann), str(len(R_list))]
    lines.append(" ".join("    1" for _ in R_list))
    for Rv in R_list:
        for m in range(1, num_wann + 1):
            for n in range(1, num_wann + 1):
                v = HR[Rv][m - 1][n - 1]
                lines.append(f"{Rv[0]:4d}{Rv[1]:4d}{Rv[2]:4d}{m:5d}{n:5d}{v.real:15.8f}{v.imag:15.8f}")
    (fix / "demo_hr.dat").write_text("\n".join(lines) + "\n")

    loaded = wx.load_wannier90(str(fix / "demo"), mode="auto")
    model = loaded.model
    print(f"loaded mode={loaded.mode}, n_orb={model.n_orb}, n_R={model.n_R}")

    # ---- validate Hermiticity --------------------------------------------
    rep = wx.validate_hermiticity(model)
    print(f"Hermiticity: {rep.is_hermitian} (max |err| = {rep.max_absolute_error:.2e})")

    # ---- band interpolation -----------------------------------------------
    kf = jnp.linspace(-0.5, 0.5, 21)
    K = jnp.stack([kf, jnp.zeros_like(kf), jnp.zeros_like(kf)], axis=-1)
    E = wx.eigh(model, K).energies
    print(f"band gap at Gamma(kf=0): {float(E[10,1]-E[10,0]):.6f} eV")

    # ---- velocity -----------------------------------------------------------
    vm = wx.velocity_matrix(model, K)
    print(f"max |group velocity| over path: {float(jnp.max(jnp.abs(jnp.diagonal(vm[:,0],axis1=-2,axis2=-1)))):.3f} m/s")

    # ---- Berry curvature + AHC ---------------------------------------------
    mesh = wx.monkhorst_pack((48, 48, 1))
    om = wx.berry_curvature(model, mesh.points, degeneracy_policy="nan")
    print(f"BZ-averaged lower-band Omega_xy: {float(np.mean(np.asarray(om[:,0,0,1]))):.6f} A^2")
    s = wx.ahc(model, mesh, mu=0.0)
    sxy_e2h = float(s[2] / wx.constants.E2_OVER_H_SI)
    print(f"AHC sheet conductance sigma_xy = {float(s[2]):.4e} S  ({sxy_e2h:.4f} e^2/h)")

    # ---- differentiate one smooth scalar observable ------------------------
    def smooth_scalar(tt):
        # hopping-like smooth parameter: scale H_R by tt
        H_R = model.H_R * tt
        mm = wx.WannierModel(model.lattice, model.R, H_R, model.weights, model.periodic, model.centers)
        eps = wx.eigh(mm, mesh.points).energies
        return jnp.sum(mesh.weights[:, None] * jnp.exp(-(eps**2)))

    g = jax.grad(smooth_scalar)(1.0)
    print(f"d(smooth BZ observable)/d(scale) at 1.0 = {float(g):.6e}")

    # ---- record provenance ---------------------------------------------------
    record = {
        "input_dataset": str(fix),
        "mesh": [48, 48, 1],
        "temperature_K": 0.0,
        "chemical_potential_eV": 0.0,
        "broadening_eV": None,
        "dtype": "float64/complex128",
        "backend_device": json.dumps(wx.config.diagnostics()["devices"]),
        "wannierx_version": wx.__version__,
        "ahc_sheet_S": float(s[2]),
        "ahc_in_e2_over_h": sxy_e2h,
    }
    (fix / "demo_record.json").write_text(json.dumps(record, indent=2))
    print(f"record -> {fix/'demo_record.json'}")
    print("=== demo complete ===")


if __name__ == "__main__":
    # Enable float64/complex128 reference precision for this process.
    jax.config.update("jax_enable_x64", True)
    main()
