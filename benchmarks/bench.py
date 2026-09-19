"""WX-027: reproducible benchmark harness.

Runs CPU/gGPU (whichever JAX resolves), x64, and reports per scenario:
    Nk, NR, Norb, dtype, device, compile_time_s, exec_time_s, peak_mem_bytes.

Scenarios: H(k), dH_dk, eigh, Berry curvature, chunked BZ integration.
Never alters scientific semantics (benchmark-only).

Usage:
    JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python benchmarks/bench.py [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

import wannierx as wx
from wannierx.kpoints.mesh import monkhorst_pack
from wannierx.models.qzhang import qiwuzhang


@dataclass
class BenchResult:
    scenario: str
    Nk: int
    NR: int
    Norb: int
    dtype: str
    device: str
    compile_time_s: float
    exec_time_s: float
    peak_mem_bytes: int | None


def _timed(fn, *args, repeat: int = 5):
    """Compile/repeat timing of a jitted function.

    Returns (compile_s, exec_s, peak_bytes). Peak memory via tracemalloc
    (host-side; device peak is backend-specific and may be None).
    """
    t0 = time.perf_counter()
    jitted = jax.jit(fn)
    out = jitted(*args)  # traces/compiles here
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = jitted(*args)
        jax.block_until_ready(out)
    exec_s = (time.perf_counter() - t0) / repeat
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return compile_s, exec_s, peak


def run(nk_side: int = 32) -> list[BenchResult]:
    model = qiwuzhang(u=-1.0)
    mesh = monkhorst_pack((nk_side, nk_side, 1))
    k = mesh.points
    dev = str(jax.devices()[0])
    dtype = str(k.dtype)
    n_R, n_orb = model.n_R, model.n_orb

    rows: list[BenchResult] = []

    def bench(name, fn):
        cs, es, pk = _timed(fn, model, k)
        rows.append(
            BenchResult(name, mesh.n_k, n_R, n_orb, dtype, dev, cs, es, pk)
        )
        print(f"{name:28s} Nk={mesh.n_k:6d} NR={n_R:3d} Norb={n_orb:2d} "
              f"compile={cs:.3f}s exec={es * 1e3:.3f}ms peak={pk}")

    bench("hamiltonian", lambda m, kk: wx.hamiltonian(m, kk))
    bench("dH_dk", lambda m, kk: wx.dH_dk(m, kk))
    bench("eigh", lambda m, kk: wx.eigh(m, kk).energies)
    bench(
        "berry_curvature",
        lambda m, kk: wx.berry_curvature(m, kk, degeneracy_policy="nan"),
    )
    from wannierx.kpoints.integration import integrate

    bench(
        "bz_integrate_eigh",
        lambda m, kk: jnp.tensordot(
            mesh.weights, jnp.sum(wx.eigh(m, kk).energies, -1), (0, 0)
        ),
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nk", type=int, default=32)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    rows = run(args.nk)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump([asdict(r) for r in rows], fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
