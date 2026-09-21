#!/usr/bin/env python
"""Benchmark / parameter-sweep driver for the Shastry-Sutherland iPEPS (PC2 Noctua2).

One SLURM array task = one value of Jp (selected from JP_VALUES by the array index).
Within a task we loop over bond dimension D and, for each D, run a staged
imaginary-time evolution, time it (with CUDA synchronization so GPU timings are
real), measure observables, and log timing + physics so we can confirm the GPU
path is actually being exercised.

Fixed conventions (do NOT change without intent):
  * J  = 1.0   (AFM nearest-neighbour, hard-coded)
  * J4 = 0.0   (parasitic coupling OFF -- canonical Shastry-Sutherland).
                Passed explicitly everywhere to guard against regressions.

TEST run  : --array=0:0 (single Jp), D = 3..6, to validate GPU acceleration.
PROD run  : widen JP_VALUES + --array range, and use --dmax 10.
"""
import os
import sys
import csv
import json
import time
import argparse
import logging
import platform

import torch

# import the model that lives in the repo root (one level up from ./driver)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolve_gs_shastry import ShastrySutherlandModel          # noqa: E402
from acetn.ipeps import Ipeps                                  # noqa: E402


# --- sweep grid: array index -> Jp -------------------------------------------
# J = 1 fixed; Jp = J'/J is the diagonal dimer coupling being swept.
# Corboz-Mila boundaries sit near J/J' ~ 0.675 and 0.765  <=>  Jp ~ 1.48, 1.31.
JP_VALUES = [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

# staged imaginary-time schedule (dtau, steps). First stage also warms up CUDA
# (cuBLAS / kernel init); steady-state ms/step is taken from the LAST stage.
DEFAULT_STAGES = [(0.1, 20), (0.02, 50)]
QUICK_STAGES = [(0.1, 3), (0.05, 3)]          # --quick: fast local smoke test

CSV_FIELDS = ["D", "chi", "Jp", "J4", "device", "gpu",
              "energy", "plaq", "dimer_ss",
              "n_steps", "evolve_time_s", "steady_s_per_step", "peak_gpu_mem_MB"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_one(D, chi, jp, j4, device_str, stages, nx, ny):
    """Build an iPEPS at bond dimension D, evolve+measure, return (row, stages, meas)."""
    assert j4 == 0.0, "J4 must be 0.0 for the canonical Shastry-Sutherland sweep"
    config = {
        "dtype": "float32",
        "device": device_str,
        "TN": {"dims": {"phys": 16, "bond": D, "chi": chi}, "nx": nx, "ny": ny},
    }
    params = {"J": 1.0, "Jp": jp, "J4": j4}

    ipeps = Ipeps(config)
    device = torch.device(ipeps.device)        # actual device (acetn falls back to cpu)
    ipeps.set_model(ShastrySutherlandModel, params)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stage_times = []
    total_steps = 0
    _sync(device)
    t0 = time.perf_counter()
    for dtau, steps in stages:
        _sync(device)
        ts = time.perf_counter()
        ipeps.evolve(dtau=dtau, steps=steps)
        _sync(device)
        dt = time.perf_counter() - ts
        stage_times.append({"dtau": dtau, "steps": steps,
                            "time_s": dt, "s_per_step": dt / steps})
        total_steps += steps
    evolve_time = time.perf_counter() - t0

    meas = ipeps.measure()

    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device.type == "cuda" else float("nan"))
    row = {
        "D": D, "chi": chi, "Jp": jp, "J4": j4,
        "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "energy": float(meas.get("Energy", float("nan"))),
        "plaq": float(meas.get("plaq", float("nan"))),
        "dimer_ss": float(meas.get("dimer_ss", float("nan"))),
        "n_steps": total_steps,
        "evolve_time_s": evolve_time,
        "steady_s_per_step": stage_times[-1]["s_per_step"],   # warmup excluded
        "peak_gpu_mem_MB": peak_mem_mb,
    }
    return row, stage_times, meas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array-index", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                    help="index into JP_VALUES (default: $SLURM_ARRAY_TASK_ID or 0)")
    ap.add_argument("--dmin", type=int, default=3)
    ap.add_argument("--dmax", type=int, default=6, help="TEST: 6 ; PROD: 10")
    ap.add_argument("--chi-factor", type=int, default=2, help="chi = factor * D^2")
    ap.add_argument("--chi", type=int, default=None, help="override: fixed chi for all D")
    ap.add_argument("--j4", type=float, default=0.0, help="FIXED 0.0 (explicit)")
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="driver/out")
    ap.add_argument("--quick", action="store_true", help="tiny schedule for smoke tests")
    ap.add_argument("--verbose", action="store_true", help="acetn per-iteration timing logs")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    assert args.j4 == 0.0, "J4 must be 0.0 for the canonical Shastry-Sutherland sweep"
    stages = QUICK_STAGES if args.quick else DEFAULT_STAGES
    jp = JP_VALUES[args.array_index]

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"jp{jp:.4f}_idx{args.array_index}"
    csv_path = os.path.join(args.outdir, f"bench_{tag}.csv")
    json_path = os.path.join(args.outdir, f"bench_{tag}.jsonl")

    print("=" * 74)
    print(f"host={platform.node()}  torch={torch.__version__}  "
          f"cuda_avail={torch.cuda.is_available()}  cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    print(f"Jp={jp}  J4={args.j4}  D={args.dmin}..{args.dmax}  "
          f"chi_factor={args.chi_factor}  device={args.device}  quick={args.quick}")
    print("=" * 74, flush=True)

    write_header = not os.path.exists(csv_path)
    for D in range(args.dmin, args.dmax + 1):
        chi = args.chi if args.chi else args.chi_factor * D * D
        row, stage_times, meas = run_one(D, chi, jp, args.j4, args.device,
                                         stages, args.nx, args.ny)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)
        with open(json_path, "a") as f:
            f.write(json.dumps({"row": row, "stages": stage_times,
                                "measurements": {k: float(v) for k, v in meas.items()}}) + "\n")
        print(f"[D={D:2d} chi={chi:4d}] dev={row['device']} "
              f"E={row['energy']:.6f} plaq={row['plaq']:+.4f} "
              f"dimer_ss={row['dimer_ss']:+.4f} | "
              f"{row['steady_s_per_step']*1e3:8.1f} ms/step  "
              f"evolve={row['evolve_time_s']:7.1f}s  "
              f"peakGPU={row['peak_gpu_mem_MB']:.0f}MB", flush=True)

    print(f"\nwrote {csv_path}\n      {json_path}")


if __name__ == "__main__":
    main()
