# Noctua2 deployment — Shastry-Sutherland iPEPS

## Files
- `run_sweep.py` — Python driver. One SLURM array task = one `Jp` (index into
  `JP_VALUES`); loops `D = --dmin..--dmax`, runs a staged imaginary-time
  evolution, **CUDA-synced** timing, `measure()`, and logs to `out/`.
  `J = 1` and **`J4 = 0`** are fixed and asserted (canonical Shastry-Sutherland).
- `sweep.sbatch` — SLURM array job. Placeholders marked `TODO` / `FILL FROM YOUR
  OLD PC2 SCRIPTS` (`-A` account, `-p` partition, `-q` QOS, `module load`, venv).
- `out/` — CSV (`bench_*.csv`), JSONL (`bench_*.jsonl`), and SLURM logs.

## Before submitting — fill placeholders in `sweep.sbatch`
1. `-A <account>` (`hpc-prf-…`), `-p <gpu partition>`, `-q <qos>` — copy from an
   old GPU script (check `sinfo -s` / `scontrol show partition`).
2. `module load …` (CUDA matching your torch wheel + Python) and the
   `source …/activate` line for your env.
3. `--cpus-per-task`, `--mem-per-cpu`, `--time`.
   GPUs: `--gres=gpu:a100:COUNT` (COUNT 1–4). This job is single-process → **1**.

## Test run (validate GPU acceleration)
As shipped: `--array=0-0`, `D=3..6`.
```bash
sbatch driver/sweep.sbatch
```
Check `out/slurm-*.out`: `nvidia-smi` shows the A100, the banner prints
`cuda_avail=True`, and each `[D=..]` line reports `dev=cuda`, `ms/step`, and
`peakGPU`. If `dev=cpu` appears, the GPU path is **not** active (wrong module /
CPU-only torch).

Local smoke test (no GPU needed, tiny schedule):
```bash
python driver/run_sweep.py --device cpu --dmin 3 --dmax 3 --quick --outdir /tmp/ssbench
```

## Going to production
- Widen `JP_VALUES` in `run_sweep.py` (e.g. `np.round(np.arange(0.80,2.01,0.05),3)`).
- `sweep.sbatch`: `--array=0-<len(JP_VALUES)-1>` and change `--dmax 6` → `--dmax 10`.
- Size `--time` from the measured D-scaling (below); each task runs D=3..10 serially.

## Suggested benchmarks BEFORE the big run
Run these from the test allocation and read `out/bench_*.csv`:

1. **CPU vs GPU** at fixed `D`: run once `--device cuda` and once `--device cpu`
   (`--dmin 5 --dmax 5`). The `ms/step` ratio is your speedup — confirm it's large
   (else the GPU isn't helping and prod on GPU is wasteful).
2. **D-scaling → walltime/memory extrapolation**: from `D=3..6` `ms/step` and
   `peakGPU`, fit the growth and predict `D=7..10`. Decide whether `D=10` fits the
   A100 (40 vs 80 GB) and how long a full D=3..10 task takes → set `--time`.
3. **chi-scaling**: fix `D`, vary `--chi-factor 1 2 3 4` (CTMRG cost ~ χ³). Pick the
   smallest χ where `energy`/observables are converged — this dominates prod cost.
4. **float32 vs float64**: try `dtype="float32"` (edit `run_sweep.py` config) — ~2×
   faster and half the memory on A100; verify `energy` is unchanged to the precision
   you need before trusting it for prod.
5. **Physics sanity / convergence**: at a deep-dimer `Jp` (e.g. 3.0) confirm
   `energy/site → -3/8·Jp` and `dimer_ss → -0.75`; and check `energy` is flat vs
   number of steps so the schedule is actually converged (not under/over-evolved).
6. **Determinism**: run the same `(D, Jp)` twice → identical `energy`. Rules out
   nondeterministic GPU kernels muddying comparisons.
7. **Multi-GPU (optional)**: only worthwhile if acetn's distributed path speeds up a
   single `(D,Jp)`. Otherwise keep 1 GPU/task and parallelise across `Jp` via the
   array (usually the better throughput). Test 1 vs 2 vs 4 GPUs before committing.

## Notes
- The first CUDA calls include cuBLAS/kernel init; the driver's first schedule stage
  is a warmup and `steady_s_per_step` is taken from the **last** stage.
- Re-running an array index appends to the same `bench_<jp>_<idx>.csv` (by design).
