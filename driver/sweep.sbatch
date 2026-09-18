#!/bin/bash
# Shastry-Sutherland iPEPS bond-dimension benchmark / Jp sweep on PC2 Noctua2.
#
# TEST (this file as-is): 1 array task, D = 3..6, to confirm GPU acceleration.
# PROD: see the marked lines below (widen --array, --dmax 10, longer --time).
#
#SBATCH --job-name=ss_sweep
#SBATCH --array=0-0                       # TEST: single Jp (index 0). PROD: 0-<len(JP_VALUES)-1>
#SBATCH --nodes=1
#SBATCH --ntasks=1                         # acetn is a single process (1 GPU); array-parallelise over Jp
#SBATCH --cpus-per-task=16                 # TODO adjust
#SBATCH --gres=gpu:a100:1                  # a100 count 1..4 (single-process job -> 1)
#SBATCH --mem-per-cpu=8G                   # TODO adjust
#SBATCH --time=01:00:00                    # TEST walltime; size PROD from the D-scaling below
#SBATCH --output=driver/out/slurm-%A_%a.out
#SBATCH --error=driver/out/slurm-%A_%a.err
#
# ===== FILL FROM YOUR OLD PC2 SCRIPTS (these ###SBATCH lines are inert until de-commented) =====
###SBATCH -A <YOUR_PROJECT_ACCOUNT>        # e.g. hpc-prf-xxxxx  (required if in >1 project)
###SBATCH -p <GPU_PARTITION>               # e.g. gpu / a100 -- check `sinfo -s`
###SBATCH -q <QOS_IF_REQUIRED>             # some GPU partitions need a QOS
# ================================================================================================

set -euo pipefail

# ----- module / environment: FILL FROM YOUR OLD PC2 SCRIPTS ------------------------------------
# module reset
# module load system/CUDA/12.x            # match the CUDA your torch wheel expects
# module load lang/Python/3.x             #   ... or a toolchain that provides Python
# source <PATH_TO_YOUR_VENV>/bin/activate #   ... or: source <miniforge3>/bin/activate <env>
# ----------------------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"
mkdir -p driver/out

echo "host=$(hostname)  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true    # sanity: confirm a GPU is actually visible to the job

# One task -> one Jp (by array index); the driver loops D internally.
# PROD: change --dmax 6 -> --dmax 10.
srun python driver/run_sweep.py \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --dmin 3 --dmax 6 \
    --chi-factor 2 \
    --j4 0.0 \
    --device cuda \
    --outdir driver/out
