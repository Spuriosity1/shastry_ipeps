#!/bin/bash
# Shastry-Sutherland iPEPS bond-dimension X Jprime sweep
#
# TEST (this file as-is): 1 array task, D = 3..6, to confirm GPU acceleration.
# PROD: see the marked lines below (widen --array, --dmax 10, longer --time).
#
#SBATCH --job-name=ss_sweep
#SBATCH --array=0-10                       # TEST: single Jp (index 0). PROD: 0-<len(JP_VALUES)-1>
#SBATCH --nodes=1
#SBATCH --ntasks=1                         # acetn is a single process (1 GPU); array-parallelise over Jp
#SBATCH --cpus-per-gpu=1                 # TODO adjust
#SBATCH --gres=gpu:1                  # a100 count 1..4 (single-process job -> 1)
#SBATCH --mem-per-cpu=4G                   # TODO adjust
#SBATCH --time=2:00:00                   
#SBATCH --output=driver/track/slurm-%A_%a.out
#SBATCH --error=driver/track/slurm-%A_%a.err
#SBATCH -p gpu
#SBATCH --chdir=/scratch/alaric/ipeps/shastry_ipeps

set -euo pipefail

# ----- module / environment: FILL FROM YOUR OLD PC2 SCRIPTS ------------------------------------

source /home/alaric/py/bin/activate
module load cuda/12.5.0

out_dir="out"

mkdir -p $out_dir

echo "host=$(hostname)  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true    # sanity: confirm a GPU is actually visible to the job

# One task -> one Jp (by array index); the driver loops D internally.
# PROD: change --dmax 6 -> --dmax 10.
torchrun --nproc_per_node=$SLURM_NTASKS driver/run_sweep.py \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --dmin 3 --dmax 6 \
    --chi-factor 2 \
    --j4 0.0 \
    --device cuda \
    --outdir $out_dir
