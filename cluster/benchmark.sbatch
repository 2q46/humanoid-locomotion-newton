#!/usr/bin/env bash
#SBATCH --job-name=g1-scaling
#SBATCH --account=tritondroids
#SBATCH --partition=debug
#SBATCH --gres=rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=g1-scaling-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export G1_AMP_WORKDIR="${G1_AMP_WORKDIR:-$SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python_bin="${G1_AMP_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
srun "$python_bin" cluster/benchmark.py "$@"
