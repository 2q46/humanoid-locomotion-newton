#!/usr/bin/env bash
#SBATCH --job-name=g1-amp
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=g1-amp-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export G1_AMP_WORKDIR="${G1_AMP_WORKDIR:-$SLURM_SUBMIT_DIR}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python_bin="${G1_AMP_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
seed="${SLURM_ARRAY_TASK_ID:-0}"
run_dir="$G1_AMP_WORKDIR/runs/slurm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_seed_$seed"
args=(--num-envs "${NUM_ENVS:-64}" --iterations "${ITERATIONS:-10000}"
      --seed "$seed" --output "$run_dir" --wandb-mode online
      --wandb-project "${WANDB_PROJECT:-humanoid-locomotion-newton}"
      --wandb-group "${WANDB_RUN_GROUP:-atlantis}" --checkpoint-every 50
      --video-every "${VIDEO_EVERY:-500}" --video-steps "${VIDEO_STEPS:-300}")
if [[ -f "$run_dir/checkpoint.msgpack" ]]; then
    args+=(--resume "$run_dir/checkpoint.msgpack")
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun "$python_bin" -m g1_amp.train "${args[@]}" "$@"
