# SDSC Atlantis

Access follows the [Atlantis guide](https://wiki.scc.sdsc.edu/s/guide). The verified username is i1agrawal, Slurm account tritondroids, and GPU resource rtx2080ti:1. The live scheduler lists eight RTX 2080 Ti cards on zixian. Each process uses one card; arrays run independent seeds.

## Setup

```bash
ssh i1agrawal@login.scc.sdsc.edu
git clone https://github.com/2q46/humanoid-locomotion-newton.git
cd humanoid-locomotion-newton
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/g1-amp-prepare
.venv/bin/wandb login
```

Authenticate on the cluster before submitting training. Keep credentials out of source and chat. The launch script requires W&B online mode; authentication failure stops collection. Set WANDB_ENTITY to target another team.

Keep the environment, assets, motions, caches, and checkpoints under /home/i1agrawal or /projects/tritondroids, which are shared across nodes. G1_AMP_WORKDIR can select shared project storage. A compute-node probe on 2026-10-18 reported RTX 2080 Ti, 11 GiB, driver 610.43.02, compatible with the pinned Warp wheel.

## Validate, then train

Submit from the repository root:

```bash
sbatch cluster/validate.sbatch
sbatch cluster/train.sbatch
squeue -u i1agrawal
```

Override the account using sbatch -A ACCOUNT. For independent seeds use an array, for example sbatch --array=0-3%2 cluster/train.sbatch, which allows two simultaneous jobs. Adjust exported NUM_ENVS and ITERATIONS. Start with one GPU and measure throughput before scaling.

W&B receives metrics and versioned checkpoint artifacts. Outputs persist wandb_run.json for resuming the same W&B run. Slurm sends USR1 two minutes before timeout; the learner finishes its update and saves. Requeue restarts episodes and replay while restoring the learner. Iterations are additional updates, including after restart. A hard node failure recovers the last periodic checkpoint.

Periodic W&B videos are enabled: six seconds at startup, every 500 updates, and at finish. Override exported VIDEO_EVERY and VIDEO_STEPS when submitting. Captions identify the checkpoint, command, seed, and reset behavior; eval/falls and eval/task_reward accompany each clip. Headless NVIDIA EGL rendering must work in the allocation.
