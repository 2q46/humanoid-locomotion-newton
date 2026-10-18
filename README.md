# Unitree G1: Newton, Brax PPO, and AMP

A 29-DOF humanoid locomotion example using Newton 1.7 / MuJoCo-Warp physics, Brax PPO optimization, and adversarial motion priors (AMP) from public G1 walking references.

The custom Python rollout driver calls Newton directly and passes on-policy sequences to Brax’s compute_ppo_loss, GAE, networks, and tanh-normal action distribution. It trains the discriminator between rollouts. It does not call brax.training.agents.ppo.train; this makes AMP updates explicit and avoids hiding a stateful simulator inside JAX callbacks.

## Setup

The workspace-local virtual environment inherits Newton, Warp, and CUDA from the existing newton Conda environment and adds learner dependencies. It leaves that Conda environment unchanged.

```bash
conda activate newton
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m g1_amp.prepare
```

For a fresh cluster environment, clone the repository, create/activate a Python 3.11 Conda environment named newton, and run pip install -e . followed by g1-amp-prepare. Use pip install -e ".[viewer]" for the optional desktop viewer. The console commands g1-amp-train, g1-amp-play, g1-amp-validate, and g1-amp-diagnose also work outside the checkout. Downloads and outputs default to the current directory; set G1_AMP_WORKDIR to choose a persistent workspace.

This workspace has already been prepared. The tested stack is Python 3.11, Newton 1.7.0, Warp 1.18.0, MuJoCo 3.15.0, Brax 0.14.2, and JAX 0.10.2 on an RTX 5060 Laptop GPU with 8 GiB VRAM. Newton runs on CUDA; JAX defaults to CPU to leave GPU memory for simulation. CUDA graphs accelerate even-numbered physics substeps. With a CUDA JAX installation, JAX_PLATFORMS=cuda selects a GPU learner.

## Reference motions and assets

Preparation downloads the [MuJoCo Menagerie G1 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) and three clips from the [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

| Clip | Duration | Median planar speed |
| --- | ---: | ---: |
| walk1_subject1 | 261.3 s | 0.44 m/s |
| walk1_subject2 | 261.3 s | 0.84 m/s |
| walk2_subject1 | 238.2 s | 0.50 m/s |

This is about 12.7 minutes of motion. Source revisions are pinned in g1_amp/prepare.py; source URLs, SHA-256 checksums, and conversion statistics are stored in data/lafan_g1/manifest.json.

The CSV contains root XYZ, quaternion XYZW, and 29 joint positions at 30 Hz. Conversion interpolates positions and joints, applies quaternion SLERP at 50 Hz, computes world angular and root-COM linear velocity, and runs feet forward kinematics. MuJoCo performs this offline FK calculation; Newton performs training simulation. Sampling excludes collapsed/low reference states and never constructs transitions across clip boundaries.

The upstream MJX XML disables automatic contacts and expects scene-level explicit pairs. Preparation creates g1_newton.xml with collision geometry enabled and visual geometry excluded, so the robot can contact the ground added by Newton.

The dataset card identifies LAFAN1 motion content as CC BY-NC-ND 4.0 and notes that retargeting accounts for kinematics rather than actuator dynamics. Downloads and converted files remain local and are ignored by Git. Preserve the source attribution and applicable terms when using the motions. The robot asset license is retained in assets/unitree_g1/LICENSE.

## Train

```bash
.venv/bin/python -m g1_amp.train --num-envs 1024 --iterations 10000 --output runs/my_g1
```

At 1,024 environments and a 24-step horizon, 10,000 iterations collect 245.76 million transitions. A cluster benchmark measured approximately 21,500 transitions/second at 1,024 environments versus about 3,700 at 64. This is a starting recipe; gait quality requires evaluation and tuning. Short runs verify the pipeline and do not establish stable walking.

W&B online logging is enabled by default. Authenticate once with wandb login (or set WANDB_API_KEY securely) before training. Use --wandb-project, --wandb-entity, --wandb-name, and --wandb-group to organize runs. Metrics use environment transitions as the chart axis. Checkpoints and configuration are uploaded as versioned model artifacts. The local wandb_run.json preserves the run ID on resume. For isolated tests use --wandb-mode offline; disabled is available for debugging.

Training saves config.json, per-update metrics.jsonl, and an atomic checkpoint.msgpack containing both networks, both optimizers, discriminator feature statistics, learner RNG, and iteration. Ctrl-C or SIGTERM finishes the current update and saves a checkpoint.

```bash
.venv/bin/python -m g1_amp.train --num-envs 1024 --iterations 1000 --resume runs/my_g1/checkpoint.msgpack --output runs/my_g1
```

Iterations are additional updates when resuming. Resume restores the learner and starts fresh simulation episodes and replay; simulation continuation is not bit-exact. Keep the same motions, horizon, environment count, and optimizer settings for comparable learning curves. Use a new output directory for a new run.

Defaults use reference-state initialization, PD position targets, robot joint/actuator limits, velocity commands of 0.4–1.2 m/s with small lateral/yaw commands, fall termination, and 20-second episode limits. Self-collisions are disabled. The task reward combines planar/yaw velocity tracking, survival, smoothness, and vertical/angular velocity costs. Per-joint action ranges cover the reference motion distribution.

AMP uses pairs of 74-dimensional states: root height, projected gravity, local COM linear/angular velocity, joint positions/velocities, and local feet positions. A least-squares discriminator has expert target +1 and policy target -1, an expert gradient penalty, fixed shared normalization, and policy replay. Style reward is max(0, 1 - 0.25 * (D - 1)^2). Discriminator parameters remain fixed during each rollout. Terminal features are captured before resetting, so reset jumps never become AMP examples. Brax handles terminal and timeout masks in its PPO loss.

## Evaluate

Evaluation uses deterministic actions and starts from standing by default. Use --reference-init to evaluate the training initialization distribution separately.

```bash
.venv/bin/python -m g1_amp.play --checkpoint runs/my_g1/checkpoint.msgpack --viewer
.venv/bin/python -m g1_amp.play --checkpoint runs/my_g1/checkpoint.msgpack --steps 1000 --video runs/my_g1/eval.mp4 --trajectory runs/my_g1/eval.npz
```

The viewer uses Newton. Video renders the Newton trajectory through MuJoCo’s offscreen EGL renderer. Evaluation reports falls, completed episode lengths, and mean task reward. Training reports task and discriminator rewards separately.

## Validate

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m g1_amp.validate --steps 100
.venv/bin/python -m g1_amp.diagnose
.venv/bin/python -m g1_amp.train --num-envs 4 --iterations 2 --horizon 8 --epochs 2 --minibatches 2 --disc-updates 2 --disc-batch-size 32 --episode-length 6 --output runs/smoke_test
```

Checks cover heading/translation invariance, clip boundaries, replay wraparound, expert/Newton feature agreement including feet FK, finite CUDA stepping, partial-reset isolation, graph equivalence, physical ground support, and real PPO/discriminator updates through episode timeouts. Initial Warp kernel compilation can take a minute. Caches live under .cache/.

Implementation: env.py is the Newton task, motion.py handles references, amp.py contains the discriminator and replay, train.py drives Brax PPO, and play.py evaluates checkpoints.

## Hardware

The trainer currently uses one simulation GPU per process. A 10 × RTX 2080 Ti cluster is best used for independent seeds or parameter sweeps; it does not automatically distribute one policy across the cards. Benchmark contact-enabled rollouts before extrapolating training time. The default Warp 1.18 wheel requires a Turing-or-newer GPU and R580+ driver; CUDA 12 wheels are available upstream for older driver environments.

Atlantis access, validated GPU resources, W&B setup, and Slurm launch scripts are documented in [cluster/README.md](cluster/README.md).

Training records 32 distinct six-second evaluation videos at startup, every 500 updates, and at the end. W&B's eval/rollouts table contains every clip, seed, initialization, fall count, and reward; eval/video is only a preview. Half start standing with reproducible 0.02-radian joint perturbations, and half use independently sampled reference states. All use the fixed 0.7 m/s command by default. Batched Newton physics is followed by sequential encoding with one encoder at a time. Videos and metrics remain in videos/iteration-N/ under the run directory. Configure --video-envs, --video-every, --video-steps, --video-fps, --video-velocity, and --video-seed. --video-every 0 disables recording. Evaluation does not alter learner state or RNG. A rendering failure saves the checkpoint and leaves an evaluation log before stopping.
