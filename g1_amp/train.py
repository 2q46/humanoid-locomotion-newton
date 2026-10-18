"""Brax PPO updates driven by Newton rollouts, with joint AMP training."""
import argparse
from dataclasses import asdict
from datetime import datetime
import functools
import json
import os
import signal
from pathlib import Path
import time

# Preserve GPU memory for Newton on small GPUs. PPO runs on CPU by default.
# Set JAX_PLATFORMS=cuda with a CUDA JAX installation to move the learner to GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import types
from brax.training.agents.ppo import losses, networks
from .amp import AMP, Replay
from .env import EnvConfig, G1Env
from .motion import MotionLibrary, JOINT_NAMES
from .prepare import ROOT, MOTION_DIR
from .tracking import RunTracker
from .evaluation import record_video


def make_networks(observation_size):
    return networks.make_ppo_networks(
        observation_size, 29, policy_hidden_layer_sizes=(256, 128),
        value_hidden_layer_sizes=(256, 128), activation=jax.nn.elu,
        init_noise_std=0.5, state_dependent_std=False)


def make_sampler(net):
    dist = net.parametric_action_distribution

    @jax.jit
    def sample(params, obs, key):
        logits = net.policy_network.apply(None, params.policy, obs)
        raw = dist.sample_no_postprocessing(logits, key)
        return dist.postprocess(raw), {"raw_action": raw, "log_prob": dist.log_prob(logits, raw),
                                      "distribution_params": logits}
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000, help="Additional rollout/update iterations")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--disc-updates", type=int, default=4)
    parser.add_argument("--disc-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--amp-weight", type=float, default=1.0)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0", help="Newton device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motions", type=Path, default=MOTION_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb-project", default="humanoid-locomotion-newton")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--video-every", type=int, default=500, help="Record at startup, each N updates, and at finish; 0 disables")
    parser.add_argument("--video-steps", type=int, default=300)
    parser.add_argument("--video-fps", type=int, choices=(10, 25, 50), default=25)
    parser.add_argument("--video-velocity", type=float, default=0.7)
    parser.add_argument("--video-seed", type=int, default=123)
    parser.add_argument("--video-timeout", type=int, default=180)
    args = parser.parse_args()
    if args.video_every < 0 or args.video_steps < 1 or args.video_timeout < 1:
        parser.error("video-every must be nonnegative and video steps/timeout positive")
    if args.num_envs % args.minibatches or min(args.iterations, args.horizon, args.epochs,
            args.minibatches, args.disc_updates, args.disc_batch_size, args.checkpoint_every, args.log_every) < 1:
        parser.error("Use positive counts and num-envs divisible by minibatches")
    if args.amp_weight < 0 or args.task_weight < 0 or args.learning_rate <= 0:
        parser.error("Reward weights must be nonnegative and learning rate positive")
    output = args.output or ROOT / "runs" / datetime.now().strftime("g1_amp_%Y%m%d_%H%M%S")
    if any((output / name).exists() for name in ("checkpoint.msgpack", "metrics.jsonl")) and not args.resume:
        raise FileExistsError(f"Run already exists: {output}; choose another output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(num_envs=args.num_envs, device=args.device, seed=args.seed,
                    substeps=args.substeps, episode_length=args.episode_length)
    motions = MotionLibrary(args.motions)
    env = G1Env(cfg, motions)
    net = make_networks(env.observation_size)
    key = jax.random.PRNGKey(args.seed)
    key, kp, kv, kd = jax.random.split(key, 4)
    params = losses.PPONetworkParams(policy=net.policy_network.init(kp), value=net.value_network.init(kv))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = optimizer.init(params)
    amp = AMP(motions, kd)
    replay = Replay()
    rng = np.random.default_rng(args.seed + 1)
    sampler = make_sampler(net)
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=net, entropy_cost=0.005,
                               discounting=0.99, gae_lambda=0.95, clipping_epsilon=0.2,
                               normalize_advantage=True, vf_coefficient=1.0)

    @jax.jit
    def update(params, opt_state, batch, key):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, None, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    def training_state(iteration):
        return {"params": params, "optimizer": opt_state, "disc_params": amp.params,
                "disc_optimizer": amp.opt_state, "key": key, "iteration": iteration,
                "disc_mean": amp.mean, "disc_std": amp.std}

    start = 0
    start_env_steps = 0
    if args.resume:
        restored = flax.serialization.from_bytes(training_state(0), args.resume.read_bytes())
        params, opt_state = restored["params"], restored["optimizer"]
        amp.params, amp.opt_state = restored["disc_params"], restored["disc_optimizer"]
        if not np.allclose(restored["disc_mean"], motions.mean) or not np.allclose(restored["disc_std"], motions.std):
            raise ValueError("Checkpoint reference feature statistics differ from these motions")
        key, start = restored["key"], int(restored["iteration"])
        meta_path = args.resume.parent / "checkpoint.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("iteration") == start:
            start_env_steps = int(meta["env_steps"])
        else:
            previous = json.loads((args.resume.parent / "config.json").read_text())["args"]
            start_env_steps = start * previous["horizon"] * previous["num_envs"]
        print(f"Resuming learner from iteration {start}; starting fresh simulation episodes", flush=True)
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "env": asdict(cfg), "observation_size": env.observation_size,
              "joint_names": list(JOINT_NAMES),
              "motions": [str(p) for p in motions.paths], "brax_version": __import__("brax").__version__,
              "learner_devices": [str(d) for d in jax.devices()]}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tracker = RunTracker(args, output, config)
    obs = env.observe()
    wall = time.monotonic()
    print(f"Newton {args.device}; JAX {jax.devices()}; {len(motions.pairs)} expert transitions; output={output}", flush=True)

    def save(iteration):
        target = output / "checkpoint.msgpack"
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(flax.serialization.to_bytes(training_state(iteration)))
        tmp.replace(target)
        env_steps = start_env_steps + (iteration - start) * args.horizon * args.num_envs
        (output / "checkpoint.json").write_text(json.dumps({"iteration": iteration, "env_steps": env_steps}, indent=2) + "\n")
        tracker.checkpoint(target, iteration, env_steps)

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stopping after this update and saving checkpoint", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    try:
        for iteration in range(start + 1, start + args.iterations + 1):
            records, amp_pairs, tasks, styles, episodes = [], [], [], [], []
            for _ in range(args.horizon):
                before = env.amp_features()
                key, sample_key = jax.random.split(key)
                action, policy_extra = sampler(params, jnp.asarray(obs), sample_key)
                terminal_obs, task_reward, done, metrics = env.step(np.asarray(action))
                pair = np.concatenate([before, env.amp_features()], axis=-1)
                style_reward = amp.reward(pair)
                reward = args.task_weight * task_reward + args.amp_weight * style_reward
                extras = {"policy_extras": policy_extra,
                          "state_extras": {"truncation": jnp.asarray(metrics["timeout"], jnp.float32)}}
                records.append(types.Transition(observation=jnp.asarray(obs), action=action,
                    reward=jnp.asarray(reward), discount=jnp.asarray(~done, jnp.float32),
                    next_observation=jnp.asarray(terminal_obs), extras=extras))
                tasks.append(task_reward); styles.append(style_reward); amp_pairs.append(pair)
                if done.any():
                    episodes.extend(zip(metrics["episode_return"][done].tolist(), metrics["episode_length"][done].tolist()))
                obs = env.reset(done) if done.any() else terminal_obs
            # Keep each time sequence intact for Brax GAE. Shuffle environment sequences only.
            batch = jax.tree_util.tree_map(lambda *xs: jnp.swapaxes(jnp.stack(xs), 0, 1), *records)
            ppo_metrics = []
            for _ in range(args.epochs):
                permutation = rng.permutation(args.num_envs)
                for ids in np.split(permutation, args.minibatches):
                    minibatch = jax.tree_util.tree_map(lambda x: x[ids], batch)
                    key, loss_key = jax.random.split(key)
                    params, opt_state, metric = update(params, opt_state, minibatch, loss_key)
                    ppo_metrics.append({k: float(v) for k, v in metric.items()})
            pairs = np.concatenate(amp_pairs)
            replay.add(pairs)
            disc_metrics = []
            for _ in range(args.disc_updates):
                size = args.disc_batch_size
                current = pairs[rng.integers(len(pairs), size=size // 2)]
                fake = np.concatenate([current, replay.sample(rng, size - len(current))])
                disc_metrics.append(amp.update(motions.sample_pairs(rng, size), fake))
            report = {"iteration": iteration, "env_steps": start_env_steps + (iteration - start) * args.horizon * args.num_envs,
                      "task_reward": float(np.mean(tasks)), "style_reward": float(np.mean(styles)),
                      "completed_episodes": len(episodes),
                      "episode_task_return": float(np.mean([x[0] for x in episodes])) if episodes else None,
                      "episode_length": float(np.mean([x[1] for x in episodes])) if episodes else None,
                      "steps_per_second": (iteration - start) * args.horizon * args.num_envs / (time.monotonic() - wall),
                      **{k: float(np.mean([m[k] for m in ppo_metrics])) for k in ppo_metrics[0]},
                      **{k: float(np.mean([m[k] for m in disc_metrics])) for k in disc_metrics[0]}}
            if not all(np.isfinite(v) for v in report.values() if isinstance(v, (float, int))):
                raise FloatingPointError(f"Non-finite training metrics: {report}")
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(report) + "\n")
            if iteration % args.log_every == 0 or iteration in (start + 1, start + args.iterations) or stop_requested:
                print(json.dumps(report), flush=True)
                tracker.log(report)
            record = args.video_every > 0 and not stop_requested and (
                iteration == start + 1 or iteration % args.video_every == 0 or iteration == start + args.iterations)
            if record or stop_requested or iteration % args.checkpoint_every == 0 or iteration == start + args.iterations:
                save(iteration)
            if record:
                print(f"Recording evaluation video at update {iteration}", flush=True)
                video, evaluation = record_video(output / "checkpoint.msgpack", args.motions, output, iteration,
                    device=args.device, steps=args.video_steps, fps=args.video_fps,
                    velocity=args.video_velocity, seed=args.video_seed, timeout=args.video_timeout)
                tracker.video(video, evaluation, iteration, report["env_steps"], args.video_fps)
                print(f"Logged evaluation video {video}", flush=True)
            if stop_requested:
                break
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish()
    print(f"Saved {output / 'checkpoint.msgpack'}", flush=True)


if __name__ == "__main__":
    main()
