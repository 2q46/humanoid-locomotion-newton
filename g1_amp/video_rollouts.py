"""Record distinct evaluation rollouts using one batched Newton simulation."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import flax.serialization
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import newton
import numpy as np
import warp as wp
import xml.etree.ElementTree as ET
from .env import EnvConfig, G1Env
from .motion import JOINT_NAMES, MotionLibrary
from .prepare import MODEL_DIR, MOTION_DIR
from .train import make_networks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=MOTION_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, choices=(10, 25, 50), default=25)
    parser.add_argument("--velocity", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.steps) < 1:
        parser.error("num-envs and steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    cfg = EnvConfig(**{**config["env"], "num_envs": args.num_envs,
                       "device": args.device, "seed": args.seed, "reference_init": True})
    motions = MotionLibrary(args.motions)
    env = G1Env(cfg, motions)
    saved = flax.serialization.msgpack_restore(args.checkpoint.read_bytes())
    if not np.allclose(saved["disc_mean"], motions.mean) or not np.allclose(saved["disc_std"], motions.std):
        raise ValueError("Checkpoint motion statistics do not match")
    net = make_networks(env.observation_size)
    policy = jax.jit(lambda obs: net.parametric_action_distribution.mode(
        net.policy_network.apply(None, saved["params"]["policy"], obs)))
    seeds = args.seed + np.arange(args.num_envs)
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    standing = np.arange(args.num_envs) % 2 == 0

    def reset(mask):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return
        for i in ids:
            if standing[i]:
                q, qd = np.zeros(36, np.float32), np.zeros(35, np.float32)
                q[2], q[6] = 0.78, 1
                q[7:] = np.clip(env.nominal + rngs[i].normal(0, 0.02, 29), env.lower, env.upper)
            else:
                qs, vs = motions.sample_states(rngs[i], 1)
                q, qd = qs[0], vs[0]
                q[:2], q[2] = 0, q[2] + 0.02
            env.q[i, :7], env.q[i, env.q_indices] = q[:7], q[7:]
            env.qd[i, :6], env.qd[i, env.v_indices] = qd[:6], qd[6:]
        env.age[ids], env.returns[ids], env.last_action[ids] = 0, 0, 0
        with wp.ScopedDevice(args.device):
            env.state.joint_q.assign(env.q.reshape(-1))
            env.state.joint_qd.assign(env.qd.reshape(-1))
            newton.eval_fk(env.model, env.state.joint_q, env.state.joint_qd, env.state)
            if args.device != "cpu":
                warm = env.solver.mjw_data.qacc_warmstart
                values = warm.numpy()
                values[ids] = 0
                warm.assign(values)
            else:
                env.solver.mj_data.qacc_warmstart[:] = 0

    reset(np.ones(args.num_envs, bool))
    rewards = np.zeros(args.num_envs)
    falls = np.zeros(args.num_envs, int)
    lengths = [[] for _ in range(args.num_envs)]
    poses = []
    for step in range(args.steps):
        env.commands[:] = [args.velocity, 0, 0]
        action = np.asarray(policy(jnp.asarray(env.observe())))
        _, reward, done, metric = env.step(action)
        rewards += reward
        falls += metric["fallen"].astype(int)
        if step % (50 // args.fps) == 0:
            poses.append(env._canonical()[0].copy())
        for i in np.flatnonzero(done):
            lengths[i].append(int(metric["episode_length"][i]))
        reset(done)
    poses = np.asarray(poses)
    # Render each clip sequentially to avoid starting 32 encoders at once.
    root = ET.parse(MODEL_DIR / "g1_mjx.xml").getroot()
    root.find("compiler").set("meshdir", str(MODEL_DIR / "assets"))
    ET.SubElement(root.find("worldbody"), "geom", type="plane", size="200 200 0.1", rgba="0.3 0.35 0.4 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -15
    renderer = mujoco.Renderer(model, height=480, width=640)
    rollouts = []
    try:
        for i in range(args.num_envs):
            filename = f"rollout-{i:02d}.mp4"
            with imageio.get_writer(str(args.output_dir / filename), fps=args.fps,
                                   ffmpeg_params=["-threads", "1"]) as writer:
                for q in poses[:, i]:
                    data.qpos[:3], data.qpos[3:7] = q[:3], q[[6, 3, 4, 5]]
                    data.qpos[model.jnt_qposadr[joints]] = q[7:]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = q[:3]
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
            rollout = {"index": i, "seed": int(seeds[i]), "command_velocity": args.velocity,
                       "initialization": "standing + joint noise" if standing[i] else "reference",
                       "joint_noise_std": 0.02 if standing[i] else 0.0,
                       "mean_task_reward": float(rewards[i] / args.steps), "falls": int(falls[i]),
                       "completed_episode_lengths": lengths[i], "video": filename}
            rollouts.append(rollout)
            print(f"Rendered {i + 1}/{args.num_envs}: {filename}", flush=True)
    finally:
        renderer.close()
    report = {"iteration": int(saved["iteration"]), "video_count": args.num_envs,
              "steps": args.steps, "fps": args.fps, "rollouts": rollouts}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.num_envs} evaluation videos", flush=True)


if __name__ == "__main__":
    main()
