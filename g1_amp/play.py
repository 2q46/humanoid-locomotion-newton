"""Evaluate a saved PPO policy in Newton; optionally view or record video."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from .env import EnvConfig, G1Env
from .motion import MotionLibrary, JOINT_NAMES
from .prepare import MOTION_DIR, MODEL_DIR
from .train import make_networks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motions", type=Path, default=MOTION_DIR)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--reference-init", action="store_true", help="Start from a reference state instead of standing")
    p.add_argument("--velocity", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    cfg = EnvConfig(**{**config["env"], "num_envs": 1, "device": args.device,
                       "seed": args.seed, "reference_init": args.reference_init})
    motions = MotionLibrary(args.motions)
    env = G1Env(cfg, motions)
    saved = flax.serialization.msgpack_restore(args.checkpoint.read_bytes())
    if not np.allclose(saved["disc_mean"], motions.mean) or not np.allclose(saved["disc_std"], motions.std):
        raise ValueError("Checkpoint was trained with a different reference motion library")
    net = make_networks(env.observation_size)
    policy = jax.jit(lambda obs: net.parametric_action_distribution.mode(
        net.policy_network.apply(None, saved["params"]["policy"], obs)))
    viewer = None
    if args.viewer:
        import newton.viewer
        viewer = newton.viewer.ViewerGL()
        viewer.set_model(env.model)
    writer, renderer = None, None
    if args.video:
        import imageio.v2 as imageio
        import mujoco
        import xml.etree.ElementTree as ET
        root = ET.parse(MODEL_DIR / "g1_mjx.xml").getroot()
        root.find("compiler").set("meshdir", str(MODEL_DIR / "assets"))
        ET.SubElement(root.find("worldbody"), "geom", type="plane", size="200 200 0.1", rgba="0.3 0.35 0.4 1")
        render_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        render_data = mujoco.MjData(render_model)
        render_joint_ids = [mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        renderer = mujoco.Renderer(render_model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -15
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.video), fps=50)
    poses, rewards, fall_count, lengths = [], [], 0, []
    try:
        for step in range(args.steps):
            if viewer and not viewer.is_running():
                break
            env.commands[:] = [args.velocity, 0, 0]
            obs = env.observe()
            action = np.asarray(policy(jnp.asarray(obs)))
            _, reward, done, metric = env.step(action)
            q, _ = env._canonical()
            poses.append(q[0].copy()); rewards.append(float(reward[0]))
            if viewer:
                viewer.begin_frame(step * 0.02)
                viewer.log_state(env.state)
                viewer.end_frame()
            if writer:
                render_data.qpos[:3] = q[0, :3]
                render_data.qpos[3:7] = q[0, [6, 3, 4, 5]]
                render_data.qpos[render_model.jnt_qposadr[render_joint_ids]] = q[0, 7:]
                mujoco.mj_forward(render_model, render_data)
                camera.lookat[:] = q[0, :3]
                renderer.update_scene(render_data, camera=camera)
                writer.append_data(renderer.render())
            if done[0]:
                fall_count += int(metric["fallen"][0])
                lengths.append(int(metric["episode_length"][0]))
                env.reset(done)
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        if viewer: viewer.close()
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trajectory, q=np.asarray(poses), dt=np.float32(0.02),
                            joint_names=np.asarray(JOINT_NAMES))
    print(json.dumps({"steps": len(rewards), "mean_task_reward": float(np.mean(rewards)),
                      "falls": fall_count, "completed_episode_lengths": lengths}, indent=2))


if __name__ == "__main__":
    main()
