"""Check reference/physics coordinate agreement, finite stepping, and partial resets."""
import argparse
import numpy as np
import warp as wp
import newton
from .env import G1Env, EnvConfig
from .motion import MotionLibrary
from .prepare import MOTION_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    motions = MotionLibrary(MOTION_DIR)
    env = G1Env(EnvConfig(num_envs=2, device=args.device), motions)
    # Place exactly the same reference frame into Newton and compare all 74 features,
    # including FK feet positions and COM velocities, against the MuJoCo converter.
    q, qd, expected = motions.clips[0]
    for frame in (10, len(q) // 2, len(q) - 10):
        env.q[:, :7] = q[frame, :7]
        env.q[:, env.q_indices] = q[frame, 7:]
        env.qd[:, :6] = qd[frame, :6]
        env.qd[:, env.v_indices] = qd[frame, 6:]
        env.state.joint_q.assign(env.q.reshape(-1))
        env.state.joint_qd.assign(env.qd.reshape(-1))
        newton.eval_fk(env.model, env.state.joint_q, env.state.joint_qd, env.state)
        np.testing.assert_allclose(env.amp_features(), np.tile(expected[frame], (2, 1)), atol=2e-4)
    env.reset(np.ones(2, bool))
    terminal_count = 0
    for _ in range(args.steps):
        obs, reward, done, metrics = env.step(np.zeros((2, 29), np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        terminal_count += int(done.sum())
        env.reset(done)
    q_before, qd_before = env.q[1].copy(), env.qd[1].copy()
    age_before = env.age[1]
    env.reset(np.array([True, False]))
    np.testing.assert_array_equal(env.q[1], q_before)
    np.testing.assert_array_equal(env.qd[1], qd_before)
    assert env.age[1] == age_before and env.age[0] == 0
    print(f"PASS: expert/Newton feature agreement, {args.steps} finite GPU steps, partial reset isolation; terminals={terminal_count}", flush=True)


if __name__ == "__main__":
    main()
