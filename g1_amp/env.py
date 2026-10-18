"""Vectorized Newton G1 locomotion with native MuJoCo-Warp contacts."""
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import newton
from .motion import JOINT_NAMES, FEET, DT, features
from .prepare import MODEL_DIR, ROOT


@dataclass
class EnvConfig:
    num_envs: int = 64
    device: str = "cuda:0"
    substeps: int = 4
    episode_length: int = 1000
    action_scale: float = 1.0
    seed: int = 0
    reference_init: bool = True
    cuda_graph: bool = True


class G1Env:
    action_size = 29

    def __init__(self, cfg, motions):
        if cfg.num_envs < 1 or cfg.substeps < 1 or cfg.episode_length < 1:
            raise ValueError("Environment count, substeps, and episode length must be positive")
        self.cfg, self.motions = cfg, motions
        self.graph = None
        self.rng = np.random.default_rng(cfg.seed)
        os.environ.setdefault("NEWTON_CACHE_PATH", str(ROOT / ".cache/newton"))
        wp.config.kernel_cache_dir = str(ROOT / ".cache/warp")
        wp.init()
        with wp.ScopedDevice(cfg.device):
            robot = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
            model_path = MODEL_DIR / "g1_newton.xml"
            if not model_path.exists():
                raise FileNotFoundError("Run python -m g1_amp.prepare first")
            robot.add_mjcf(str(model_path), enable_self_collisions=False,
                           collapse_fixed_joints=True, parse_sites=False)
            labels = [n.rsplit("/", 1)[-1] for n in robot.joint_label]
            self.q_indices = np.asarray([robot.joint_q_start[labels.index(n)] for n in JOINT_NAMES])
            self.v_indices = np.asarray([robot.joint_qd_start[labels.index(n)] for n in JOINT_NAMES])
            if robot.joint_coord_count != 36 or robot.joint_dof_count != 35:
                raise ValueError("Expected floating-base 29-DOF G1")
            self.nq, self.nv = robot.joint_coord_count, robot.joint_dof_count
            # Free root has world pose and parent-frame (world) COM linear/angular velocity.
            free = next(i for i, t in enumerate(robot.joint_type) if t == newton.JointType.FREE)
            if free != 0 or not np.allclose(np.asarray(robot.joint_X_p[free])[:3], 0):
                raise ValueError("Expected a world-anchored free root")
            reference_joints = np.concatenate([clip[0][:, 7:] for clip in motions.clips])
            self.nominal = np.median(reference_joints, axis=0).astype(np.float32)
            lo, hi = np.quantile(reference_joints, [0.01, 0.99], axis=0)
            self.action_range = np.maximum(0.5, np.maximum(self.nominal - lo, hi - self.nominal)).astype(np.float32)
            for i, vi in enumerate(self.v_indices):
                name = JOINT_NAMES[i]
                # Moderate impedance; imported model supplies per-joint effort limits.
                robot.joint_target_ke[vi] = 80.0 if ("hip" in name or "knee" in name or "waist" in name) else 30.0
                robot.joint_target_kd[vi] = 2.0
                robot.joint_target_mode[vi] = int(newton.JointTargetMode.POSITION)
            self.lower = np.asarray(robot.joint_limit_lower)[self.v_indices]
            self.upper = np.asarray(robot.joint_limit_upper)[self.v_indices]
            builder = newton.ModelBuilder()
            builder.replicate(robot, cfg.num_envs)
            builder.add_ground_plane()
            self.model = builder.finalize()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=cfg.device == "cpu", use_mujoco_contacts=True,
                solver="newton", integrator="implicitfast", iterations=50, ls_iterations=50,
                njmax=128, nconmax=64, update_data_interval=1)
            self.state = self.model.state()
            self.spare = self.model.state()
            self.control = self.model.control()
            self.contacts = newton.CollisionPipeline(self.model).contacts()
        body_labels = [n.rsplit("/", 1)[-1] for n in robot.body_label]
        self.nb = robot.body_count
        self.feet_ids = [body_labels.index(n) for n in FEET]
        self.age = np.zeros(cfg.num_envs, np.int32)
        self.last_action = np.zeros((cfg.num_envs, 29), np.float32)
        self.commands = np.zeros((cfg.num_envs, 3), np.float32)
        self.returns = np.zeros(cfg.num_envs, np.float32)
        self.q = self.model.joint_q.numpy().reshape(cfg.num_envs, self.nq).copy()
        self.qd = self.model.joint_qd.numpy().reshape(cfg.num_envs, self.nv).copy()
        self.reset(np.ones(cfg.num_envs, bool))
        self.observation_size = self.observe().shape[-1]

    def _canonical(self):
        q = np.concatenate([self.q[:, :7], self.q[:, self.q_indices]], axis=-1)
        qd = np.concatenate([self.qd[:, :6], self.qd[:, self.v_indices]], axis=-1)
        return q, qd

    def amp_features(self):
        q, qd = self._canonical()
        bq = self.state.body_q.numpy().reshape(self.cfg.num_envs, self.nb, 7)
        return features(q, qd, bq[:, self.feet_ids, :3])

    def observe(self):
        q, qd = self._canonical()
        inv = Rotation.from_quat(q[:, 3:7]).inv()
        gravity = inv.apply(np.tile([0., 0., -1.], (len(q), 1)))
        return np.concatenate([inv.apply(qd[:, :3]), 0.25 * inv.apply(qd[:, 3:6]), gravity,
                               self.commands, q[:, 7:] - self.nominal,
                               0.05 * qd[:, 6:], self.last_action], axis=-1).astype(np.float32)

    def reset(self, mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return self.observe()
        if self.cfg.reference_init:
            q, qd = self.motions.sample_states(self.rng, len(idx))
            # Random reference-state initialization, with horizontal position recentered.
            q[:, :2] = 0
            q[:, 2] += 0.02
        else:
            q = np.zeros((len(idx), 36), np.float32)
            q[:, 2], q[:, 6], q[:, 7:] = 0.78, 1.0, self.nominal
            qd = np.zeros((len(idx), 35), np.float32)
        self.q[idx, :7] = q[:, :7]
        self.q[np.ix_(idx, self.q_indices)] = q[:, 7:]
        self.qd[idx, :6] = qd[:, :6]
        self.qd[np.ix_(idx, self.v_indices)] = qd[:, 6:]
        self.age[idx], self.last_action[idx], self.returns[idx] = 0, 0, 0
        self.commands[idx] = self.rng.uniform([0.4, -0.15, -0.35], [1.2, 0.15, 0.35], (len(idx), 3))
        with wp.ScopedDevice(self.cfg.device):
            self.state.joint_q.assign(self.q.reshape(-1))
            self.state.joint_qd.assign(self.qd.reshape(-1))
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            # Clear the reset worlds' solver warm-start state without altering live worlds.
            if self.cfg.device != "cpu":
                warm = self.solver.mjw_data.qacc_warmstart
                values = warm.numpy()
                values[idx] = 0
                warm.assign(values)
            else:
                self.solver.mj_data.qacc_warmstart[:] = 0
        return self.observe()

    def _simulate(self):
        for _ in range(self.cfg.substeps):
            self.state.clear_forces()
            self.solver.step(self.state, self.spare, self.control, self.contacts, DT / self.cfg.substeps)
            self.state, self.spare = self.spare, self.state

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != self.last_action.shape or not np.isfinite(action).all():
            raise ValueError("Invalid action batch")
        action = np.clip(action, -1, 1)
        coord_targets = self.control.joint_target_q.size == self.cfg.num_envs * self.nq
        targets = np.zeros((self.cfg.num_envs, self.nq if coord_targets else self.nv), np.float32)
        targets[:, self.q_indices if coord_targets else self.v_indices] = np.clip(self.nominal + self.cfg.action_scale * self.action_range * action, self.lower, self.upper)
        with wp.ScopedDevice(self.cfg.device):
            self.control.joint_target_q.assign(targets.reshape(-1))
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._simulate()
                # Warm up kernels before capture. Even substeps preserve state buffer identities.
                if self.cfg.cuda_graph and self.cfg.device != "cpu" and self.cfg.substeps % 2 == 0:
                    with wp.ScopedCapture() as capture:
                        self._simulate()
                    self.graph = capture.graph
        self.q = self.state.joint_q.numpy().reshape(self.cfg.num_envs, self.nq).copy()
        self.qd = self.state.joint_qd.numpy().reshape(self.cfg.num_envs, self.nv).copy()
        if not np.isfinite(self.q).all() or not np.isfinite(self.qd).all():
            raise FloatingPointError("Non-finite Newton state")
        q, qd = self._canonical()
        inv = Rotation.from_quat(q[:, 3:7]).inv()
        v = inv.apply(qd[:, :3])
        w = inv.apply(qd[:, 3:6])
        gravity = inv.apply(np.tile([0., 0., -1.], (len(q), 1)))
        tracking = np.exp(-np.sum((v[:, :2] - self.commands[:, :2]) ** 2, axis=-1) / 0.25)
        turning = np.exp(-(w[:, 2] - self.commands[:, 2]) ** 2 / 0.25)
        smoothness = np.sum((action - self.last_action) ** 2, axis=-1)
        reward = (tracking + 0.5 * turning + 0.2 - 0.002 * smoothness
                  - 0.01 * v[:, 2] ** 2 - 0.01 * np.sum(w[:, :2] ** 2, axis=-1)
                  - 0.00002 * np.sum(qd[:, 6:] ** 2, axis=-1))
        self.age += 1
        fallen = (q[:, 2] < 0.45) | (q[:, 2] > 1.3) | (gravity[:, 2] > -0.45)
        timeout = (self.age >= self.cfg.episode_length) & ~fallen
        done = fallen | timeout
        reward -= fallen.astype(np.float32)
        self.last_action = action.copy()
        self.returns += reward
        metrics = {"tracking": tracking, "turning": turning, "fallen": fallen,
                   "timeout": timeout, "episode_return": self.returns.copy(), "episode_length": self.age.copy()}
        # Terminal observations/features are returned BEFORE resetting; AMP never sees reset jumps.
        return self.observe(), reward.astype(np.float32), done, metrics
