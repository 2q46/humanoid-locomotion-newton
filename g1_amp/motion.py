"""Shared motion features: all quaternions are XYZW, velocities are world/COM."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

JOINT_NAMES = tuple(
    [f"{side}_{name}_joint" for side in ("left", "right")
     for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{name}_joint" for side in ("left", "right")
       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
DT = 0.02


def features(q, qd, feet_world):
    """Heading/translation invariant AMP state, with identical expert/policy scaling."""
    rot = Rotation.from_quat(q[:, 3:7])
    inv = rot.inv()
    gravity = inv.apply(np.tile([0., 0., -1.], (len(q), 1)))
    feet = np.stack([inv.apply(feet_world[:, i] - q[:, :3]) for i in range(2)], axis=1)
    return np.concatenate([q[:, 2:3], gravity, inv.apply(qd[:, :3]),
                           0.25 * inv.apply(qd[:, 3:6]), q[:, 7:],
                           0.05 * qd[:, 6:], feet.reshape(len(q), 6)], axis=-1).astype(np.float32)


def convert_csv(src, dst):
    """LAFAN G1 XYZ/XYZW/29-joint CSV -> 50 Hz motion with FK features."""
    import mujoco
    from .prepare import MODEL_DIR
    raw = np.loadtxt(src, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 4 or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite LAFAN G1 CSV [frames,36]: {src}")
    if np.max(np.abs(np.linalg.norm(raw[:, 3:7], axis=-1) - 1)) > 0.01:
        raise ValueError("Reference root quaternion must be normalized XYZW")
    source_t = np.arange(len(raw)) / 30.0
    target_t = np.arange(0, source_t[-1], DT)
    q = np.empty((len(target_t), 36), np.float64)
    for k in list(range(3)) + list(range(7, 36)):
        q[:, k] = np.interp(target_t, source_t, raw[:, k])
    q[:, 3:7] = Slerp(source_t, Rotation.from_quat(raw[:, 3:7]))(target_t).as_quat()
    rot = Rotation.from_quat(q[:, 3:7])
    omega = (rot[1:] * rot[:-1].inv()).as_rotvec() / DT
    omega = np.concatenate([omega, omega[-1:]])
    mj = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "g1_mjx.xml"))
    md = mujoco.MjData(mj)
    ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    if any(i < 0 for i in ids):
        raise ValueError("G1 model is missing a reference joint")
    pelvis = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    feet_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET]
    qd = np.concatenate([np.gradient(q[:, :3], DT, axis=0) + np.cross(omega, rot.apply(np.tile(mj.body_ipos[pelvis], (len(q), 1)))),
                         omega, np.gradient(q[:, 7:], DT, axis=0)], axis=-1)
    feet = np.empty((len(q), 2, 3))
    for frame in range(len(q)):
        md.qpos[:3] = q[frame, :3]
        md.qpos[3:7] = q[frame, [6, 3, 4, 5]]  # MuJoCo uses WXYZ
        md.qpos[mj.jnt_qposadr[ids]] = q[frame, 7:]
        mujoco.mj_forward(mj, md)
        feet[frame] = md.xpos[feet_ids]
    f = features(q, qd, feet)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, q=q.astype(np.float32), qd=qd.astype(np.float32), features=f,
                        joint_names=np.asarray(JOINT_NAMES), dt=np.float32(DT))
    speed = np.linalg.norm(qd[:, :2], axis=-1)
    report = {"frames": len(q), "seconds": float(target_t[-1]),
              "height_range": [float(q[:, 2].min()), float(q[:, 2].max())],
              "median_speed": float(np.median(speed)), "feature_dim": f.shape[-1]}
    print(f"Converted {src}: {report}", flush=True)
    return report


class MotionLibrary:
    def __init__(self, directory):
        self.clips = []
        self.reset_frames = []
        self.paths = sorted(Path(directory).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No reference NPZ motions in {directory}; run python -m g1_amp.prepare")
        pairs = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as d:
                q, qd, f = d["q"].copy(), d["qd"].copy(), d["features"].copy()
                if tuple(d["joint_names"].tolist()) != JOINT_NAMES or not np.isclose(d["dt"], DT):
                    raise ValueError(f"Joint order or sample period mismatch: {path}")
            if q.shape != (len(q), 36) or qd.shape != (len(q), 35) or f.shape != (len(q), 74):
                raise ValueError(f"Invalid motion array shape: {path}")
            if len(q) < 3 or not all(np.isfinite(x).all() for x in (q, qd, f)):
                raise ValueError(f"Invalid reference motion: {path}")
            # Do not make artificial transitions between clips or wrap the last frame.
            valid = (q[:, 2] > 0.60) & (q[:, 2] < 1.1) & (f[:, 3] < -0.8)
            valid_pairs = valid[:-1] & valid[1:]
            if not valid_pairs.any():
                raise ValueError(f"No upright walking transitions in {path}")
            pairs.append(np.concatenate([f[:-1], f[1:]], axis=-1)[valid_pairs])
            self.clips.append((q, qd, f))
            self.reset_frames.append(np.flatnonzero(valid[1:-1]) + 1)
        self.pairs = np.concatenate(pairs).astype(np.float32)
        self.mean = self.pairs.mean(axis=0)
        self.std = np.maximum(self.pairs.std(axis=0), 0.1)

    def sample_pairs(self, rng, size):
        return self.pairs[rng.integers(len(self.pairs), size=size)]

    def sample_states(self, rng, size):
        # Uniform clips, then uniform frames; initialization excludes differentiation edges.
        qs, vs = [], []
        for c in rng.integers(len(self.clips), size=size):
            q, qd, _ = self.clips[c]
            frame = rng.choice(self.reset_frames[c])
            qs.append(q[frame]); vs.append(qd[frame])
        return np.asarray(qs).copy(), np.asarray(vs).copy()
