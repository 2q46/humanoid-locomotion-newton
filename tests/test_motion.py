import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from g1_amp.motion import features, MotionLibrary, JOINT_NAMES
from g1_amp.amp import Replay


class MotionTests(unittest.TestCase):
    def test_heading_translation_invariance(self):
        rng = np.random.default_rng(12)
        q = rng.normal(size=(4, 36))
        q[:, 3:7] = Rotation.random(4, random_state=rng).as_quat()
        qd = rng.normal(size=(4, 35))
        feet = rng.normal(size=(4, 2, 3))
        expected = features(q, qd, feet)
        yaw = Rotation.from_euler("z", 1.2)
        q2, v2 = q.copy(), qd.copy()
        offset = np.array([12, -3, 0])
        q2[:, :3] = yaw.apply(q[:, :3]) + offset
        q2[:, 3:7] = (yaw * Rotation.from_quat(q[:, 3:7])).as_quat()
        v2[:, :3], v2[:, 3:6] = yaw.apply(qd[:, :3]), yaw.apply(qd[:, 3:6])
        f2 = yaw.apply(feet.reshape(-1, 3)).reshape(4, 2, 3) + offset
        np.testing.assert_allclose(features(q2, v2, f2), expected, atol=1e-6)

    def test_clip_boundaries_and_reset_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            for c in range(2):
                q, v, f = np.zeros((5, 36)), np.zeros((5, 35)), np.zeros((5, 74))
                q[:, 2], q[:, 6], f[:, 3] = 0.75, 1, -1
                f[:, 0] = np.arange(5) + 100 * c
                np.savez(Path(tmp) / f"clip{c}.npz", q=q, qd=v, features=f,
                         joint_names=JOINT_NAMES, dt=0.02)
            m = MotionLibrary(tmp)
            self.assertEqual(len(m.pairs), 8)
            np.testing.assert_array_equal(m.pairs[:, 74] - m.pairs[:, 0], 1)
            q, qd = m.sample_states(np.random.default_rng(0), 50)
            self.assertEqual(q.shape, (50, 36))
            self.assertTrue((q[:, 2] > 0.6).all())

    def test_replay_wrap(self):
        r = Replay(capacity=5, feature_dim=2)
        r.add(np.arange(16, dtype=np.float32).reshape(8, 2))
        self.assertEqual(r.size, 5)
        self.assertEqual(set(r.data[:, 0].tolist()), {6., 8., 10., 12., 14.})
        r.add(np.array([[20, 21]], np.float32))
        self.assertIn(20, r.sample(np.random.default_rng(0), 100)[:, 0])


if __name__ == "__main__":
    unittest.main()
