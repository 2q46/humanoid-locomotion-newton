"""Small deterministic check of CUDA graph equivalence and standing dynamics."""
import numpy as np
from .env import G1Env, EnvConfig
from .motion import MotionLibrary
from .prepare import MOTION_DIR


def main():
    motions = MotionLibrary(MOTION_DIR)
    a = G1Env(EnvConfig(num_envs=2, cuda_graph=True, reference_init=False), motions)
    b = G1Env(EnvConfig(num_envs=2, cuda_graph=False, reference_init=False), motions)
    max_difference = 0.0
    for i in range(30):
        a.step(np.zeros((2, 29)))
        b.step(np.zeros((2, 29)))
        max_difference = max(max_difference, float(np.max(np.abs(a.q - b.q))))
        np.testing.assert_allclose(a.q, b.q, atol=5e-3, rtol=5e-3)
        if i % 5 == 0:
            print(i, 'z', a.q[0, 2], 'v', a.qd[0, :6], 'joints', a.q[0, a.q_indices[:6]], flush=True)
    assert np.all(a.q[:, 2] > 0.6), 'Standing robot lost ground support'
    contacts = int(a.solver.mjw_data.nacon.numpy().sum())
    assert contacts > 0, 'No physical contacts detected'
    print(f'PASS: graph agreement within contact solver tolerance; max delta={max_difference:.6f}, contacts={contacts}, supported height={a.q[:, 2]}', flush=True)


if __name__ == '__main__':
    main()
