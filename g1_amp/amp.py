"""Least-squares AMP discriminator and a bounded policy-transition replay buffer."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        for width in (256, 128):
            x = nn.leaky_relu(nn.Dense(width)(x), negative_slope=0.2)
        return nn.Dense(1)(x)[..., 0]


class AMP:
    def __init__(self, motions, key, learning_rate=1e-4, gradient_penalty=10.0):
        self.net = Discriminator()
        self.mean, self.std = jnp.asarray(motions.mean), jnp.asarray(motions.std)
        self.params = self.net.init(key, jnp.zeros((1, 148)))
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
        self.opt_state = self.optimizer.init(self.params)

        def normalized(x):
            return jnp.clip((x - self.mean) / self.std, -10, 10)

        def objective(params, expert, policy):
            expert, policy = normalized(expert), normalized(policy)
            real = self.net.apply(params, expert)
            fake = self.net.apply(params, policy)
            classification = 0.5 * (jnp.mean((real - 1) ** 2) + jnp.mean((fake + 1) ** 2))
            # R1-style gradient regularization on expert features, as used by AMP.
            grad = jax.vmap(jax.grad(lambda x: self.net.apply(params, x)))(expert)
            gp = jnp.mean(jnp.sum(grad ** 2, axis=-1))
            loss = classification + gradient_penalty * gp
            return loss, {"disc_loss": loss, "disc_expert": real.mean(), "disc_policy": fake.mean(), "disc_grad_penalty": gp}

        @jax.jit
        def update(params, opt_state, expert, policy):
            (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params, expert, policy)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, metrics

        self._update = update
        self._reward = jax.jit(lambda params, x: jnp.maximum(0, 1 - 0.25 * (self.net.apply(params, normalized(x)) - 1) ** 2))

    def reward(self, transitions):
        return np.asarray(self._reward(self.params, jnp.asarray(transitions)))

    def update(self, expert, policy):
        self.params, self.opt_state, metrics = self._update(self.params, self.opt_state, jnp.asarray(expert), jnp.asarray(policy))
        return {k: float(v) for k, v in metrics.items()}


class Replay:
    def __init__(self, capacity=100000, feature_dim=148):
        self.data = np.empty((capacity, feature_dim), np.float32)
        self.size, self.cursor = 0, 0

    def add(self, x):
        x = np.asarray(x, np.float32)[-len(self.data):]
        ids = (np.arange(len(x)) + self.cursor) % len(self.data)
        self.data[ids] = x
        self.cursor = (self.cursor + len(x)) % len(self.data)
        self.size = min(len(self.data), self.size + len(x))

    def sample(self, rng, size):
        if not self.size:
            raise ValueError("Cannot sample an empty replay buffer")
        return self.data[rng.integers(self.size, size=size)]
