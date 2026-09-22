"""Small bounded residual actor, PPO update and atomic single-file checkpoints."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from locomotion_laptop import ACTION_SIZE, OBS_SIZE, VERSION, Config, model_hash

HIDDEN = 128
LEARNING_RATE = 1e-4
LOG_STD_MIN, LOG_STD_MAX = -4.0, -1.5


def init_params(seed=42):
    rng = np.random.default_rng(seed)
    def weight(a, b):
        return jnp.asarray(rng.normal(0, np.sqrt(1/a), (a, b)), dtype=jnp.float32)
    return dict(w1=weight(OBS_SIZE, HIDDEN), b1=jnp.zeros(HIDDEN),
                w2=weight(HIDDEN, HIDDEN), b2=jnp.zeros(HIDDEN),
                wm=jnp.zeros((HIDDEN, ACTION_SIZE)), bm=jnp.zeros(ACTION_SIZE),
                wv=jnp.zeros((HIDDEN, 1)), bv=jnp.zeros(1),
                log_std=jnp.full(ACTION_SIZE, np.log(0.12)))


def network(params, obs, xp=jnp):
    x = xp.tanh(obs @ params['w1'] + params['b1'])
    x = xp.tanh(x @ params['w2'] + params['b2'])
    mean = x @ params['wm'] + params['bm']
    value = (x @ params['wv'] + params['bv'])[..., 0]
    return mean, xp.clip(params['log_std'], LOG_STD_MIN, LOG_STD_MAX), value


def normal_log_prob(latent, mean, log_std, xp=jnp):
    return (-0.5*((latent-mean)*xp.exp(-log_std))**2
            - log_std - 0.5*np.log(2*np.pi)).sum(axis=-1)


class NumpyPolicy:
    """Identical tiny network on CPU for rollouts/viewing, no GPU round trips.

    PPO differentiates the same equations in JAX. Policy parameters are copied
    once per rollout, not at each 50 Hz simulation tick.
    """
    def __init__(self, params):
        self.params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32), params)

    def act(self, obs, rng=None):
        mean, log_std, value = network(self.params, np.asarray(obs, np.float32), xp=np)
        latent = mean if rng is None else mean + np.exp(log_std)*rng.standard_normal(mean.shape).astype(np.float32)
        log_prob = normal_log_prob(latent, mean, log_std, xp=np)
        return np.tanh(latent).astype(np.float32), latent, log_prob, value

    def values(self, obs):
        return network(self.params, np.asarray(obs, np.float32), xp=np)[2]


def make_optimizer():
    return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LEARNING_RATE, eps=1e-5))


def make_update(optimizer):
    @jax.jit
    def update(params, opt_state, batch):
        def loss_fn(p):
            mean, log_std, values = network(p, batch['obs'])
            log_prob = normal_log_prob(batch['latent'], mean, log_std)
            log_ratio = log_prob-batch['log_prob']
            ratio = jnp.exp(log_ratio)
            # The tanh change-of-variables Jacobian cancels in the PPO ratio
            # since the exact pre-tanh latent action is stored in the rollout.
            surrogate = jnp.minimum(ratio*batch['advantages'],
                                    jnp.clip(ratio, 0.9, 1.1)*batch['advantages'])
            value_clipped = batch['values'] + jnp.clip(values-batch['values'], -0.2, 0.2)
            value_loss = 0.5*jnp.maximum((values-batch['returns'])**2,
                                        (value_clipped-batch['returns'])**2).mean()
            # Keep a small correction around V5, not an entropy-driven new gait.
            anchor = 0.05*jnp.mean(jnp.tanh(mean)**2)
            loss = -surrogate.mean() + 0.5*value_loss + anchor
            approximate_kl = jnp.mean((ratio-1)-log_ratio)
            return loss, (approximate_kl, value_loss)
        (loss, (kl, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params['log_std'] = jnp.clip(params['log_std'], LOG_STD_MIN, LOG_STD_MAX)
        return params, opt_state, jnp.array([loss, kl, value_loss])
    return update


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          gamma=0.99, lam=0.95):
    """Bootstrap time limits, but never carry GAE across an environment reset."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma*(1-terminated[t])*next_values[t] - values[t]
        carry = delta + gamma*lam*(1-np.maximum(terminated[t], truncated[t]))*carry
        advantages[t] = carry
    returns = advantages + values
    advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)
    return advantages, returns


def save_checkpoint(path, params, opt_state, metadata, model_xml):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, version=VERSION, observation_size=OBS_SIZE,
                    hidden_size=HIDDEN, model_sha256=model_hash(model_xml))
    # One atomic file binds weights, optimizer, config and exact physics XML.
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,
            params=np.frombuffer(serialization.to_bytes(params), dtype=np.uint8),
            optimizer=np.frombuffer(serialization.to_bytes(opt_state), dtype=np.uint8),
            metadata=np.array(json.dumps(metadata, allow_nan=False)),
            model_xml=np.array(model_xml))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        if meta.get('version') != VERSION or meta.get('observation_size') != OBS_SIZE:
            raise ValueError('Incompatible checkpoint: start a fresh V3 run; V1/V2 cannot be resumed.')
        xml = str(archive['model_xml'])
        if meta['model_sha256'] != model_hash(xml):
            raise ValueError('Checkpoint physics checksum mismatch')
        params = serialization.from_bytes(init_params(), archive['params'].tobytes())
        optimizer = make_optimizer()
        opt_state = serialization.from_bytes(optimizer.init(params), archive['optimizer'].tobytes())
    if not all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(params)):
        raise ValueError('Non-finite checkpoint weights')
    return params, opt_state, meta, xml
