"""Multi-GPU Mesh sharding hang repro.

Inline Python from the README, extracted so we can run it directly.
Trivial 2-layer MLP sharded across all visible devices via Mesh +
NamedSharding(P('batch')).
"""
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


def loss_fn(params, x, y):
    w1, b1, w2, b2 = params
    h = jnp.maximum(0, x @ w1.T + b1)
    logits = h @ w2.T + b2
    return jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])


def main():
    devices = jax.devices()
    n_devices = len(devices)
    print(f"devices: {n_devices} ({devices})")
    if n_devices < 2:
        print("WARNING: fewer than 2 devices; this test needs 2+ to reproduce.")
        return

    mesh = Mesh(np.array(devices), axis_names=('batch',))
    data_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())

    key = random.PRNGKey(0)
    params = (
        random.normal(key, (32, 16)) * 0.1, jnp.zeros(32),
        random.normal(key, (3, 32)) * 0.1, jnp.zeros(3),
    )
    params = jax.device_put(params, replicated)

    bs = n_devices * 4
    x = jax.device_put(jnp.ones((bs, 16)), data_sharding)
    y = jax.device_put(jnp.zeros(bs, dtype=jnp.int32), data_sharding)

    val, g = jit(value_and_grad(loss_fn))(params, x, y)
    print(f"loss: {float(val)}")
    print(f"grad leaf shapes: {[a.shape for a in jax.tree.leaves(g)]}")
    print("OK")


if __name__ == "__main__":
    main()
