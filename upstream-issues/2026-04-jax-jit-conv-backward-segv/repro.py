"""Conv + reshape + matmul backward JIT segfault repro.

Inline Python from the README, extracted so we can run it directly.
"""
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit


def loss_fn(params, x, y):
    w_conv, b_conv, w_dense, b_dense = params
    x = jax.lax.conv_general_dilated(
        x, w_conv, (1, 1), 'SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    x = x + b_conv.reshape(1, -1, 1, 1)
    x = x.reshape(x.shape[0], -1)
    x = x @ w_dense.T + b_dense
    return jnp.mean(x)


def main():
    key = jax.random.PRNGKey(0)
    params = (
        jax.random.normal(key, (2, 1, 3, 3)) * 0.01,
        jnp.zeros(2),
        jax.random.normal(key, (3, 32)) * 0.01,
        jnp.zeros(3),
    )
    x = jnp.ones((2, 1, 4, 4))
    y = jnp.array([0, 1])

    val, g = jit(value_and_grad(loss_fn))(params, x, y)
    print("loss:", float(val))
    print("grad shapes:", [a.shape for a in jax.tree.leaves(g)])
    print("OK")


if __name__ == "__main__":
    main()
