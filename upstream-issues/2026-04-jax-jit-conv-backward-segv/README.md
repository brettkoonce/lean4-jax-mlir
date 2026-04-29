# JIT segfault: conv + reshape + matmul backward on gfx1100 (ROCm 7.2)

**Status: Fixed.** Confirmed fixed by upgrading to `jax 0.10.0` /
`jaxlib 0.10.0` / `jax-rocm7-{pjrt,plugin} 0.9.1.post4`. ROCm /
MIOpen versions unchanged (7.2.0 / 3.5.1). The minimal reproducer
(`repro.py`) now runs to completion: returns gradient tensors of the
expected shapes, no SIGSEGV. The previous `JAX_DISABLE_JIT=1`
workaround is no longer needed on 0.10.0+.

## Summary

`jax.jit(value_and_grad(f))` segfaults when `f` contains a convolution followed by a reshape (flatten) and a matmul (dense layer). The crash occurs during XLA compilation, not at runtime. Eager mode (`JAX_DISABLE_JIT=1`) works correctly. Forward-only JIT also works. The crash is specifically in the backward pass compilation.

## Environment

- GPU: 2x AMD Radeon RX 7900 XTX (gfx1100, RDNA 3)
- ROCm: 7.2.0
- OS: Ubuntu, Linux 6.8.0-106-generic
- Python: 3.12.3
- jax: 0.9.2
- jaxlib: 0.9.2
- jax-rocm7-plugin: 0.9.1.post3
- jax-rocm7-pjrt: 0.9.1.post3

## Minimal reproducer

```python
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit

def loss_fn(params, x, y):
    w_conv, b_conv, w_dense, b_dense = params
    # Conv
    x = jax.lax.conv_general_dilated(
        x, w_conv, (1, 1), 'SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    x = x + b_conv.reshape(1, -1, 1, 1)
    # Flatten
    x = x.reshape(x.shape[0], -1)
    # Dense
    x = x @ w_dense.T + b_dense
    return jnp.mean(x)

key = jax.random.PRNGKey(0)
params = (
    jax.random.normal(key, (2, 1, 3, 3)) * 0.01,
    jnp.zeros(2),
    jax.random.normal(key, (3, 32)) * 0.01,
    jnp.zeros(3),
)
x = jnp.ones((2, 1, 4, 4))
y = jnp.array([0, 1])

# SEGFAULT:
val, g = jit(value_and_grad(loss_fn))(params, x, y)
```

## What works vs. what crashes

| Test | JIT | Result |
|------|-----|--------|
| Conv forward only | yes | OK |
| Conv forward + backward (grad w.r.t. conv weights) | yes | OK |
| Conv + pool forward + backward | yes | OK |
| Dense (matmul) forward + backward | yes | OK (MLP trains fine) |
| Conv + flatten + dense forward + backward | **yes** | **SEGFAULT** |
| Conv + pool + flatten + dense forward + backward | **yes** | **SEGFAULT** |
| Conv + flatten + dense forward + backward | no (eager) | OK |
| Full CNN training (12 epochs MNIST) | no (eager) | OK, 97.6% accuracy |

The crash is **not** related to:
- Pool type (max pool, avg pool, or no pool -- all crash when combined with conv+flatten+dense)
- Tensor size (crashes with 2x1x4x4 input, 2 filters, 3 output classes)
- Data layout (NCHW and NHWC both crash)
- Autotune (`--xla_gpu_autotune_level=0` fixes standalone conv segfault but not this one)
- Triton GEMM (`--xla_gpu_enable_triton_gemm=false` does not help)
- Command buffers (`--xla_gpu_enable_command_buffer=` does not help)
- `jax.remat` / `jax.checkpoint` (does not help)

## Workarounds

1. **`JAX_DISABLE_JIT=1`** -- works but ~15x slower (MNIST CNN: 350s eager vs ~23s JIT on CUDA)
2. **CPU backend** -- `JAX_PLATFORMS=cpu` works correctly
3. **MLP-only models** work with JIT on GPU (no conv ops)

## Additional context

The segfault appears to be in the XLA compiler when fusing the backward pass for conv_general_dilated through a reshape into a dot_general. The backward pass for the dense layer produces a gradient tensor that must be reshaped back to the conv output shape before computing the conv weight gradient (transposed convolution). This specific fusion pattern crashes on gfx1100.

Also observed: multi-GPU `jax.sharding.Mesh` + `NamedSharding` causes XLA compilation to hang indefinitely (18+ minutes, never completes) on gfx1100 even for MLP-only models. Single GPU with `HIP_VISIBLE_DEVICES=0` works.

## Where to report

- JAX ROCm plugin: https://github.com/ROCm/jax/issues
- JAX main: https://github.com/jax-ml/jax/issues
