# Multi-GPU Mesh sharding hangs indefinitely on gfx1100 (ROCm 7.2)

**Status: Fixed.** Confirmed fixed by upgrading to `jax 0.10.0` /
`jaxlib 0.10.0` / `jax-rocm7-{pjrt,plugin} 0.9.1.post4`. ROCm /
MIOpen versions unchanged (7.2.0 / 3.5.1). The minimal reproducer
(`repro.py`) now compiles and runs to completion in seconds across
both GPUs of the same dual-7900-XTX node where it previously hung
indefinitely (18+ minutes, never completed). The previous
`ROCR_VISIBLE_DEVICES=0` (single-GPU) workaround is no longer needed.

## Summary

`jax.jit(value_and_grad(f))` hangs indefinitely during XLA compilation when data is distributed across multiple GPUs using `jax.sharding.Mesh` + `NamedSharding`. This affects **all** models, including trivial dense-only (MLP) networks. Single-GPU JIT works fine, and multi-GPU with no sharding also works fine. The hang is specifically triggered by `PartitionSpec('batch')` data sharding across a multi-device `Mesh`.

## Environment

- GPU: 2× AMD Radeon RX 7900 XTX (gfx1100, RDNA 3)
- ROCm: 7.2.0
- OS: Ubuntu 24.04, Linux 6.8.0-106-generic
- Python: 3.12.3
- jax: 0.9.2
- jaxlib: 0.9.2
- jax-rocm7-plugin: 0.9.1.post3
- jax-rocm7-pjrt: 0.9.1.post3

## Minimal reproducer

```python
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

devices = jax.devices()
n_devices = len(devices)
mesh = Mesh(np.array(devices), axis_names=('batch',))
data_sharding = NamedSharding(mesh, P('batch'))
replicated = NamedSharding(mesh, P())

def loss_fn(params, x, y):
    w1, b1, w2, b2 = params
    h = jnp.maximum(0, x @ w1.T + b1)
    logits = h @ w2.T + b2
    return jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])

key = random.PRNGKey(0)
params = (
    random.normal(key, (32, 16)) * 0.1, jnp.zeros(32),
    random.normal(key, (3, 32)) * 0.1, jnp.zeros(3),
)
params = jax.device_put(params, replicated)

bs = n_devices * 4
x = jax.device_put(jnp.ones((bs, 16)), data_sharding)
y = jax.device_put(jnp.zeros(bs, dtype=jnp.int32), data_sharding)

# HANGS INDEFINITELY:
val, g = jit(value_and_grad(loss_fn))(params, x, y)
```

## What works vs. what hangs

| Test | Devices | Sharding | Result |
|------|---------|----------|--------|
| MLP JIT, 1 GPU visible | 1 | none | **OK** |
| MLP JIT, 2 GPUs visible | 2 | none | **OK** |
| MLP JIT, 2 GPUs + Mesh | 2 | `P('batch')` | **HANGS** |
| MLP eager, 2 GPUs + Mesh | 2 | `P('batch')` | **OK** (with `JAX_DISABLE_JIT=1`) |
| CNN JIT, 1 GPU | 1 | none | SEGFAULT (separate bug, see [conv backward bug report](../2026-04-jax-jit-conv-backward-segv/README.md)) |

The model complexity is irrelevant — even a 2-layer MLP with 675 parameters hangs when sharded across 2 devices.

## Behavior

- The process hangs silently after printing device info — no output, no error, no timeout
- CPU usage stays near 0% (not spinning on compilation)
- GPU memory is allocated but no kernels run
- Left running for 18+ minutes with no progress
- `timeout 60` confirms it does not complete within 60 seconds

## Workarounds

1. **`ROCR_VISIBLE_DEVICES=0`** — restrict to single GPU, no sharding needed
2. **`JAX_DISABLE_JIT=1`** — eager mode works but is ~100–180× slower
3. **Use 2 GPUs without sharding** — `jit` works if data is not explicitly partitioned via `Mesh`

## Additional context

This was discovered while running Lean 4 → JAX generated training scripts (https://github.com/brettkoonce/lean4-mlir). The codegen produces multi-GPU data-parallel training by default using `Mesh` + `NamedSharding`. All models (MLP, CNN, ResNet, etc.) hang on gfx1100 when using this pattern.

Separately, single-GPU CNN models also crash with a JIT segfault during conv backward pass compilation — that is a different bug tracked in our [conv backward report](../2026-04-jax-jit-conv-backward-segv/README.md).

## Where to report

- JAX ROCm plugin: https://github.com/ROCm/jax/issues
- JAX main: https://github.com/jax-ml/jax/issues
