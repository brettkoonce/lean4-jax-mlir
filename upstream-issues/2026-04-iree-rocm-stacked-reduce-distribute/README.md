# IREE/HIP `failed to distribute` for stacked `[0, 2, 3]` reductions in BN backward

**Filed upstream:** [iree-org/iree#24282](https://github.com/iree-org/iree/issues/24282)

## Summary

A function containing four `reduce(..., dimensions = [0, 2, 3])` ops on
an NCHW tensor — the natural shape of BatchNorm's backward (β-grad,
γ-grad, sum(dn), sum(dn·norm)) — followed by the standard three-term
LayerNorm-style dx rebuild fails the `Distribute` pass when targeting
the HIP backend on gfx1100.

```
error: 'func.func' op failed to distribute
note: see current operation:
"func.func"() <{ ..., sym_name = "<...>_dispatch_4_reduction_32x2x256_f32"}>
```

A *single* `reduce(..., dimensions = [0, 2, 3])` of the same tensor
compiles fine on HIP — IREE's existing `convBn` lowering has used this
exact pattern for years. The failure surfaces only when **multiple**
such reductions live in one function and feed a follow-up
broadcast/multiply/subtract chain that produces a same-shape output.
IREE appears to fuse the cluster into a single dispatch with a tile
shape (`<batch>×<small>×<spatial>` here) the Distribute pass can't
tile for GPU codegen.

The same MLIR compiles cleanly for `--iree-hal-target-backends=llvm-cpu`.

## Environment

- GPU: AMD Radeon RX 7900 XTX (gfx1100, RDNA 3)
- ROCm: 7.2.0
- IREE compiler: 3.11.0rc20260316 @ `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`
- LLVM: 23.0.0git
- OS: Ubuntu 24.04, Linux 6.8.0-106-generic

## Minimal reproducer

[`repro_stacked_reduce.mlir`](repro_stacked_reduce.mlir) — 60 lines of
StableHLO that emit exactly the BN three-term backward (`dy → dx` for
shape `2×32×16×16`).

```bash
iree-compile repro_stacked_reduce.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target=gfx1100 \
    -o /tmp/x.vmfb
```

Result on this environment: failure on
`bn_backward_dx_dispatch_4_reduction_32x2x256_f32`.

```bash
iree-compile repro_stacked_reduce.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    -o /tmp/x_cpu.vmfb
```

Result: clean compile, vmfb produced.

## Notes on isolation

- Removing reductions until only one remains: compiles fine.
- Removing the broadcast-back-and-rebuild chain (returning only the
  four `[C]`-vector sums): compiles fine.
- It's the *combination* (4× reductions + the dx rebuild that consumes
  them via broadcast/multiply/subtract) that triggers the fusion that
  Distribute can't handle.

## Workaround we use

Each `reduce(..., dimensions=[0, 2, 3])` is split into a two-step
reduction: first `reduce(..., dimensions=[2, 3])` producing `[B, C]`,
then `reduce(..., dimensions=[0])` producing `[C]`. Mathematically
identical, two passes per reduction. Compiles cleanly on HIP. This
mirrors what IREE's existing `convBn` lowering has been doing for the
forward variance/mean computation since long before this bug surfaced
(comments in `MlirCodegen.lean` note "IREE can't distribute [0,2,3]"
as an established workaround).

(Implementation reference: the BN backward in `emitTrainStepBody` for
`.convNextStage` arms in `LeanMlir/MlirCodegen.lean`, commit `a5c7db3`.)

## Related

The companion bug in this directory
([`../2026-04-iree-rocm-ln-channel-reduction-distribute/`](../2026-04-iree-rocm-ln-channel-reduction-distribute/))
is a different distribute-pass failure on the same backend, triggered
by a single LayerNorm-over-channels of an NCHW tensor. Both are
GPU-codegen tiling/fusion issues; both have small MLIR reproducers.
