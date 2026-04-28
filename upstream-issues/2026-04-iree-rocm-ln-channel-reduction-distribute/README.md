# IREE/HIP `failed to distribute` on LayerNorm-over-channels of NCHW

**Filed upstream:** [iree-org/iree#24283](https://github.com/iree-org/iree/issues/24283)

**Status: Fixed.** Confirmed fixed. Tested with `iree-base-compiler 3.12.0rc20260428 @ af030e43d8343263a6c869eae32f958f229ff7af`. Same MLIR previously failed on `3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25` (Mar 16); the fix landed somewhere in `e4a3b04…af030e4`.

## Summary

A straightforward LayerNorm computed over the channel axis of an NCHW
tensor (`reduce(..., dimensions=[1])` over `tensor<BxCxHxW>`) fails the
`Distribute` pass when targeting the HIP backend on gfx1100. The error
manifests as

```
error: 'func.func' op failed to distribute
note: see current operation:
"func.func"() <{ ..., sym_name = "<...>_dispatch_0_reduction_32x3136x96_f32"}>
```

The dispatch name `_reduction_32x3136x96_f32` reflects IREE's internal
flattening of the spatial dims (`56*56 = 3136`) and an attempt to tile a
reduction over the inner `C=96` axis. The pattern is exactly what arises
in ConvNeXt-style architectures (LN-over-channels per spatial location).

The same MLIR compiles cleanly for `--iree-hal-target-backends=llvm-cpu`.

## Environment

- GPU: AMD Radeon RX 7900 XTX (gfx1100, RDNA 3)
- ROCm: 7.2.0
- IREE compiler: 3.11.0rc20260316 @ `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`
- LLVM: 23.0.0git
- OS: Ubuntu 24.04, Linux 6.8.0-106-generic

## Minimal reproducer

[`repro_ln_nchw.mlir`](repro_ln_nchw.mlir) — 60 lines of StableHLO that
emits exactly the LayerNorm sequence (mean → broadcast → diff → var →
rsqrt → normalize → γ/β affine).

```bash
iree-compile repro_ln_nchw.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target=gfx1100 \
    -o /tmp/x.vmfb
```

Result on this environment: `failed to distribute` as quoted above.

```bash
iree-compile repro_ln_nchw.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    -o /tmp/x_cpu.vmfb
```

Result: clean compile, vmfb produced.

## Notes on isolation

A *single* `reduce(..., dimensions=[1])` of the same tensor compiles
fine on HIP — the failure requires the surrounding LN op chain
(broadcast/subtract/multiply/etc.). IREE appears to fuse the chain
into one dispatch and then can't tile the resulting `BxHWxC` shape
for GPU codegen.

## Workaround we use

Pre-transpose `[B, C, H, W] → [B, H, W, C]` (NHWC), reshape to
`[B, H*W, C]`, then run a transformer-style LN over the now-inner
`C` axis — the same algebraic LN with reductions over the innermost
axis (the well-supported case for IREE GPU codegen). Reshape +
transpose the result back to NCHW for downstream layers. The math
is identical; only the data layout changes.

(Implementation reference in our codegen:
`emitLayerNormForwardNCHW` in `LeanMlir/MlirCodegen.lean`, commit
`780dc48`.)

## Related

The companion bug in this directory
([`../2026-04-iree-rocm-stacked-reduce-distribute/`](../2026-04-iree-rocm-stacked-reduce-distribute/))
is a different distribute-pass failure on the same backend, triggered
by stacking 4+ `[0, 2, 3]` reductions inside one function. Both are
GPU-codegen tiling/fusion issues; both have small MLIR reproducers.
