# upstream-issues

Isolated reproducers and diagnostic notes for bugs we've hit in
dependencies while developing this project. Each subfolder is one bug,
dated by when we filed it, with a minimal repro script, full
environment info, and a GDB backtrace where we have one.

The common theme: **JAX 0.9.2 + ROCm 7.2 + gfx1100 (Radeon RX 7900
XTX)**. None of the bugs here are our code. They're filed upstream,
reproduced locally, and the workaround for each is documented in the
relevant folder's README.

## Currently open

| folder | upstream issue | workaround |
|---|---|---|
| [`2026-04-rocm-miopen-conv-segv/`](2026-04-rocm-miopen-conv-segv/) | [ROCm/MIOpen#3955](https://github.com/ROCm/MIOpen/issues/3955) | `JAX_PLATFORMS=cpu` for conv workloads on gfx1100 |
| [`2026-04-jax-jit-conv-backward-segv/`](2026-04-jax-jit-conv-backward-segv/) | [ROCm/jax#745](https://github.com/ROCm/jax/issues/745) | `JAX_DISABLE_JIT=1` (eager mode, ~15× slower) |
| [`2026-04-jax-rocm-multigpu-mesh-hang/`](2026-04-jax-rocm-multigpu-mesh-hang/) | [ROCm/jax#746](https://github.com/ROCm/jax/issues/746) | `ROCR_VISIBLE_DEVICES=0` (single GPU) |

## Overview of each

**MIOpen#3955 — conv SIGSEGV.** Any `jax.lax.conv_general_dilated` on
gfx1100 SIGSEGVs deep in MIOpen. Crash lands in
`miopen::kernels[abi:cxx11]()` via `GetKernelSrc` during the
compile-from-source fallback. MIOpen's kernel-DB query completes; the
blob comes back unusable; `LoadBinary` fails; fallback path crashes.
6-line reproducer. Blocks all convolution-using JAX workloads on
gfx1100 unless you fall back to CPU. This is the one that forced us
to run phase-2 JAX on CPU for all CNN cross-backend traces on mars.

**ROCm/jax#745 — conv+backward JIT compile segfault.** Earlier bug
(predates #3955, probably the same underlying MIOpen issue manifesting
via a different codepath). `jax.jit(value_and_grad(f))` segfaults when
`f` contains conv+reshape+matmul backward. Eager mode works. Filed
before we narrowed things down to MIOpen; worth keeping around in
case the two turn out to be separate fixes.

**ROCm/jax#746 — multi-GPU Mesh hang.** `jax.sharding.Mesh` +
`NamedSharding` with `PartitionSpec('batch')` across 2× gfx1100 hangs
indefinitely during XLA compile for every model (even trivial MLP).
Single-GPU works fine; multi-GPU with no sharding works fine. The
hang is specifically the Mesh sharding pass.

## Why these are here, not just in GitHub issues

1. **Reproducible offline.** If the GitHub issues get deleted or the
   repro code links rot, the important artifacts (minimal repro
   scripts, version matrices, backtraces) are still here in the repo.
2. **Onboarding.** Anyone picking this project up on AMD hardware
   hits at least #3955 on any conv work. Having the repro + workaround
   next to the code that uses ROCm means they can diagnose their own
   failure in 2 minutes instead of 2 hours.
3. **Paper trail.** If any of these get fixed upstream and a later
   ROCm version makes the workaround unnecessary, the git history
   here documents when each bug went away.
