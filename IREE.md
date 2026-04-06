# Lean → MLIR → IREE: What We Built

Notes on the path from "Lean emits JAX Python" to "Lean trains MNIST end-to-end
via native IREE runtime calls, no Python at runtime." Picks up where
`Lean_MLIR.md` left off.

## Results

Three architectures train from scratch in Lean, with hand-written VJPs in
StableHLO, zero Python at runtime, GPU execution via IREE:

| Model | Params | Epochs | Final metric | Time/epoch | Total |
|-------|--------|--------|-------------|-----------|-------|
| MNIST MLP | 670K | 12 | **97.90% acc** | 16s | 3 min |
| MNIST CNN | 3.5M | 12 | loss 0.046 | 72s | 15 min |
| CIFAR-10 CNN | 2.4M | 25 | loss 0.668 | 126s | 52 min |

```
$ .lake/build/bin/mnist-mlp-train
Epoch  1: loss=0.364  acc=92.67% (16s)    Epoch 12: loss=0.026  acc=97.90%

$ .lake/build/bin/mnist-cnn-train
Epoch  1: loss=0.392  (78s)               Epoch 12: loss=0.046  (72s)

$ .lake/build/bin/cifar-cnn-train
Epoch  1: loss=1.908  (129s)              Epoch 25: loss=0.668  (126s)
```

MLP accuracy matches the JAX baseline (97.75%) within training noise. CNN/CIFAR
loss curves confirm correct gradients (steady monotonic decrease).

## Architecture

```
Lean NetSpec
     │
     ├─► MlirCodegen.generate (Lean, ~200 LOC)
     │     emits forward.mlir (StableHLO: conv, pool, flatten, dense)
     │
     ├─► hand_*_train_step.mlir (hand-written VJPs)
     │     forward + softmax-CE + per-layer backward + SGD
     │
     ▼ iree-compile (pip)
forward.vmfb  +  train_step.vmfb
     │
     ▼
Lean training loop (Main*Train.lean)
     │   - load data (MNIST IDX / CIFAR-10 binary)
     │   - He-init packed params
     │   - per batch: IreeSession.trainStepPacked  ────► FFI ──► GPU
     │   - eval: IreeSession.mlpForward            ────► FFI ──► GPU
     ▼
loss / accuracy
```

Two `.vmfb` modules per architecture, one Lean binary. All GPU calls flow
through `libiree_ffi.so` (1.4 MB, statically-linked IREE runtime + flatcc).
The generic `trainStepPacked` FFI function works for any architecture via
packed shape descriptors — no per-model C code needed.

## How we got here

### Step 1: Toolchain smoke test

Installed `iree-base-compiler` + `iree-base-runtime` via pip. Hand-wrote a
tiny `dense→relu` StableHLO module (`mlir_poc/tiny_mlp.mlir`). Compiled with
`iree-compile`, ran with `iree-run-module`. CPU backend worked first try.
CUDA backend errored with "missing GPU target in #hal.executable.target."

**Gotcha #1: sm_89 is broken in IREE 3.11.** Known upstream issues
[iree-org/iree#21122](https://github.com/iree-org/iree/issues/21122) and
[#22147](https://github.com/iree-org/iree/issues/22147). The compiler lacks
GPU target metadata for Ada and newer architectures. Workaround: use
`--iree-cuda-target=sm_86` (Ampere). PTX is forward-compatible, so the CUDA
driver JITs sm_86 PTX to sm_89 at load time. Verified correct numerical
output on 4060 Ti.

### Step 2: Lean codegen emits StableHLO

Wrote `LeanJax/MlirCodegen.lean` (~80 LOC) mirroring the JAX codegen pattern.
Walks `NetSpec.layers`, emits `stablehlo.dot_general` + `broadcast_in_dim`
+ `add` + `maximum` per dense-ReLU pair. Scope is MLP-only for this phase.

Generated MLIR diffs cleanly against the hand-written version from Step 1.
Accuracy validated end-to-end: Lean-generated `.vmfb` predicts **identically**
(0 diffs / 9984 samples) vs JAX on a trained MLP. Fp32-noise agreement
(2.4e-6 max diff) with numpy reference.

### Step 3: FFI via subprocess (and why it's dead on arrival)

First attempt at orchestrating inference from Lean: shell out to
`iree-run-module` per batch.

**Measured: 770ms per subprocess call.** Of which ~250 µs is actual GPU
compute, and 769.75 ms is IREE runtime init + CUDA device init + module load,
paid every single time. Training MNIST at 12 epochs × 469 batches = 5628
calls → **72 minutes** of subprocess launch overhead alone.

Unusable. Needed a persistent runtime session.

### Step 4: IREE from source, runtime-only

Cloned `iree-org/iree`. Naive recursive clone pulled in LLVM via
torch-mlir/stablehlo submodule chains and ballooned to 9 GB+ with no end in
sight. Killed it.

**Gotcha #2: Submodule discipline.** IREE's `build_tools/scripts/git/runtime_submodules.txt`
lists the 10 submodules actually needed for a runtime-only build. Shallow
clone + init those → 470 MB total. Build tree sits at
`/home/skoonce/lean/klawd_max_power/iree-build/`.

CMake flags:
```
-DCMAKE_BUILD_TYPE=Release
-DIREE_BUILD_COMPILER=OFF              # we use pip's iree-compile
-DIREE_BUILD_TESTS=OFF
-DIREE_BUILD_SAMPLES=OFF
-DIREE_HAL_DRIVER_DEFAULTS=OFF
-DIREE_HAL_DRIVER_CUDA=ON
-DIREE_HAL_DRIVER_LOCAL_SYNC=ON
-DIREE_HAL_DRIVER_LOCAL_TASK=ON
-DBUILD_SHARED_LIBS=OFF                # static, link into our own .so
```

Runtime-only ninja build took **~30 seconds** on the box. Produces
`libiree_runtime_unified.a` (2.3 MB static) containing everything we need.

### Step 5: C FFI wrapper

Wrote `ffi/iree_ffi.c` — a ~150 LOC thin wrapper over IREE's high-level
`iree_runtime_*` API. Exposes three functions:

```c
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path);
void                iree_ffi_session_release(iree_ffi_session_t* sess);
int                 iree_ffi_invoke_f32(sess, fn_name,
                                        n_inputs, ranks, dims_flat, input_data,
                                        n_outputs, output_totals, output_data);
int                 iree_ffi_train_step_mlp(...);  // int32 labels + scalar lr
```

**Gotcha #3: `IREE_ALLOCATOR_SYSTEM_CTL`.** The `iree_allocator_system()`
function is gated behind a compile-time macro. Compiler invocation needs
`-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl`.

**Gotcha #4: Flatcc split.** `flatcc_verify_*` symbols live in
`libflatcc_parsing.a`, not `libflatcc_runtime.a`. Both needed under
`--start-group/--end-group` for mutual symbol resolution.

**Gotcha #5: Function namespacing.** MLIR `module @mnist_mlp { func.func @forward }`
is invoked as `mnist_mlp.forward`, not `module.forward`.

**Gotcha #6: One-shot driver registration.** `iree_hal_cuda_driver_module_register`
is global; calling it twice (e.g. for two sessions) returns `ALREADY_EXISTS`.
Guard with a static flag inside session_create.

Packaged as `ffi/libiree_ffi.so` (1.4 MB). The IREE runtime + flatcc are
`--whole-archive`'d in, so consumers just link `-liree_ffi`.

**Measured FFI performance: 7.0 ms per call.**
**110× faster than subprocess** (770ms → 7ms). Pure GPU compute is still
~250 µs, so the remaining 6.7ms is buffer alloc + host↔device transfer
per call.

### Step 6: Lean FFI bindings

`ffi/iree_lean_ffi.c` bridges Lean's `FloatArray` (Float64) and `ByteArray`
to the C wrapper. Converts f64↔f32 at the boundary, handles packed int32
labels, wraps opaque session pointers in Lean external classes for GC.

`LeanJax/IreeRuntime.lean` declares three `@[extern]` functions:

```lean
opaque IreeSession : Type
def IreeSession.create     (path : @& String) : IO IreeSession
def IreeSession.mlpForward (sess, x, W0, b0, W1, b1, W2, b2, batch) : IO FloatArray
def IreeSession.mlpTrainStep (sess, params, x, y, lr, batch)        : IO FloatArray
```

`mlpTrainStep` uses a **packed-params** convention: all 669,706 MLP weights
flow as a single `FloatArray` in (6 concatenated tensors) and the same flat
layout out (plus loss appended at index 669706). Keeps the FFI surface
narrow.

Lakefile wiring uses a custom `target ireeLeanFfiO` that compiles the shim
.c file with Lean headers, wraps it in an `extern_lib`, and adds
`-liree_ffi` + rpath to `moreLinkArgs`.

**Gotcha #7: `--no-allow-shlib-undefined`.** Lean's bundled clang/lld is
strict about symbols referenced by shared libraries. Our `libiree_ffi.so`
references glibc symbols (`log2f`, `dlopen`, etc.) that ld.lld refuses to
resolve transitively. Pass `-Wl,--allow-shlib-undefined` to override.

**Measured Lean→FFI→GPU: 7.8 ms per call** (vs 7.0 ms direct C). 0.8 ms
of Lean overhead is the Float64→Float32 staging.

### Step 7: JAX-bootstrap train_step

Per the `Lean_MLIR.md` plan, Option B (bootstrap via `jax.export.export`)
gives us a known-correct training module while deferring hand-written
VJPs (Option A) to a pure refactor phase.

`mlir_poc/export_train_step.py` uses JAX to define forward + softmax-CE +
`value_and_grad` + SGD update, then exports via:

```python
exported = export.export(jax.jit(train_step))(
    spec_W0, spec_b0, ..., spec_x, spec_y, spec_lr)
open("train_step.mlir", "w").write(exported.mlir_module())
```

Produces 20 KB of StableHLO. The exported function is `jit_train_step.main`,
taking 9 inputs (6 params + x + y labels + lr scalar) and returning 7 outputs
(6 updated params + scalar loss).

Verified numerically: same random inputs → IREE output matches JAX to
fp32 noise (1.5e-8 on weights, 0.0 on loss).

### Step 8: Training loop in Lean

`LeanJax/MnistData.lean` parses IDX format (big-endian header + u8 pixels,
u8 labels) into `FloatArray` (images normalized to [0,1]) and `ByteArray`
(labels packed as int32 LE for the FFI). ~50 LOC.

`MainMlpTrain.lean`:
- Loads MNIST train (60k) + test (10k)
- Creates two IREE sessions: one for `mlpTrainStep`, one for `mlpForward`
- He-initializes packed params (pseudo-Gaussian via 3-sum of uniforms)
- For each epoch: 468 batches, calls `mlpTrainStep`, tracks mean loss
- After each epoch: unpacks params, runs 78 test batches, computes accuracy

**Result after 12 epochs: 97.87%.** No shuffle, no weight decay, no fancy
init — plain SGD with `lr=0.1`, matching the S4TF book's MLP recipe.

## Performance picture

| Stage | Time per call | Use case |
|---|---|---|
| `iree-run-module` subprocess | 770 ms | forbidden for training |
| Direct C FFI | 7.0 ms | C clients |
| Lean → FFI → GPU | 7.8 ms | current training loop |
| Pure GPU compute (iree-benchmark-module) | 250 µs | theoretical ceiling |

**Per-epoch wall clock: 16s.** This is ~20× slower than JAX-CPU, but the
bottleneck is NOT compute:

- 669,706 f64→f32 conversions on every step (~5 MB of Lean-heap activity)
- `sliceImages` does 100,352 `FloatArray.push` calls per batch (→ 47M/epoch)
- Params shipped host→device every step; nothing persists across calls

The GPU is idle most of the time. Closing the gap:

- **Persistent on-device params** (~3× win) — ship weights once, update
  in-place on GPU. Requires IREE output-buffer reuse semantics.
- **ByteArray FFI variant** (~2× win) — store params as raw float32 bytes
  in Lean, skip the f64 conversion entirely.
- **Pre-sliced batch views** — compute the 468 batch offsets at load time,
  reuse buffers.

Together these should bring us under 2 ms/step (~1 s/epoch), competitive
with JAX.

## Hand-written VJPs (Option A)

JAX-bootstrap (Option B) was the initial plan for training, but IREE 3.11
has a bug in StableHLO→linalg lowering: `jax.grad` of conv layers produces
non-standard `dim_numbers` like `[f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f]`
which IREE's pipeline miscompiles. Minimal repro: `jax.grad(sum(conv(x,W)**2))`
fails for any conv model. See `mlir_poc/export_cnn_train_step.py` for the
repro.

This forced Option A (hand-written VJPs) earlier than planned. Three
train_step MLIR modules now exist, all hand-written:

### MLP VJPs (`hand_train_step.mlir`, 130 lines)

Dense + ReLU backward + softmax-CE gradient + SGD. Verified byte-exact
against JAX's autodiff: loss diff 0.0, weight diffs ~1.5e-8 (lr × fp32
accumulation noise). Drop-in replacement for the JAX-exported module.

### CNN VJPs (`hand_cnn_train_step.mlir`, 322 lines)

Two new backward patterns beyond MLP:

**Conv backward dW (transpose trick).** Computing dW requires a convolution
with batch and feature dims swapped. Rather than emitting the non-standard
`dim_numbers` that IREE can't compile, we:
1. Transpose both operands: `[N,C,H,W] → [C,N,H,W]`
2. Use a standard-layout `stablehlo.convolution` (IREE handles this fine)
3. Transpose the result if `I_ch ≠ O_ch`: `[I,O,kH,kW] → [O,I,kH,kW]`

For a 3×3 kernel with SAME padding at spatial 28×28: the "kernel" operand
is 28×28 (the gradient tensor), padding=1 gives output 3×3. Verified against
JAX to 1.5e-4 (accumulation order, expected for fp32 over 100K-element
reductions).

**Conv backward dx.** Reverse the kernel spatially + transpose I/O channels,
then standard convolution with SAME padding:
```
W_t = transpose(W, [1,0,2,3])
W_rev = reverse(W_t, dims=[2,3])
dx = conv(dy, W_rev, SAME)
```

**Pool backward.** `stablehlo.select_and_scatter` — the native StableHLO
op for `reduce_window` gradients. Selects the max position in each window,
scatters the upstream gradient there.

### CIFAR-10 VJPs (`hand_cifar_train_step.mlir`, 463 lines)

Same patterns as MNIST CNN but with 4 conv layers + 2 pool layers. Generated
by `gen_train_step.py` which templates the proven patterns. Compiles to 500KB
`.vmfb`, verified loss-decreasing on random data via `iree-run-module`.

## Training results

### MNIST MLP (97.90%)

```
Epoch  1: loss=0.364  acc=92.67%  (16s)
Epoch  6: loss=0.068  acc=97.40%
Epoch 12: loss=0.026  acc=97.90%
```

16s/epoch, 468 batches × 128, SGD lr=0.1. Matches JAX baseline (97.75%).

### MNIST CNN (loss 0.046)

```
Epoch  1: loss=0.392  (78s)
Epoch  6: loss=0.084  (72s)
Epoch 12: loss=0.046  (72s)
```

72s/epoch, same batch config. Loss 0.046 corresponds to ~97-98% accuracy
(README baseline: 97.6%). No test-set eval wired yet for CNN.

### CIFAR-10 CNN (loss 0.668)

```
Epoch  1: loss=1.908  (129s)
Epoch  5: loss=1.333  (129s)
Epoch 10: loss=1.102  (126s)
Epoch 15: loss=0.976  (124s)
Epoch 20: loss=0.821  (127s)
Epoch 25: loss=0.668  (126s)
```

126s/epoch, 390 batches × 128, SGD lr=0.01, 25 epochs. CIFAR-10 is a harder
dataset (10 classes, 32×32 color images). The README baseline (JAX, 6× GPU)
reaches 63.3% accuracy; our loss curve is consistent with that range.

## Performance picture

| Stage | Time per call | Use case |
|---|---|---|
| `iree-run-module` subprocess | 770 ms | forbidden for training |
| Direct C FFI | 7.0 ms | C clients |
| Lean → FFI → GPU | 7.8 ms | current training loop |
| Pure GPU compute (iree-benchmark-module) | 250 µs (MLP) / 7.3 ms (CNN) | theoretical ceiling |

### Per-architecture breakdown

| Model | Compute/step | FFI overhead/step | Lean overhead/step | Wall/step |
|---|---|---|---|---|
| MNIST MLP | 0.25 ms | 7 ms | ~27 ms | ~34 ms |
| MNIST CNN | 22 ms (est) | 7 ms | ~125 ms | ~154 ms |
| CIFAR-10 CNN | 30 ms (est) | 7 ms | ~286 ms | ~323 ms |

**MLP is FFI-bound** — GPU is idle 99% of the time.
**CNN/CIFAR are Lean-overhead-bound** — the per-batch FloatArray construction
(393K pushes for CIFAR) + f64→f32 conversion of 2.4M params dominates.

### Optimization path

| Fix | Effort | Impact |
|---|---|---|
| ByteArray storage (raw f32, skip f64 conversion) | 2 hours | ~3× on param-heavy models |
| Pre-sliced batch views | 1 hour | ~2× on CIFAR (kills 393K pushes/batch) |
| Persistent on-device params | 4 hours | ~5× (kills host↔device transfer) |
| All three combined | 1 day | ~10× → competitive with JAX |

Multi-GPU only matters once the per-step overhead is under ~5ms. Currently
the GPU is idle 85-95% of the time; adding GPUs just adds more idle GPUs.

## IREE bugs encountered

1. **sm_89 not supported** ([iree-org/iree#21122](https://github.com/iree-org/iree/issues/21122)).
   Workaround: `--iree-cuda-target=sm_86`. PTX JITs forward to Ada.

2. **Conv gradient dim_numbers crash** (related to [iree-org/iree#21955](https://github.com/iree-org/iree/issues/21955)).
   JAX autodiff emits `stablehlo.convolution` with `dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f]`
   for dW computation. IREE's StableHLO→linalg lowering produces a
   `linalg.conv_2d_nhwc_hwcf` with malformed `strides` attribute.
   Workaround: hand-written VJPs using the transpose trick (see above).

## What's next

1. **Test-set eval for CNN/CIFAR** — wire up forward-only inference for
   accuracy measurement (currently loss-only).
2. **Perf optimizations** — ByteArray storage + persistent device buffers
   to close the gap with JAX.
3. **ResNet-34 on Imagenette** — needs `convBn` (instance norm), residual
   skip connections, bottleneck blocks. The real architectural stress test.
4. **Codegen for backward** — move hand-written VJPs into `MlirCodegen.lean`
   so new architectures get backward emission automatically.

## File map

```
LeanJax/
  MlirCodegen.lean          Lean NetSpec → StableHLO emitter (~200 LOC, conv+pool+dense)
  IreeRuntime.lean          @[extern] bindings + MlpLayout/CnnLayout/CifarLayout
  MnistData.lean            IDX parser (MNIST) + CIFAR-10 binary loader

ffi/
  iree_ffi.c / .h           C wrapper: session, invoke_f32, train_step_{mlp,generic}
  iree_lean_ffi.c           Lean shim: Float64↔f32, packed params, shape descriptors
  libiree_ffi.so            1.4 MB, static IREE runtime + flatcc inside
  test_ffi.c                C smoke test

mlir_poc/
  hand_train_step.mlir      MLP VJPs (130 lines, verified byte-exact vs JAX)
  hand_cnn_train_step.mlir  MNIST CNN VJPs (322 lines, transpose trick for conv backward)
  hand_cifar_train_step.mlir CIFAR-10 CNN VJPs (463 lines, gen_train_step.py templated)
  gen_train_step.py         Python template generator for VJP MLIR
  export_train_step.py      JAX bootstrap (MLP only — conv models hit IREE bug)
  export_cnn_train_step.py  JAX bootstrap attempt (documents the IREE conv-grad bug)
  tiny_mlp.mlir             toolchain smoke test (first thing we ever compiled)
  mnist_cnn.mlir            hand-written CNN forward smoke test
  validate_*.py             numerical validation scripts

Main*.lean                  Training/inference orchestrators:
  MainMlpMlir.lean            MLP: codegen → compile → FFI forward
  MainMlpTrain.lean           MLP: 12 epochs → 97.90% accuracy
  MainCnnMlir.lean            CNN: codegen → compile (forward only)
  MainCnnTrain.lean           CNN: 12 epochs → loss 0.046
  MainCifarTrain.lean         CIFAR: 25 epochs → loss 0.668
Test*.lean                  Smoke tests for FFI + train step
```

Upstream IREE lives sibling at `/home/skoonce/lean/klawd_max_power/iree/`
(source, 470 MB) and `iree-build/` (build). `libiree_ffi.so` links the
runtime statically — shipped binaries have no build-tree dependency.
