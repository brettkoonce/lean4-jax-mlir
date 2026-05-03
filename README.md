# Lean 4 → MLIR → GPU

**Interactive proof blueprint: [brettkoonce.github.io/lean4-mlir/blueprint/](https://brettkoonce.github.io/lean4-mlir/blueprint/)**
(or [PDF](https://brettkoonce.github.io/lean4-mlir/blueprint.pdf))
— clickable dependency DAG for the full VJP proof suite (72 theorems, no `sorry`s),
from `pdiv` primitives up to `vit_body_has_vjp_mat`.

Lean 4 as a specification language for neural networks. Declare architecture
in Lean, generate StableHLO MLIR (forward + loss + backward + optimizer all
in one fused function), compile to GPU via IREE, train end-to-end. No Python
runtime, no autograd library — the gradients are computed at codegen time
in Lean.

Companion code for the upcoming book *Verified Deep Learning with Lean4*
(follow-up to [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2), Apress).

**Current version: `v0.5.6`** — Chapter 9 lands its ConvNeXt-T
worked example (84.94% val on Imagenette, paper-faithful recipe);
Chapter 10 gets a Data Augmentation section with a 9-row ViT
recipe ablation table — CutMix is the load-bearing knob at 9.5K
images, and stacking RandAugment + Random Erasing on top of it
*hurts* val accuracy. Bestiary gets paper-exact entries for VGG,
ResNet-50/101/152, WRN, and DenseNet, plus the "N new primitives"
claim reframed around the Ch 2-10 reader's toolbox (what's free)
rather than the codebase (what's already in `Types.lean`). Found
and fixed a long-standing eval-pipeline bug along the way:
`centerCrop` was running on already-224 val data, reading past
per-image bounds and making heavy-aug runs appear to collapse.
New `LEAN_MLIR_EVAL_ONLY=1` mode re-evals saved checkpoints in
~5 sec each.

The `v0.5.5` headline still holds: Swish/SiLU as a first-class
activation (forward + backward + proved `swish_has_vjp_correct`)
plus the independent-kernel comparator re-check covering 38
theorems via public `*_has_vjp_correct` wrappers, and Ch 2's
"Why VJPs, not Jacobians?" bridge + canonical-pdiv witness
explainer + three-pillar TikZ spine diagram.

The `v0.5.4` headline still holds: MNIST MLP *and* CNN training
traces agree at the **float32 ULP floor** between two independent
compilation pipelines (Lean→IREE→GPU vs Lean→JAX→XLA) on both
NVIDIA and AMD hardware. See
[`traces/CROSS_BACKEND_RESULTS.md`](traces/CROSS_BACKEND_RESULTS.md)
for the full four-corner verification tables.

On top of that, a differential-test suite in
[`tests/vjp_oracle/`](tests/vjp_oracle/) uses JAX's `value_and_grad`
as an oracle for the hand-derived VJPs in `LeanMlir/Proofs/`. Nine
test cases cover every axiom family — dense, conv, BN, maxPool,
residual (biPath), depthwise, SE (elementwise product), attention,
and the transformer block — each verified to 1–2 ULP of JAX autodiff.

## Three phases

This project went through three implementations of the same idea — "Lean 4 as a
specification language for deep learning" — each shedding more dependencies
than the last.

**Phase 1 — Pure Lean 4.** [`mnist-lean4/`](mnist-lean4/): everything in Lean,
`Float64` as the only datatype, hand-written gradients, C FFI to OpenBLAS /
hipBLAS for the matmuls. Worked end-to-end on MNIST through ResNet-34 but
performance was poor — every operation crossed the FFI boundary, no fusion,
no autodiff, no JIT.

**Phase 2 — Lean → JAX.** [`jax/`](jax/): Lean as a metaprogramming layer
that emits idiomatic JAX Python (`jax/Jax/Codegen.lean`, ~1100 lines). The
generated script gets `value_and_grad` autodiff and XLA JIT for free, runs
on any JAX-supported device. Trades the pure-Lean story for a working stack
and real GPU performance. See [`jax/README.md`](jax/README.md) for details.

**Phase 3 — Lean → StableHLO → MLIR → device.** *(this README)* No Python
runtime at all. Lean directly emits StableHLO MLIR, IREE compiles it to a
GPU flatbuffer, a thin C FFI loads and runs it. The pure-math version of
phase 2 — autodiff is done at codegen time in Lean (`LeanMlir/MlirCodegen.lean`,
~5000 lines), not at runtime by a framework. See [`RESULTS.md`](RESULTS.md)
for the per-architecture numbers.

The proofs that the generated MLIR is mathematically correct live in
[`LeanMlir/Proofs/`](LeanMlir/Proofs/) — chapter-by-chapter VJP correctness
proofs for tensor ops, MLP, CNN, residual, batch norm, depthwise, SE,
LayerNorm, and attention. The codegen and the proofs were written
independently and arrived at the same decomposition: every backward pass
factors through the standalone gradient of one new primitive per
architecture (softmax for attention, the spatial reductions for BN, the
rank-1 collapse for SE), and everything else is composition via the chain
rule on tools from earlier chapters.

## Pipeline

```
Lean NetSpec  (~15 lines)
   │
   │  MlirCodegen.generateTrainStep
   ▼
StableHLO MLIR  (500 KB - 2 MB of text, forward+loss+backward+Adam fused)
   │
   │  iree-compile (~10-15 min for ROCm gfx1100)
   ▼
VMFB flatbuffer  (1.8-3 MB)
   │
   │  IREE runtime via libiree_ffi.so
   ▼
GPU execution  (HIP/ROCm or CUDA)
```

The same Lean → MLIR pipeline handles every architecture. Adding a new
architecture means extending `LeanMlir/MlirCodegen.lean` with:
- forward emission for the new layer types
- VJP / backward emission
- `FwdRec` recording for backward intermediates

The training executable, FFI, and IREE runtime are unchanged.

## Cross-backend verification

Phase 2 and Phase 3 share the same Lean `NetSpec` ADT but compile through
*completely independent* stacks (JAX/XLA vs IREE). Differential testing
confirms both stacks produce the same training dynamics on the same input,
for both MLP (670K params, 12 epochs) and CNN (1.7M params with conv+BN,
15 epochs):

| diff                              | MLP step 1 Δ | CNN step 1 Δ |
|-----------------------------------|--------------|--------------|
| phase 2 (JAX)  vs phase 3 (IREE)  | ~2e-7        | ~1e-5 to 1e-4 |
| phase 3 ROCm   vs phase 3 CUDA    | **0**        | **0**        |
| phase 2 CPU    vs phase 2 CUDA    | ~4e-6        | ~1e-4        |

MLP hits the float32 ULP floor because it's dense-only. CNN's noise
floor is looser by ~100× because each conv-BN layer does two
reductions over ~100k-element tensors and XLA's reduction trees differ
from IREE's — both pipelines do correct math, just with different
summation orders. Phase 3 ROCm ≡ Phase 3 CUDA is bit-identical at
step 1 on both networks. Reproducible in 5 minutes via
[`traces/CROSS_BACKEND_RESULTS.md`](traces/CROSS_BACKEND_RESULTS.md).

### VJP oracle

A separate per-axiom differential test in
[`tests/vjp_oracle/`](tests/vjp_oracle/) uses JAX's `value_and_grad` as
a correctness oracle for every hand-derived backward pass in
`LeanMlir/Proofs/`. Each test case is a minimal NetSpec exercising one
axiom in isolation; the oracle compares step-2 loss (the first step
whose value depends on the backward pass) against phase 2's
autodiff-derived gradients.

Nine cases, all green on mars (ROCm + CPU) and ares (CUDA):

| case | axiom | step 2 Δ |
|---|---|---|
| `dense` | `dense_has_vjp` + `softmaxCE_grad` | 2.7e-07 |
| `dense-relu` | `relu_has_vjp` + `vjp_comp` | 4.8e-07 |
| `conv` | `conv2d_has_vjp` + `flatten_has_vjp` | 2.2e-07 |
| `convbn` | `convBn_has_vjp` (BN-mode) | 2.2e-06 |
| `conv-pool` | `maxPool_has_vjp` (argmax tiebreaks) | 1.2e-04 |
| `residual` | `biPath_has_vjp` (additive fan-in) | 3.1e-07 |
| `depthwise` | depthwise-conv VJP via `.invertedResidual` | 1.1e-05 |
| `mbconv` | `elemwiseProduct_has_vjp` (SE gate) + Swish | 1.6e-06 |
| `attention` | patchEmbed + `transformerBlock_has_vjp_mat` + classifier | 1.8e-07 |

Run with `tests/vjp_oracle/run.sh`. Adding a new axiom means dropping
a minimal Lean spec under `tests/vjp_oracle/phase{2,3}/` plus one
line in the lakefiles — see
[`tests/vjp_oracle/README.md`](tests/vjp_oracle/README.md).

The oracle also surfaced a real `heInitParams` bug (shape-peek
heuristic misfiring at patchEmbed + transformer-block boundaries) and
a JAX-ROCm crash on gfx1100 (filed as
[ROCm/MIOpen#3955](https://github.com/ROCm/MIOpen/issues/3955); repro
lives at [`upstream-issues/2026-04-rocm-miopen-conv-segv/`](upstream-issues/2026-04-rocm-miopen-conv-segv/)).

## Results (Imagenette, 10 classes, 224×224)

Trained from scratch on a single AMD 7900 XTX (gfx1100), Adam, batch 32,
cosine LR + 3-epoch warmup, label smoothing 0.1, weight decay 1e-4, random
crop (256→224) + horizontal flip, **running BN stats for eval**.

| Model | Params | Val accuracy |
|---|---|---|
| ResNet-34 | 21.3M | **90.29%** |
| ResNet-50 | 23.5M | **89.40%** |
| EfficientNetV2-S | 38.2M | **88.50%** |
| EfficientNet-B0 | 7.2M | **87.58%** |
| MobileNetV2 | 2.2M | **87.09%** |
| MobileNetV3-Large | 3.0M | **86.48%** |
| MobileNetV4-Medium | 4.1M | **84.58%** |
| ViT-Tiny | 5.5M | **71.70%** |

Per-epoch eval histories and ablation tables in [`RESULTS.md`](RESULTS.md).

## Quick start

### 1. Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Install IREE

You need the IREE runtime built for your GPU (CUDA or ROCm). The FFI shim
in `ffi/` links against `libiree_runtime_unified.a` from the IREE build tree.
See [`IREE_BUILD.md`](IREE_BUILD.md) for build instructions.

### 3. Get data

```bash
./download_mnist.sh        # MNIST (Ch 3-4 trainers)
./download_cifar.sh        # CIFAR-10 (Ch 5 trainers)
./download_imagenette.sh   # Imagenette 320px → preprocessed binary (Ch 6+)
```

### 4. Build a trainer

```bash
lake build resnet34-train
```

This compiles the Lean trainer (which generates MLIR + drives IREE + runs
the training loop). Other targets, in roughly book order:
`mnist-mlp-train`, `mnist-cnn-train`, `cifar-cnn-train`, `cifar-bn-train`,
`resnet50-train`, `mobilenet-v2-train`, `mobilenet-v3-train`,
`mobilenet-v4-train`, `efficientnet-train`, `efficientnet-v2-train`,
`vgg-train`, `vit-tiny-train`.

### 5. Run

The first invocation generates and compiles the vmfbs (slow — IREE
compilation takes 10-15 min for ResNet-sized models). Subsequent
runs reuse the cached vmfbs unless you clear `.lake/build/`.

```bash
HIP_VISIBLE_DEVICES=0 IREE_BACKEND=rocm .lake/build/bin/resnet34-train

# Or via the included shell wrapper that sets the env vars correctly
bash run.sh resnet34                  # GPU 0, ROCm (defaults)
bash run.sh efficientnet-v2 1 cuda    # GPU 1, CUDA
```

For CUDA, set `IREE_BACKEND=cuda` (the default) and use `CUDA_VISIBLE_DEVICES`.

## Lean specs

The same `NetSpec` type is used by all three phases. A spec is a list of
`Layer` values:

```lean
def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity
  ]

def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .fusedMbConv 32 48 4 3 2 1 false,
    .uib  48  80 4 2 3 5,    -- ExtraDW
    .uib  80 160 6 2 0 3,    -- IB (= MBConv)
    .uib 160 160 4 1 5 0,    -- ConvNeXt
    .uib 160 160 4 1 0 0,    -- FFN
    -- ... 11 more UIB blocks
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]
```

## Project structure

```
lean4-mlir/
├── README.md               -- this file (phase 3)
├── RESULTS.md              -- per-architecture eval histories + ablations
├── IREE_BUILD.md           -- how to build libiree_ffi.so from scratch
├── ROCM.md                 -- ROCm setup notes
├── BENCHMARK.md            -- ROCm vs CUDA performance comparison
├── lakefile.lean           -- Lake build config (libraries + ~30 execs)
│
├── LeanMlir.lean           -- umbrella module
├── LeanMlir/
│   ├── MlirCodegen.lean    -- ~5000 lines, NetSpec → StableHLO MLIR
│   ├── IreeRuntime.lean    -- Lean ↔ libiree_ffi.so bindings
│   ├── F32Array.lean       -- ByteArray-backed float32 helpers
│   ├── Spec.lean           -- NetSpec / Layer / param-counting
│   ├── Types.lean          -- core types (Layer, Activation, Padding, ...)
│   ├── MnistData.lean      -- IDX file loader (older training paths)
│   └── Proofs/             -- VJP correctness proofs (~2100 lines)
│       ├── Tensor.lean
│       ├── MLP.lean
│       ├── CNN.lean
│       ├── Residual.lean
│       ├── BatchNorm.lean
│       ├── Depthwise.lean
│       ├── SE.lean
│       ├── LayerNorm.lean
│       └── Attention.lean
│
├── Main*Train.lean         -- phase 3 trainers (one per architecture)
│   ├── MainResnetTrain.lean
│   ├── MainResnet50Train.lean
│   ├── MainMobilenetV2Train.lean
│   ├── MainMobilenetV3Train.lean
│   ├── MainMobilenetV4Train.lean
│   ├── MainEfficientNetTrain.lean
│   ├── MainEfficientNetV2Train.lean
│   ├── MainVitTrain.lean
│   ├── MainVggTrain.lean
│   ├── MainMnistMlpTrain.lean
│   ├── MainMnistCnnTrain.lean
│   ├── MainCifarCnnBnTrain.lean
│   ├── MainCifarCnnTrain.lean
│   └── MainAblation.lean
│
├── tests/                  -- unit tests + smoke tests + differential tests
│   ├── Test*.lean          -- runtime / FFI / codegen sanity tests
│   ├── BenchResnet.lean
│   ├── diff_traces.py      -- JSONL trace diff helper
│   ├── cross_backend_mnist_mlp.sh
│   └── vjp_oracle/         -- JAX-autodiff oracle for hand-derived VJPs
│       ├── README.md
│       ├── run.sh
│       ├── diff_step.py
│       ├── phase3/         -- Lean→IREE test trainers
│       └── phase2/         -- (mirrored at jax/tests/vjp_oracle/phase2/)
│
├── upstream-issues/        -- isolated reproducers + backtraces for bugs
│   └── 2026-04-rocm-miopen-conv-segv/  -- ROCm/MIOpen#3955
│
├── ffi/
│   ├── iree_ffi.{c,h}      -- IREE runtime wrapper
│   ├── iree_lean_ffi.c     -- Lean FFI bindings
│   ├── f32_helpers.c       -- data loading, He init, EMA, augmentation
│   └── libiree_ffi.so      -- compiled shared library
│
├── jax/                    -- phase 2 (Lean → JAX Python)
│   ├── README.md
│   ├── Jax.lean
│   ├── Jax/{Codegen,Runner}.lean
│   ├── Main*.lean          -- 14 JAX-driven architecture specs
│   └── tests/vjp_oracle/phase2/  -- phase-2 mirror of oracle specs
│
├── mnist-lean4/            -- phase 1 (pure Lean 4 + C BLAS)
│
├── traces/                 -- committed cross-backend training traces
│   ├── CROSS_BACKEND_RESULTS.md
│   ├── TRACE_FORMAT.md
│   └── mnist_{mlp,cnn}.*.jsonl
│
├── data/                   -- downloaded + preprocessed datasets
└── run_*.sh                -- shell wrappers for tmux env propagation
```

## Supported layers (phase 3 codegen)

| Layer | Description |
|-------|-------------|
| `dense` | Fully connected (with optional activation) |
| `conv2d` | Standard convolution |
| `convBn` | Conv + batch norm + ReLU/ReLU6/Swish/h-swish |
| `residualBlock` | BasicBlock (ResNet-18/34) |
| `bottleneckBlock` | Bottleneck (ResNet-50/101/152) |
| `invertedResidual` | Expand → depthwise → project + skip (MobileNetV2) |
| `mbConv` | + Squeeze-Excitation, Swish (EfficientNet) |
| `mbConvV3` | + h-swish + h-sigmoid SE (MobileNetV3, exact math) |
| `fusedMbConv` | k×k regular conv replaces (1×1 expand + depthwise) (EfficientNetV2) |
| `uib` | Universal Inverted Bottleneck — pre-DW? + expand + post-DW? + project (MobileNetV4) |
| `patchEmbed` | Conv patch projection + CLS token + positional embedding (ViT) |
| `transformerEncoder` | LN → MHSA → + → LN → MLP → +, with exact tanh-form GELU |
| `maxPool`, `globalAvgPool`, `flatten` | Structural |

Activations supported with exact backward: ReLU, ReLU6, Swish, h-swish,
h-sigmoid, GELU (tanh form). Layer-norm and batch-norm both have proper
VJPs and (for BN) running statistics for eval.

## Lean version

Tested with Lean 4.29.0 / Lake 5.0.0, IREE built from source against
ROCm 7.2.0 / gfx1100.

## Citing this work

```bibtex
@software{koonce2026leanmlir,
  author  = {Brett Koonce and Claude Code},
  title   = {Verified Deep Learning with Lean4: Formal Backpropagation from MLP to Attention, via MLIR},
  url     = {https://github.com/brettkoonce/lean4-mlir},
  version = {0.5.6},
  year    = {2026},
}
```
