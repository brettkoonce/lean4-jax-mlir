# Phase 2 — Lean → JAX

Lean 4 as a metaprogramming layer that emits idiomatic JAX Python.
The generated script gets `value_and_grad` autodiff and XLA JIT for
free, runs on any JAX-supported device.

This is the working middle ground between [`mnist-lean4/`](../mnist-lean4)
(pure Lean 4 with hand-written gradients, slow) and the phase 3 IREE
pipeline at the project root (no Python at all, MLIR straight to GPU).

## What's here

```
jax/
├── Jax.lean              -- umbrella: re-exports LeanMlir.Types/Spec + Jax.Codegen/Runner
├── Jax/
│   ├── Codegen.lean      -- ~1100 lines, walks NetSpec → JAX Python
│   └── Runner.lean       -- find Python, write generated_*.py, exec
├── MainMlp.lean          -- MNIST MLP (S4TF book Ch. 1)
├── MainCnn.lean          -- MNIST CNN (S4TF book Ch. 2)
├── MainCifar.lean        -- CIFAR-10 CNN (S4TF book Ch. 3)
├── MainResnet.lean       -- ResNet-34
├── MainResnet50.lean     -- ResNet-50
├── MainMobilenet.lean    -- MobileNet v1 (depthwise separable)
├── MainMobilenetV2.lean  -- MobileNet v2 (inverted residual)
├── MainMobilenetV3.lean  -- MobileNet v3 (hard-swish + SE)
├── MainMobilenetV4.lean  -- MobileNet v4 (UIB)
├── MainEfficientNet.lean -- EfficientNet-B0 (MBConv + SE + Swish)
├── MainEfficientNetV2.lean -- EfficientNet V2-S (Fused-MBConv early)
├── MainSqueezeNet.lean   -- SqueezeNet (fire modules)
├── MainVgg.lean          -- VGG-16-BN
└── MainVit.lean          -- ViT-Tiny (patch embed + transformer)
```

The Lean spec types (`NetSpec`, `Layer`, `TrainConfig`) live in
`LeanMlir/Types.lean` and `LeanMlir/Spec.lean` — they're shared with
the phase 3 backend at the project root.

## Quick start

### 1. Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Install JAX

```bash
python3 -m venv .venv
.venv/bin/pip install jax jaxlib
```

For GPU (ROCm — see ROCm status below for caveats):
```bash
.venv/bin/pip install jax==0.9.2 jax-rocm7-plugin==0.9.1.post3 jax-rocm7-pjrt==0.9.1.post3
```

### 3. Get data

```bash
./download_mnist.sh        # MNIST (MLP, CNN)
./download_cifar.sh        # CIFAR-10
./download_imagenette.sh   # Imagenette (ResNet, MobileNet, ViT, etc.)
```

### 4. Build and run

From the project root:

```bash
lake build mnist-mlp mnist-cnn cifar-cnn squeezenet mobilenet-v1 mobilenet-v2 \
      mobilenet-v3 mobilenet-v4 efficientnet-b0 efficientnet-v2s vgg16bn \
      resnet34 resnet50 vit-tiny

.lake/build/bin/mnist-mlp        # 7.5s
.lake/build/bin/mnist-cnn        # 23s on GPU
.lake/build/bin/resnet34         # 50 min on 6× GPU
.lake/build/bin/vit-tiny         # 45 min on 6× GPU
# etc.

# Custom data dir
.lake/build/bin/resnet34 /path/to/imagenette
```

## How it works

1. Lean defines the network as a `NetSpec` — a list of `Layer` values
2. `Jax.Codegen.generate` walks the layer list and emits idiomatic JAX Python
   - Conv layers → `jax.lax.conv_general_dilated`
   - Residual blocks → `basic_block` / `bottleneck_block` with skip connections
   - Depthwise separable convs → `feature_group_count` in JAX
   - MBConv blocks → inverted residuals + squeeze-excitation + Swish/hard-swish
   - Fire modules → squeeze + parallel expand + channel concat
   - Transformer blocks → multi-head self-attention + layer norm + MLP
   - Patch embedding → conv projection + CLS token + positional embedding
   - Pool layers → `jax.lax.reduce_window`
   - Dense layers → `x @ w.T + b`
   - Instance normalization, layer normalization, activation, init, loss, training loop — all generated
3. `runJax` writes the script to `.lake/build/generated_*.py` and runs it via `python3`
4. JAX handles autodiff (`value_and_grad`), JIT compilation, XLA

The generated Python is readable and auditable — open `.lake/build/generated_resnet34.py` after a build.

## GPU / multi-GPU

The generated Python is GPU-ready — **zero code changes needed**. JAX
auto-dispatches to GPU when one is available:

```bash
# CUDA (NVIDIA)
pip install jax[cuda12]

# ROCm (AMD) — requires manual install, see below
pip install jax==0.9.2 jax-rocm7-plugin==0.9.1.post3 jax-rocm7-pjrt==0.9.1.post3
```

**Multi-GPU data parallelism** is automatic via `jax.sharding`. The codegen
emits a `Mesh` + `NamedSharding` setup that detects all available GPUs,
replicates params, and shards batches across devices. No changes to the
Lean spec or training config — just add more GPUs.

### ROCm status (April 2026)

Tested on 2× RX 7900 XTX (gfx1100) with ROCm 7.2.0:

| Model | ROCm status | Notes |
|-------|-------------|-------|
| MNIST MLP | JIT works, 17s (1 GPU) | Matmul-only models work with JIT |
| MNIST CNN | Eager only, 350s | Conv+flatten+dense backward segfaults under JIT |
| CIFAR-10 CNN | Eager only, 659s | Same conv fusion bug |

**Known issues:**
- **Conv JIT segfault** — `jax.jit(value_and_grad(f))` segfaults when `f`
  combines conv + reshape (flatten) + matmul in the backward pass. Affects all
  conv models. Workaround: `JAX_DISABLE_JIT=1` (eager mode, ~15× slower).
  See [`../bug_report.md`](../bug_report.md) for minimal reproducer and details.
  Reported upstream: [ROCm/jax#745](https://github.com/ROCm/jax/issues/745).
- **Multi-GPU sharding hangs** — `jax.sharding.Mesh` causes XLA compilation
  to hang indefinitely on gfx1100, even for trivial MLP models. Workaround:
  `ROCR_VISIBLE_DEVICES=0`. See [`../bug_report_sharding.md`](../bug_report_sharding.md) for reproducer.
- **`jax[rocm]` pip extra broken** — JAX 0.9.2 defines `rocm7-local` but
  requires `jax-rocm7-plugin==0.9.2.*` which doesn't exist yet. Install
  the 0.9.1.post3 packages manually.

Run conv models on ROCm with:
```bash
LLVM_PATH=/opt/rocm/llvm HIP_VISIBLE_DEVICES=0 JAX_DISABLE_JIT=1 \
  .lake/build/bin/mnist-cnn
```

## Why this exists, and why we moved on

[`mnist-lean4/`](../mnist-lean4) (phase 1) built neural nets from scratch in Lean 4
with C FFI → OpenBLAS → hipBLAS. That works, but requires hand-written gradients
and manual BLAS calls for every operation.

The JAX path (phase 2) gives you:
- **Automatic differentiation** — JAX's `grad` replaces 100s of lines of manual backward passes
- **JIT compilation** — XLA compiles the compute graph, no manual optimization
- **GPU for free** — swap `jax[rocm]` or `jax[cuda]` and it just works
- **Lean as the spec** — type-checked architecture definitions

Compare: the Lean CNN backward pass in phase 1 is ~100 lines of hand-written gradient code.
The JAX version: zero — `value_and_grad(loss_fn)` does it all.

The phase 3 IREE pipeline (project root) goes one step further: instead of
generating JAX *Python* and running it through XLA, the codegen emits
StableHLO MLIR directly and IREE compiles it to a GPU flatbuffer. No Python
runtime, no autodiff library — the gradients are computed at codegen time
in Lean. See the project root README for that path.

## Models trained via this path

| Model | File | Params | Accuracy | GPU time | Optimizer |
|-------|------|--------|----------|---------|-----------|
| MNIST MLP | `MainMlp.lean` | 670K | 97.9% | 7.5s | SGD |
| MNIST CNN | `MainCnn.lean` | 3.5M | 97.6% | 23s | SGD |
| CIFAR-10 CNN | `MainCifar.lean` | 2.4M | 63.3% | 53s | SGD |
| SqueezeNet v1.1 | `MainSqueezeNet.lean` | 730K | 77.2% | 45 min | Adam |
| MobileNet v1 | `MainMobilenet.lean` | 3.2M | 79.5% | 47 min | Adam |
| MobileNet v2 | `MainMobilenetV2.lean` | 2.2M | 79.2% | 48 min | Adam |
| MobileNet v3-Large | `MainMobilenetV3.lean` | 3.0M | 81.1% | 46 min | Adam |
| MobileNet V4-Medium | `MainMobilenetV4.lean` | 4.1M | 77.3% | 46 min | Adam |
| EfficientNet-B0 | `MainEfficientNet.lean` | 7.2M | 82.5% | 48 min | Adam |
| EfficientNet V2-S | `MainEfficientNetV2.lean` | 38.2M | 83.1% | 58 min | Adam |
| **VGG-16-BN** | `MainVgg.lean` | **14.7M** | **88.3%** | **42 min** | Adam |
| ResNet-34 | `MainResnet.lean` | 21.3M | 84.9% | 50 min | Adam |
| ResNet-50 | `MainResnet50.lean` | 23.5M | 85.0% | 55 min | Adam |
| ViT-Tiny | `MainVit.lean` | 5.5M | 65.2% | 45 min | Adam |

All Imagenette models trained from scratch on 6× RTX 4060 Ti, no pretrained weights, random crop augmentation. The phase 3 numbers (which use a different training recipe and proper running BN stats) are in [`../RESULTS.md`](../RESULTS.md) and tend to be 5-10 points higher.

## Differences from the S4TF book

All architectures are **structurally identical** to the Swift for TensorFlow
implementations — same layer configs, same channel counts. The training recipes
differ due to multi-GPU batching and optimizer discoveries:

| Model | Book config | Our config | Why |
|-------|------------|------------|-----|
| MNIST MLP | SGD 0.1, bs=128, 12ep | Same | Exact match |
| MNIST CNN | SGD 0.1 | SGD 0.01 | LR tuned for stability |
| CIFAR-10 | SGD 0.1, 12ep | SGD 0.01, 25ep | LR tuned, more epochs |
| ResNet-34 | SGD 0.002, mom=0.9, bs=32 | Adam 0.001 + cosine/WD | Adam outperforms SGD with instance norm |
| ResNet-50 | SGD 0.002, mom=0.9, bs=32 | Adam 0.001 + cosine/WD + zero-init residual | Bottleneck blocks need adaptive LR |
| MobileNets | SGD 0.002, mom=0.9 | Adam 0.001 + cosine/WD | SGD can't converge without skip connections |
| EfficientNet | SGD 0.002, mom=0.9 | Adam 0.001 + cosine/WD | Same — depthwise convs need adaptive LR |
| SqueezeNet | SGD 0.0001, bs=128, 100ep | Adam 0.001, bs=192, 50ep | Adam converges in half the epochs |
| VGG-16 | SGD 0.002, bs=32, 10ep, no BN | Adam 0.001, bs=192, 50ep, +BN, GAP | Added BN + GAP (14.7M vs ~134M params) |

**Other differences:**
- **Normalization** — book uses batch norm with running stats (per-sample forward).
  This path uses instance norm (spatial stats only) — batch norm diverged with multi-GPU sharding.
  Phase 3 has proper batch norm with running stats, see [`../RESULTS.md`](../RESULTS.md).
- **Book processes one image at a time** with gradient accumulation over the batch.
  We do true batched forward/backward across 6 GPUs.
- **We add** cosine LR decay, linear warmup, weight decay, random horizontal flip —
  none of which are in the book.
- **Book adds** dropout (MobileNet v1) and running BN stats for eval — phase 2 doesn't, phase 3 does.
