# Lean 4 → JAX

Lean 4 as a specification language for neural networks. Declare architecture
and training config in Lean, generate idiomatic JAX Python, run training.

Replicating the models from [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2) (Apress).

## Models

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

All Imagenette models trained from scratch on 6× RTX 4060 Ti, no pretrained weights, random crop augmentation.

## Lean specs

```lean
-- MLP (S4TF book Ch. 1)
def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- CNN (S4TF book Ch. 2)
def mnistCnn : NetSpec where
  name := "MNIST CNN"
  layers := [
    .conv2d  1 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 6272 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

-- CIFAR-10 CNN (S4TF book Ch. 3)
def cifarCnn : NetSpec where
  name := "CIFAR-10 CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

-- ResNet-34
def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

-- MobileNet v2 — 2.2M params, inverted residuals
def mobilenetV2 : NetSpec where
  name := "MobileNet v2"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .invertedResidual  32  16 1 1 1,
    .invertedResidual  16  24 6 2 2,
    .invertedResidual  24  32 6 2 3,
    .invertedResidual  32  64 6 2 4,
    .invertedResidual  64  96 6 1 3,
    .invertedResidual  96 160 6 2 3,
    .invertedResidual 160 320 6 1 1,
    .convBn 320 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]
-- ViT-Tiny — first transformer in Lean → JAX
def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity
  ]
```

Lean generates a complete JAX training script and runs it. The generated
Python is readable and auditable at `.lake/build/generated_*.py`.

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

For GPU (ROCm):
```bash
.venv/bin/pip install jax[rocm]
```

### 3. Get data

```bash
./download_mnist.sh        # MNIST (MLP, CNN)
./download_cifar.sh        # CIFAR-10
./download_imagenette.sh   # Imagenette (ResNet-34, requires Pillow)
```

### 4. Build and run

```bash
lake build mnist-mlp mnist-cnn cifar-cnn squeezenet mobilenet-v1 mobilenet-v2 \
      mobilenet-v3 mobilenet-v4 efficientnet-b0 efficientnet-v2s vgg16bn \
      resnet34 resnet50 vit-tiny

.lake/build/bin/mnist-mlp       # 7.5s
.lake/build/bin/mnist-cnn       # 23s on GPU
.lake/build/bin/cifar-cnn       # 53s on GPU
.lake/build/bin/squeezenet      # 45 min on 6× GPU
.lake/build/bin/mobilenet-v1    # 47 min on 6× GPU
.lake/build/bin/mobilenet-v2    # 48 min on 6× GPU
.lake/build/bin/mobilenet-v3    # 46 min on 6× GPU
.lake/build/bin/mobilenet-v4    # 46 min on 6× GPU
.lake/build/bin/efficientnet-b0 # 48 min on 6× GPU
.lake/build/bin/efficientnet-v2s # 58 min on 6× GPU
.lake/build/bin/vgg16bn         # 42 min on 6× GPU
.lake/build/bin/resnet34        # 50 min on 6× GPU
.lake/build/bin/resnet50        # 55 min on 6× GPU
.lake/build/bin/vit-tiny        # 45 min on 6× GPU

# Custom data dir
.lake/build/bin/mnist-mlp /path/to/data
.lake/build/bin/resnet34 /path/to/imagenette
```

## Project structure

```
LeanJax.lean              Types + JAX codegen + runner (~1200 lines)
Main*.lean                Model specs (15 architectures)
lakefile.lean             Build config (15 executables, 1 library)
download_mnist.sh         Download MNIST dataset
download_cifar.sh         Download CIFAR-10 dataset
download_imagenette.sh    Download + preprocess Imagenette
preprocess_imagenette.py  Resize JPEGs to 224×224 binary format
bug_report.md             ROCm XLA conv fusion bug reproducer
```

## How it works

1. Lean defines the network as a `NetSpec` — a list of `Layer` values
2. `JaxCodegen.generate` walks the layer list and emits idiomatic JAX Python
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
3. `runJax` writes the script to `.lake/build/` and runs it via `python3`
4. JAX handles autodiff (`value_and_grad`), JIT compilation, XLA

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
  See [`bug_report.md`](bug_report.md) for minimal reproducer and details.
- **Multi-GPU sharding hangs** — `jax.sharding.Mesh` causes XLA compilation
  to hang indefinitely on gfx1100. Workaround: `HIP_VISIBLE_DEVICES=0`.
- **`jax[rocm]` pip extra broken** — JAX 0.9.2 defines `rocm7-local` but
  requires `jax-rocm7-plugin==0.9.2.*` which doesn't exist yet. Install
  the 0.9.1.post3 packages manually.

Run conv models on ROCm with:
```bash
LLVM_PATH=/opt/rocm/llvm HIP_VISIBLE_DEVICES=0 JAX_DISABLE_JIT=1 \
  .lake/build/bin/mnist-cnn
```

## Why Lean → JAX?

The [mnist-lean4](../mnist-lean4) project built neural nets from scratch in Lean 4
with C FFI → OpenBLAS → hipBLAS. That works, but requires hand-written gradients
and manual BLAS calls for every operation.

Lean → JAX gives you:
- **Automatic differentiation** — JAX's `grad` replaces 100s of lines of manual backward passes
- **JIT compilation** — XLA compiles the compute graph, no manual optimization
- **GPU for free** — swap `jax[rocm]` or `jax[cuda]` and it just works
- **Lean as the spec** — type-checked architecture definitions, eventually provable properties

Compare: the Lean CNN backward pass is ~100 lines of hand-written gradient code.
The JAX version: zero — `value_and_grad(loss_fn)` does it all.

## Differences from the S4TF book

All 11 architectures are **structurally identical** to the Swift for TensorFlow
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
  We use instance norm (spatial stats only) — batch norm diverged with multi-GPU sharding
- **Book processes one image at a time** with gradient accumulation over the batch.
  We do true batched forward/backward across 6 GPUs
- **We add** cosine LR decay, linear warmup, weight decay, random horizontal flip —
  none of which are in the book
- **Book adds** dropout (MobileNet v1) and running BN stats for eval — we don't

## Supported layer types

| Layer | Description |
|-------|-------------|
| `dense` | Fully connected |
| `conv2d` | Standard convolution |
| `convBn` | Conv + instance norm + ReLU |
| `residualBlock` | BasicBlock (ResNet-18/34) |
| `bottleneckBlock` | Bottleneck (ResNet-50/101/152) |
| `separableConv` | Depthwise + pointwise (MobileNet v1) |
| `invertedResidual` | Expand → depthwise → project + skip (MobileNet v2) |
| `mbConv` | MBConv + SE + Swish (EfficientNet) |
| `mbConvV3` | Hard-swish, hard-sigmoid SE (MobileNet v3) |
| `fireModule` | Squeeze → parallel expand → concat (SqueezeNet) |
| `patchEmbed` | Conv patch projection + CLS token + positional embedding (ViT) |
| `transformerEncoder` | Multi-head self-attention + layer norm + MLP blocks (ViT) |
| `maxPool`, `globalAvgPool`, `flatten` | Structural layers |

## Lean version

Tested with Lean 4.29.0 / Lake 5.0.0, JAX 0.9.2.

## License

Public domain.
