# Lean 4 → JAX

Lean 4 as a specification language for neural networks. Declare architecture
and training config in Lean, generate idiomatic JAX Python, run training.

Replicating the models from [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2) (Apress).

## Models

| Model | File | Params | Accuracy | GPU time | Optimizer |
|-------|------|--------|----------|---------|-----------|
| MNIST MLP (Ch.1) | `MainMlp.lean` | 670K | 97.9% | 7.5s | SGD |
| MNIST CNN (Ch.2) | `MainCnn.lean` | 3.5M | 97.6% | 23s | SGD |
| CIFAR-10 CNN (Ch.3) | `MainCifar.lean` | 2.4M | 63.3% | 53s | SGD |
| ResNet-34 | `MainResnet.lean` | 21.3M | 72.8% | 17 min | SGD+momentum |
| ResNet-50 | `MainResnet50.lean` | 23.5M | 61.2% | 31 min | SGD+momentum |
| MobileNet v1 | `MainMobilenet.lean` | 3.2M | 52.4% | 15 min | Adam |
| MobileNet v2 | `MainMobilenetV2.lean` | 2.2M | 59.4% | 15 min | Adam |
| EfficientNet-B0 | `MainEfficientNet.lean` | 7.2M | 55.9% | 16 min | Adam |

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
cd ../mnist-lean4
./download_mnist.sh        # MNIST (MLP, CNN)
./download_cifar.sh        # CIFAR-10
./download_imagenette.sh   # Imagenette (ResNet-34, requires Pillow)
cd ../lean4-jax
```

### 4. Build and run

```bash
lake build mnist-mlp mnist-cnn cifar-cnn resnet34 resnet50 mobilenet-v1 mobilenet-v2 efficientnet-b0

.lake/build/bin/mnist-mlp       # 7.5s
.lake/build/bin/mnist-cnn       # 23s on GPU
.lake/build/bin/cifar-cnn       # 53s on GPU
.lake/build/bin/resnet34        # 17 min on 6× GPU
.lake/build/bin/resnet50        # 31 min on 6× GPU
.lake/build/bin/mobilenet-v1    # 15 min on 6× GPU
.lake/build/bin/mobilenet-v2    # 15 min on 6× GPU
.lake/build/bin/efficientnet-b0 # 16 min on 6× GPU

# Custom data dir
.lake/build/bin/mnist-mlp /path/to/data
.lake/build/bin/resnet34 /path/to/imagenette
```

## Project structure

```
LeanJax.lean         Types + JAX codegen + runner (~600 lines)
MainMlp.lean         MNIST MLP spec
MainCnn.lean         MNIST CNN spec
MainCifar.lean       CIFAR-10 CNN spec
MainResnet.lean      ResNet-34 spec
MainResnet50.lean    ResNet-50 spec (bottleneck blocks)
MainMobilenet.lean   MobileNet v1 spec (depthwise separable)
MainMobilenetV2.lean MobileNet v2 spec (inverted residuals)
MainEfficientNet.lean EfficientNet-B0 spec (MBConv + SE + Swish)
lakefile.lean        Build config (8 executables, 1 library)
```

## How it works

1. Lean defines the network as a `NetSpec` — a list of `Layer` values
2. `JaxCodegen.generate` walks the layer list and emits idiomatic JAX Python
   - Conv layers → `jax.lax.conv_general_dilated`
   - Residual blocks → `basic_block` / `bottleneck_block` with skip connections
   - Depthwise separable convs → `feature_group_count` in JAX
   - MBConv blocks → inverted residuals + squeeze-excitation + Swish
   - Pool layers → `jax.lax.reduce_window`
   - Dense layers → `x @ w.T + b`
   - Instance normalization, activation, init, loss, training loop — all generated
3. `runJax` writes the script to `.lake/build/` and runs it via `python3`
4. JAX handles autodiff (`value_and_grad`), JIT compilation, XLA

## GPU / multi-GPU

The generated Python is GPU-ready — **zero code changes needed**. JAX
auto-dispatches to GPU when one is available:

```bash
# CUDA (NVIDIA)
pip install jax[cuda12]

# ROCm (AMD)
pip install jax[rocm]
```

**Multi-GPU data parallelism** is automatic via `jax.sharding`. The codegen
emits a `Mesh` + `NamedSharding` setup that detects all available GPUs,
replicates params, and shards batches across devices. No changes to the
Lean spec or training config — just add more GPUs.

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

## Lean version

Tested with Lean 4.29.0 / Lake 5.0.0, JAX 0.9.2.

## License

Public domain.
