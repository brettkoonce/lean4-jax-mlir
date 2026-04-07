# Benchmark: ROCm vs CUDA

Head-to-head comparison of the Lean4 -> MLIR -> IREE training pipeline
on AMD and NVIDIA hardware. Same codebase, same MLIR, different backends.

## Hardware

| | AMD (mars) | NVIDIA (klawd) |
|---|---|---|
| GPU | Radeon RX 7900 XTX | GeForce RTX 4060 Ti |
| VRAM | 24 GB | 16 GB |
| f32 TFLOPS | ~61 | ~22 |
| Driver | ROCm 7.2 | CUDA 12.9 |
| IREE backend | `rocm` / `gfx1100` | `cuda` / `sm_86` |

## Benchmarks to run

All use `IREE_BACKEND=rocm` on AMD, default (cuda) on NVIDIA.
Delete any existing `.vmfb` files first so compile is fresh.

### 1. MNIST MLP (669K params, batch 128)

```bash
rm -f .lake/build/train_step.vmfb
time IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-train-f32
```

Report: epoch time (ms), final loss after 12 epochs.

### 2. CIFAR-10 CNN (2.4M params, batch 128)

```bash
rm -f .lake/build/cifar_cnn.vmfb .lake/build/cifar_train_step.vmfb
time IREE_BACKEND=rocm .lake/build/bin/cifar-cnn-train-f32
```

Report: epoch time (ms), final loss after 25 epochs.

### 3. ResNet-34 on Imagenette (21.3M params, batch 16)

```bash
# Generate vmfb first
IREE_BACKEND=rocm .lake/build/bin/test-resnet-residual
# Train
IREE_BACKEND=rocm .lake/build/bin/resnet34-train data/imagenette
```

Report: ms/step, epoch time, loss curve.

## ROCm results (7900 XTX, ROCm 7.2, IREE 3.11)

| Model | Epoch time | ms/step | Loss (final) |
|---|---|---|---|
| MNIST MLP (12 ep) | ~7.6s | ~16ms | 0.005 |
| CIFAR CNN (25 ep) | ~23s | ~60ms | 0.018 |
| ResNet-34 (80 ep) | ~7.3min | ~720ms | training... |

## CUDA results (4060 Ti, CUDA 12.9, IREE 3.12)

| Model | Epoch time | ms/step | Loss (final) |
|---|---|---|---|
| MNIST MLP (12 ep) | ~17s | ~36ms | 0.026 |
| MNIST CNN BN (15 ep) | ~96s | ~205ms | 0.011 |
| CIFAR CNN BN (30 ep) | ~99s | ~254ms | 0.035 |
| ResNet-34 (80 ep) | ~24min | ~2430ms | training (0.34 @ ep7) |

## Notes

- First step is always slower (JIT warmup)
- The FFI pushes tensors one-by-one (220 for R34 with momentum),
  so transfer overhead dominates small models
- ROCm compile flags: `--iree-hal-target-backends=rocm --iree-rocm-target=gfx1100`
- CUDA compile flags: `--iree-hal-target-backends=cuda --iree-cuda-target=sm_86`
