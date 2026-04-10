# Imagenette Training Results

All models trained on Imagenette (10 classes, ~9.5K train / 3.9K val) at 224×224
input via the Lean → MLIR → IREE pipeline. Training: Adam, batch 32, cosine LR
schedule with 3-epoch warmup, label smoothing 0.1, weight decay 1e-4, random
crop (256→224) + horizontal flip. Eval uses running BN statistics (EMA momentum
0.1) — not batch statistics.

Hardware: AMD Radeon 7900 XTX (gfx1100) via ROCm 7.2 / IREE.

## Final accuracies

| Model | Params | Val accuracy | Notes |
|---|---|---|---|
| ResNet-34 | 21.3M | **90.29%** | basic + bottleneck residual blocks |
| ResNet-50 | 23.5M | **89.40%** | bottleneck residual blocks |
| MobileNetV2 | 2.2M | **87.09%** | depthwise separable + inverted residual |
| MobileNetV3-Large | 3.0M | _in progress_ | exact h-swish + h-sigmoid SE |
| EfficientNet-B0 | 7.2M | _in progress_ | MBConv with swish + sigmoid SE |

## Per-epoch eval history (running BN stats)

### ResNet-34

| Epoch | Val acc |
|---|---|
| 10 | 75.74% |
| 20 | 81.99% |
| 30 | 87.81% |
| 40 | 87.04% |
| 50 | 88.93% |
| 60 | 90.14% |
| 70 | 90.19% |
| 80 | **90.29%** |

### ResNet-50

| Epoch | Val acc |
|---|---|
| 10 | 73.87% |
| 20 | 77.15% |
| 30 | 85.07% |
| 40 | 87.47% |
| 50 | 88.42% |
| 60 | 89.11% |
| 70 | 89.73% |
| 80 | **89.40%** |

### MobileNetV2

| Epoch | Val acc |
|---|---|
| 10 | 75.03% |
| 20 | 79.00% |
| 30 | 81.86% |
| 40 | 84.68% |
| 50 | 86.37% |
| 60 | 86.76% |
| 70 | 87.19% |
| 80 | **87.09%** |

## Training recipe ablation (ResNet-34)

| Config | Final val | Notes |
|---|---|---|
| SGD+momentum, batch 16, batch-stats eval | 24.36% | initial baseline |
| SGD + WD 5e-4 + hflip + label smooth + crop | 26.53% | partial regularization |
| Adam, batch 32, cosine LR (batch-stats eval) | 52.54% | optimizer + batch fix |
| Adam, batch 32, cosine LR + **running BN eval** | **90.29%** | full pipeline |

The single biggest jump (+38 points) came from switching eval from
mini-batch BN statistics to running BN stats. The model was learning
fine the whole time — the eval was broken because batch statistics
from 32 images are too noisy to normalize correctly.

## Pipeline

```
Lean 4 NetSpec (~15 lines)
  ↓  MlirCodegen.generateTrainStep (forward + loss + VJPs + Adam)
StableHLO MLIR (~500 KB - 760 KB)
  ↓  iree-compile (~10 min for ROCm gfx1100)
VMFB flatbuffer (1.8 - 2.4 MB)
  ↓  IREE runtime via libiree_ffi.so
GPU execution (HIP / ROCm)
```

The same Lean → MLIR pipeline handles all 5 architectures. Adding a new
architecture requires extending `MlirCodegen.lean` with:
- Forward emission for the new layer types
- VJP / backward emission
- FwdRec recording for backward intermediates

The training executable, FFI, and IREE runtime are unchanged across
architectures. Only the codegen grows.
