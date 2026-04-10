# CIFAR-10 CNN — README additions

Add this to your existing README.md:

---

## CIFAR-10

A third model extends the CNN to the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset (32×32 RGB images, 10 classes), matching the architecture from
[Chapter 3](https://github.com/Apress/convolutional-neural-networks-with-swift-for-tensorflow/blob/main/CIFAR/main.swift)
of the book.

| Model | File | Architecture | Params | Expected Accuracy |
|-------|------|-------------|--------|-------------------|
| CIFAR CNN | `Main_cifar_s4tf.lean` | Conv²×32→Pool→Conv²×64→Pool→512→512→10 | ~2.43M | ~70% |

### Architecture detail

```
Input: 3×32×32 (RGB)
  → Conv2D 3×3 (3→32),  same padding, ReLU
  → Conv2D 3×3 (32→32), same padding, ReLU
  → MaxPool 2×2 stride 2                       → 32×16×16
  → Conv2D 3×3 (32→64), same padding, ReLU
  → Conv2D 3×3 (64→64), same padding, ReLU
  → MaxPool 2×2 stride 2                       → 64×8×8
  → Flatten                                     → 4096
  → Dense 4096→512, ReLU
  → Dense 512→512, ReLU
  → Dense 512→10, Softmax
```

### Parameter count breakdown

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| conv1a (3→32) | 864 | 32 | 896 |
| conv1b (32→32) | 9,216 | 32 | 9,248 |
| conv2a (32→64) | 18,432 | 64 | 18,496 |
| conv2b (64→64) | 36,864 | 64 | 36,928 |
| dense1 (4096→512) | 2,097,152 | 512 | 2,097,664 |
| dense2 (512→512) | 262,144 | 512 | 262,656 |
| dense3 (512→10) | 5,120 | 10 | 5,130 |
| **Total** | | | **2,431,018** |

### Quick start

```bash
# Download CIFAR-10
chmod +x download_cifar.sh
./download_cifar.sh

# Build & run
lake build cifar-cnn
.lake/build/bin/cifar-cnn ./data
```

### Estimated training time

The CIFAR CNN does ~2.45× more FLOPs per sample than the MNIST CNN (53.7M vs
21.9M forward, ~161M total per sample including backward). With 50K training
images × 12 epochs, expect roughly **12–20 hours** on a 24-core Intel
workstation (2455X). The extra conv block with 64 channels adds significant
cache pressure beyond the raw FLOP increase.

### Changes from the MNIST CNN

The `conv2dFwd`, `conv2dBwd`, `maxpool2dFwd`, and `maxpool2dBwd` functions are
reused as-is — they're already parameterized by channel counts and spatial
dimensions. The main additions are:

- CIFAR-10 binary data loader (different format from MNIST's IDX files)
- Two additional conv layers (conv2a 32→64, conv2b 64→64) + second pool
- Input changes from 1×28×28 to 3×32×32
- Flatten dimension changes from 6,272 to 4,096
- Learning rate 0.1 (vs 0.01 for MNIST) matching the S4TF reference

### Updated lakefile.lean

Add to your existing lakefile:

```lean
lean_exe «cifar-cnn» where
  root := `Main_cifar_s4tf
```

### Updated table for top of README

| Model | File | Architecture | Params | Accuracy |
|-------|------|-------------|--------|----------|
| MLP | `Main_working_1d_s4tf.lean` | 784 → 512 → 512 → 10 | ~670K | ~97% |
| CNN | `Main_working_2d_s4tf.lean` | Conv²×32→Pool→512→512→10 | ~3.5M | ~98% |
| CIFAR CNN | `Main_cifar_s4tf.lean` | Conv²×32→Pool→Conv²×64→Pool→512→512→10 | ~2.43M | ~70% |
