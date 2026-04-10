# MNIST Neural Network in Pure Lean 4

A complete, self-contained neural network that trains on the MNIST handwritten
digit dataset.  No dependencies beyond Lean 4.

## Background

I am new to Lean, trying to replicate the [MNIST solvers](https://github.com/Apress/convolutional-neural-networks-with-swift-for-tensorflow)
from the first two chapters of my book [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2).

Two models are included:

| Model | File | Architecture | Params | Accuracy |
|-------|------|-------------|--------|----------|
| MLP | `Main_working_1d_s4tf.lean` | 784 → 512 → 512 → 10 | ~670K | ~97% |
| CNN | `Main_working_2d_s4tf.lean` | Conv3×3(1→32) → Conv3×3(32→32) → MaxPool2×2 → 512 → 512 → 10 | ~3.5M | ~98% |

## Claude Code

When I asked Claude Opus (extra reasoning) about this, it found a book by Tomáš Skřivan with most of the required Lean code:

https://lecopivo.github.io/scientific-computing-lean/Working-with-Arrays/Tensor-Operations/#Scientific-Computing-in-Lean--Working-with-Arrays--Tensor-Operations

From there it was able to build a basic model, but we were unable to train anything because the way the data was modelled added too much overhead.  After a few rounds of back and forth we rebuilt the data pipe, after which I was able to successfully train two different models matching the architecture used in my book.  Claude generated all the code, in particular the workers to do parallel training.

The basic MLP takes about 5 minutes to train on a 24-core Intel workstation (2455X).

The basic CNN takes about ~300 minutes / 5 hours to train on a 24-core Intel workstation.

## Quick Start

### 1. Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Download MNIST data

```bash
chmod +x download_mnist.sh
./download_mnist.sh
```

This creates a `data/` directory with the four uncompressed IDX files.

### 3. Build & run

Build both models:

```bash
lake build
```

Or build them individually:

```bash
lake build mnist-mlp
lake build mnist-cnn
```

Run the MLP (~5 minutes):

```bash
.lake/build/bin/mnist-mlp ./data
```

Run the CNN (~5 hours):

```bash
.lake/build/bin/mnist-cnn ./data
```

## Understanding the output

Each epoch prints two lines:

```
  epoch 8  loss=0.054526  train_acc=98.511667%
[Epoch 8] Accuracy: 9759/10000 (0.975900) Loss: 0.078128
```

The first line is **training accuracy** — measured on the 60K training images as the model is being updated batch by batch.  The second line is **test accuracy** — measured on the 10K held-out test images after the epoch finishes.  Training accuracy being higher than test accuracy is normal and expected (the model fits its own training data slightly better than unseen data).

## How it relates to SciLean

The MLP forward pass is mathematically identical to:

```lean
import SciLean
open SciLean

def net (w₁,b₁,w₂,b₂,w₃,b₃) (x : Float^[784]) :=
  x |> dense 512 w₁ b₁
    |>.mapMono (fun x => max x 0)
    |> dense 512 w₂ b₂
    |>.mapMono (fun x => max x 0)
    |> dense 10 w₃ b₃
    |> softMax 0.1
```

In SciLean, the training step would ideally be:

```lean
let grad := ∇ p := params, loss p batch
params := params - lr • grad
```

However, SciLean's `autodiff` tactic cannot yet reliably differentiate through
a full network composition.  This project implements the same gradient
computation by hand (the `Net.backward` function).

## Lean version

Tested against `leanprover/lean4:v4.28.0`.  If you hit build issues, update
the `lean-toolchain` file to match your installed Lean version.

## License

Public domain — do whatever you want with it.
