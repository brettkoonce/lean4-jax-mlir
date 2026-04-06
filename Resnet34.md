# ResNet-34: Path to Imagenette Training

Status and plan for ResNet-34 on Imagenette via the Lean → MLIR → IREE pipeline.

## What's done

**Forward codegen compiles.** `MlirCodegen.generate` on an expanded ResNet-34
(individual `convBn` layers, no `residualBlock` sugar) produces 20,699 chars
of StableHLO that IREE compiles to a `.vmfb` for CUDA. Validated 2026-04-06.

New ops added to `MlirCodegen.lean` this session:

| Op | Status | Notes |
|---|---|---|
| `convBn` forward | ✓ works | Conv (strided OK) + instance norm (mean/var/rsqrt/affine) + ReLU |
| `globalAvgPool` forward | ✓ works | reduce-sum spatial / (H×W) |
| Strided conv | ✓ works | `window_strides = array<i64: S, S>` in convAttrBlock |
| Strided maxPool (3/2) | ✓ works | SAME padding computed for kSize ≠ stride |

## What's not done

### 1. `residualBlock` forward emission (~40 LOC)

The `residualBlock ic oc nBlocks firstStride` layer type expands to N basic
blocks. Each basic block is:

```
block(x):
    out = convBn(x, W1, g1, b1)         # conv 3×3 + inst norm + relu
    out = convBn_no_relu(out, W2, g2, b2)  # conv 3×3 + inst norm (NO relu yet)
    return relu(out + x)                  # skip connection + relu
```

First block when `firstStride > 1` or `ic != oc`:
```
block_down(x):
    out = convBn(x, W1, g1, b1, stride=firstStride)
    out = convBn_no_relu(out, W2, g2, b2)
    shortcut = convBn_no_relu(x, Wp, gp, bp, stride=firstStride, k=1)  # 1×1 projection
    return relu(out + shortcut)
```

**Params per basic block:** 2 convBn = 2 × (W + gamma + beta) = 6 tensors.
**Params for downsampling block:** 3 convBn = 9 tensors (extra 1×1 projection).

The emission walks `nBlocks`, emitting the first block specially if it
downsamples, then N-1 identity blocks. The skip connection is just
`stablehlo.add` of the block input and the conv output.

**Key detail:** The second convBn in each block has NO ReLU. ReLU is applied
AFTER the skip addition. This means `emitConvBn` needs a `relu : Bool`
parameter (currently always emits ReLU). Or emit a variant `emitConvBnNoRelu`.

### 2. `residualBlock` backward emission (~60 LOC)

The backward through a residual block:
```
d_out → d_after_relu = d_out * (pre_relu > 0)
       ├→ d_conv_path = d_after_relu  (gradient flows to conv path)
       └→ d_skip = d_after_relu       (gradient flows to skip path)
           ├→ d_x_direct = d_skip     (identity shortcut, no downsampling)
           └→ d_x_proj = convBn_backward(d_skip, ...)  (projection shortcut)

d_conv2_pre = d_after_relu
→ convBn backward (no relu — inst norm backward only)
→ d_conv1_pre
→ convBn backward (with relu)
→ d_block_input = d_conv_path + d_skip_path
```

The skip connection means the gradient ADDS: `d_input = d_via_convs + d_via_skip`.
This is the "gradient highway" that makes ResNets trainable — the skip path
provides an unimpeded gradient flow.

### 3. `convBn` backward emission (~60 LOC)

Instance norm backward is the trickiest new VJP. For `y = (x - μ) / σ * γ + β`:

```
d_γ = reduce_sum(d_y * x_norm, dims=[0,2,3])    -- per-channel
d_β = reduce_sum(d_y, dims=[0,2,3])
d_norm = d_y * γ_broadcast

-- Instance norm backward (per-sample, per-channel):
N = H * W
d_var = sum(d_norm * (x - μ) * -0.5 * (var + ε)^(-1.5), spatial)
d_μ = sum(d_norm * -1/σ, spatial) + d_var * sum(-2*(x-μ), spatial) / N
d_x_norm = d_norm / σ + d_var * 2*(x-μ)/N + d_μ/N
```

Then `d_x_conv = d_x_norm` flows into the conv backward (same transpose trick
we already have).

**Total per convBn backward:** ~20 lines for d_γ/d_β, ~25 lines for instance
norm backward, ~15 lines for conv backward via transpose trick.

### 4. `globalAvgPool` backward (~5 LOC)

```
d_x = broadcast(d_out, dims=[0,1]) / (H * W)
```

Broadcast the (B, C) gradient back to (B, C, H, W) and divide by spatial count.

### 5. Imagenette data loader

Binary format from `preprocess_imagenette.py`:
- 4-byte header: sample count (little-endian u32)
- Per sample: 1 byte label + 224×224×3 bytes pixels (CHW, RGB)
- Total per sample: 1 + 150528 = 150529 bytes

Need `F32.loadImagenette` in C:
- Read binary file, normalize pixels to [0,1]
- Apply ImageNet mean/std normalization: `(pixel/255 - mean) / std`
  - mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- Store as float32 in ByteArray (CHW order, matches NCHW convention)

### 6. Training config

From `MainResnet.lean`:
```lean
learningRate := 0.001
batchSize := 192      -- but we'll use 128 for single-GPU
epochs := 80
useAdam := true       -- need Adam optimizer in MLIR (not just SGD)
weightDecay := 0.0001
cosineDecay := true
warmupEpochs := 3
```

**Adam optimizer** is a bigger change than SGD — needs moment buffers (m, v)
per parameter, bias correction, and the update rule:
```
m = β1 * m + (1-β1) * grad
v = β2 * v + (1-β2) * grad²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
W = W - lr * m_hat / (sqrt(v_hat) + ε)
```

Options:
- **Emit Adam in MLIR:** train_step takes (params, moments) and returns
  (new_params, new_moments, loss). Doubles the FFI data per step.
- **Adam in Lean:** keep moments in Lean, do the update in Lean after
  getting gradients from IREE. Needs a new FFI function that returns
  gradients instead of updated params.
- **Start with SGD:** simpler, may still converge (just slower/worse accuracy).
  The book uses Adam but SGD with good LR schedule can get close.

Recommendation: start with SGD (lr=0.01, cosine decay) and see what accuracy
we get. Add Adam later if SGD plateaus too early.

## Architecture summary

ResNet-34 layer structure:
```
convBn 3 64 7 2 .same          → (B, 64, 112, 112)   stem
maxPool 3 2                     → (B, 64, 56, 56)
residualBlock 64  64  3 1       → (B, 64, 56, 56)     3 blocks, no downsample
residualBlock 64  128 4 2       → (B, 128, 28, 28)    4 blocks, stride-2 first
residualBlock 128 256 6 2       → (B, 256, 14, 14)    6 blocks
residualBlock 256 512 3 2       → (B, 512, 7, 7)      3 blocks
globalAvgPool                   → (B, 512)
dense 512 10 .identity          → (B, 10)
```

Total params: ~21.3M (mostly in the 256→512 and 128→256 conv stages).

Param count breakdown:
- Stem (7×7 conv): 3×64×7×7 = 9408 + 2×64 = 9536
- Stage 1 (3 blocks × 2 conv): 6 × (64×64×3×3 + 2×64) = 222,336
- Stage 2 (4 blocks + proj): more...
- Total: ~21.3M

## Estimated implementation effort

| Task | LOC | Time |
|---|---|---|
| `residualBlock` forward emission | ~60 | 2 hours |
| `convBn` backward emission (inst norm) | ~60 | 3 hours |
| `residualBlock` backward emission | ~40 | 2 hours |
| `globalAvgPool` backward | ~5 | 15 min |
| Imagenette data loader (C) | ~80 | 1 hour |
| Training loop + layout constants | ~60 | 1 hour |
| Testing + debugging | — | 2-3 hours |
| **Total** | **~300** | **~2 days** |

## Estimated training time

Per-step: ~850ms (800ms GPU compute + 50ms FFI overhead)
Per-epoch (74 batches × 128): ~63s
80 epochs with SGD: **~84 minutes**

JAX baseline on 6× GPU: 50 min (so ~300 min single-GPU equivalent).
Our estimate of ~84 min is much faster because we use static batch,
no data augmentation, and simpler optimizer (SGD vs Adam). The JAX
version with Adam + cosine + augmentation + 6-GPU sharding does more
work per step.

For a fair comparison with SGD on 1 GPU: expect 60-75% accuracy
(vs 84.9% with Adam). Adding Adam would push accuracy higher but
requires moment buffers (~2× the parameter data per step).

## Dependencies

- Imagenette dataset: `./download_imagenette.sh` (downloads + preprocesses)
- IREE compile: `iree-base-compiler` 3.11 pip package (`--iree-cuda-target=sm_86`)
- IREE runtime: `libiree_ffi.so` from source build at `../iree-build/`

## Open questions

1. **Adam in MLIR vs Lean?** Emitting Adam in MLIR keeps everything on GPU
   but the train_step function gets larger and the FFI needs to pass moment
   buffers (doubling data per step). Adam in Lean is simpler but adds
   per-param element-wise updates on CPU. For 21.3M params × 2 moments × 4
   bytes = 170MB of moment data — significant either way.

2. **Batch size 128 vs 192?** The JAX version uses 192 (split across 6 GPUs
   = 32 per GPU). On 1 GPU with batch 128, each step does less work but we
   do more steps per epoch. Memory might be tight for 192 on a single 4060 Ti
   (16GB).

3. **Data augmentation?** The JAX version uses random crop. Implementing
   random crop in Lean (before FFI) or in MLIR (in the train_step) adds
   complexity. Start without augmentation; add if accuracy is too low.

4. **Weight decay?** Can be emitted as part of the SGD update:
   `W_new = W * (1 - wd * lr) - lr * grad`. One extra multiply per param.
   Easy to add.
