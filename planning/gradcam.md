# gradcam.md — Class Activation Map / GradCAM plan

Goal: produce "what spatial regions did the network look at" heatmap
overlays for a trained network, target class `c`. Output: PPM/PNG per
image, suitable for embedding in the blueprint as Ch 9 / Ch 10
visualization.

## The catch — and the shortcut

GradCAM (Selvaraju et al. 2017) needs `∂(logit_c) / ∂(last conv map)`.
Our pipeline does **not** have inference-time autodiff:

- `forward` MLIR returns logits; no intermediate activations exposed.
- `train_step` MLIR backprops to **params**, not activations, and the
  starting gradient is `(softmax − onehot)/B`, not `−onehot` of a
  single class.

A naive port would require a new `gradcam_step` MLIR module that runs
forward + selective backward to a chosen activation. That's nontrivial.

**However:** for any network that ends `... → globalAvgPool → dense`
— ResNet, EfficientNet, MobileNet-V{2,3}, ConvNeXt, the MNIST CNN, the
CIFAR CNN — GradCAM **collapses to a closed form**:

```
y_c = Σ_k W[c, k] · pooled_k = Σ_k W[c, k] · (1/HW) Σ_{i,j} A_{ij}^k

⇒  ∂y_c / ∂A_{ij}^k = W[c, k] / HW       (constant in i, j)
⇒  α_c^k = (1/HW) Σ_{i,j} ∂y_c / ∂A_{ij}^k = W[c, k] / HW
⇒  GradCAM_c[i,j] = ReLU(Σ_k α_c^k · A_{ij}^k)
                  = (1/HW) · ReLU(Σ_k W[c, k] · A_{ij}^k)
```

Up to a positive constant, this is exactly Zhou et al. 2016 CAM —
**no backward pass required**. Just the forward activation `A` and
a slice of the dense weight matrix.

This covers ~80% of our zoo. Phase 1 cashes in on this; Phase 2
handles the holdouts (ViT, MobileViT, multi-head DETR-likes) only if
demand justifies the codegen cost.

## Phase 1 — CAM for GAP + dense networks (small, ~3 hr)

### Scope

In: ResNet-{18,34,50}, MobileNet-V2/V3, EfficientNet-B0/V2,
    ConvNeXt-{Tiny,Mini}, MNIST/CIFAR CNNs.
Out: ViT-Tiny, MobileViT, anything with a non-trivial head.

For Phase 1, **the spec must end in exactly `[…, .globalAvgPool, .dense ic oc act]`.**
Detect violators and refuse with a clear error rather than producing
silently-wrong heatmaps.

### Files to add

- `LeanMlir/Cam.lean` — pure Lean library:
  - `cam : (denseW : ByteArray) (lastConv : ByteArray) (B C H W NC : Nat) (targetClass : Nat) → ByteArray`
    Output shape `[B, H, W]` f32, ReLU-clamped, not yet upsampled.
  - `bilinearUpsample : ByteArray → Nat → Nat → Nat → Nat → Nat → ByteArray`
    Standard bilinear, NCHW or NHW input.
  - `colormap : Float → (UInt8 × UInt8 × UInt8)` — viridis (perceptually
    uniform, prints OK in greyscale). Lookup table of 256 entries.
  - `overlay : (img : ByteArray) (heatmap : ByteArray) (alpha : Float) → ByteArray`
    Blend in RGB.
  - `savePPM : ByteArray → Nat → Nat → String → IO Unit`
    Binary P6 format. ~150 KB per 224×224. Convert to PNG outside if
    final asset size matters.

- `MainGradCAM.lean` — exe that orchestrates:
  1. Parse args: `--spec resnet34 --ckpt runs/r34/final.bin --image
     data/imagenette/val/.../n01440764.JPEG --class auto --out
     blueprint/figures/gradcam/r34_n01440764.ppm`
  2. Load checkpoint, build spec, compile forward-with-capture vmfb.
  3. Forward image, capture last-conv activation + logits.
  4. `targetClass := if class == "auto" then argmax logits else class`.
  5. Compute heatmap, upsample, colormap, overlay, save.

- `tests/TestCam.lean` — sanity test:
  - Synthetic `[B=1, C=2, H=3, W=3]` activation + 2-class dense.
  - Hand-compute expected heatmap, compare to `Cam.cam` output.
  - Numerical match within ε.

### Forward-with-capture

The one piece of MLIR work in Phase 1: a forward variant that also
returns the last conv activation. Two implementation options:

**Option A (preferred):** new `generateForwardCam` that emits the
forward up to `globalAvgPool`, returns `(lastConv, logits)`. Stops the
emit walk one step early, captures the SSA of the pre-pool activation,
then runs the GAP + dense tail and returns both.

About ~30 lines of codegen, no backward.

**Option B:** split the existing forward at the GAP and chain two
separate vmfbs. Cleaner conceptually but needs param-loading twice
and double the compile time per network.

Go with Option A.

### Heatmap math (one screenful)

```lean
def cam (W : ByteArray) (A : ByteArray) (B C H W_ NC tgt : Nat) : ByteArray := Id.run do
  -- W: [NC, C] dense weights, row-major
  -- A: [B, C, H, W_] last conv, NCHW float32
  let mut out : ByteArray := ByteArray.mkEmpty (B * H * W_ * 4)
  for b in [:B] do
    for i in [:H] do
      for j in [:W_] do
        let mut s : Float := 0.0
        for k in [:C] do
          let wck := F32.read W (tgt * C + k).toUSize
          let aijk := F32.read A (b * C * H * W_ + k * H * W_ + i * W_ + j).toUSize
          s := s + wck * aijk
        let v := if s < 0.0 then 0.0 else s
        out := out.push 0 |>.push 0 |>.push 0 |>.push 0  -- placeholder, write f32
        -- (real impl writes 4 bytes via F32.write)
  return out
```

Then normalize per-image to [0, 1] before upsample.

### Output spec

For each `(spec, image, class)` triple emit:

- `<name>.ppm` — 224×224×3 binary PPM, heatmap-overlaid.
- `<name>.json` — sidecar with `{ predicted_class, target_class,
  prob_target, image_path, spec_name, checkpoint_id }`.

Stash under `blueprint/src/figures/gradcam/`.

### Visualization in the blueprint

Probably 2 figures:

1. One image, four CNNs (R34 / EnetB0 / ConvNeXt-T / MnV2) — same input,
   different "what each network sees". Demonstrates that architecture
   choice affects spatial attention.

2. One CNN, four images per ImageNette class — qualitative spread.

Skip if either gets ugly; one panel of "here's a CAM, it works" may be
enough.

## Phase 2 — true GradCAM for non-CAM networks (deferred)

### Why it's not Phase 1

For a head other than `GAP → dense` — e.g. ViT's `slice CLS token →
dense`, MobileViT's `conv-fold-fold → dense`, or DETR's per-query
prediction heads — GradCAM no longer collapses to a weight read. We
need actual `∂y_c / ∂A`.

Networks affected: ViT-Tiny, MobileViT (when added), any future
DETR-like / multi-head architecture.

### Sub-options

**2a — `gradcam_step` MLIR module.** New codegen target. Forward
to logits, then backward starting from `−onehot[target_class]`,
short-circuit at the chosen feature map (no param gradients). Reuses
~70% of the train backward emit but with a different starting
gradient and a "stop here" point.

Effort: high (~6 hr). Touches every per-layer backward emit (each
needs a "skip me, just propagate the gradient through" path that
already exists in some form for non-trainable layers).

Risk: same kind of MLIR signature plumbing as dropout/stochastic-depth
— the user has explicitly punted that pattern before.

**2b — checkpoint export to JAX, run there.** Save params in a JAX-
readable format, run a parallel forward + `jax.grad` in Python.

Effort: medium (~3 hr) including weight-format converter. Bypasses
our pipeline entirely; the visualization carries no codegen pedigree.

**2c — punt.** Phase 1 covers the headline networks the blueprint
needs visualizations for. ViT GradCAM is its own paper-spawning
research direction (Attention Rollout, AttCAT, etc.) and not a clean
single-knob deliverable.

Default to 2c. Revisit only if a specific blueprint figure demands
ViT spatial attention.

## Decisions to make before starting

| Question | Default | Notes |
|---|---|---|
| Output format | PPM binary | Trade size for zero-deps; convert to PNG outside |
| Colormap | viridis | Jet is classic, viridis prints better in greyscale |
| Class selection | argmax | Add `--class N` flag; document but don't make it mandatory |
| Upsample | bilinear | Bicubic is overkill for a 7×7 → 224×224 viz |
| Multi-image | one-shot per call | Loop in a shell script for batch; keep the exe simple |

## Risks / unknowns

- **Forward-with-capture vmfb size:** capturing a [B, 512, 7, 7] activation
  might bloat the IREE buffer transfer cost. Single-image forward
  should be fine; if batching is added later, profile.
- **Last-conv detection across architectures:** MBConv with SE blocks
  has a "last conv" that's the post-SE projection — make sure the
  emit walk identifies it correctly. Add a unit test per spec.
- **Stride / receptive field:** for R34, last conv is 7×7 from a
  224×224 input — coarse. For EnetB0 it's 7×7 too. ConvNeXt-T is 7×7.
  Bilinear upsample to 224×224 will look soft; that's intrinsic to
  GradCAM, not a bug.

## Out of scope

- GradCAM++ (Chattopadhay et al. 2018): refines weighting with second
  derivatives. Phase 2 dependency. Skip.
- ScoreCAM, Eigen-CAM: gradient-free alternatives, fine for ViT but
  separate plumbing. Skip.
- Counterfactual or contrastive heatmaps. Out.

## Estimated scope summary

| Phase | Effort | Coverage | Codegen change |
|---|---|---|---|
| Phase 1 (CAM) | ~3 hr | ~80% of zoo | Forward-with-capture variant only |
| Phase 2a (full GradCAM) | ~6 hr | +ViT, +MobileViT | New backward target |
| Phase 2b (JAX export) | ~3 hr | Same | None (sidesteps pipeline) |

Recommendation: ship Phase 1; defer Phase 2 until a concrete blueprint
figure asks for ViT spatial attention.
