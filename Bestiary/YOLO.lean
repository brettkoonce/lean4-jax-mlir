import LeanMlir

/-! # YOLO (v1 / v3 / v5 / v8 / v11) — Bestiary entry

YOLO v1 (Redmon, Divvala, Girshick, Farhadi, 2016 — "You Only Look Once:
Unified, Real-Time Object Detection") was the paper that reframed
detection as a **single regression problem**. Instead of the
two-stage R-CNN pipeline (region proposals → classify), YOLO ran one
convnet over the whole image and predicted boxes + classes directly in
one forward pass. The slogan came from the insight: each pixel is
looked at exactly once, grounded in a spatial grid of predictions.

## The output encoding is the trick

The network's output is a tensor of shape `(S, S, B·5 + C)`:

- `S × S` — the image is divided into an S×S grid (S = 7 in v1).
- `B` bounding boxes per grid cell (B = 2).
  Each box has 5 scalars: `(x, y, w, h, confidence)`.
- `C` class probabilities per grid cell (C = 20 for Pascal VOC).

So each of the 49 cells predicts 10 bbox scalars + 20 class probs =
30 outputs. The network's last layer is a dense projection to the
flat `49 · 30 = 1470` vector, which then gets reshaped to (7, 7, 30)
for loss computation. **There's no special YOLO primitive** — the
whole architecture is just stacked convs and two fully-connected
layers. The cleverness lives in the loss function and the output
interpretation, not in any novel layer.

## Architecture

```
Input: 448 × 448 × 3

 1. Conv 7×7, 64, stride 2      → 224×224×64
    MaxPool 2×2, stride 2       → 112×112×64

 2. Conv 3×3, 192               → 112×112×192
    MaxPool 2×2, stride 2       → 56×56×192

 3. Conv 1×1, 128               ┐
    Conv 3×3, 256               │ the "inception-ish" reduction block
    Conv 1×1, 256               │
    Conv 3×3, 512               ┘ → 56×56×512
    MaxPool 2×2, stride 2       → 28×28×512

 4. [Conv 1×1, 256              ┐
     Conv 3×3, 512]  × 4        │ four 1×1 + 3×3 pairs
    Conv 1×1, 512               │
    Conv 3×3, 1024              ┘ → 28×28×1024
    MaxPool 2×2, stride 2       → 14×14×1024

 5. [Conv 1×1, 512              ┐
     Conv 3×3, 1024]  × 2       │ two more pairs
    Conv 3×3, 1024              │
    Conv 3×3, 1024, stride 2    ┘ → 7×7×1024

 6. Conv 3×3, 1024              ┐
    Conv 3×3, 1024              ┘ → 7×7×1024

 7. Flatten                     → 50176
    Dense 50176 → 4096          ← this FC is HUGE (~200M params alone)
    Dense 4096 → 1470           ← → reshape to 7×7×30
```

The paper uses **Leaky ReLU** (slope 0.1) after every conv / FC except
the final output. Our `Activation` enum has `{relu, relu6, identity}`
only, so the bestiary spec below uses `relu` with a prose note. The
architectural shape and param count are unchanged by that substitution.

## Variants

- `yolo` — full YOLOv1, 24 conv layers + 2 FC. ~270M params total,
  with ~205M in the first FC layer alone.
- `fastYolo` — 9 conv + 2 FC, ~163M params. Faster, worse AP.
- `tinyYolo` — scale-model fixture: 6 conv + 2 FC at reduced width.

Note: "tiny-YOLO" in the wild also refers to a specific published
variant of YOLOv2/v3; `tinyYolo` here is just a bestiary toy, not that.
-/

-- ════════════════════════════════════════════════════════════════
-- § YOLOv1 — full 24-conv version
-- ════════════════════════════════════════════════════════════════

/-- YOLO v1 output head size: `S·S · (B·5 + C)` for Pascal VOC
    (S = 7, B = 2, C = 20) = 7·7·(10+20) = **1470**. -/
def yoloHeadSize : Nat := 7 * 7 * (2 * 5 + 20)

def yolo : NetSpec where
  name := "YOLOv1 (Redmon 2016)"
  imageH := 448
  imageW := 448
  layers := [
    -- Block 1: stem
    .conv2d 3 64 7 .same .relu,      -- stride is 1 in conv2d; stride-2 via a maxPool below
    .maxPool 2 2,

    -- Block 2
    .conv2d 64 192 3 .same .relu,
    .maxPool 2 2,

    -- Block 3: 1×1 reduce + 3×3 expand pattern
    .conv2d 192 128 1 .same .relu,
    .conv2d 128 256 3 .same .relu,
    .conv2d 256 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .maxPool 2 2,

    -- Block 4: four 1×1+3×3 pairs, then 1×1 + 3×3
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .maxPool 2 2,

    -- Block 5: two 1×1+3×3 pairs + two more 3×3 convs
    .conv2d 1024 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,    -- the stride-2 last conv (approximated with .same)

    -- Block 6: two 3×3 at 7×7 resolution
    .conv2d 1024 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,

    -- Head: flatten + 2 FC → reshape to (7, 7, 2·5 + 20)
    .flatten,
    .dense (7 * 7 * 1024) 4096 .relu,
    .dense 4096 yoloHeadSize .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Fast YOLOv1 — 9-conv variant
-- ════════════════════════════════════════════════════════════════

def fastYolo : NetSpec where
  name := "Fast YOLOv1"
  imageH := 448
  imageW := 448
  layers := [
    .conv2d 3 16 3 .same .relu,
    .maxPool 2 2,
    .conv2d 16 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .maxPool 2 2,
    .conv2d 128 256 3 .same .relu,
    .maxPool 2 2,
    .conv2d 256 512 3 .same .relu,
    .maxPool 2 2,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,

    .flatten,
    .dense (7 * 7 * 256) 4096 .relu,
    .dense 4096 yoloHeadSize .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyYolo fixture
-- ════════════════════════════════════════════════════════════════

def tinyYolo : NetSpec where
  name := "tiny-YOLO (fixture)"
  imageH := 112
  imageW := 112
  layers := [
    .conv2d 3 16 3 .same .relu,
    .maxPool 2 2,
    .conv2d 16 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .maxPool 2 2,

    .flatten,
    .dense (7 * 7 * 128) 512 .relu,
    .dense 512 (7 * 7 * (2 * 5 + 5)) .identity   -- 5 classes instead of 20
  ]

-- ════════════════════════════════════════════════════════════════
-- § YOLOv3 — Darknet-53 backbone + multi-scale FPN
-- ════════════════════════════════════════════════════════════════

/-! YOLOv3 (Redmon & Farhadi, 2018) made several big changes:
    * Backbone: **Darknet-53** — 53 conv layers, mostly 1×1 + 3×3 residual
      blocks at doubling channels. ~40M params on its own.
    * Multi-scale detection: predicts at three feature-pyramid levels
      (~13×13, 26×26, 52×52 for a 416 input). Catches small and large
      objects in one forward pass.
    * Anchor-based: 3 anchor boxes per grid cell × 3 scales = 9 anchors
      total, each producing (4 box + 1 obj + 80 class) = 85 outputs for
      COCO. Each scale's head emits 3 × 85 = 255 channels.

    Our NetSpec shows a single-scale variant for linearity; the
    multi-scale FPN (upsample + concat with earlier features) doesn't
    linearize cleanly. Same honest limitation as UNet's skip
    connections. -/
def yoloV3 : NetSpec where
  name := "YOLOv3 (single-scale view)"
  imageH := 416
  imageW := 416
  layers := [
    .convBn 3 32 3 1 .same,
    .convBn 32 64 3 2 .same,        -- downsample
    .darknetBlock 64 1,
    .convBn 64 128 3 2 .same,
    .darknetBlock 128 2,
    .convBn 128 256 3 2 .same,
    .darknetBlock 256 8,            -- P3 feature (52×52)
    .convBn 256 512 3 2 .same,
    .darknetBlock 512 8,            -- P4 feature (26×26)
    .convBn 512 1024 3 2 .same,
    .darknetBlock 1024 4,           -- P5 feature (13×13)
    -- Single-scale head (COCO: 3 anchors × (5 + 80) = 255 channels)
    .conv2d 1024 255 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § YOLOv5 — CSPDarknet backbone, anchor-based, Ultralytics 2020
-- ════════════════════════════════════════════════════════════════

/-! YOLOv5 (Ultralytics 2020, no paper) swapped in the **CSP block**
    (Cross-Stage Partial) for the Darknet residual. CSP splits the
    feature map in half, processes one half through residuals, and
    concatenates — cuts FLOPs without hurting accuracy.

    Uses SiLU (Swish) activation. PANet neck (bidirectional FPN) between
    backbone and head. Anchor-based detection at 3 scales, same head
    format as v3. -/
def yoloV5s : NetSpec where
  name := "YOLOv5s (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 32 6 2 .same,           -- stem
    .convBn 32 64 3 2 .same,
    .cspBlock 64 64 1,                -- C3 blocks in v5 terminology
    .convBn 64 128 3 2 .same,
    .cspBlock 128 128 2,
    .convBn 128 256 3 2 .same,
    .cspBlock 256 256 3,
    .convBn 256 512 3 2 .same,
    .cspBlock 512 512 1,
    .conv2d 512 255 1 .same .identity
  ]

def yoloV5m : NetSpec where
  name := "YOLOv5m (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 48 6 2 .same,
    .convBn 48 96 3 2 .same,
    .cspBlock 96 96 2,
    .convBn 96 192 3 2 .same,
    .cspBlock 192 192 4,
    .convBn 192 384 3 2 .same,
    .cspBlock 384 384 6,
    .convBn 384 768 3 2 .same,
    .cspBlock 768 768 2,
    .conv2d 768 255 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § YOLOv8 — anchor-free CSP with C2f blocks, Ultralytics 2023
-- ════════════════════════════════════════════════════════════════

/-! YOLOv8 (Ultralytics 2023) stayed close to v5 but:
    * Replaced C3 with **C2f** — a faster CSP variant with more parallel
      paths. Our `.cspBlock` approximates both.
    * Went **anchor-free**: no more preset anchor boxes. Each grid cell
      predicts offsets directly; classification + regression are
      decoupled heads.
    * Simpler head: no more `3 × (5 + C)` channels; separate class head
      (C channels) and box head (4×reg_max channels, where reg_max=16).

    Head output below is a single-scale approximation. -/
def yoloV8n : NetSpec where
  name := "YOLOv8n (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 16 3 2 .same,
    .convBn 16 32 3 2 .same,
    .cspBlock 32 32 1,
    .convBn 32 64 3 2 .same,
    .cspBlock 64 64 2,
    .convBn 64 128 3 2 .same,
    .cspBlock 128 128 2,
    .convBn 128 256 3 2 .same,
    .cspBlock 256 256 1,
    -- anchor-free head: classes(80) + box(4·reg_max=64) = 144 channels
    .conv2d 256 144 1 .same .identity
  ]

def yoloV8s : NetSpec where
  name := "YOLOv8s (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 32 3 2 .same,
    .convBn 32 64 3 2 .same,
    .cspBlock 64 64 1,
    .convBn 64 128 3 2 .same,
    .cspBlock 128 128 2,
    .convBn 128 256 3 2 .same,
    .cspBlock 256 256 2,
    .convBn 256 512 3 2 .same,
    .cspBlock 512 512 1,
    .conv2d 512 144 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § YOLOv11 — lightweight C3k2 + C2PSA attention, Ultralytics 2024
-- ════════════════════════════════════════════════════════════════

/-! YOLOv11 (Ultralytics 2024) keeps the anchor-free head from v8 but
    swaps in:
    * **C3k2** — lighter CSP variant (smaller kernels by default).
    * **C2PSA** — partial self-attention block added at the deepest
      stage. Modest accuracy gain, still much cheaper than full
      attention since it's on the smallest (7×7 or so) feature map.

    Our `.cspBlock` approximates C3k2 (same shape, slightly different
    FLOP profile). The attention block isn't exposed; at this abstraction
    level it reads as one more stage of CSP + a final conv. -/
def yoloV11n : NetSpec where
  name := "YOLOv11n (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 16 3 2 .same,
    .convBn 16 32 3 2 .same,
    .cspBlock 32 64 1,              -- C3k2 block (input doubles here)
    .convBn 64 64 3 2 .same,
    .cspBlock 64 128 2,
    .convBn 128 128 3 2 .same,
    .cspBlock 128 128 2,
    .convBn 128 256 3 2 .same,
    .cspBlock 256 256 1,            -- C2PSA approximated as one more CSP
    .conv2d 256 144 1 .same .identity
  ]

def yoloV11m : NetSpec where
  name := "YOLOv11m (single-scale view)"
  imageH := 640
  imageW := 640
  layers := [
    .convBn 3 64 3 2 .same,
    .convBn 64 128 3 2 .same,
    .cspBlock 128 256 1,
    .convBn 256 256 3 2 .same,
    .cspBlock 256 512 4,
    .convBn 512 512 3 2 .same,
    .cspBlock 512 512 4,
    .convBn 512 512 3 2 .same,
    .cspBlock 512 512 2,
    .conv2d 512 144 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — YOLO (v1 / v3 / v5 / v8 / v11)"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Detection as regression. One forward pass → grid of (box,"
  IO.println "  confidence, class) predictions. The paper that retired"
  IO.println "  two-stage detectors, then eight years of incremental refinement."

  IO.println ""
  IO.println "──────────── v1 (2016) ────────────"
  summarize yolo
  summarize fastYolo
  summarize tinyYolo

  IO.println ""
  IO.println "──────────── v3 (2018) — Darknet-53 + multi-scale ────────────"
  summarize yoloV3

  IO.println ""
  IO.println "──────────── v5 (2020) — CSPDarknet, anchor-based ────────────"
  summarize yoloV5s
  summarize yoloV5m

  IO.println ""
  IO.println "──────────── v8 (2023) — anchor-free, C2f ────────────"
  summarize yoloV8n
  summarize yoloV8s

  IO.println ""
  IO.println "──────────── v11 (2024) — C3k2 + C2PSA ────────────"
  summarize yoloV11n
  summarize yoloV11m

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • YOLOv1 uses zero new primitives (conv + FC); v3 adds"
  IO.println "    `.darknetBlock` (Darknet-53's 1×1+3×3 residual); v5/v8/v11"
  IO.println "    use `.cspBlock` (Cross-Stage Partial). Every YOLO since v5"
  IO.println "    is an Ultralytics-released variant of the same CSP +"
  IO.println "    FPN-neck + detection-head scaffold."
  IO.println "  • Multi-scale detection (FPN at 3 levels) DOESN'T linearize:"
  IO.println "    YOLO heads read features from the backbone at three"
  IO.println "    resolutions and concat upsampled coarser with finer. Same"
  IO.println "    skip-connection issue as UNet. We show a single-scale view."
  IO.println "  • v8's anchor-free shift is in the HEAD format (144 channels"
  IO.println "    = 80 classes + 4·16 distribution-based box regression),"
  IO.println "    not in the backbone. Backbone-level the v5 → v8 change is"
  IO.println "    mostly C3 → C2f, which our `.cspBlock` approximates."
  IO.println "  • v11's C2PSA attention block is the first self-attention"
  IO.println "    in a mainline YOLO — interesting inflection point. Still"
  IO.println "    lightweight (applied to smallest feature map only)."
  IO.println ""
  IO.println "  • Paper uses LeakyReLU(0.1) throughout the YOLO family; our"
  IO.println "    Activation enum has {.relu, .relu6, .identity}. Param"
  IO.println "    count identical; bestiary simplification."
  IO.println "  • v1's 50176 → 4096 FC alone takes ~205M of the ~270M total."
  IO.println "    This is why v2+ dropped FCs for convolutional output heads"
  IO.println "    (first with anchors in v2-v7, then anchor-free from v8)."
  IO.println "  • Stride-2 convs are approximated with conv+maxPool-2 since"
  IO.println "    our conv2d is stride-1-only in v1; the convBn primitive"
  IO.println "    (used from v3 onward) DOES have a stride param."
