import LeanMlir

/-! # MobileViT — Bestiary entry

MobileViT (Mehta & Rastegari, 2022 — "MobileViT: Light-weight,
General-purpose, and Mobile-friendly Vision Transformer") is a
hybrid of CNNs and transformers designed for edge devices. It
combines the **local inductive bias** of convolutions with the
**global context** of transformers, without paying the full
$O((HW)^2)$ attention cost.

## The MobileViT block

The secret sauce is a block structure that mixes conv and
transformer at the **spatial-patch granularity**:

```
         x : (ic, H, W)                        ← input feature map
             │
             ▼
     Conv 3×3 + BN + ReLU                     ← local mixing ("local rep")
             │
             ▼
     Conv 1×1  (ic → d)                       ← project to transformer dim
             │
             ▼
     Unfold into 2D patches: (d, H, W) → (d, P, N)
             │                                    P = patch area, N = n patches
             ▼
     Transformer × L                          ← global mixing across patches
             │
             ▼
     Fold back: (d, P, N) → (d, H, W)
             │
             ▼
     Conv 1×1  (d → ic)                       ← back to feature dim
             │
             ▼
     concat with input x along channels       ← fuse local + global
             │
             ▼
     Conv 3×3  (2·ic → ic) + BN + ReLU        ← fusion
             │
             ▼
         y : (ic, H, W)
```

**Why unfold into patches?** The paper's insight: most of the work a
standard ViT does is redundant at each spatial location. MobileViT
runs a transformer across **patches** (not pixels), so each patch sees
a global view, but the transformer is small because there are few
patches. The local convs handle fine-grained within-patch mixing.

Between MobileViT blocks, the backbone uses **MV2** (MobileNetV2
inverted-residual) blocks for efficient channel expansion and spatial
downsampling. Our existing `.invertedResidual` constructor handles MV2
directly.

## Variants (paper Table 1)

| Variant       | Channels [stages 1-6, out]       | Tx dim [d4, d5, d6] | L [L4, L5, L6] | Params |
|---------------|----------------------------------|---------------------|----------------|--------|
| MobileViT-XXS | 16, 16, 24, 48, 64, 80, 320      | 64 / 80 / 96        | 2 / 4 / 3      | 1.3M   |
| MobileViT-XS  | 16, 32, 48, 64, 80, 96, 384      | 96 / 120 / 144      | 2 / 4 / 3      | 2.3M   |
| MobileViT-S   | 16, 32, 64, 96, 128, 160, 640    | 144 / 192 / 240     | 2 / 4 / 3      | 5.6M   |

Trained on ImageNet-1k at 256×256 input (vs the usual 224×224).
-/

-- ════════════════════════════════════════════════════════════════
-- § MobileViT-S (canonical)
-- ════════════════════════════════════════════════════════════════

def mobileViTS : NetSpec where
  name := "MobileViT-S"
  imageH := 256
  imageW := 256
  layers := [
    -- Stem: 3×3 stride-2 conv (3 → 16)
    .convBn 3 16 3 2 .same,

    -- Stage 1: MV2 (16 → 32)
    .invertedResidual 16 32 4 1 1,

    -- Stage 2: MV2 × 3 with downsample (32 → 64)
    .invertedResidual 32 64 4 2 3,

    -- Stage 3: MV2 downsample + MobileViT block (64 → 96, d=144, L=2)
    .invertedResidual 64 96 4 2 1,
    .mobileVitBlock 96 144 4 288 2,

    -- Stage 4: MV2 downsample + MobileViT block (96 → 128, d=192, L=4)
    .invertedResidual 96 128 4 2 1,
    .mobileVitBlock 128 192 4 384 4,

    -- Stage 5: MV2 downsample + MobileViT block (128 → 160, d=240, L=3)
    .invertedResidual 128 160 4 2 1,
    .mobileVitBlock 160 240 4 480 3,

    -- Output: 1×1 expansion conv + GAP + classifier
    .conv2d 160 640 1 .same .relu,
    .globalAvgPool,
    .dense 640 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § MobileViT-XS (smaller, same shape)
-- ════════════════════════════════════════════════════════════════

def mobileViTXS : NetSpec where
  name := "MobileViT-XS"
  imageH := 256
  imageW := 256
  layers := [
    .convBn 3 16 3 2 .same,
    .invertedResidual 16 32 4 1 1,
    .invertedResidual 32 48 4 2 3,
    .invertedResidual 48 64 4 2 1,
    .mobileVitBlock 64 96 4 192 2,
    .invertedResidual 64 80 4 2 1,
    .mobileVitBlock 80 120 4 240 4,
    .invertedResidual 80 96 4 2 1,
    .mobileVitBlock 96 144 4 288 3,
    .conv2d 96 384 1 .same .relu,
    .globalAvgPool,
    .dense 384 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § MobileViT-XXS (smallest, aggressive compression)
-- ════════════════════════════════════════════════════════════════

def mobileViTXXS : NetSpec where
  name := "MobileViT-XXS"
  imageH := 256
  imageW := 256
  layers := [
    .convBn 3 16 3 2 .same,
    .invertedResidual 16 16 2 1 1,             -- narrower expansion (2× not 4×)
    .invertedResidual 16 24 2 2 3,
    .invertedResidual 24 48 2 2 1,
    .mobileVitBlock 48 64 4 128 2,
    .invertedResidual 48 64 2 2 1,
    .mobileVitBlock 64 80 4 160 4,
    .invertedResidual 64 80 2 2 1,
    .mobileVitBlock 80 96 4 192 3,
    .conv2d 80 320 1 .same .relu,
    .globalAvgPool,
    .dense 320 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyMobileViT fixture
-- ════════════════════════════════════════════════════════════════

def tinyMobileViT : NetSpec where
  name := "tiny-MobileViT"
  imageH := 64
  imageW := 64
  layers := [
    .convBn 3 16 3 2 .same,
    .invertedResidual 16 32 2 1 1,
    .invertedResidual 32 48 2 2 2,
    .mobileVitBlock 48 64 2 128 2,
    .conv2d 48 128 1 .same .relu,
    .globalAvgPool,
    .dense 128 10 .identity
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
  IO.println "  Bestiary — MobileViT"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Hybrid CNN + transformer for mobile. Convs for local detail,"
  IO.println "  a small transformer across patches for global context. Best"
  IO.println "  accuracy-per-parameter in its weight class."

  summarize mobileViTXXS
  summarize mobileViTXS
  summarize mobileViTS
  summarize tinyMobileViT

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.mobileVitBlock (ic dim heads mlpDim nTxBlocks)` bundles"
  IO.println "    the whole unit: local 3×3 → 1×1 project → fold into patches"
  IO.println "    → transformer × L → fold back → 1×1 project → concat with"
  IO.println "    input → 3×3 fusion."
  IO.println "  • MV2 blocks reuse the existing `.invertedResidual` constructor"
  IO.println "    — MobileViT is literally MobileNetV2 with MobileViT blocks"
  IO.println "    replacing some of the deeper MV2 stages."
  IO.println "  • The unfold/fold operations between the local-rep conv and"
  IO.println "    the inner transformer are `pdiv_reindex`-flavored shape"
  IO.println "    transformations — no new calculus primitive needed."
  IO.println "  • Variants differ almost entirely by channel / transformer-dim"
  IO.println "    widths; the structural shape is identical across S/XS/XXS."
