import LeanMlir

/-! # Swin Transformer — Bestiary entry

Swin Transformer (Liu et al., 2021) is a hierarchical vision transformer
that makes ViT-style architectures scale to dense prediction tasks
(segmentation, detection) by combining three ideas:

1. **Hierarchical pyramid** — downsample spatial dims and double channels
   across four stages, like a CNN backbone. Cheap per-token attention at
   high resolution; more channels at low resolution.
2. **Windowed attention** — compute self-attention within local
   `windowSize × windowSize` patches (default 7×7). Cost is `O(M·HW)`
   instead of `O((HW)²)`, where M = windowSize².
3. **Shifted windows** — alternate blocks shift the window grid by
   `windowSize/2` so information crosses window boundaries every other
   layer. Keeps the linear cost, restores global receptive field.

Between stages, a **patch merging** op concatenates 2×2 neighbor tokens
and runs a linear projection — the transformer analogue of strided
convolution. Halves spatial, doubles channels (typically).

```
  Input (3, 224, 224)
       │
       ▼  patchEmbed 4×4       → (96, 56, 56)        token seq: 3136 × 96
       │
       ▼  swinStage ×2         → same shape          Stage 1: 2 blocks, W-MSA+SW-MSA
       │
       ▼  patchMerging 96→192  → (192, 28, 28)       784 × 192
       │
       ▼  swinStage ×2         → same shape          Stage 2: 2 blocks
       │
       ▼  patchMerging 192→384 → (384, 14, 14)       196 × 384
       │
       ▼  swinStage ×6         → same shape          Stage 3: 6 blocks
       │
       ▼  patchMerging 384→768 → (768, 7, 7)         49 × 768
       │
       ▼  swinStage ×2         → same shape          Stage 4: 2 blocks
       │
       ▼  globalAvgPool + dense(768 → 1000)
```

Every `swinStage` alternates W-MSA and SW-MSA internally (handled inside
the layer). The head count doubles per stage: [3, 6, 12, 24] for Swin-T,
giving a fixed `d_head = 32` throughout.

## Variants (paper defaults)

| Model | Stage depths    | Embed dim | Heads       | Params |
|-------|-----------------|-----------|-------------|--------|
| Swin-T (tiny)  | [2, 2, 6, 2]  | 96  | [3, 6, 12, 24] | ~28M |
| Swin-S (small) | [2, 2, 18, 2] | 96  | [3, 6, 12, 24] | ~50M |
| Swin-B (base)  | [2, 2, 18, 2] | 128 | [4, 8, 16, 32] | ~88M |
| Swin-L (large) | [2, 2, 18, 2] | 192 | [6, 12, 24, 48] | ~197M |

MLP ratio = 4 everywhere; window size = 7 (12 for `^` variants trained at
384×384).
-/

-- ════════════════════════════════════════════════════════════════
-- § Swin-T (tiny) — 28M params, the canonical imagenet classifier
-- ════════════════════════════════════════════════════════════════

def swinT : NetSpec where
  name := "Swin-T"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 96 4 3136,                 -- 56×56 = 3136 patches, dim 96
    .swinStage 96 3 384 7 2,                  -- stage 1: 2 blocks
    .patchMerging 96 192,                     -- 56×56×96 → 28×28×192
    .swinStage 192 6 768 7 2,                 -- stage 2: 2 blocks
    .patchMerging 192 384,                    -- 28×28×192 → 14×14×384
    .swinStage 384 12 1536 7 6,               -- stage 3: 6 blocks (the deep one)
    .patchMerging 384 768,                    -- 14×14×384 → 7×7×768
    .swinStage 768 24 3072 7 2,               -- stage 4: 2 blocks
    .globalAvgPool,
    .dense 768 1000 .identity                 -- ImageNet-1k head
  ]

-- ════════════════════════════════════════════════════════════════
-- § Swin-S (small) — stage 3 blows up to 18 blocks (~50M)
-- ════════════════════════════════════════════════════════════════

def swinS : NetSpec where
  name := "Swin-S"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 96 4 3136,
    .swinStage 96 3 384 7 2,
    .patchMerging 96 192,
    .swinStage 192 6 768 7 2,
    .patchMerging 192 384,
    .swinStage 384 12 1536 7 18,              -- 18 blocks at stage 3
    .patchMerging 384 768,
    .swinStage 768 24 3072 7 2,
    .globalAvgPool,
    .dense 768 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Swin-B (base) — dim 128, same depth as Swin-S (~88M)
-- ════════════════════════════════════════════════════════════════

def swinB : NetSpec where
  name := "Swin-B"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 128 4 3136,
    .swinStage 128 4 512 7 2,
    .patchMerging 128 256,
    .swinStage 256 8 1024 7 2,
    .patchMerging 256 512,
    .swinStage 512 16 2048 7 18,
    .patchMerging 512 1024,
    .swinStage 1024 32 4096 7 2,
    .globalAvgPool,
    .dense 1024 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinySwin — 4 stages × 2 blocks, dim 32, bestiary fixture
-- ════════════════════════════════════════════════════════════════

def tinySwin : NetSpec where
  name := "tiny-Swin"
  imageH := 32
  imageW := 32
  layers := [
    .patchEmbed 3 32 2 256,                   -- 16×16 patches from a 32×32 input
    .swinStage 32 2 128 4 2,                  -- window size 4 for a smaller grid
    .patchMerging 32 64,
    .swinStage 64 4 256 4 2,
    .patchMerging 64 128,
    .swinStage 128 8 512 4 2,
    .patchMerging 128 256,
    .swinStage 256 16 1024 4 2,
    .globalAvgPool,
    .dense 256 10 .identity                    -- CIFAR-10 head
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
  IO.println "  Bestiary — Swin Transformer"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Hierarchical ViT with windowed attention + shifted windows."
  IO.println "  The cheap-but-global attention variant. Not trained here —"
  IO.println "  just the architecture, as NetSpec values."

  summarize swinT
  summarize swinS
  summarize swinB
  summarize tinySwin

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.swinStage` and `.patchMerging` are NEW Layer constructors"
  IO.println "    added for this bestiary. Codegen emits UNSUPPORTED for both;"
  IO.println "    implementing them would require windowed-MHSA + shifted-"
  IO.println "    window cyclic-shift + attention-masking kernels in StableHLO."
  IO.println "  • Within a swinStage, alternating blocks use W-MSA and SW-MSA"
  IO.println "    (shifted-window). That alternation is internal to the block;"
  IO.println "    NetSpec just says \"N Swin blocks at this resolution.\""
  IO.println "  • Param count is approximate; relative-position bias term is"
  IO.println "    included as `(2·ws-1)² · heads` per block."
  IO.println "  • Patch merging is the \"2x downsample + channel projection\""
  IO.println "    transformer-style. In a CNN it'd be a stride-2 conv."
