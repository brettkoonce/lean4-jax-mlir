import LeanMlir

/-! # ResNet — Bestiary entry (R-50 / R-101 / R-152)

ResNet (He et al.\ 2015,
[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) is the residual
revolution. Chapter 6 covers ResNet-34 (the basic-block variant —
two 3×3 convs per block + skip). This entry covers the **bottleneck
variants**: ResNet-50, -101, -152, where each block is

    1×1 conv (c → c/4)  →  3×3 conv (c/4 → c/4)  →  1×1 conv (c/4 → c)

with a residual skip. The 1×1 reduce/expand sandwich around a 3×3
keeps the parameter count manageable when going deeper, since the
expensive 3×3 conv runs at 1/4 the input channels.

Why three sizes? The paper's headline finding is that residual
connections let you keep training as you go deeper, well past the
point where a plain CNN would degrade. R-50 / R-101 / R-152 demonstrate
that on ImageNet: each successive depth tier adds 1-2 points of top-1
accuracy at proportional compute cost. The plateau eventually arrives
(R-200 is roughly equivalent to R-152), but the residual primitive
doesn't degrade gracefully — it *just keeps working*.

ResNet-34 is in Chapter 6 because it's the cleanest residual block to
follow through the math (no bottleneck). ResNet-50 onwards goes here
because the bottleneck is a tweak on the same residual primitive — same
chain rule, same fan-in, just three convs per block instead of two.

## Variants

| Name        | Stage block counts   | Params | ImageNet top-1 |
|-------------|----------------------|--------|---------------|
| ResNet-50   | (3, 4, 6, 3)         | 25.6M  | 76.0          |
| ResNet-101  | (3, 4, 23, 3)        | 44.5M  | 77.4          |
| ResNet-152  | (3, 8, 36, 3)        | 60.2M  | 78.3          |

All three share the same stem (7×7 stride-2 conv + 3×3 stride-2 max
pool), the same 4-stage layout (channels 256 → 512 → 1024 → 2048),
and the same GAP + single-FC head. They differ only in how many
bottleneck blocks each stage has.
-/

-- ════════════════════════════════════════════════════════════════
-- § ResNet-50 (paper canonical bottleneck variant)
-- ════════════════════════════════════════════════════════════════

def resNet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,                    -- stem 7×7 stride 2 → 112
    .maxPool 3 2,                              -- → 56
    .bottleneckBlock  64  256 3 1,             -- stage 1: 3 blocks @ 56
    .bottleneckBlock 256  512 4 2,             -- stage 2: 4 blocks @ 28 (stride 2 first)
    .bottleneckBlock 512 1024 6 2,             -- stage 3: 6 blocks @ 14
    .bottleneckBlock 1024 2048 3 2,            -- stage 4: 3 blocks @ 7
    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ResNet-101 (deeper)
-- ════════════════════════════════════════════════════════════════

def resNet101 : NetSpec where
  name := "ResNet-101"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .bottleneckBlock  64  256 3 1,
    .bottleneckBlock 256  512 4 2,
    .bottleneckBlock 512 1024 23 2,            -- stage 3 jumps to 23 blocks
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ResNet-152 (deepest standard variant)
-- ════════════════════════════════════════════════════════════════

def resNet152 : NetSpec where
  name := "ResNet-152"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .bottleneckBlock  64  256 3 1,
    .bottleneckBlock 256  512 8 2,             -- stage 2 → 8 blocks
    .bottleneckBlock 512 1024 36 2,            -- stage 3 → 36 blocks
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny ResNet (CIFAR fixture, basic blocks instead of bottleneck)
-- ════════════════════════════════════════════════════════════════

def tinyResNet : NetSpec where
  name := "tiny-ResNet (basic blocks)"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .residualBlock 16 16 2 1,
    .residualBlock 16 32 2 2,
    .residualBlock 32 64 2 2,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams}"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK (channel dims chain cleanly)"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — ResNet (R-50 / R-101 / R-152)"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bottleneck residual blocks (1×1 reduce → 3×3 → 1×1 expand)."
  IO.println "  ResNet-34 with basic blocks lives in Chapter 6 of the book."

  summarize resNet50
  summarize resNet101
  summarize resNet152
  summarize tinyResNet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Bottleneck = 3 convs per block; basic = 2. Same residual"
  IO.println "    chain rule applies to both. Bottleneck wins on params/depth"
  IO.println "    once you go past ResNet-50."
  IO.println "  • Final stage channels grow to 2048 in bottleneck variants"
  IO.println "    vs 512 in ResNet-34. The '4× channel growth across stages'"
  IO.println "    trick is bottleneck-specific."
  IO.println "  • Modern variants (ResNeXt, ResNeSt, RegNet) all preserve the"
  IO.println "    bottleneck residual structure; that's the load-bearing"
  IO.println "    architectural primitive of post-2015 vision."
