import LeanMlir

/-! # VGG — Bestiary entry

VGG (Simonyan & Zisserman 2014,
[arXiv:1409.1556](https://arxiv.org/abs/1409.1556)) was the deep-CNN
era's reference architecture for two years between AlexNet (2012) and
ResNet (2015). Stack 3×3 convs separated by 2×2 max pools, dump into
three fully-connected layers. Nothing fancier — but the "small kernel,
deep stack" recipe stuck.

Two same-shape 3×3 convs in sequence cover the same receptive field as
one 5×5 conv but with fewer parameters (`2·9·c² < 25·c²`) and an extra
nonlinearity in the middle. Three 3×3s ≈ one 7×7 with the same
receptive-field math. VGG's whole design philosophy is "instead of
choosing kernel sizes, just always use 3×3 and add depth."

The downside: 138M params for VGG-16, of which **~80%** live in the
first FC layer (`7×7×512 → 4096`). Pre-GAP, pre-modern-classifier-head
era. This is precisely the dense-head-bloat that GAP + single-FC heads
fixed in ResNet onward.

## Variants

The five paper variants differ only in conv-stack depth at each
spatial resolution:

| Name   | Configuration                              | Params |
|--------|--------------------------------------------|--------|
| VGG-11 | (1, 1, 2, 2, 2) convs per stage            | 133M   |
| VGG-13 | (2, 2, 2, 2, 2)                            | 133M   |
| VGG-16 | (2, 2, 3, 3, 3) — paper canonical          | 138M   |
| VGG-19 | (2, 2, 4, 4, 4)                            | 144M   |

VGG-11 has only one conv per early stage; VGG-19 has four convs in the
deeper stages. All five share the same FC head (4096 → 4096 → 1000),
which is where most of the params live.

We model VGG-16 (canonical) and VGG-19 here, plus a tiny CIFAR fixture.
-/

-- ════════════════════════════════════════════════════════════════
-- § VGG-16 (paper canonical, ~138M params)
-- ════════════════════════════════════════════════════════════════

def vgg16 : NetSpec where
  name := "VGG-16"
  imageH := 224
  imageW := 224
  layers := [
    -- stage 1: 64 channels, 224×224
    .conv2d 3 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,                                  -- → 112×112
    -- stage 2: 128 channels
    .conv2d 64 128 3 .same .relu,
    .conv2d 128 128 3 .same .relu,
    .maxPool 2 2,                                  -- → 56×56
    -- stage 3: 256 channels, 3 convs
    .conv2d 128 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .maxPool 2 2,                                  -- → 28×28
    -- stage 4: 512 channels, 3 convs
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .maxPool 2 2,                                  -- → 14×14
    -- stage 5: 512 channels, 3 convs
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .maxPool 2 2,                                  -- → 7×7
    -- FC head — where 80% of the params live
    .flatten,
    .dense (7 * 7 * 512) 4096 .relu,
    .dense 4096 4096 .relu,
    .dense 4096 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § VGG-19 (deeper, ~144M params)
-- ════════════════════════════════════════════════════════════════

def vgg19 : NetSpec where
  name := "VGG-19"
  imageH := 224
  imageW := 224
  layers := [
    .conv2d 3 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .conv2d 128 128 3 .same .relu,
    .maxPool 2 2,
    -- stage 3: 4 convs
    .conv2d 128 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .maxPool 2 2,
    -- stage 4: 4 convs
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .maxPool 2 2,
    -- stage 5: 4 convs
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .conv2d 512 512 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense (7 * 7 * 512) 4096 .relu,
    .dense 4096 4096 .relu,
    .dense 4096 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny VGG (CIFAR-sized fixture)
-- ════════════════════════════════════════════════════════════════

def tinyVgg : NetSpec where
  name := "tiny-VGG"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d 3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense (8 * 8 * 64) 256 .relu,
    .dense 256 10 .identity
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
  IO.println "  Bestiary — VGG"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Deep stacks of 3×3 convs + max pool + heavy FC head."
  IO.println "  The dense-head era; the FC layers hold ~80% of params."

  summarize vgg16
  summarize vgg19
  summarize tinyVgg

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • VGG predates BatchNorm (2015) and residual connections (2015)."
  IO.println "    Trained with SGD + momentum + weight decay + dropout."
  IO.println "  • The first FC (7×7×512 → 4096 = 102M params) dominates."
  IO.println "    Every modern CNN uses GAP + single-FC instead, an idea that"
  IO.println "    arrived with NIN (Lin 2013) but only became standard with"
  IO.println "    ResNet (He 2015)."
  IO.println "  • VGG-16/19 are still cited today mainly as feature extractors"
  IO.println "    (perceptual loss, style transfer) — those FC features turn"
  IO.println "    out to capture useful image structure."
