import LeanMlir

/-! # WRN (Wide ResNet) — Bestiary entry

WRN (Zagoruyko & Komodakis 2016,
[arXiv:1605.07146](https://arxiv.org/abs/1605.07146)) is the
"don't go deeper, go wider" answer to ResNet. Same residual block,
same chain rule, multiply every channel count by a widening factor
`k` and trim the depth. The paper's headline finding: WRN-28-10
(28 layers, k=10) **matches** ResNet-1001 (1001 layers, no
widening) on CIFAR-10/100 at half the training time and a quarter
the parameters. Width turned out to be a more efficient axis to
scale than depth for the ResNet family.

The architecture is exactly Chapter 6's ResNet-34 with two knobs
turned: depth pulled in (40 or 28 layers instead of 34), channels
multiplied by k (10× the original 16-32-64 widths). No new
primitive — `Layer.residualBlock` is the same one Ch 6 proves.
The widening factor is a per-instance parameter on the existing
constructor.

## WRN-N-k notation

- `N` = depth (must be `6n + 4` for some `n`; 28, 40, 22 etc.).
- `k` = widening factor; channel widths become 16, 16k, 32k, 64k
  across the 3 stages.

For CIFAR (32×32 input):
- Stem: 3 → 16 channels (3×3 conv).
- 3 stages of `(N-4)/6` basic residual blocks each.
  - Stage 1: 16 → 16k channels, stride 1.
  - Stage 2: 16k → 32k channels, stride 2.
  - Stage 3: 32k → 64k channels, stride 2.
- GAP → dense(64k → 10).

## Variants

| Name      | Depth | k  | Params  | CIFAR-10 |
|-----------|-------|----|---------|----------|
| WRN-28-10 | 28    | 10 | 36.5M   | 96.0%    |
| WRN-40-2  | 40    | 2  |  2.2M   | 95.4%    |
| WRN-22-8  | 22    | 8  | 17.2M   | 95.7%    |

We pack the canonical CIFAR variants here plus a fixture. ImageNet-
style WRN (WRN-50-2, WRN-101-2) uses bottleneck blocks instead;
those are a one-line variant on `.bottleneckBlock` and aren't worth
their own entry given Ch 6 already covers the bottleneck path.
-/

-- ════════════════════════════════════════════════════════════════
-- § WRN-28-10 (paper canonical, CIFAR)
-- ════════════════════════════════════════════════════════════════

/-- WRN-28-10 on CIFAR-10. Depth 28 = stem (1 conv) + 3 stages × 4
    blocks × 2 convs + final BN-ReLU-FC head. k=10 widens the
    16-32-64 channel sequence to 160-320-640. -/
def wrn28_10 : NetSpec where
  name := "WRN-28-10"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,                   -- stem
    .residualBlock 16 160 4 1,                -- stage 1: 4 blocks @ 32×32
    .residualBlock 160 320 4 2,               -- stage 2: 4 blocks @ 16×16
    .residualBlock 320 640 4 2,               -- stage 3: 4 blocks @ 8×8
    .globalAvgPool,
    .dense 640 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § WRN-40-2 (lighter, CIFAR)
-- ════════════════════════════════════════════════════════════════

/-- WRN-40-2 on CIFAR. Depth 40 = 6 blocks per stage; k=2 widens
    to 32-64-128 channels. ~2.2M params, 95.4% on CIFAR-10. -/
def wrn40_2 : NetSpec where
  name := "WRN-40-2"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .residualBlock 16 32 6 1,
    .residualBlock 32 64 6 2,
    .residualBlock 64 128 6 2,
    .globalAvgPool,
    .dense 128 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § WRN-22-8 (mid-tier, CIFAR)
-- ════════════════════════════════════════════════════════════════

/-- WRN-22-8 on CIFAR. Depth 22 = 3 blocks per stage; k=8 widens
    to 128-256-512 channels. -/
def wrn22_8 : NetSpec where
  name := "WRN-22-8"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .residualBlock 16 128 3 1,
    .residualBlock 128 256 3 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny WRN (fixture)
-- ════════════════════════════════════════════════════════════════

def tinyWrn : NetSpec where
  name := "tiny-WRN (10-2)"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .residualBlock 16 32 1 1,
    .residualBlock 32 64 1 2,
    .residualBlock 64 128 1 2,
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
  IO.println s!"  params      : {spec.totalParams}"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK (channel dims chain cleanly)"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — WRN (Wide ResNet)"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Same .residualBlock as ResNet-34 in Chapter 6, just"
  IO.println "  widened. Wider-not-deeper turns out to be the more"
  IO.println "  efficient ResNet scaling axis."

  summarize wrn28_10
  summarize wrn40_2
  summarize wrn22_8
  summarize tinyWrn

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • WRN-N-k uses N=6n+4 to balance the 3-stage design."
  IO.println "  • The paper also adds dropout between the two convs in each"
  IO.println "    block; we omit that here (static-graph scope, no per-step"
  IO.println "    masks). The dropout-free variant is also reported in the"
  IO.println "    paper and lands within ~0.3% of the dropout one on CIFAR."
  IO.println "  • For ImageNet WRN, bottleneck-block variants (WRN-50-2,"
  IO.println "    WRN-101-2) live one .bottleneckBlock-channel-tweak away"
  IO.println "    from the ResNet bestiary entry."
