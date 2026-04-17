import LeanMlir

/-! # AlphaZero — Bestiary entry

AlphaZero (Silver et al., 2018) is a self-play reinforcement-learning system
that plays Go, chess, and shogi from scratch. At its heart is a very
simple neural network: a stack of residual blocks feeding **two heads** —
one predicting the move probability distribution (policy), one predicting
the expected game outcome (value).

The body is "just a ResNet." The novelty is the self-play training loop,
not the architecture. From a `NetSpec` perspective, AlphaZero is exactly
the ResNet body from Chapter-N, forked at the end into two tiny heads.

```
           Input planes (board state encoding)
                       │
                       ▼
          ┌─────────────────────────┐
          │  convBn: ic → 256, 3×3  │
          │                         │
          │    residualBlock × N    │   ← "tower"
          │      256 → 256, s=1     │      (N = 19 for original AlphaGo Zero,
          │                         │       40 for AlphaZero chess/shogi)
          └──────────┬──────────────┘
                     │
          ┌──────────┴──────────────┐
          ▼                         ▼
  Policy head (2 filters)    Value head (1 filter)
   → flatten                  → flatten
   → dense(2·H·W → nMoves)    → dense(H·W → 256, ReLU)
                              → dense(256 → 1)
                              → tanh   (not in Activation enum yet;
                                        identity here + externally
                                        applied, same as the book's LN
                                        γ/β scalar simplification)
```

Our `NetSpec` is a linear list of layers, so we represent the two heads
as two independent specs sharing the body conceptually. A reader who
wants to think "what does AlphaZero look like in Lean" sees **both**
branches laid out — the shared body in each, the heads diverging only at
the last few layers. In the codegen, sharing the body parameters is an
orchestration concern; the pure shape/architecture story is the two
specs below.

## Variants

- `alphaGoZeroPolicy` / `alphaGoZeroValue` — the original 19-block /
  256-channel / 19×19 board / 17-plane Go network (~23M params).
- `alphaZeroChessPolicy` / `alphaZeroChessValue` — the 40-block /
  256-channel / 8×8 board / 119-plane chess network (~46M params).
- `tinyAlphaZeroPolicy` / `tinyAlphaZeroValue` — a scale-model with 3
  blocks for quick inspection / testing. Useful as a fixture.

All are pure `NetSpec` values; no training runs here. The book's Part 2
(Bestiary) uses these as read-only examples of architecture idioms.
-/

-- ════════════════════════════════════════════════════════════════
-- § AlphaGo Zero (original, Go board 19×19, 17 input planes)
-- ════════════════════════════════════════════════════════════════

/-- Policy branch: outputs 19·19 + 1 = 362 move probabilities (361 board
    positions + 1 pass). -/
def alphaGoZeroPolicy : NetSpec where
  name := "AlphaGo Zero (policy head)"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 17 256 3 1 .same,                        -- 17 planes → 256
    .residualBlock 256 256 19 1,                     -- 19 residual blocks
    .conv2d 256 2 1 .same .identity,                 -- policy head: 2 filters
    .flatten,
    .dense (2 * 19 * 19) 362 .identity               -- 361 moves + pass
  ]

/-- Value branch: outputs a single scalar (expected outcome, pre-tanh). -/
def alphaGoZeroValue : NetSpec where
  name := "AlphaGo Zero (value head)"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 17 256 3 1 .same,
    .residualBlock 256 256 19 1,
    .conv2d 256 1 1 .same .identity,                 -- value head: 1 filter
    .flatten,
    .dense (1 * 19 * 19) 256 .relu,
    .dense 256 1 .identity                            -- tanh applied downstream
  ]

-- ════════════════════════════════════════════════════════════════
-- § AlphaZero chess (8×8 board, 119 input planes, 40 blocks)
-- ════════════════════════════════════════════════════════════════

/-- Policy: 73 × 8 × 8 = 4672 move encodings (AlphaZero's dense move rep). -/
def alphaZeroChessPolicy : NetSpec where
  name := "AlphaZero chess (policy head)"
  imageH := 8
  imageW := 8
  layers := [
    .convBn 119 256 3 1 .same,
    .residualBlock 256 256 40 1,
    .conv2d 256 73 1 .same .identity,
    .flatten,
    .dense (73 * 8 * 8) (73 * 8 * 8) .identity
  ]

def alphaZeroChessValue : NetSpec where
  name := "AlphaZero chess (value head)"
  imageH := 8
  imageW := 8
  layers := [
    .convBn 119 256 3 1 .same,
    .residualBlock 256 256 40 1,
    .conv2d 256 1 1 .same .identity,
    .flatten,
    .dense (1 * 8 * 8) 256 .relu,
    .dense 256 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny AlphaZero (fixture for testing / small-scale pedagogy)
-- ════════════════════════════════════════════════════════════════

/-- 3 residual blocks, 64 channels, 9×9 tiny-Go-style board. -/
def tinyAlphaZeroPolicy : NetSpec where
  name := "Tiny AlphaZero (policy head)"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 17 64 3 1 .same,
    .residualBlock 64 64 3 1,
    .conv2d 64 2 1 .same .identity,
    .flatten,
    .dense (2 * 9 * 9) 82 .identity
  ]

def tinyAlphaZeroValue : NetSpec where
  name := "Tiny AlphaZero (value head)"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 17 64 3 1 .same,
    .residualBlock 64 64 3 1,
    .conv2d 64 1 1 .same .identity,
    .flatten,
    .dense (1 * 9 * 9) 64 .relu,
    .dense 64 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary of every Bestiary entry in this file.
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
  IO.println "  Bestiary — AlphaZero"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Two-headed network: shared residual body + policy / value heads."
  IO.println "  Not trained here — just the architecture, as NetSpec values."

  summarize alphaGoZeroPolicy
  summarize alphaGoZeroValue
  summarize alphaZeroChessPolicy
  summarize alphaZeroChessValue
  summarize tinyAlphaZeroPolicy
  summarize tinyAlphaZeroValue

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Policy and value heads share the first two layers (body),"
  IO.println "    which would share parameters in a real training run. NetSpec"
  IO.println "    as a linear list can't express that sharing; we show the"
  IO.println "    two forks as separate specs."
  IO.println "  • Value head ends with dense(→1, identity). The original paper"
  IO.println "    applies tanh afterwards; `Activation` doesn't include tanh"
  IO.println "    yet — same book-simplification stance as LN's scalar γ/β."
  IO.println "  • Chess policy dense is `73·8·8 → 73·8·8 = identity` — the"
  IO.println "    network's output channels already encode every move class"
  IO.println "    directly (no further projection needed)."
