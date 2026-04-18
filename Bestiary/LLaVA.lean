import LeanMlir

/-! # LLaVA — Bestiary entry

LLaVA (Liu, Li, Wu, Lee, 2023 — "Visual Instruction Tuning") is the
reference open-source vision-language model. It is also the cleanest
exhibit of the pattern behind essentially every modern VLM:

**Freeze a pretrained vision encoder. Freeze a pretrained LLM.
Train a small adapter between them.**

That's the entire architectural contribution. The innovations that
made LLaVA matter were: (a) showing a single-linear-layer projector
is enough to get useful vision-language behavior, (b) the
instruction-tuning data construction that used GPT-4 to generate
image-grounded instruction examples, and (c) the two-stage training
recipe (pretrain the projector only, then fine-tune projector + LLM
on instructions).

None of those are layers. All the "visual instruction tuning" magic
lives in the data and the training procedure.

## Anatomy of LLaVA-1.5

```
  Image (336×336×3)                   Text prompt
         │                                  │
         ▼                                  │
  CLIP ViT-L/14 @ 336px                     │
  (24×24 = 576 patch tokens, dim=1024)      │
         │                                  │
         ▼  MLP projector                   │
  (576 tokens, dim=4096)                    │
         │                                  │
         └────────── prepend ───────┬───────┘
                                    ▼
                            Vicuna / LLaMA LM
                            (frozen during stage 1,
                             fine-tuned in stage 2)
                                    │
                                    ▼
                             generated response
```

Numbers that matter: for LLaVA-1.5 7B the total is $\sim$7.0B
parameters, of which:

- $\sim$304M in the vision encoder (frozen CLIP ViT-L),
- $\sim$21M in the MLP projector (trained), and
- $\sim$6.7B in the language model (fine-tuned in stage 2).

The \textbf{projector is 0.3\%} of the total. That ratio is the
whole story.

## LLaVA-1 vs LLaVA-1.5 projector

- **LLaVA-1** used a \emph{single} linear layer
  $1024 \to 4096$ as the projector. Roughly 4M parameters.
- **LLaVA-1.5** bumped that to a 2-layer MLP with GELU
  ($1024 \to 4096 \to 4096$). Roughly 21M parameters. This single
  change accounted for a meaningful jump in instruction-following
  quality; it's one of those moments where a trivially small
  architectural change mattered disproportionately.

## Variants (shown as separate components)

- `llavaVisionEncoder`   — CLIP ViT-L/14 @ 336px (shared across versions)
- `llava1Projector`      — LLaVA-1 single-linear projector
- `llava15Projector`     — LLaVA-1.5 two-layer MLP projector
- `llavaLLM7B`           — LLaMA-7B / Vicuna-7B decoder
- `llavaLLM13B`          — LLaMA-13B / Vicuna-13B decoder
- `tinyLlava`            — end-to-end fixture

## NetSpec simplifications

- Vision encoder shown at LLaVA-1.5's 336px; LLaVA-1 used 224px (same
  architecture, fewer patches).
- LLM is expressed via \texttt{.transformerEncoder}. Real LLaMA uses
  SwiGLU (3 linear projections in the FFN) and RMSNorm instead of
  LayerNorm; our \texttt{.transformerEncoder} uses the standard
  2-projection FFN + LayerNorm. \textbf{Consequence: our LLM param
  counts undershoot real LLaMA by about 20--25\%} (5.2B vs 6.7B at
  7B; 10.0B vs 13.0B at 13B). The spec still shows the right depth,
  width, and head count --- the gap is in the FFN's internal wiring.
- Causal mask for the LLM is a training-time detail, not a parameter.
-/

-- ════════════════════════════════════════════════════════════════
-- § Vision encoder: CLIP ViT-L/14 @ 336px (shared across LLaVA versions)
-- ════════════════════════════════════════════════════════════════
-- 336/14 = 24 patches per side, so 576 image tokens fed into the LM.

def llavaVisionEncoder : NetSpec where
  name := "LLaVA vision encoder (CLIP ViT-L/14 @ 336px)"
  imageH := 336
  imageW := 336
  layers := [
    .patchEmbed 3 1024 14 576,
    .transformerEncoder 1024 16 4096 24
  ]

-- ════════════════════════════════════════════════════════════════
-- § LLaVA-1 projector: single linear layer 1024 → 4096
-- ════════════════════════════════════════════════════════════════
-- ~4M params. Shockingly, enough to get meaningful vision-language
-- behavior in the original LLaVA paper. Followed later by LLaVA-1.5's
-- 2-layer MLP which added another ~17M and meaningfully improved
-- instruction following.

def llava1Projector : NetSpec where
  name := "LLaVA-1 projector (single linear)"
  imageH := 576       -- visual token count
  imageW := 1
  layers := [
    .dense 1024 4096 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § LLaVA-1.5 projector: 2-layer MLP with GELU
-- ════════════════════════════════════════════════════════════════

def llava15Projector : NetSpec where
  name := "LLaVA-1.5 projector (2-layer MLP)"
  imageH := 576
  imageW := 1
  layers := [
    .dense 1024 4096 .identity,    -- GELU in between (activation is
    .dense 4096 4096 .identity     -- zero-param; we model as identity)
  ]

-- ════════════════════════════════════════════════════════════════
-- § LLaVA 7B language model: Vicuna / LLaMA-7B decoder
-- ════════════════════════════════════════════════════════════════
-- 32 layers, dim 4096, heads 32, mlp 11008, vocab 32000. Standard
-- TransformerEncoder stand-in undercounts by ~23% vs real LLaMA
-- because SwiGLU has 3 FFN projections where our block has 2.

def llavaLLM7B : NetSpec where
  name := "LLaVA 7B LM (LLaMA-7B backbone)"
  imageH := 2048       -- context length (LLaVA-1.5 uses 2048 / 4096)
  imageW := 1
  layers := [
    -- Text-token embedding (vocab → dim). Tied to the LM head output,
    -- same convention as GPT: one matrix, used at both ends.
    .dense 32000 4096 .identity,
    .transformerEncoder 4096 32 11008 32
  ]

-- ════════════════════════════════════════════════════════════════
-- § LLaVA 13B language model: Vicuna / LLaMA-13B decoder
-- ════════════════════════════════════════════════════════════════
-- 40 layers, dim 5120, heads 40, mlp 13824, vocab 32000.

def llavaLLM13B : NetSpec where
  name := "LLaVA 13B LM (LLaMA-13B backbone)"
  imageH := 2048
  imageW := 1
  layers := [
    .dense 32000 5120 .identity,
    .transformerEncoder 5120 40 13824 40
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyLlava — tiny end-to-end fixture (encoder + projector + LM)
-- ════════════════════════════════════════════════════════════════
-- Compressed into one NetSpec for convenience; real LLaVA would be
-- three separate networks with the projector connecting them.

def tinyLlava : NetSpec where
  name := "tiny-LLaVA"
  imageH := 112
  imageW := 112
  layers := [
    .patchEmbed 3 128 14 64,           -- tiny vision encoder
    .transformerEncoder 128 4 512 4,
    .dense 128 256 .identity,          -- projector into LM dim
    .transformerEncoder 256 4 1024 4   -- tiny LM
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
  IO.println "  Bestiary — LLaVA"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Frozen CLIP ViT + tiny MLP projector + (mostly) frozen LLaMA."
  IO.println "  Every modern open-source VLM demo is a LLaVA descendant."

  summarize llavaVisionEncoder
  summarize llava1Projector
  summarize llava15Projector
  summarize llavaLLM7B
  summarize llavaLLM13B
  summarize tinyLlava

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Vision encoder is ViT kit,"
  IO.println "    projector is plain .dense, LM is .transformerEncoder."
  IO.println "  • The projector is 0.3% of total model params for LLaVA-1.5 7B"
  IO.println "    (21M out of ~7B). Pre-training does the work; the tiny"
  IO.println "    adapter just aligns modalities."
  IO.println "  • LLaMA-7B / 13B specs here undercount by ~23%: real LLaMA"
  IO.println "    uses SwiGLU (3 FFN projections) + RMSNorm, our"
  IO.println "    .transformerEncoder uses a 2-projection FFN + LayerNorm."
  IO.println "    Depth / width / heads still match exactly."
  IO.println "  • Two-stage training (projector-only pretrain, then joint"
  IO.println "    fine-tune) lives entirely in the training procedure, not"
  IO.println "    in the architecture."
  IO.println "  • LLaVA-NeXT / LLaVA-1.6 adds image-splitting for higher res"
  IO.println "    (AnyRes); same components, different preprocessing."
  IO.println "  • The pattern generalizes: BLIP-2 replaces the MLP with a"
  IO.println "    Q-Former, Flamingo replaces it with a Perceiver resampler"
  IO.println "    + gated xattn-dense, but the frozen-backbone-plus-adapter"
  IO.println "    template is the same."
