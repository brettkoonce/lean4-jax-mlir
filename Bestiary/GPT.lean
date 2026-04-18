import LeanMlir

/-! # GPT-1 / GPT-2 — Bestiary entry

GPT-1 (Radford et al., 2018 — "Improving Language Understanding by
Generative Pre-Training") and GPT-2 (Radford et al., 2019 — "Language
Models are Unsupervised Multitask Learners") are the decoder-only
counterparts to BERT. Same transformer block, flipped role: GPT
predicts the next token given all previous tokens (causal
self-attention), BERT fills in a masked token given both left and
right context (bidirectional attention).

Once you've seen BERT, GPT is a two-line delta at the architecture
level:

- **Causal mask.** Self-attention can only attend backwards in the
  sequence. Implemented as an upper-triangular additive mask of
  $-\infty$ on the attention scores. Not a parameter; a training-time
  mask.
- **No pooler, no CLS token.** GPT pools by "take the last token's
  hidden state" rather than by running a [CLS] head. Nothing lives
  between the transformer stack and the LM head.

GPT-2 also moves LayerNorm to the pre-attention / pre-FFN position
(pre-norm) instead of after (post-norm, BERT's choice). Pre-norm
trains deeper stacks more stably; zero parameter difference.

## Weight tying (Press & Wolf 2017)

GPT ties the output projection to the input token embedding:
`logits = hidden @ W_embed^T`. The `vocab × dim` matrix is used
twice and counted once in the param total. Our NetSpec approximates
token embedding via `.dense vocab → D`; since GPT ties the LM head,
no extra `.dense D → vocab` is needed --- the embedding already
pays for both. This is the main reason GPT-2 small lands at 124M
rather than 163M.

## Karpathy's nanoGPT

If you want the minimal working instance of everything in this file,
Andrej Karpathy's `nanoGPT` is ~300 lines of PyTorch that trains a
GPT-2 small on a single GPU. It's the reference implementation most
practitioners mentally cite when they say ``GPT''. Every line in
nanoGPT corresponds to something already covered by Part 1's VJP
primitives; this entry exists to make that explicit.

## Variants

- `gpt1`        — 12 layers, dim 768,  heads 12, vocab 40478, ctx  512  → 117M (paper: 117M)
- `gpt2Small`   — 12 layers, dim 768,  heads 12, vocab 50257, ctx 1024  → 124M (paper: 124M)
- `gpt2Medium`  — 24 layers, dim 1024, heads 16                         → 354M (paper: 345M)
- `gpt2Large`   — 36 layers, dim 1280, heads 20                         → 773M (paper: 774M)
- `gpt2XL`      — 48 layers, dim 1600, heads 25                         → 1.56B (paper: 1.5B)
- `tinyGPT`     — 4  layers, dim 128,  heads 2, vocab 1000              → ~1M fixture
-/

-- ════════════════════════════════════════════════════════════════
-- § GPT-1: 12 × 768, 117M params (the "first GPT paper")
-- ════════════════════════════════════════════════════════════════
-- vocab = 40478 (BPE), ctx = 512

def gpt1 : NetSpec where
  name := "GPT-1"
  imageH := 512       -- context length
  imageW := 1
  layers := [
    -- Token embedding (vocab → D). Also serves as the (tied) LM head
    -- projection at the output, so no final D→vocab dense is needed.
    .dense 40478 768 .identity,
    -- 12 decoder blocks, post-norm (GPT-1 used post-norm before GPT-2
    -- switched to pre-norm). Same parameter count either way.
    .transformerEncoder 768 12 3072 12
  ]

-- ════════════════════════════════════════════════════════════════
-- § GPT-2 Small: 12 × 768, 124M — the nanoGPT target
-- ════════════════════════════════════════════════════════════════

def gpt2Small : NetSpec where
  name := "GPT-2 small"
  imageH := 1024
  imageW := 1
  layers := [
    .dense 50257 768 .identity,
    .transformerEncoder 768 12 3072 12
  ]

-- ════════════════════════════════════════════════════════════════
-- § GPT-2 Medium: 24 × 1024, 350M
-- ════════════════════════════════════════════════════════════════

def gpt2Medium : NetSpec where
  name := "GPT-2 medium"
  imageH := 1024
  imageW := 1
  layers := [
    .dense 50257 1024 .identity,
    .transformerEncoder 1024 16 4096 24
  ]

-- ════════════════════════════════════════════════════════════════
-- § GPT-2 Large: 36 × 1280, 774M
-- ════════════════════════════════════════════════════════════════

def gpt2Large : NetSpec where
  name := "GPT-2 large"
  imageH := 1024
  imageW := 1
  layers := [
    .dense 50257 1280 .identity,
    .transformerEncoder 1280 20 5120 36
  ]

-- ════════════════════════════════════════════════════════════════
-- § GPT-2 XL: 48 × 1600, 1.5B (the largest variant)
-- ════════════════════════════════════════════════════════════════

def gpt2XL : NetSpec where
  name := "GPT-2 XL"
  imageH := 1024
  imageW := 1
  layers := [
    .dense 50257 1600 .identity,
    .transformerEncoder 1600 25 6400 48
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyGPT fixture — ~1M params for quick inspection
-- ════════════════════════════════════════════════════════════════

def tinyGPT : NetSpec where
  name := "tiny-GPT"
  imageH := 128
  imageW := 1
  layers := [
    .dense 1000 128 .identity,
    .transformerEncoder 128 2 512 4
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  context     : {spec.imageH} tokens"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — GPT-1 / GPT-2"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Decoder-only transformer family. BERT with a causal mask and"
  IO.println "  no pooler. Karpathy's nanoGPT is the 300-line reference."

  summarize gpt1
  summarize gpt2Small
  summarize gpt2Medium
  summarize gpt2Large
  summarize gpt2XL
  summarize tinyGPT

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Zero new Layer primitives. The encoder stack is the same"
  IO.println "    .transformerEncoder used by BERT, ViT, and DETR — the"
  IO.println "    differences between encoder-only and decoder-only models"
  IO.println "    live in masking and training objective, not layer shapes."
  IO.println "  • Weight tying: GPT's LM head reuses the token-embedding"
  IO.println "    matrix (transposed). Our .dense vocab→D stand-in counts"
  IO.println "    this once, matching the real tied-weights param budget."
  IO.println "  • GPT-1 predates BPE; GPT-2 introduced the 50257-token BPE"
  IO.println "    vocab reused by nanoGPT, GPT-3, and basically everything"
  IO.println "    since."
  IO.println "  • For Karpathy-style single-GPU training, start from GPT-2"
  IO.println "    small (124M). It's small enough to train from scratch on"
  IO.println "    a 24GB consumer card and large enough to produce coherent"
  IO.println "    text after a few epochs on reasonable corpora."
