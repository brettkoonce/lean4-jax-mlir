import LeanMlir

/-! # BERT / RoBERTa — Bestiary entry

BERT (Devlin et al., 2018 — "Pre-training of Deep Bidirectional
Transformers for Language Understanding") is the paper that set the
encoder side of NLP on its modern course: masked-language-modeling
pretraining on a huge unlabeled corpus, then fine-tune on downstream
tasks. RoBERTa (Liu et al., 2019 — "A Robustly Optimized BERT
Pretraining Approach") is "same architecture, better training":
dynamic masking, more data, bigger batches, longer training, and
dropping the next-sentence-prediction objective.

Architecturally the two are nearly identical. All of the delta lives
in (a) vocabulary — RoBERTa uses a 50265-token BPE vocab vs BERT's
30522 WordPiece vocab, and (b) the disappearance of the segment
embedding in RoBERTa (NSP is gone). This bestiary entry covers both
under one file because the lesson is precisely that the big NLP
breakthroughs of 2018–2019 were training-side, not architectural.

## Anatomy

```
  Input: token IDs (L,) — L ≤ 512
       │
       ▼
  Token embedding (vocab → D) — by far the largest single param block
       + position embedding (512 × D), + segment embedding (2 × D, BERT only)
       + LayerNorm
       │
       ▼
  N × transformer encoder block:
       - MHSA (post-norm in BERT, same in RoBERTa)
       - Residual + LayerNorm
       - FFN (D → 4D → D), GELU
       - Residual + LayerNorm
       │
       ▼
  [CLS]-token pooler (Dense D → D, tanh) — used for classification
       │
       ▼
  Task head (classification: Dense D → nClasses;
             MLM:           Dense D → vocab, tied to token embedding)
```

## NetSpec shape caveat

We approximate the token embedding by a `.dense vocab → D` layer:
param count is the same (vocab × D + D ≈ vocab × D), shape semantics
are a well-known bestiary simplification (real embedding is a lookup
table; we stand in a dense matmul because the shape chain
`L → L × D` would require a non-linear spec anyway). The position
and segment embeddings are small (< 1% of the model) and we fold
them into the transformer-encoder block's LayerNorm budget.

## Variants

- `bertBase`      — 12 layers, dim 768,  heads 12, 110M (paper: 110M, exact)
- `bertLarge`     — 24 layers, dim 1024, heads 16, 335M (paper: 340M, -1.5%)
- `robertaBase`   — 12 layers, dim 768,  heads 12, 124M (paper: 125M, exact)
- `robertaLarge`  — 24 layers, dim 1024, heads 16, 355M (paper: 355M, exact)
- `tinyBERT`      — 4 layers, dim 128, heads 2, ~1M (CIFAR-ish fixture)
-/

-- ════════════════════════════════════════════════════════════════
-- § BERT-base: 12 × 768, 110M params
-- ════════════════════════════════════════════════════════════════
-- vocab = 30522 (WordPiece), max context = 512 tokens.

def bertBase : NetSpec where
  name := "BERT-base"
  imageH := 512       -- max context length
  imageW := 1         -- unused; sequence is 1D
  layers := [
    -- Token embedding (vocab → hidden). Stands in for the lookup
    -- table + position + segment embeddings; param count matches
    -- within 1% because position/segment are small.
    .dense 30522 768 .identity,
    -- 12 layers of (MHSA + FFN), post-norm, GELU. mlpDim = 4·D.
    .transformerEncoder 768 12 3072 12,
    -- [CLS] pooler: project and tanh. The pooled output goes to any
    -- task-specific head (NSP, GLUE classification, …).
    .dense 768 768 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § BERT-large: 24 × 1024, 340M params
-- ════════════════════════════════════════════════════════════════

def bertLarge : NetSpec where
  name := "BERT-large"
  imageH := 512
  imageW := 1
  layers := [
    .dense 30522 1024 .identity,
    .transformerEncoder 1024 16 4096 24,
    .dense 1024 1024 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § RoBERTa-base: 12 × 768, 125M (BPE 50265 vocab swells embedding)
-- ════════════════════════════════════════════════════════════════

def robertaBase : NetSpec where
  name := "RoBERTa-base"
  imageH := 512
  imageW := 1
  layers := [
    -- BPE vocab is 50265 tokens; this is where the param-count delta
    -- over BERT-base lives (≈15M extra in the embedding table alone).
    .dense 50265 768 .identity,
    .transformerEncoder 768 12 3072 12,
    .dense 768 768 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § RoBERTa-large: 24 × 1024, 355M params
-- ════════════════════════════════════════════════════════════════

def robertaLarge : NetSpec where
  name := "RoBERTa-large"
  imageH := 512
  imageW := 1
  layers := [
    .dense 50265 1024 .identity,
    .transformerEncoder 1024 16 4096 24,
    .dense 1024 1024 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyBERT fixture — ~1M params, for quick pedagogy
-- ════════════════════════════════════════════════════════════════

def tinyBERT : NetSpec where
  name := "tiny-BERT"
  imageH := 128       -- short context
  imageW := 1
  layers := [
    .dense 1000 128 .identity,            -- tiny vocab
    .transformerEncoder 128 2 512 4,      -- 4 layers, dim 128, 2 heads
    .dense 128 128 .identity              -- pooler
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
  IO.println "  Bestiary — BERT / RoBERTa"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The encoder-only transformer family. Same architecture for"
  IO.println "  BERT and RoBERTa — the delta was better pretraining, not"
  IO.println "  better layers."

  summarize bertBase
  summarize bertLarge
  summarize robertaBase
  summarize robertaLarge
  summarize tinyBERT

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Zero new Layer primitives. The encoder stack is the same"
  IO.println "    .transformerEncoder used by ViT and DETR — the NLP/vision"
  IO.println "    convergence happened years ago at the kernel level, even if"
  IO.println "    the communities didn't admit it until ~2020."
  IO.println "  • Token embedding is approximated by .dense vocab→D. Param"
  IO.println "    count is faithful (within 1%); shape semantics cheat"
  IO.println "    because a linear NetSpec can't express the L → L×D lookup"
  IO.println "    non-linearity. See Mamba.lean for the same simplification."
  IO.println "  • RoBERTa = BERT architecturally. All of RoBERTa's gains"
  IO.println "    came from: (1) BPE vs WordPiece tokenizer, (2) dynamic"
  IO.println "    masking, (3) no NSP task, (4) 10× more data, (5) bigger"
  IO.println "    batches, (6) longer training. None of that lives in the"
  IO.println "    NetSpec — it's all training-procedure."
  IO.println "  • Decoder-only cousins (GPT) use the same transformer block"
  IO.println "    with a causal mask; architecturally they're a handful of"
  IO.println "    lines different. BERT-the-architecture is the scaffolding;"
  IO.println "    what you pretrain it on is what makes it do the task."
