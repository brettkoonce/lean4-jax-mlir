import LeanMlir

/-! # Mamba — Bestiary entry 🐍

Mamba (Gu & Dao, 2023) is a **selective state-space model** — a linear-time
alternative to attention for long-context sequence modeling. Where a
transformer computes `softmax(QKᵀ)V` at `O(L²)` cost in sequence length,
Mamba runs a **hardware-aware selective scan** at `O(L)`. It's the cleanest
recent win against attention for language modeling at ≥ 1B params.

The neat trick: state-space models (SSMs) in the `S6` formulation let the
state-transition dynamics be **input-dependent** (selective), which recovers
attention-like gating while keeping the recurrent / convolutional efficiency.

## Anatomy of one Mamba block

```
        x : (L, D)                    ← sequence of L tokens, each dim D
            │
    ┌───────┴───────┐
    │               │
RMSNorm             │                  ← residual path taps in before norm
    │               │
Linear(D → 2ED)     │                  ← "input projection", expand by factor E
    │               │
split → (u, z)      │                  ← u is the main path, z is the gate
    │               │
Conv1D(kernel=4)    │                  ← depthwise short-range token mixing
    │               │
SiLU                │                  ← swish activation
    │               │
SSM selective scan  │                  ← ⭐ the one novel primitive ⭐
    │               │                   (state dim N, input-dependent Δ, B, C)
u * SiLU(z)         │                  ← elementwise gate
    │               │
Linear(ED → D)      │                  ← output projection back to D
    │               │
    └──────┬────────┘
           ▼
      x + residual                     ← add back the skip
```

## Why `.mambaBlock` is a single Layer

Every sub-op (RMSNorm, linear, depthwise conv, SiLU, gate) already has
a reasonable NetSpec analogue or is trivially derivable. The one **genuinely
new primitive** is the selective state-space scan; everything else is just
orchestration around it. Following the same philosophy as our treatment of
multi-head attention (one bundled `mhsa_layer`) and the transformer encoder
(one bundled `transformerEncoder`), we expose the whole Mamba block as a
single `Layer` constructor. The book's Chapter-N can unpack the innards
when it matters; the `NetSpec` stays readable.

For the proof side (Chapter-VJP), proving `mambaBlock_has_vjp_mat` would
require an axiom for the selective-scan VJP plus composition with our
existing pieces (dense, SiLU via `pdiv_relu`-style swap, elementwise gate
via `elemwiseProduct_has_vjp`, residual via `biPath`). Not done here —
the bestiary is pure architecture, no VJP commitment.

## Variants

- `mamba130M` — 24 blocks, dim 768 — matches the original Mamba-130M paper weights
- `mamba370M` — 48 blocks, dim 1024 — the Mamba-370M sibling
- `mamba790M` — 48 blocks, dim 1536 — the Mamba-790M variant
- `tinyMamba` — 4 blocks, dim 128 — smallest pedagogical fixture

All use `stateSize = 16` and `expand = 2`, matching the paper's defaults.
-/

-- ════════════════════════════════════════════════════════════════
-- § Mamba-130M: 24 blocks × dim 768  (matches Gu & Dao weights)
-- ════════════════════════════════════════════════════════════════

def mamba130M : NetSpec where
  name := "Mamba-130M"
  -- For LMs we treat input as (L, D) pre-embedded tokens. imageH=seqLen, imageW=dim.
  imageH := 2048       -- context length (typical)
  imageW := 1
  layers := [
    -- Real Mamba has a token embedding (vocab → D) at the start and an LM
    -- head (D → vocab) at the end. We skip both for bestiary brevity —
    -- Chapter on language models covers the embedding/head separately.
    .mambaBlock 768 16 2 24,     -- dim=768, state=16, expand=2, 24 blocks
    .dense 768 50280 .identity   -- LM head: project to vocab size (GPT-NeoX vocab)
  ]

-- ════════════════════════════════════════════════════════════════
-- § Mamba-370M: 48 × 1024
-- ════════════════════════════════════════════════════════════════

def mamba370M : NetSpec where
  name := "Mamba-370M"
  imageH := 2048
  imageW := 1
  layers := [
    .mambaBlock 1024 16 2 48,
    .dense 1024 50280 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Mamba-790M: 48 × 1536
-- ════════════════════════════════════════════════════════════════

def mamba790M : NetSpec where
  name := "Mamba-790M"
  imageH := 2048
  imageW := 1
  layers := [
    .mambaBlock 1536 16 2 48,
    .dense 1536 50280 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyMamba: 4 × 128  (bestiary fixture for quick inspection)
-- ════════════════════════════════════════════════════════════════

def tinyMamba : NetSpec where
  name := "tiny-Mamba"
  imageH := 256
  imageW := 1
  layers := [
    .mambaBlock 128 16 2 4,
    .dense 128 1000 .identity    -- small vocab for demo
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
  | none     => IO.println s!"  validate    : OK (dim chains cleanly)"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Mamba 🐍"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Selective state-space models. Linear-time alternative to"
  IO.println "  attention for language modeling. Not trained here — just"
  IO.println "  the architecture, as NetSpec values."

  summarize mamba130M
  summarize mamba370M
  summarize mamba790M
  summarize tinyMamba

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • .mambaBlock is a NEW Layer constructor added for this"
  IO.println "    bestiary. The codegen emits `// UNSUPPORTED` for it; this"
  IO.println "    entry is shape/param only. A real Mamba trainer would need"
  IO.println "    MlirCodegen.emitMambaBlock implementing the selective scan"
  IO.println "    (the hardware-aware scan kernel from the paper)."
  IO.println "  • Parameter count is approximate: 3·E·D² (in+out proj) +"
  IO.println "    3·E·D·N (dt/B/C) + 5·E·D (conv + bias) + D (RMSNorm)"
  IO.println "    per block. Matches Gu & Dao's reported counts within ~5%."
  IO.println "  • Input is treated as pre-embedded (L, D) tokens; token-"
  IO.println "    embedding layer not shown (same simplification as ViT's"
  IO.println "    patch embedding being bundled into one axiom)."
