import LeanMlir

/-! # Nyströmformer — Bestiary entry

Nyströmformer (Xiong, Zeng, Chakraborty, Tan, Fung, Li, Singh, AAAI
2021 --- "Nyströmformer: A Nyström-Based Algorithm for Approximating
Self-Attention") is one of the strangest entries in the efficient-
attention literature. Most efficient-attention papers come from ML
methodology: Linformer uses learned low-rank projections, Performer
uses random features, Longformer uses sparse patterns. Nyströmformer
reaches back to \textbf{numerical linear algebra from 1928}: the
Nyström approximation of a Gram matrix via landmark points.

The idea in one paragraph: given a softmax-attention matrix
$A \in \mathbb{R}^{n \times n}$ of sequence length $n$, pick $m$
landmark tokens where $m \ll n$ (typically $m = 64$ for $n =
4096$). Compute three small softmax matrices --- query-to-landmark
$(n \times m)$, landmark-to-landmark $(m \times m)$, and
landmark-to-key $(m \times n)$ --- and approximate the full
attention as $A \approx A_{QL} \cdot A_{LL}^{+} \cdot A_{LK}$,
where $A_{LL}^{+}$ is the Moore--Penrose pseudoinverse. That's the
Nyström method from 1928, lifted from kernel methods into attention.
Cost: $O(n m + m^3)$, which is $O(n)$ when $m$ is fixed.

## The pedagogical point: same params, different compute

Nyströmformer's \texttt{NetSpec} is architecturally \emph{identical}
to a standard BERT. Same $W_Q$, $W_K$, $W_V$, $W_O$ projections per
head, same FFN, same LayerNorms. The Nyström algorithm is a
drop-in replacement for the \texttt{softmax(QK}$^T$\texttt{)V}
kernel --- it computes \emph{the same logical output} using a
different algorithm. From a parameter-counting standpoint there is
nothing new to report.

This is true of almost every efficient-attention paper:

| Paper              | Attention complexity | New params? |
|--------------------|----------------------|-------------|
| Nyströmformer      | O(n) via landmarks   | \textbf{none}  |
| Linformer          | O(n) via low-rank K,V| learned projections |
| Performer (FAVOR+) | O(n) via random feat | \textbf{none}  |
| Longformer         | O(n) sparse pattern  | \textbf{none}  |
| BigBird            | O(n) sparse pattern  | \textbf{none}  |
| Reformer (LSH)     | O(n log n) via LSH   | \textbf{none}  |
| FlashAttention     | still O(n$^2$), IO-aware  | \textbf{none}  |

Only Linformer changes the parameter count. The rest are all
compute-optimization papers, and we can't differentiate them at the
\texttt{NetSpec} level.

## Why include Nyströmformer specifically

Because the bestiary already covers BERT / GPT / ViT / Swin
(standard attention), Mamba (state-space, the ``throw attention
out'' alternative), and DETR / SAM's transformer decoder
(cross-attention). Adding \emph{one} efficient-attention entry as a
representative is valuable both because the family has become
essential for long-context models (Mamba's main rival circa 2023)
and because Nyströmformer is the one where the connection to
classical math is most obvious. Readers see ``efficient attention
is just a compute choice'' and also ``this 90-year-old trick from
numerical linear algebra is still useful.''

## Variants

- `nystromformerBase`  --- BERT-base scale: 12 layers, dim 768
- `nystromformerLarge` --- BERT-large scale: 24 layers, dim 1024
- `tinyNystromformer`  --- fixture

Paper reports results on the Long Range Arena benchmark using
BERT-base-scale Nyströmformer configurations; exact param counts
match BERT-base (~110M) and BERT-large (~340M) when we include
token embeddings.
-/

-- ════════════════════════════════════════════════════════════════
-- § Nyströmformer base (BERT-base shape, ~110M)
-- ════════════════════════════════════════════════════════════════

def nystromformerBase : NetSpec where
  name := "Nyströmformer base (BERT-base-shaped, O(n) attention)"
  imageH := 4096      -- long context — what the paper was designed for
  imageW := 1
  layers := [
    -- Token embedding: same trick as BERT / GPT, .dense vocab → D.
    .dense 30522 768 .identity,
    -- 12 transformer encoder blocks. At the NetSpec level these are
    -- identical to BERT's blocks; the Nyström approximation lives
    -- inside the attention kernel, not the layer definition.
    .transformerEncoder 768 12 3072 12
  ]

-- ════════════════════════════════════════════════════════════════
-- § Nyströmformer large (BERT-large shape, ~335M)
-- ════════════════════════════════════════════════════════════════

def nystromformerLarge : NetSpec where
  name := "Nyströmformer large (BERT-large-shaped)"
  imageH := 4096
  imageW := 1
  layers := [
    .dense 30522 1024 .identity,
    .transformerEncoder 1024 16 4096 24
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyNystromformer — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinyNystromformer : NetSpec where
  name := "tiny-Nyströmformer"
  imageH := 512
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
  IO.println "  Bestiary — Nyströmformer"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  1928 numerical-linear-algebra trick meets 2017 transformer."
  IO.println "  O(n²) softmax attention → O(n) via landmark approximation."

  summarize nystromformerBase
  summarize nystromformerLarge
  summarize tinyNystromformer

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Param counts are identical to"
  IO.println "    BERT at each scale (base 110M, large 340M). The entire"
  IO.println "    Nyström contribution lives in how attention is computed,"
  IO.println "    not in what layers the network has."
  IO.println "  • The Nyström approximation is a 1928 result from integral"
  IO.println "    equations — it came to ML via kernel methods in the early"
  IO.println "    2000s (Williams & Seeger 2001 for Gaussian process"
  IO.println "    speedups), then got lifted into transformer attention in"
  IO.println "    2021. A genuinely long arc for a single math trick."
  IO.println "  • Most of the 2020–2022 'efficient attention' literature is"
  IO.println "    architecturally-equivalent papers (Linformer is the one"
  IO.println "    exception — it adds learned low-rank projections). All"
  IO.println "    compete on the compute / memory / quality / implementation-"
  IO.println "    complexity axes. At the NetSpec level they're all"
  IO.println "    indistinguishable from BERT."
  IO.println "  • FlashAttention (Dao et al. 2022) won most of this race in"
  IO.println "    practice: keep the exact O(n²) softmax but restructure the"
  IO.println "    computation to be IO-aware on GPU. All the approximate-"
  IO.println "    attention papers optimize for theoretical complexity; FA"
  IO.println "    optimized for real-world GPU memory bandwidth, which is"
  IO.println "    usually the bottleneck anyway."
  IO.println "  • Mamba (see Mamba.lean) is the parallel-track answer:"
  IO.println "    replace attention entirely with a selective state-space"
  IO.println "    scan. O(n) time, different architecture, different params."
