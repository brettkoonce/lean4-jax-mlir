import LeanMlir

/-! # QANet — Bestiary entry

QANet (Yu, Dohan, Luong, Zhao, Chen, Norouzi, Le, ICLR 2018 ---
"QANet: Combining Local Convolution with Global Self-Attention for
Reading Comprehension") is the reading-comprehension architecture
that first showed you could kill the LSTM from SQuAD-style question-
answering pipelines and \emph{speed training up 3--4$\times$} without
losing accuracy.

The context matters. Pre-2018, BiDAF (BiDirectional Attention Flow)
and its BiLSTM cousins dominated SQuAD. They scored well, but BiLSTMs
don't parallelize across timesteps --- each token has to wait for
its predecessor. On SQuAD-sized contexts (400+ tokens) this meant
hours-per-epoch training runs on already-expensive GPU hardware.
QANet's pitch was \textbf{same accuracy, much faster}, achieved by
throwing out recurrence entirely.

## The QANet encoder block

The core architectural unit --- what makes QANet QANet --- is a
hybrid ``conv + attention'' block:

```
  Input token representations (n × 128)
        │
        ▼  (repeated 4 times)
   LayerNorm → depthwise separable conv (kernel 7, 128→128)
        │      (+ residual)
        ▼
   LayerNorm → multi-head self-attention (8 heads, 128-dim)
        │      (+ residual)
        ▼
   LayerNorm → FFN (128 → 512 → 128)
        │      (+ residual)
        ▼
  Output (n × 128)
```

The separable convs give \emph{local} context (sliding window of 7
tokens); the self-attention gives \emph{global} context (every
token to every other). Both in the same block, both residual-
connected. This ``local conv + global attention'' hybrid pre-dates
MobileViT (2022), ConvNeXt's 7$\times$7 depthwise (2022), and the
general 2020--2022 trend of reintroducing convolutions into vision
transformers. QANet had it in 2018 because convolutions were fast
and parallelizable, which was the whole point.

## Overall QANet structure

```
  Question                 Context
     │                        │
     ▼ char + word emb         ▼ char + word emb
  Embedding encoder        Embedding encoder
  (1 × encoder block)      (1 × encoder block)
     │                        │
     └──────┬──────────────────┘
            ▼
  Context-query attention (BiDAF-style bidirectional attention)
            │
            ▼
  Model encoder × 3 stacks (each 7 × encoder block)
            │
            ▼
  Output: two pointer networks predicting (start, end) in context
```

The embedding encoder appears at the input (question and context
sides share weights). The model encoder is 7 blocks stacked,
repeated 3 times, for the main body. Output is two small softmax
heads that predict span start and end token positions.

## Why QANet matters pedagogically

Two reasons. First, the ``conv + attention hybrid'' shape was
novel in 2018 and has since become the default for efficient
architectures that want the inductive biases of both. Second, it's
a clean case of \textbf{hardware forcing a design choice}: the
primary argument for QANet vs BiDAF wasn't accuracy (comparable),
it was training speed. The paper explicitly frames the contribution
as ``3--4x speedup on the same hardware,'' and the architectural
decisions (no recurrence, heavy use of parallelizable convs, self-
attention instead of LSTM gating) all trace back to that constraint.

## Variants

- `qanetEncoderBlock`       --- the hybrid architectural unit
                                (one "conv + attention" block)
- `qanetModelEncoderStack`  --- 7 encoder blocks stacked
                                (repeated 3$\times$ in the full model)
- `tinyQANet`               --- fixture

## NetSpec simplifications

- Character/word embedding tables are omitted (shape-only; real QANet
  uses GloVe for words + a character-CNN for subword features, maybe
  100M combined at typical SQuAD vocab sizes).
- BiDAF-style context-query attention doesn't fit a linear NetSpec
  cleanly (it's bidirectional between two sequences); described in
  prose only.
- Output pointer networks (two softmax heads over context positions)
  also omitted --- they're task-specific heads at low parameter cost.
-/

-- ════════════════════════════════════════════════════════════════
-- § QANet encoder block — the hybrid architectural unit
-- ════════════════════════════════════════════════════════════════
-- Four separable convs (local context) + one transformer encoder
-- block (global context via self-attention + FFN).

def qanetEncoderBlock : NetSpec where
  name := "QANet encoder block (4× sep-conv + self-attn + FFN)"
  imageH := 400        -- representative context length
  imageW := 1
  layers := [
    -- 4 depthwise-separable convs at kernel 7, dim 128 → 128.
    -- Our .separableConv uses kernel 3 internally; we approximate
    -- by stacking four calls (captures the count, not the k=7 receptive
    -- field). Param count per call: depthwise (3 * 128) + pointwise
    -- (128 * 128 + 128) + BN ≈ 17K.
    .separableConv 128 128 1,
    .separableConv 128 128 1,
    .separableConv 128 128 1,
    .separableConv 128 128 1,
    -- Self-attention + FFN + 2 LayerNorms (one transformer encoder
    -- block with 1 layer). Dim 128, 8 heads, mlpDim 512 (= 4 * dim).
    .transformerEncoder 128 8 512 1
  ]

-- ════════════════════════════════════════════════════════════════
-- § QANet model encoder stack (7 encoder blocks)
-- ════════════════════════════════════════════════════════════════
-- The paper's main body: 7 encoder blocks in sequence, the stack
-- repeated 3 times after context-query attention. We show one copy
-- of the 7-block stack; multiply by 3 for the full model.

def qanetModelEncoderStack : NetSpec where
  name := "QANet model encoder stack (7 blocks, deep body)"
  imageH := 400
  imageW := 1
  layers := [
    -- Approximate the 7-block stack by giving each "block" its
    -- 4 separable convs and one transformer-encoder block. For spec
    -- compactness we stack 7 copies inline.
    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1,

    .separableConv 128 128 1, .separableConv 128 128 1,
    .separableConv 128 128 1, .separableConv 128 128 1,
    .transformerEncoder 128 8 512 1
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyQANet — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinyQANet : NetSpec where
  name := "tiny-QANet (2 blocks, dim 32)"
  imageH := 64
  imageW := 1
  layers := [
    .separableConv 32 32 1,
    .separableConv 32 32 1,
    .transformerEncoder 32 2 128 1,
    .separableConv 32 32 1,
    .separableConv 32 32 1,
    .transformerEncoder 32 2 128 1
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  context     : {spec.imageH} tokens"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000} K)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — QANet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The SQuAD-era reading-comprehension architecture that killed"
  IO.println "  the BiLSTM. Conv + attention hybrid, pre-dates MobileViT by"
  IO.println "  four years."

  summarize qanetEncoderBlock
  summarize qanetModelEncoderStack
  summarize tinyQANet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Encoder block is 4 × .separableConv"
  IO.println "    + 1 × .transformerEncoder (with nBlocks=1 covering the self-"
  IO.println "    attn + FFN + LNs). The conv + attention hybrid shape that"
  IO.println "    QANet shipped in 2018 can be expressed with primitives the"
  IO.println "    bestiary already had from Xception and BERT."
  IO.println "  • Full QANet model: embedding encoder (1 block), context-query"
  IO.println "    attention (BiDAF-style, omitted here — doesn't linearize),"
  IO.println "    model encoder stack × 3 (the qanetModelEncoderStack spec"
  IO.println "    above is one of these three copies)."
  IO.println "  • The paper's headline number was training speed, not accuracy:"
  IO.println "    3-4× speedup over BiDAF on the same hardware. Reading comp"
  IO.println "    didn't get substantially better; training got much faster."
  IO.println "    Hardware-driven design choice in the purest form."
  IO.println "  • The 'conv + attention hybrid' shape later became fashionable"
  IO.println "    across vision (MobileViT 2022, ConvNeXt's 7×7 depthwise 2022,"
  IO.println "    CoAtNet 2021). QANet had it first; the NLP community mostly"
  IO.println "    moved on to pure-attention BERT (Nov 2018, 7 months after"
  IO.println "    QANet) before the hybrid idea had time to catch on in text."
  IO.println "  • BERT landed in November 2018 and essentially ended the"
  IO.println "    SQuAD-as-benchmark era by brute-force scaling. QANet's"
  IO.println "    architectural ideas lived on in vision instead of NLP —"
  IO.println "    where they ultimately mattered more."
