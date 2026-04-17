import LeanMlir

/-! # CLIP — Bestiary entry

CLIP (Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell,
Mishkin, Clark, Krueger, Sutskever, 2021 — "Learning Transferable
Visual Models From Natural Language Supervision") turned out to be
one of the most influential ML papers of the early 2020s. Zero-shot
image classification, cross-modal retrieval, generative-model guidance
(CLIP-guided SD, DALL·E 2 prior), everything that needs a shared
image-text embedding.

## The architecture is boring on purpose

CLIP is **two encoders** trained together:

- An **image encoder** (ResNet-50 with attention pooling, or a ViT-B).
- A **text encoder** (a 12-layer causal transformer).

Both project their output to a **shared 512-dim embedding space** via a
final linear layer. Training uses a **contrastive loss**: in each batch
of $N$ image-text pairs, compute the $N \times N$ matrix of cosine
similarities and drive the diagonal up (matched pair) and off-diagonal
down (mismatched). Temperature-scaled softmax cross-entropy in both
directions (image→text and text→image).

```
      Image (3, 224, 224)                    Text (tokens, seq=77)
             │                                       │
             ▼                                       ▼
        Image encoder                          Text encoder
     (ResNet-50 or ViT-B)                   (12-layer transformer)
             │                                       │
             ▼                                       ▼
      (2048 or 768,)-dim                       (512,)-dim
             │                                       │
         Linear(→ 512)                         Linear(→ 512)
             │                                       │
             ▼                                       ▼
      image_features                          text_features
      (512-dim, L2-normalized)                (512-dim, L2-normalized)
             │                                       │
             └──────────────┬────────────────────────┘
                            │
             cosine similarity × temperature
                            │
                            ▼
                   N × N similarity matrix
                            │
              Contrastive softmax cross-entropy
              (image-to-text + text-to-image)
```

What's new in the paper:

1. **Training signal** — 400M image-caption pairs from the web.
2. **Contrastive objective** — cheaper than predicting captions word-
   by-word (what earlier vision-language models did), and transfers
   well to zero-shot classification: reformulate a classification task
   as "which class name has highest text-embedding similarity to this
   image-embedding?"
3. **The observation itself** — that this scales remarkably well, and
   transfers to downstream tasks with **zero fine-tuning**.

Architecturally, it's two textbook networks glued together with a
cosine similarity. That's the whole diagram.

## Variants

CLIP shipped multiple image backbones; the ViT-B/32 and ResNet-50
variants are the canonical references.

| Variant       | Image encoder    | Text encoder | Shared dim | Params |
|---------------|------------------|--------------|------------|--------|
| CLIP-RN50     | ResNet-50 + attn | 12×512 trans | 1024       | 102M   |
| CLIP-ViT-B/32 | ViT-B/32         | 12×512 trans | 512        | 151M   |
| CLIP-ViT-L/14 | ViT-L/14         | 12×768 trans | 768        | 428M   |

The ViT image encoders dominate for downstream tasks; most modern
CLIP descendants (OpenCLIP, SigLIP, EVA-CLIP) use ViT backbones.
-/

-- ════════════════════════════════════════════════════════════════
-- § CLIP-RN50: ResNet-50 image encoder + transformer text encoder
-- ════════════════════════════════════════════════════════════════

/-- Image encoder for CLIP-RN50: ResNet-50 body + GAP + linear projection
    to the shared 1024-dim embedding space. Actual CLIP replaces the GAP
    with a self-attention pool; our bestiary uses GAP for brevity
    (same output dim, different pooling statistics). -/
def clipRN50ImageEncoder : NetSpec where
  name := "CLIP-RN50 (image encoder)"
  imageH := 224
  imageW := 224
  layers := [
    -- ResNet-50 body
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock 64   256  3 1,
    .bottleneckBlock 256  512  4 2,
    .bottleneckBlock 512  1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    -- Pool + projection to shared embedding space
    .globalAvgPool,
    .dense 2048 1024 .identity   -- shared 1024-dim for RN50 variant
  ]

/-- Text encoder for CLIP-RN50 / ViT-B/32: 12-layer causal transformer
    at dim 512 with 8 heads, followed by a projection to the shared
    embedding space. NetSpec treats the transformer as `.transformerEncoder`
    (our primitive doesn't distinguish causal vs non-causal masking — same
    params, same layer shape; mask is a training-time attention pattern). -/
def clipTextEncoder : NetSpec where
  name := "CLIP (text encoder)"
  imageH := 77                    -- context length (CLIP uses 77 tokens)
  imageW := 1
  layers := [
    -- Token embedding (vocab 49408 → 512). Realized as a dense layer;
    -- in practice this is an embedding lookup, but param count matches.
    .dense 49408 512 .identity,
    -- 12-layer transformer encoder, dim=512, 8 heads, mlp=2048
    .transformerEncoder 512 8 2048 12,
    -- [EOS]-token pooling + projection to shared embedding space.
    -- Paper uses the [EOS] position's output; we approximate with the
    -- first-token pool (flatten + dense does the rough shape).
    .dense 512 512 .identity      -- shared embedding dim for RN50/ViT-B
  ]

-- ════════════════════════════════════════════════════════════════
-- § CLIP-ViT-B/32: ViT image encoder + same text encoder
-- ════════════════════════════════════════════════════════════════

def clipViTB32ImageEncoder : NetSpec where
  name := "CLIP-ViT-B/32 (image encoder)"
  imageH := 224
  imageW := 224
  layers := [
    -- ViT-B/32: patch size 32 → 49 patches, dim 768, 12 blocks, 12 heads
    .patchEmbed 3 768 32 49,
    .transformerEncoder 768 12 3072 12,
    -- Take [CLS] token, project to 512-dim shared space
    .dense 768 512 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § CLIP-ViT-L/14 (the big one)
-- ════════════════════════════════════════════════════════════════

def clipViTL14ImageEncoder : NetSpec where
  name := "CLIP-ViT-L/14 (image encoder)"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 1024 14 256,
    .transformerEncoder 1024 16 4096 24,
    .dense 1024 768 .identity
  ]

/-- Text encoder for ViT-L/14 uses wider dim (768). -/
def clipViTL14TextEncoder : NetSpec where
  name := "CLIP-ViT-L/14 (text encoder)"
  imageH := 77
  imageW := 1
  layers := [
    .dense 49408 768 .identity,
    .transformerEncoder 768 12 3072 12,
    .dense 768 768 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyCLIP (small-scale fixture)
-- ════════════════════════════════════════════════════════════════

def tinyCLIPImageEncoder : NetSpec where
  name := "tiny-CLIP (image encoder)"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 3 2 .same,
    .residualBlock 32 64 2 2,
    .globalAvgPool,
    .dense 64 128 .identity
  ]

def tinyCLIPTextEncoder : NetSpec where
  name := "tiny-CLIP (text encoder)"
  imageH := 32
  imageW := 1
  layers := [
    .dense 1000 64 .identity,            -- small vocab for demo
    .transformerEncoder 64 4 256 4,
    .dense 64 128 .identity
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
  IO.println "  Bestiary — CLIP"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Image encoder + text encoder + contrastive loss."
  IO.println "  Zero new layers. All the magic is in the training signal."

  IO.println ""
  IO.println "──────────── CLIP-RN50 ────────────"
  summarize clipRN50ImageEncoder
  summarize clipTextEncoder

  IO.println ""
  IO.println "──────────── CLIP-ViT-B/32 ────────────"
  summarize clipViTB32ImageEncoder
  summarize clipTextEncoder

  IO.println ""
  IO.println "──────────── CLIP-ViT-L/14 ────────────"
  summarize clipViTL14ImageEncoder
  summarize clipViTL14TextEncoder

  IO.println ""
  IO.println "──────────── tiny-CLIP ────────────"
  summarize tinyCLIPImageEncoder
  summarize tinyCLIPTextEncoder

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. CLIP is two textbook encoders"
  IO.println "    (ResNet / ViT + transformer) glued together with a"
  IO.println "    contrastive loss. The paper's contribution is the training"
  IO.println "    pipeline, the 400M-pair dataset, and the observation that"
  IO.println "    the result zero-shot-transfers remarkably well."
  IO.println "  • The text encoder is a CAUSAL transformer (GPT-style),"
  IO.println "    but our `.transformerEncoder` doesn't distinguish causal"
  IO.println "    vs non-causal masking. Params and layer shape are"
  IO.println "    identical; the mask is an attention pattern, not a layer."
  IO.println "  • Token embedding is expressed as `.dense (vocab × dim)`."
  IO.println "    In practice it's an embedding lookup, but param count"
  IO.println "    matches exactly (a token-embedding table IS a dense layer"
  IO.println "    applied to one-hot inputs)."
  IO.println "  • [EOS] / [CLS] token pooling is approximated with a dense"
  IO.println "    projection — the exact pooling op doesn't change param"
  IO.println "    count."
  IO.println "  • Descendants: OpenCLIP (open reproduction), SigLIP (sigmoid"
  IO.println "    loss instead of softmax), EVA-CLIP (better vision backbones),"
  IO.println "    ALIGN (Google's simultaneous version with 1.8B pairs)."
  IO.println "    All same architecture shape; just different training diets."
