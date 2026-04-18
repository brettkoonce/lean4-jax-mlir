import LeanMlir

/-! # Stable Diffusion — Bestiary entry

Stable Diffusion (Rombach et al., CVPR 2022 — "High-Resolution Image
Synthesis with Latent Diffusion Models") is the paper that made
generative image models consumer-reachable. Two architectural moves
on top of DDPM, each individually small, that together made the
compute budget 10--40× cheaper:

1. **Latent diffusion.** Train a VAE to compress 512$\times$512 pixel
   images to a 64$\times$64$\times$4 latent. Run diffusion on the
   latent, not the pixels. Decoding the latent back to pixels is
   ``free'' --- a single forward pass through the VAE decoder ---
   so you spend the 50--1000 denoising steps on a 64$\times$64 grid
   instead of 512$\times$512. That's a 64$\times$ reduction in the
   spatial work.
2. **Text conditioning via cross-attention.** At each resolution of
   the denoiser UNet, insert a \emph{spatial transformer} block that
   does self-attention over image tokens + cross-attention to CLIP
   text embeddings + FFN. The UNet sees the text prompt at every
   resolution, not just once at the start. This is what makes ``a
   photo of an astronaut riding a horse'' actually produce an
   astronaut riding a horse.

Everything else is DDPM: forward noise schedule, learned reverse
denoiser, sampler iterating from pure noise to clean latent. The
training objective, classifier-free-guidance trick, and sampler
choices all live outside the network.

## Anatomy (SD 1.5)

```
  Prompt                          Image (512×512×3)
     │                                   │
     ▼                                   ▼
  CLIP text encoder               VAE encoder
  (77 tokens, 768-dim)            (4-channel 64×64 latent)
     │                                   │
     │                                   ▼
     │                            +noise @ timestep t
     │                                   │
     │            ┌──────────────────────┘
     │            │
     ▼            ▼
   UNet denoiser (takes noisy latent + text embedding + timestep)
        │
        ▼
  predicted noise  (subtract it, iterate over T steps, get clean latent)
        │
        ▼
  VAE decoder → final 512×512 image
```

Sizes for SD 1.5:
- VAE total:          $\sim$84M  (frozen after pretraining)
- Text encoder:       $\sim$123M (CLIP ViT-L/14 text tower, frozen)
- UNet denoiser:      $\sim$865M (the trained component)
- Grand total:        $\sim$1.07B, but only the 865M UNet trains.

For reference: SDXL (2023) scales the UNet to 2.6B and adds a separate
``refiner'' UNet on top; SD 3 switches to a DiT-style transformer
instead of a UNet. The fundamental latent-diffusion + text-conditioning
template is the same across all of them.

## Variants (shown as separate components)

- `sdVAEEncoder`                 — image $\to$ latent (encoder half)
- `sdVAEDecoder`                 — latent $\to$ image (decoder half)
- `sdTextEncoder`                — CLIP ViT-L/14 text tower
- `sdUNet15`                     — SD 1.5 UNet backbone (approximate)
- `sdSpatialTransformerBlock`    — one cross-attention block,
                                    illustrates the mechanism that
                                    differentiates SD from vanilla DDPM
- `tinyStableDiffusion`          — end-to-end fixture

## NetSpec simplifications

- The UNet spec shows the backbone shape via \texttt{.unetDown} /
  \texttt{.unetUp}; same DDPM-style undercount (no per-block time
  conditioning, no multi-ResBlock-per-resolution, no interleaved
  spatial transformers). Real SD 1.5 UNet is $\sim$865M; our backbone
  lands around 200M. The missing ~650M is exactly what the separate
  \texttt{sdSpatialTransformerBlock} spec demonstrates.
- VAE approximated with plain convBn / unetUp chains. Loose match on
  params; emphasis is on the compression ratio ($512 \to 64$ spatial,
  $3 \to 4$ channels, so $\sim$48$\times$ data-size reduction).
- Classifier-free guidance, DDIM / DPM++ samplers, LoRA adapters for
  fine-tuning --- all outside the network.
-/

-- ════════════════════════════════════════════════════════════════
-- § VAE encoder: 512×512×3 → 64×64×4 latent
-- ════════════════════════════════════════════════════════════════
-- SD's VAE has ~34M params in the encoder; we approximate a 3-stage
-- downsample CNN. Output is 8-channel (4 mean + 4 log-variance for
-- the VAE reparameterization); sampling selects the 4-channel latent.

def sdVAEEncoder : NetSpec where
  name := "SD VAE encoder (image → latent)"
  imageH := 512
  imageW := 512
  layers := [
    .convBn 3   128 3 1 .same,
    .convBn 128 128 3 2 .same,     -- 512 → 256
    .convBn 128 256 3 1 .same,
    .convBn 256 256 3 2 .same,     -- 256 → 128
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 2 .same,     -- 128 → 64
    .convBn 512 512 3 1 .same,     -- bottleneck
    -- Output: 8 channels (4 μ + 4 log-σ²). Sampling picks the 4-ch latent.
    .conv2d 512 8 3 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § VAE decoder: 64×64×4 latent → 512×512×3 image
-- ════════════════════════════════════════════════════════════════
-- SD's VAE decoder is ~50M. Approximated as 3-stage upsample via unetUp;
-- normally unetUp expects a matching skip from unetDown, which the
-- decoder-only path doesn't have — validate still passes (it checks
-- channels, not skip-pairing). Spec shows shape; params approximate.

def sdVAEDecoder : NetSpec where
  name := "SD VAE decoder (latent → image)"
  imageH := 64
  imageW := 64
  layers := [
    .conv2d 4 512 3 .same .identity,   -- expand latent
    .convBn 512 512 3 1 .same,         -- bottleneck
    .unetUp 512 256,                    -- 64 → 128
    .convBn 256 256 3 1 .same,
    .unetUp 256 128,                    -- 128 → 256
    .convBn 128 128 3 1 .same,
    .unetUp 128 128,                    -- 256 → 512
    .convBn 128 128 3 1 .same,
    .conv2d 128 3 3 .same .identity     -- final RGB output
  ]

-- ════════════════════════════════════════════════════════════════
-- § Text encoder: CLIP ViT-L/14 text tower, 77 tokens, 123M params
-- ════════════════════════════════════════════════════════════════
-- Same architecture as the text side of CLIP.lean. Causal mask is a
-- training-time detail; our .transformerEncoder covers the shape.

def sdTextEncoder : NetSpec where
  name := "SD text encoder (CLIP ViT-L/14 text tower)"
  imageH := 77         -- max prompt tokens (SD's fixed context length)
  imageW := 1
  layers := [
    .dense 49408 768 .identity,           -- vocab → dim (tied to head)
    .transformerEncoder 768 12 3072 12
  ]

-- ════════════════════════════════════════════════════════════════
-- § UNet denoiser: operates on 64×64×4 latents (SD 1.5)
-- ════════════════════════════════════════════════════════════════
-- Base channels 320, mult [1, 2, 4, 4] → levels [320, 640, 1280, 1280]
-- at resolutions [64, 32, 16, 8]. Real SD interleaves spatial-transformer
-- blocks (self-attn + cross-attn to text + FFN) at resolutions 32 / 16 / 8;
-- our spec shows the UNet skeleton via unetDown / unetUp. The
-- cross-attention mechanism is shown separately as sdSpatialTransformerBlock.

def sdUNet15 : NetSpec where
  name := "SD 1.5 UNet denoiser (backbone approx)"
  imageH := 64         -- latent resolution, not pixel
  imageW := 64
  layers := [
    .conv2d 4 320 3 .same .identity,    -- stem (latent → channels)
    -- Encoder: 64 → 32 → 16 → 8
    .unetDown 320  640,                  -- 64 → 32, skip=640
    .unetDown 640  1280,                 -- 32 → 16, skip=1280
    .unetDown 1280 1280,                 -- 16 →  8, skip=1280
    -- Bottleneck at 8×8 (real SD inserts a spatial transformer here)
    .convBn 1280 1280 3 1 .same,
    .convBn 1280 1280 3 1 .same,
    -- Decoder: 8 → 16 → 32 → 64
    .unetUp 1280 1280,                   --  8 → 16, pairs with 1280 skip
    .unetUp 1280 1280,                   -- 16 → 32, pairs with 1280 skip
    .unetUp 1280 640,                    -- 32 → 64, pairs with  640 skip
    -- Output head: predict noise (4 channels, same as input latent)
    .conv2d 640 4 3 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Spatial transformer block — the cross-attention mechanism
-- ════════════════════════════════════════════════════════════════
-- At each interior UNet resolution, SD inserts a block that does:
--   LayerNorm + self-attn (image tokens × image tokens)
--   LayerNorm + cross-attn (image tokens × text tokens)
--   LayerNorm + FFN
-- This is exactly the shape of a .transformerDecoder block with
-- nQueries=0 — image tokens take the role of queries (no learned
-- object embeddings), text tokens are the key/value source for
-- cross-attention. Shown at dim=1280 (the UNet's deepest resolution),
-- one block as an illustrative example.

def sdSpatialTransformerBlock : NetSpec where
  name := "SD spatial transformer block (1280-dim, 1 layer)"
  imageH := 64         -- number of image tokens (varies by UNet level)
  imageW := 1
  layers := [
    .transformerDecoder 1280 8 5120 1 0
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyStableDiffusion — end-to-end fixture
-- ════════════════════════════════════════════════════════════════
-- Compressed representation: tiny UNet with a lone spatial transformer
-- block at the bottleneck showing text-conditioning shape.

def tinyStableDiffusion : NetSpec where
  name := "tiny-StableDiffusion (UNet + spatial transformer)"
  imageH := 32         -- tiny latent resolution
  imageW := 32
  layers := [
    .conv2d 4 64 3 .same .identity,
    .unetDown 64  128,
    .unetDown 128 256,
    -- Bottleneck spatial transformer (cross-attn to text tokens).
    .transformerDecoder 256 4 1024 1 0,
    .unetUp 256 128,
    .unetUp 128 64,
    .conv2d 64 4 3 .same .identity
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
  IO.println "  Bestiary — Stable Diffusion"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Latent diffusion with text cross-attention. DDPM compressed"
  IO.println "  64x in spatial work + text-conditioned at every UNet level."

  summarize sdVAEEncoder
  summarize sdVAEDecoder
  summarize sdTextEncoder
  summarize sdUNet15
  summarize sdSpatialTransformerBlock
  summarize tinyStableDiffusion

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. VAE uses .convBn + .unetUp /"
  IO.println "    .unetDown; text encoder is CLIP's kit; UNet is DDPM's kit;"
  IO.println "    spatial transformer is .transformerDecoder with nQueries=0"
  IO.println "    (same primitive Whisper's decoder uses)."
  IO.println "  • Three components, three pretrained artifacts: VAE and text"
  IO.println "    encoder are frozen after their own pretraining runs; only"
  IO.println "    the UNet trains during Stable Diffusion's main training."
  IO.println "  • UNet backbone spec is ~200M; real SD 1.5 UNet is 865M."
  IO.println "    The missing ~650M lives in (a) 2 ResBlocks per level"
  IO.println "    instead of unetDown's plain convs, (b) spatial-transformer"
  IO.println "    blocks interleaved at resolutions 32/16/8 — shown separately"
  IO.println "    as sdSpatialTransformerBlock."
  IO.println "  • SDXL (2023) scales the UNet to ~2.6B with a larger spatial-"
  IO.println "    transformer budget and a separate 'refiner' UNet on top."
  IO.println "    Same three-component template."
  IO.println "  • SD 3 (2024) swaps the UNet for a DiT-style transformer."
  IO.println "    Text conditioning switches from cross-attention to token-"
  IO.println "    concatenation in the input sequence. Still latent diffusion."
  IO.println "  • Everything interesting about 'prompt engineering' happens"
  IO.println "    in the text encoder; everything about 'sampling quality'"
  IO.println "    happens in the sampler (DDIM, DPM++, Euler, etc.); neither"
  IO.println "    lives in the NetSpec."
