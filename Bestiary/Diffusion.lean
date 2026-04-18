import LeanMlir

/-! # DDPM — Bestiary entry

DDPM (Ho, Jain, Abbeel, 2020 — "Denoising Diffusion Probabilistic
Models") is the paper that made diffusion models practical for image
synthesis. Same architecture it shipped with shows up everywhere now:
Stable Diffusion, Imagen, DALL·E 2, Midjourney's older stacks. And
every one of those starts from the same uncomfortable observation
Ho et al. made:

**The denoising network is just a UNet.**

Not a new UNet. The UNet. Same encoder/decoder with skip connections
that Ronneberger shipped in 2015 for biomedical image segmentation.
Everything "diffusion" about a diffusion model lives in the *training
objective* and the *sampling procedure*, not in the architecture:

- A deterministic **forward process** gradually adds Gaussian noise
  to a real image across $T$ timesteps.
- A learned **reverse process** predicts the noise added at step $t$
  given the noisy image $x_t$ and timestep $t$. This is the UNet.
- **Sampling** runs the reverse process from pure noise $x_T$ down
  to a clean image $x_0$ by iteratively subtracting the predicted
  noise, usually with 50--1000 denoising steps.

None of that is a layer. The UNet is trained with a plain
MSE-on-noise loss; everything interesting is in the noise schedule,
loss weighting, and sampler. That's the entire diffusion story in
one paragraph.

## What the DDPM UNet actually does differently from Ronneberger

Small but real modifications:

- **Residual blocks** instead of plain $3 \times 3 + \text{BN} + \text{ReLU}$
  pairs at each resolution (typically 2 res-blocks per level).
- **GroupNorm** instead of BatchNorm (BN behaves badly when the batch
  contains independent timesteps).
- **Self-attention** inserted at the lowest 1--2 resolutions (commonly
  $16 \times 16$ and $8 \times 8$), so long-range structure can inform
  the denoise.
- **Timestep conditioning**: every residual block gets an additive
  projection of a timestep embedding (sinusoidal $\to$ dense $\to$ SiLU
  $\to$ dense). This is how the network knows "how noisy" the input is.

Our linear \texttt{NetSpec} doesn't express the per-block time
injection cleanly (it'd need a branch/merge pattern), so the diffusion
UNet spec below is the Ronneberger-style backbone only. The
\texttt{ddpmTimeEmbed} spec shows the timestep MLP separately as its
own network.

## Variants

- `ddpmCifar`     — CIFAR config: 32×32, base 128, mult [1,2,2,2]
                    (paper: 35.7M; our simplified UNet backbone undershoots).
- `ddpmHires`     — 256×256 LSUN/ImageNet config, base 128, depth 5
                    (paper: ~550M for ImageNet-256; same undershoot caveat).
- `tinyDdpm`      — 32×32 fixture, small enough to read in one pass.
- `ddpmTimeEmbed` — standalone timestep-embedding MLP
                    (sinusoidal → dense → SiLU → dense).

The headline lesson is the one CLIP and NeRF already taught: the
architectural novelty of diffusion is zero. The training procedure
does all the work.
-/

-- ════════════════════════════════════════════════════════════════
-- § DDPM CIFAR (32×32, paper target ~35.7M)
-- ════════════════════════════════════════════════════════════════
-- Channel progression mirrors Ho et al.'s CIFAR config:
--   base = 128, mult = [1, 2, 2, 2], 3 downsamples over 4 resolutions.
-- Our unetDown/unetUp count 2 plain conv3×3 per level (no res-block
-- expansion, no time conditioning, no attention), so params come in
-- below the paper's 35.7M figure. The spec's value is architectural
-- shape, not an exact param match.

def ddpmCifar : NetSpec where
  name := "DDPM (CIFAR-10, backbone approx)"
  imageH := 32
  imageW := 32
  layers := [
    -- Encoder: 32 → 16 → 8 → 4
    .unetDown 3   128,               -- 32 → 16, channels 3 → 128
    .unetDown 128 256,               -- 16 → 8 , channels 128 → 256
    .unetDown 256 256,               -- 8  → 4 , channels stay at 256
    -- Bottleneck at 4×4 (standing in for the paper's two res-blocks +
    -- self-attention at the lowest resolution).
    .convBn 256 256 3 1 .same,
    .convBn 256 256 3 1 .same,
    -- Decoder: 4 → 8 → 16 → 32
    .unetUp 256 256,                 -- 4  → 8 , expects 256-ch skip
    .unetUp 256 256,                 -- 8  → 16, expects 256-ch skip
    .unetUp 256 128,                 -- 16 → 32, expects 128-ch skip
    -- Output head: predict the added noise at every pixel (3 channels).
    .conv2d 128 3 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § DDPM Hi-res (256×256 ImageNet / LSUN, paper target ~550M)
-- ════════════════════════════════════════════════════════════════
-- 5-level UNet: 256 → 128 → 64 → 32 → 16, base 128, mult [1,1,2,2,4,4].
-- Same simplifications apply as ddpmCifar (no time cond, no attention).

def ddpmHires : NetSpec where
  name := "DDPM (256×256, backbone approx)"
  imageH := 256
  imageW := 256
  layers := [
    -- Encoder: 256 → 128 → 64 → 32 → 16 → 8
    -- Skip widths produced: 128, 128, 256, 256, 512 (top → bottom).
    .unetDown 3   128,
    .unetDown 128 128,
    .unetDown 128 256,
    .unetDown 256 256,
    .unetDown 256 512,
    -- Bottleneck at 8×8
    .convBn 512 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    -- Decoder: 8 → 16 → 32 → 64 → 128 → 256.
    -- i-th unetUp from the bottom pairs with the i-th unetDown from the
    -- top, so unetUp.oc must match that unetDown.oc.
    .unetUp 512 512,                 -- pairs with unetDown 256→512 (skip=512)
    .unetUp 512 256,                 -- pairs with unetDown 256→256 (skip=256)
    .unetUp 256 256,                 -- pairs with unetDown 128→256 (skip=256)
    .unetUp 256 128,                 -- pairs with unetDown 128→128 (skip=128)
    .unetUp 128 128,                 -- pairs with unetDown 3→128   (skip=128)
    -- Output head
    .conv2d 128 3 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyDdpm fixture — 32×32, small enough to read in one pass
-- ════════════════════════════════════════════════════════════════

def tinyDdpm : NetSpec where
  name := "tiny-DDPM"
  imageH := 32
  imageW := 32
  layers := [
    .unetDown 3  32,                 -- 32 → 16, skip=32
    .unetDown 32 64,                 -- 16 →  8, skip=64
    .convBn 64 64 3 1 .same,         -- bottleneck at 8×8
    .unetUp 64 64,                   --  8 → 16, pairs with unetDown 32→64 (skip=64)
    .unetUp 64 32,                   -- 16 → 32, pairs with unetDown 3→32 (skip=32)
    .conv2d 32 3 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Timestep-embedding MLP (the conditioning signal, not the UNet)
-- ════════════════════════════════════════════════════════════════
-- Sinusoidal positional encoding of the integer timestep t, then two
-- dense layers with a SiLU in between, producing a D-dim vector that
-- the UNet uses as per-block additive conditioning. We show it as a
-- standalone NetSpec because it doesn't sit in-line with the backbone;
-- real DDPM branches the time-embedding into every residual block.

def ddpmTimeEmbed : NetSpec where
  name := "DDPM timestep-embedding MLP"
  imageH := 1            -- scalar timestep t
  imageW := 1
  layers := [
    -- γ(t) = (sin(2⁰πt), cos(2⁰πt), ..., sin(2^(L-1)πt), cos(2^(L-1)πt))
    -- inputDim=1 (t is a scalar), numFrequencies=64 → encoded dim = 1·2·64 = 128.
    .positionalEncoding 1 64,
    -- Two dense layers with a SiLU (modeled here as identity since our
    -- Activation enum doesn't have SiLU; it's elementwise and contributes
    -- zero params either way).
    .dense 128 512 .identity,
    .dense 512 512 .identity
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
  IO.println "  Bestiary — DDPM"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The denoising network is just a UNet. Everything diffusion"
  IO.println "  is training procedure, not architecture."

  summarize ddpmCifar
  summarize ddpmHires
  summarize tinyDdpm
  summarize ddpmTimeEmbed

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. The DDPM UNet reuses"
  IO.println "    .unetDown / .unetUp exactly as UNet.lean does; the timestep"
  IO.println "    MLP reuses .positionalEncoding (from NeRF) + .dense."
  IO.println "  • Our simplified backbone undercounts vs the real DDPM UNet."
  IO.println "    Paper targets: ~35.7M (CIFAR) and ~550M (ImageNet-256);"
  IO.println "    our approximation is the Ronneberger-style backbone without"
  IO.println "    residual blocks, group norm, attention-at-low-res, or the"
  IO.println "    per-block time-embedding projection. All real, all small"
  IO.println "    additions on top of a UNet."
  IO.println "  • Forward process (add noise) and reverse process (learned"
  IO.println "    denoiser, plus a sampler choosing timesteps and σ) live"
  IO.println "    OUTSIDE the network. Whether you use DDPM (Ho 2020), DDIM"
  IO.println "    (Song 2020), or newer schedulers, the UNet is the same."
  IO.println "  • Stable Diffusion adds cross-attention to text embeddings"
  IO.println "    inside each residual block — that IS architectural, but"
  IO.println "    it's a modification on top of this backbone, not a"
  IO.println "    different kind of backbone."
