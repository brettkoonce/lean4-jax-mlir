/-! ML specification types: Layer, NetSpec, TrainConfig, DatasetKind. -/

inductive Activation where
  | relu
  | relu6
  | identity
  | swish
  | hSwish
  | gelu
deriving Repr, BEq

/-- Normalization choice — picked at the block level by primitives that
    can run with either LayerNorm or BatchNorm (e.g. ConvNeXt, in its
    original LN form or a hypothetical BN variant for ablation). -/
inductive Normalization where
  | bn
  | ln
deriving Repr, BEq

inductive Padding where
  | same
  | valid
deriving Repr, BEq

inductive Layer where
  | conv2d  (ic oc kSize : Nat) (pad : Padding) (act : Activation)
  | convBn  (ic oc kSize stride : Nat) (pad : Padding)
  | maxPool (size stride : Nat)
  | globalAvgPool
  | flatten
  | dense   (fanIn fanOut : Nat) (act : Activation)
  | residualBlock (ic oc nBlocks firstStride : Nat)
  | bottleneckBlock (ic oc nBlocks firstStride : Nat)
  | separableConv (ic oc stride : Nat)
  | invertedResidual (ic oc expand stride nBlocks : Nat)
  | mbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool) (act : Activation := .swish)
  | mbConvV3 (ic oc expandCh kSize stride : Nat) (useSE : Bool) (act : Activation := .relu)
  | fusedMbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool)
  | uib (ic oc expand stride : Nat) (preDWk postDWk : Nat)  -- Universal Inverted Bottleneck; k=0 means no DW
  | fireModule (ic squeeze expand1x1 expand3x3 : Nat)
  | patchEmbed (ic dim patchSize nPatches : Nat)
  | transformerEncoder (dim heads mlpDim nBlocks : Nat)
  -- Selective state-space block (Mamba / S6); dim = hidden, stateSize = N,
  -- expand = inner-dim multiplier. Not codegen-backed yet — used by the
  -- Bestiary as a shape-only primitive for language-model architectures.
  | mambaBlock (dim stateSize expand nBlocks : Nat)
  -- Swin Transformer stage: N blocks of (W-MSA + MLP) with residuals, plus
  -- the alternating shifted-window variant. Window-local attention at fixed
  -- resolution. Not codegen-backed yet (bestiary primitive).
  | swinStage (dim heads mlpDim windowSize nBlocks : Nat)
  -- Swin patch merging: 2×2 spatial downsample, channels go inDim → outDim
  -- (typically 4·inDim → 2·inDim in the paper, but `outDim` is stated
  -- explicitly to match any hierarchy).
  | patchMerging (inDim outDim : Nat)
  -- UNet encoder stage: 2× (conv3x3 + BN + ReLU) + maxPool-2. Saves
  -- the pre-pool feature map as a skip for the matching `unetUp` later.
  | unetDown (ic oc : Nat)
  -- UNet decoder stage: transposed-conv upsample + concat with the
  -- matching encoder skip + 2× (conv3x3 + BN + ReLU). `ic` is the input
  -- channel count (from the previous decoder or bottleneck); `oc` is
  -- both the output channel count and the expected skip channel count.
  | unetUp (ic oc : Nat)
  -- Transformer decoder stack (DETR-style): N blocks of self-attention
  -- on `nQueries` learned object queries + cross-attention with the
  -- encoder output + FFN. The object-query embedding is part of this
  -- layer's parameters.
  | transformerDecoder (dim heads mlpDim nBlocks nQueries : Nat)
  -- DETR prediction heads, applied independently to each of the
  -- decoder's `nQueries` output tokens: class head (dim → nClasses+1,
  -- with the +1 being the "no object" slot) + box head (3-layer MLP
  -- dim → dim → dim → 4, predicting (cx, cy, w, h)).
  | detrHeads (dim nClasses : Nat)
  -- ShuffleNet v1 stage: `nUnits` shuffle units at a given (ic, oc) pair,
  -- the first being a stride-2 downsampling unit (avg-pool skip) and the
  -- rest being residual units. Internally: 1×1 grouped conv → channel
  -- shuffle → 3×3 depthwise → 1×1 grouped conv, with `groups` channel
  -- partitions driving the grouped convs.
  | shuffleBlock (ic oc groups nUnits : Nat)
  -- ShuffleNet v2 stage (Ma et al.\ 2018): `nUnits` v2 units at a given
  -- (ic, oc) pair, the first being a stride-2 downsampling unit and the
  -- rest being stride-1 basic units. v2 is architecturally different from
  -- v1: no grouped 1×1 convs (violates G2), no element-wise adds (G4).
  -- Basic unit (stride 1): channel split into two halves, identity-skip
  -- the first, (1×1 → 3×3 DW → 1×1) on the second, concat, channel-shuffle.
  -- Downsample unit: both branches see the full input; left does DW-stride-2
  -- + 1×1, right does 1×1 + DW-stride-2 + 1×1; concat doubles channels.
  | shuffleV2Block (ic oc nUnits : Nat)
  -- AlphaFold-2 Evoformer block stack. Operates on a DUAL representation
  -- (MSA: s × r × msaChannels, pair: r × r × pairChannels) and updates
  -- both jointly. Per-block: MSA row attention with pair bias + MSA col
  -- attention + MSA transition + outer-product-mean → pair + triangle
  -- multiplicative update (out/in) + triangle attention (start/end) +
  -- pair transition. All bundled into one primitive here since the dual
  -- representation doesn't fit a linear NetSpec naturally.
  | evoformerBlock (msaChannels pairChannels nBlocks : Nat)
  -- AlphaFold-2 Structure Module: `nBlocks` rounds of Invariant Point
  -- Attention (IPA) over residues + backbone frame updates + side-chain
  -- χ angle prediction. Shared weights across blocks (recurrent).
  | structureModule (singleChannels pairChannels nBlocks : Nat)
  -- MobileViT block (Mehta & Rastegari 2022): local 3×3 conv + 1×1
  -- projection to transformer dim, unfold into patches, `nTxBlocks`
  -- small-transformer blocks across patches, fold back, 1×1 projection
  -- back to `ic`, concat with input, fusion 3×3 conv.
  --   ic      — spatial feature channels (block is ic→ic)
  --   dim     — transformer dim (`d` in the paper, typically 144/192/240)
  --   heads   — attention heads per transformer block
  --   mlpDim  — FFN dim of the inner transformer
  --   nTxBlocks — number of transformer blocks inside the unit (L)
  | mobileVitBlock (ic dim heads mlpDim nTxBlocks : Nat)
  -- ConvNeXt stage (Liu et al. 2022): `nBlocks` ConvNeXt residual blocks
  -- at fixed `channels`. Each block: 7×7 depthwise conv + LN + 1×1
  -- expand (×4) + GELU + 1×1 project + residual skip.
  -- Does NOT include downsampling; see `convNextDownsample` for that.
  -- The `norm` field selects LayerNorm (default, paper) vs BatchNorm;
  -- the `act` field selects GELU (default, paper) vs ReLU/etc. Both
  -- defaults reproduce the Liu et al. recipe.
  | convNextStage (channels nBlocks : Nat)
                  (norm : Normalization := .ln) (act : Activation := .gelu)
  -- ConvNeXt inter-stage downsample: LayerNorm + 2×2 conv stride 2
  -- (ic → oc). Separated from the residual blocks, following the paper's
  -- design choice (point 8 of Liu et al.'s modernization recipe).
  -- The `norm` field selects LN (default) vs BN; same purpose as in
  -- `convNextStage` so a stage and its downsample stay in sync.
  | convNextDownsample (ic oc : Nat) (norm : Normalization := .ln)
  -- WaveNet (van den Oord 2016) residual-block stack with exponentially
  -- growing dilation. One stack = `nLayers` blocks with dilation rates
  -- 2⁰, 2¹, ..., 2^(nLayers−1). Per block: dilated causal conv + gated
  -- activation (tanh ⊙ sigmoid) + 1×1 project back to residualCh + skip
  -- connection + separate 1×1 conv producing a skipCh vector (summed
  -- into the final output head).
  | waveNetBlock (residualCh skipCh nLayers : Nat)
  -- Sinusoidal positional encoding (Vaswani 2017 / NeRF 2020):
  -- γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)).
  -- Parameter-free — just a deterministic frequency basis. Output dim
  -- = inputDim · 2 · numFrequencies.
  | positionalEncoding (inputDim numFrequencies : Nat)
  -- NeRF MLP core (Mildenhall et al. 2020): 8 hidden layers of `hiddenDim`
  -- with ReLU, a mid-skip concatenating the positional encoding at layer 5,
  -- dual output heads (1-dim volume density σ + 3-dim RGB via a direction-
  -- conditioned branch). Signature params:
  --   encodedPosDim — dim of γ(x), typically 60 (=3·2·10)
  --   encodedDirDim — dim of γ(d), typically 24 (=2·2·6)  [view direction]
  --   hiddenDim     — MLP width (paper: 256)
  | nerfMLP (encodedPosDim encodedDirDim hiddenDim : Nat)
  -- Darknet residual stack (Redmon & Farhadi 2018, YOLOv3). `nBlocks`
  -- residual blocks at fixed `channels`, each being: 1×1 conv (c → c/2)
  -- + 3×3 conv (c/2 → c) + residual add. No downsample; pair with a
  -- stride-2 convBn before each stack for the Darknet-53 body.
  | darknetBlock (channels nBlocks : Nat)
  -- Cross-Stage Partial block (Wang et al. 2019, used by YOLOv4/v5/v8/v11
  -- and family). Splits input channels into two halves, processes one
  -- half through a stack of `nBlocks` bottleneck residual blocks, then
  -- concatenates with the untouched half and 1×1-projects to `oc`. The
  -- exact inner block varies across YOLO versions (C3, C2f, C3k2);
  -- this primitive approximates them all at the same abstraction level.
  | cspBlock (ic oc nBlocks : Nat)
  -- Inception module (Szegedy et al. 2014, GoogLeNet). Four parallel
  -- branches concatenated along the channel axis:
  --   b1: 1×1 conv (ic → b1out)
  --   b2: 1×1 (ic → b2reduce) + 3×3 (b2reduce → b2out)
  --   b3: 1×1 (ic → b3reduce) + 5×5 (b3reduce → b3out)
  --   b4: 3×3 maxPool + 1×1 (ic → b4out)
  -- Output channels = b1out + b2out + b3out + b4out.
  | inceptionModule (ic b1out b2reduce b2out b3reduce b3out b4out : Nat)
  -- Atrous Spatial Pyramid Pooling (Chen et al.\ 2018, DeepLab v3+).
  -- Five parallel branches that all emit `oc` channels, concatenated
  -- and fused by a 1×1 conv back to `oc`:
  --   B1: 1×1 conv (ic → oc)
  --   B2/B3/B4: 3×3 atrous convs at rates 6, 12, 18 (ic → oc)
  --     (atrous changes receptive field, not param count)
  --   B5: global avg-pool + 1×1 conv (ic → oc) + bilinear upsample
  --   fusion: 1×1 conv (5·oc → oc)
  -- All branches include BN + ReLU. The atrous rates give the module
  -- a dense multi-scale receptive field at a single feature resolution,
  -- which is why the paper could drop the encoder's last-stage stride
  -- and keep the output resolution high.
  | asppModule (ic oc : Nat)
  -- Feature Pyramid Network (Lin et al.\ 2017, FPN). Takes the last
  -- four stage feature maps of a backbone (with channels c2, c3, c4,
  -- c5 at progressively deeper stages), applies 1×1 lateral convs to
  -- project each to `target` channels, merges via top-down addition
  -- after 2× upsampling, then applies a 3×3 smoothing conv at each
  -- level. Output: four pyramid levels, each with `target` channels,
  -- at the spatial resolutions of C2/C3/C4/C5. The cross-scale
  -- addition is implicit in the bundled primitive (doesn't fit a
  -- linear NetSpec layer-by-layer). Used as the detection-and-
  -- segmentation feature backbone in Mask R-CNN, RetinaNet, and most
  -- 2-stage detection families.
  | fpnModule (c2 c3 c4 c5 target : Nat)
deriving Repr

structure NetSpec where
  name   : String
  layers : List Layer
  imageH : Nat := 28
  imageW : Nat := 28
deriving Repr

structure TrainConfig where
  learningRate : Float
  batchSize    : Nat
  epochs       : Nat
  seed         : Nat := 314159
  momentum     : Float := 0.0
  useAdam      : Bool := false
  weightDecay  : Float := 0.0
  cosineDecay  : Bool := false
  warmupEpochs : Nat := 0
  augment      : Bool := false
  labelSmoothing : Float := 0.0
deriving Repr

inductive DatasetKind where
  | mnist
  | cifar10
  | imagenette
deriving Repr, BEq

/-- IREE compile flags from environment. Defaults to CUDA (sm_86).
    Set `IREE_BACKEND=rocm` and `IREE_CHIP=gfx1100` for AMD GPUs.
    Set `IREE_BACKEND=llvm-cpu` for CPU fallback (no chip needed). -/
def ireeCompileArgs (mlirPath outPath : String) : IO (Array String) := do
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "cuda")
  let baseArgs := #[mlirPath, s!"--iree-hal-target-backends={backend}"]
  let chipArgs ← if backend == "llvm-cpu" then
    pure #[]
  else
    let defaultChip := if backend == "rocm" then "gfx1100" else "sm_86"
    let chip ← (IO.getEnv "IREE_CHIP").map (·.getD defaultChip)
    pure #[s!"--iree-{backend}-target={chip}"]
  return baseArgs ++ chipArgs ++ #["-o", outPath]
