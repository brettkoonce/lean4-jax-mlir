/-! ML specification types: Layer, NetSpec, TrainConfig, DatasetKind. -/

inductive Activation where
  | relu
  | relu6
  | identity
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
  | mbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool)
  | mbConvV3 (ic oc expandCh kSize stride : Nat) (useSE useHSwish : Bool)
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
