import LeanMlir

/-! Ablation study runner. Single binary that runs any ablation config
    based on a command-line argument.

    Usage: .lake/build/bin/ablation <name> [dataDir]

    The ablation name selects a (spec, config, dataset) triple. Each
    configuration strips one component from the full recipe to measure
    its contribution.

    Results go into .lake/build/<spec_name>_<ablation>_params.bin etc.
    The spec name includes the ablation suffix so cached vmfbs don't
    collide between runs. -/

-- ═══════════════════════════════════════════════════════════════════
-- Specs
-- ═══════════════════════════════════════════════════════════════════

-- Chapter 0: Single-layer linear classifier on MNIST
-- Literally one matmul: y = Wx + b, no activation, no hidden layer.
-- The reader can trace the entire gradient by hand on one page.
def mnistLinear : NetSpec where
  name := "MNIST-Linear"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 10 .identity
  ]

-- Chapter 0.5: Single hidden layer MLP
def mnistShallow : NetSpec where
  name := "MNIST-Shallow"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 128 .relu,
    .dense 128  10 .identity
  ]

-- Chapter 1: MLP on MNIST (3-layer, the "real" MLP)
def mnistMlp : NetSpec where
  name := "MNIST-MLP"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- Width ablation: single hidden layer, varying width
def mnistHidden (w : Nat) : NetSpec where
  name := s!"MNIST-H{w}"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 w .relu,
    .dense w  10 .identity
  ]

-- Width ablation: 2-conv CNN, varying filter count (no BN, for ch2)
def mnistCnnWidth (f : Nat) : NetSpec where
  name := s!"MNIST-CNN-f{f}"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 f 3 .same .relu,
    .conv2d f f 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense (f * 14 * 14) 128 .relu,
    .dense 128  10 .identity
  ]

-- Width ablation: 2-conv CNN on CIFAR, varying filter count (with BN, for ch3)
def cifarCnnWidth (f : Nat) : NetSpec where
  name := s!"CIFAR-BN-f{f}"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 f 3 1 .same,
    .convBn f f 3 1 .same,
    .maxPool 2 2,
    .convBn f (f*2) 3 1 .same,
    .convBn (f*2) (f*2) 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense (f * 2 * 8 * 8) 512 .relu,
    .dense 512 10 .identity
  ]

-- Chapter 2: CNN on MNIST (S4TF book match — no BN, 2 conv layers)
def mnistCnnNoBn : NetSpec where
  name := "MNIST-CNN-noBN"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 6272 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

-- Chapter 2 variant: CNN with BN (our architecture)
def mnistCnnBn : NetSpec where
  name := "MNIST-CNN-BN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

-- Chapter 3: CNN on CIFAR (no BN)
def cifarCnnNoBn : NetSpec where
  name := "CIFAR-CNN-noBN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d 3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

-- Chapter 3: CNN on CIFAR (with BN)
def cifarCnnBn : NetSpec where
  name := "CIFAR-CNN-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]


-- ═══════════════════════════════════════════════════════════════════
-- Configs (each ablation strips one thing from the full recipe)
-- ═══════════════════════════════════════════════════════════════════

-- Full recipe for small models
def fullRecipe (lr : Float := 0.001) (epochs : Nat := 15) (batch : Nat := 128) : TrainConfig where
  learningRate := lr
  batchSize    := batch
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := true
  labelSmoothing := 0.1

-- S4TF book baseline: SGD 0.1, constant LR, no regularization
def s4tfBaseline (epochs : Nat := 12) : TrainConfig where
  learningRate := 0.1
  batchSize    := 128
  epochs       := epochs
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- SGD with lower LR (0.01)
def sgdLowLr (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.01
  batchSize    := 128
  epochs       := epochs
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- SGD with book LR (0.002 + momentum)
def sgdLowLr2 (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.002
  batchSize    := 128
  epochs       := epochs
  useAdam      := false
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- Adam only (no aug, no smooth, no wd, no cosine)
def adamOnly (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- Adam + cosine (no aug, no smooth, no wd)
def adamCosine (epochs : Nat := 15) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false
  labelSmoothing := 0.0

-- Adam + cosine + augmentation (no smooth, no wd)
def adamCosineAug (epochs : Nat := 30) : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := epochs
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true
  labelSmoothing := 0.0

-- ═══════════════════════════════════════════════════════════════════
-- Chapter 5: CIFAR-CNN-BN — leave-one-out recipe ablation.
-- Same flag exposure as the R34 chapter (Adam / cosine / warmup / wd /
-- smooth / aug), so the recipe story stays consistent across scales.
-- Reuses `fullRecipe 0.001 30 128` as the baseline with all knobs on.
-- ═══════════════════════════════════════════════════════════════════

def cifarBnFull : TrainConfig := fullRecipe 0.001 30 128

-- Full minus Adam (→ SGD + 0.9 momentum). LR bumped to 0.01 to keep
-- the comparison fair — at Adam's 0.001 SGD undershoots structurally.
def cifarBnNoAdam : TrainConfig :=
  { fullRecipe 0.001 30 128 with
      learningRate := 0.01, useAdam := false, momentum := 0.9 }

def cifarBnNoCosine : TrainConfig :=
  { fullRecipe 0.001 30 128 with cosineDecay := false }

def cifarBnNoWarmup : TrainConfig :=
  { fullRecipe 0.001 30 128 with warmupEpochs := 0 }

def cifarBnNoWd : TrainConfig :=
  { fullRecipe 0.001 30 128 with weightDecay := 0.0 }

def cifarBnNoSmooth : TrainConfig :=
  { fullRecipe 0.001 30 128 with labelSmoothing := 0.0 }

def cifarBnNoAug : TrainConfig :=
  { fullRecipe 0.001 30 128 with augment := false }

-- Plain SGD + momentum, no other tricks. Mirror of `r34Bare`.
def cifarBnBare : TrainConfig where
  learningRate := 0.01
  batchSize    := 128
  epochs       := 30
  useAdam      := false
  momentum     := 0.9
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- ═══════════════════════════════════════════════════════════════════
-- Chapter 6: ResNet-34 Imagenette — leave-one-out recipe ablation.
--
-- Seven configs: one full recipe + six identical copies each missing
-- one component. Each ablation run answers "given everything else is
-- present, what does THIS ingredient earn us?" Leaving the baseline
-- out deliberately — "plain SGD at lr=0.002" from the first book
-- would need its own LR sweep to converge cleanly on ResNet-34, and
-- chasing a fair baseline is more expensive than the ablation itself.
-- The Chapter cites the first book's numbers as the implicit
-- starting point; the leave-one-out table below measures lift.
-- ═══════════════════════════════════════════════════════════════════

def resnet34Spec : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def r34Full : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

-- Leave-one-out: full recipe minus Adam (→ SGD + 0.9 momentum).
-- LR bumped to 0.01 (typical SGD+momentum LR for ResNet-scale); at
-- Adam's 0.001 SGD would underperform purely due to undershoot rather
-- than recipe-structure effects. Chosen to make the comparison fair.
def r34NoAdam : TrainConfig where
  learningRate := 0.01   -- see comment above
  batchSize    := 32
  epochs       := 80
  useAdam      := false
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

def r34NoCosine : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

def r34NoWarmup : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 0
  augment      := true
  labelSmoothing := 0.1

def r34NoWd : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

def r34NoSmooth : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.0

def r34NoAug : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := false
  labelSmoothing := 0.1

-- Plain SGD + momentum, no other tricks. Establishes the
-- "what does the modern recipe actually buy us?" baseline
-- alongside the leave-one-out runs.
def r34Bare : TrainConfig where
  learningRate := 0.01
  batchSize    := 32
  epochs       := 80
  useAdam      := false
  momentum     := 0.9
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

-- ═══════════════════════════════════════════════════════════════════
-- Chapter 8: EfficientNet-B0 Imagenette — Swish vs ReLU activation
-- ablation. The "show, don't tell" answer to "what does Swish buy us?"
--
-- Two configs sharing init seed + batch order + recipe:
--   enet-b0-swish: default Swish in every MBConv block (current B0).
--   enet-b0-relu : same architecture, all MBConv activations forced to
--                  ReLU. Measures the activation-function lift in
--                  isolation.
-- ═══════════════════════════════════════════════════════════════════

def enetB0SwishSpec : NetSpec where
  name := "EfficientNet-B0-Swish"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .mbConv  32  16 1 3 1 1 true (act := .swish),
    .mbConv  16  24 6 3 2 2 true (act := .swish),
    .mbConv  24  40 6 5 2 2 true (act := .swish),
    .mbConv  40  80 6 3 2 3 true (act := .swish),
    .mbConv  80 112 6 5 1 3 true (act := .swish),
    .mbConv 112 192 6 5 2 4 true (act := .swish),
    .mbConv 192 320 6 3 1 1 true (act := .swish),
    .convBn 320 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def enetB0ReluSpec : NetSpec where
  name := "EfficientNet-B0-ReLU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .mbConv  32  16 1 3 1 1 true (act := .relu),
    .mbConv  16  24 6 3 2 2 true (act := .relu),
    .mbConv  24  40 6 5 2 2 true (act := .relu),
    .mbConv  40  80 6 3 2 3 true (act := .relu),
    .mbConv  80 112 6 5 1 3 true (act := .relu),
    .mbConv 112 192 6 5 2 4 true (act := .relu),
    .mbConv 192 320 6 3 1 1 true (act := .relu),
    .convBn 320 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

-- Standard EfficientNet-B0 recipe (matches MainEfficientNetTrain.lean).
def enetB0Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

-- ═══════════════════════════════════════════════════════════════════
-- Chapter 10: ViT-Tiny on Imagenette — DeiT-style augmentation ablation.
-- Original ViT was data-hungry (needed JFT-300M); DeiT (Touvron 2020)
-- showed the recipe is the architecture: Mixup + CutMix + RandAugment +
-- Random Erasing + EMA + Stochastic Depth lifted ViT-S from ~78% to
-- 79.8% on ImageNet-1K from scratch. We ship the dataloader-only
-- subset (Mixup, CutMix, Random Erasing) — the truly cheap subset of
-- the recipe — to test whether the recipe gap explains why our
-- bare-recipe ViT-Tiny lands at ~72% Imagenette while CNNs hit ~88%.
-- ═══════════════════════════════════════════════════════════════════

def vitTinyAblationSpec : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,
    .transformerEncoder 192 3 768 12,
    .dense 192 10 .identity
  ]

def vitTinyBareConfig : TrainConfig where
  learningRate := 0.0003
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true
  labelSmoothing := 0.1

def vitTinyEraseConfig : TrainConfig :=
  { vitTinyBareConfig with randomErasing := true }

def vitTinyRandAugConfig : TrainConfig :=
  { vitTinyBareConfig with useRandAugment := true, randAugmentN := 2, randAugmentM := 9.0 }

def vitTinyMixupConfig : TrainConfig :=
  { vitTinyBareConfig with useMixup := true, mixupAlpha := 0.8 }

def vitTinyCutmixConfig : TrainConfig :=
  { vitTinyBareConfig with useCutmix := true, cutmixAlpha := 1.0 }

def vitTinyKnnMixupConfig : TrainConfig :=
  { vitTinyBareConfig with useKnnMixup := true, knnMixupAlpha := 1.0 }

-- Focal loss requires no smoothing + no soft-label aug; disable both.
def vitTinyFocalConfig : TrainConfig :=
  { vitTinyBareConfig with
      labelSmoothing := 0.0,
      useFocal := true, focalGamma := 2.0 }

-- "Full" = mixup XOR cutmix (paper convention picks one per batch);
-- here we use mixup + RE since CutMix needs its own per-batch decision
-- and our codegen path picks Mixup when both are flagged.
def vitTinyFullConfig : TrainConfig :=
  { vitTinyBareConfig with
      useMixup := true, mixupAlpha := 0.8,
      randomErasing := true }

-- DeiT-recipe knob ablations: EMA, SWA, and stacks-on-top-of-CutMix.
-- Each tests a single new training-loop knob in isolation, then layered
-- onto the best aug variant (CutMix) to see if the gains compound.

def vitTinyEmaConfig : TrainConfig :=
  { vitTinyBareConfig with useEMA := true }

def vitTinySwaConfig : TrainConfig :=
  { vitTinyBareConfig with useSWA := true, swaStartEpoch := 60 }

def vitTinyCutmixEmaConfig : TrainConfig :=
  { vitTinyCutmixConfig with useEMA := true }

def vitTinyCutmixSwaConfig : TrainConfig :=
  { vitTinyCutmixConfig with useSWA := true, swaStartEpoch := 60 }

-- SWAG (extends SWA with the diagonal+low-rank covariance) and TTA
-- (M-pass augmented eval). Plus a "kitchen-sink" cell that stacks
-- everything: CutMix + EMA + SWA + SWAG + TTA.

def vitTinySwagConfig : TrainConfig :=
  { vitTinyBareConfig with
      useSWA := true, swaStartEpoch := 60,
      useSWAG := true, swagK := 20, swagSamples := 30 }

def vitTinyTtaConfig : TrainConfig :=
  { vitTinyBareConfig with useTTA := true, ttaSamples := 5 }

def vitTinyKitchenSinkConfig : TrainConfig :=
  { vitTinyBareConfig with
      useCutmix := true, cutmixAlpha := 1.0,
      useEMA := true,
      useSWA := true, swaStartEpoch := 60,
      useSWAG := true, swagK := 20, swagSamples := 30,
      useTTA := true, ttaSamples := 5 }

-- ═══════════════════════════════════════════════════════════════════
-- Chapter 9: ConvNeXt-Tiny on Imagenette — LayerNorm + GELU on a
-- pure-CNN backbone. The "can a CNN still compete in 2022" answer
-- (Liu et al. 2022). 1D activation ablation: GELU vs ReLU, both
-- with LN as the per-block norm.
--
--   convnext-tiny-gelu : paper recipe (LN + GELU + LayerScale).
--   convnext-tiny-relu : same architecture, ReLU instead of GELU.
--                        Measures the activation lift in isolation.
--
-- Plus a CIFAR-sized "convnext-mini" pair using a single ConvNeXt
-- stage at 32×32 — small enough that the train-step vmfb compiles
-- in seconds for fast smoke.
-- ═══════════════════════════════════════════════════════════════════

def convNextTinyGeluSpec : NetSpec where
  name := "ConvNeXt-T-GELU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .ln .gelu,
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def convNextTinyReluSpec : NetSpec where
  name := "ConvNeXt-T-ReLU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .ln .relu,
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .relu,
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .relu,
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .relu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def convNextTinyConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true
  labelSmoothing := 0.1

-- Mini CIFAR-sized ConvNeXt: one stage at 32 channels and one at 64.
-- Same primitives, fast to compile, useful for the activation ablation
-- on CIFAR-10 if Imagenette compute isn't available.
def convNextMiniGeluSpec : NetSpec where
  name := "ConvNeXt-Mini-GELU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .ln .gelu,
    .convNextDownsample 32 64,
    .convNextStage 64 2 .ln .gelu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

def convNextMiniReluSpec : NetSpec where
  name := "ConvNeXt-Mini-ReLU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .ln .relu,
    .convNextDownsample 32 64,
    .convNextStage 64 2 .ln .relu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

-- BN variants: same architecture, only the per-block norm differs
-- (per-channel BN over batch+spatial vs per-spatial LN over channels).
-- Completes the 2D activation × normalization ablation cell.

def convNextTinyBnGeluSpec : NetSpec where
  name := "ConvNeXt-T-BN-GELU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .bn .gelu,
    .convNextDownsample 96 192 .bn,
    .convNextStage 192 3 .bn .gelu,
    .convNextDownsample 192 384 .bn,
    .convNextStage 384 9 .bn .gelu,
    .convNextDownsample 384 768 .bn,
    .convNextStage 768 3 .bn .gelu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def convNextTinyBnReluSpec : NetSpec where
  name := "ConvNeXt-T-BN-ReLU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .bn .relu,
    .convNextDownsample 96 192 .bn,
    .convNextStage 192 3 .bn .relu,
    .convNextDownsample 192 384 .bn,
    .convNextStage 384 9 .bn .relu,
    .convNextDownsample 384 768 .bn,
    .convNextStage 768 3 .bn .relu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def convNextMiniBnGeluSpec : NetSpec where
  name := "ConvNeXt-Mini-BN-GELU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .bn .gelu,
    .convNextDownsample 32 64 .bn,
    .convNextStage 64 2 .bn .gelu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

def convNextMiniBnReluSpec : NetSpec where
  name := "ConvNeXt-Mini-BN-ReLU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .bn .relu,
    .convNextDownsample 32 64 .bn,
    .convNextStage 64 2 .bn .relu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

def convNextMiniConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 30
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true
  labelSmoothing := 0.1

-- ═══════════════════════════════════════════════════════════════════
-- Ablation registry
-- ═══════════════════════════════════════════════════════════════════

structure AblationRun where
  spec     : NetSpec
  config   : TrainConfig
  dataset  : DatasetKind
  dataDir  : String  -- default data dir

def ablations : List (String × AblationRun) := [
  -- Chapter 0: Linear classifier (one matmul, ~7850 params)
  ("linear-sgd",       ⟨mnistLinear,  s4tfBaseline 12,  .mnist, "data"⟩),
  ("linear-adam",      ⟨mnistLinear,  adamOnly 12,      .mnist, "data"⟩),

  -- Chapter 0.5: Shallow MLP (one hidden layer, ~101K params)
  ("shallow-sgd",      ⟨mnistShallow, s4tfBaseline 15,  .mnist, "data"⟩),
  ("shallow-adam",     ⟨mnistShallow, adamOnly 15,      .mnist, "data"⟩),

  -- Width ablation: single hidden layer MLP, SGD 0.002
  ("width-h64",        ⟨mnistHidden 64,  sgdLowLr2 15, .mnist, "data"⟩),
  ("width-h128",       ⟨mnistHidden 128, sgdLowLr2 15, .mnist, "data"⟩),
  ("width-h512",       ⟨mnistHidden 512, sgdLowLr2 15, .mnist, "data"⟩),

  -- Width ablation: MNIST CNN (no BN), SGD 0.002
  ("width-cnn-f8",     ⟨mnistCnnWidth 8,  sgdLowLr2 15, .mnist, "data"⟩),
  ("width-cnn-f64",    ⟨mnistCnnWidth 64, sgdLowLr2 15, .mnist, "data"⟩),

  -- Width ablation: CIFAR CNN (with BN), SGD 0.002
  ("width-cifar-f8",   ⟨cifarCnnWidth 8,  sgdLowLr2 30, .cifar10, "data"⟩),
  ("width-cifar-f64",  ⟨cifarCnnWidth 64, sgdLowLr2 30, .cifar10, "data"⟩),

  -- Chapter 1: 3-layer MLP (~670K params)
  ("mlp-sgd",         ⟨mnistMlp,     s4tfBaseline 12,  .mnist, "data"⟩),
  ("mlp-sgd-low",     ⟨mnistMlp,     sgdLowLr 15,     .mnist, "data"⟩),
  ("mlp-adam",         ⟨mnistMlp,     adamOnly 12,     .mnist, "data"⟩),
  ("mlp-full",         ⟨mnistMlp,     fullRecipe 0.001 12 128, .mnist, "data"⟩),

  -- Chapter 2: CNN on MNIST (no BN, S4TF architecture)
  ("cnn-nobn-sgd",     ⟨mnistCnnNoBn, s4tfBaseline 12, .mnist, "data"⟩),

  -- Chapter 2: CNN on MNIST (with BN, our architecture)
  ("cnn-bn-sgd",       ⟨mnistCnnBn,   s4tfBaseline 15, .mnist, "data"⟩),
  ("cnn-bn-full",      ⟨mnistCnnBn,   fullRecipe 0.001 15 128, .mnist, "data"⟩),

  -- Chapter 3: CIFAR (no BN — should struggle). Direct s4tf baseline
  -- match: SGD 0.1, no augmentation, no recipe tricks.
  ("cifar-nobn-sgd",   ⟨cifarCnnNoBn, s4tfBaseline 30, .cifar10, "data"⟩),

  -- Chapter 5: CIFAR-CNN-BN leave-one-out recipe ablation. Same flag
  -- exposure as the R34 chapter; cifar-bn-full has all knobs on, each
  -- cifar-bn-no-X removes one ingredient. Compare rows to measure lift.
  ("cifar-bn-bare",      ⟨cifarCnnBn, cifarBnBare,     .cifar10, "data"⟩),
  ("cifar-bn-no-adam",   ⟨cifarCnnBn, cifarBnNoAdam,   .cifar10, "data"⟩),
  ("cifar-bn-no-cosine", ⟨cifarCnnBn, cifarBnNoCosine, .cifar10, "data"⟩),
  ("cifar-bn-no-warmup", ⟨cifarCnnBn, cifarBnNoWarmup, .cifar10, "data"⟩),
  ("cifar-bn-no-wd",     ⟨cifarCnnBn, cifarBnNoWd,     .cifar10, "data"⟩),
  ("cifar-bn-no-smooth", ⟨cifarCnnBn, cifarBnNoSmooth, .cifar10, "data"⟩),
  ("cifar-bn-no-aug",    ⟨cifarCnnBn, cifarBnNoAug,    .cifar10, "data"⟩),
  ("cifar-bn-full",      ⟨cifarCnnBn, cifarBnFull,     .cifar10, "data"⟩),

  -- Chapter 6: ResNet-34 Imagenette leave-one-out recipe ablation.
  -- r34-full is the headline; each r34-no-X removes one ingredient
  -- from the full recipe. Compare rows to measure marginal lift.
  ("r34-full",      ⟨resnet34Spec, r34Full,     .imagenette, "data/imagenette"⟩),
  ("r34-no-adam",   ⟨resnet34Spec, r34NoAdam,   .imagenette, "data/imagenette"⟩),
  ("r34-no-cosine", ⟨resnet34Spec, r34NoCosine, .imagenette, "data/imagenette"⟩),
  ("r34-no-warmup", ⟨resnet34Spec, r34NoWarmup, .imagenette, "data/imagenette"⟩),
  ("r34-no-wd",     ⟨resnet34Spec, r34NoWd,     .imagenette, "data/imagenette"⟩),
  ("r34-no-smooth", ⟨resnet34Spec, r34NoSmooth, .imagenette, "data/imagenette"⟩),
  ("r34-no-aug",    ⟨resnet34Spec, r34NoAug,    .imagenette, "data/imagenette"⟩),
  ("r34-bare",      ⟨resnet34Spec, r34Bare,     .imagenette, "data/imagenette"⟩),

  -- Chapter 8: EfficientNet-B0 activation ablation. Swish (default) vs
  -- ReLU. Same recipe as r34-full to factor out optimizer/schedule
  -- effects; the only knob varied is the MBConv activation function.
  ("enet-b0-swish", ⟨enetB0SwishSpec, enetB0Config, .imagenette, "data/imagenette"⟩),
  ("enet-b0-relu",  ⟨enetB0ReluSpec,  enetB0Config, .imagenette, "data/imagenette"⟩),

  -- Chapter 10: ViT-Tiny augmentation ablation. Same architecture +
  -- recipe; only the dataloader's augmentation pack varies. Tests
  -- whether ViT's "needs data scale" reputation reduces to "needs
  -- the DeiT recipe" — at our Imagenette scale, with only the cheap
  -- subset of the DeiT recipe.
  ("vit-tiny-bare",        ⟨vitTinyAblationSpec, vitTinyBareConfig,        .imagenette, "data/imagenette"⟩),
  ("vit-tiny-erase",       ⟨vitTinyAblationSpec, vitTinyEraseConfig,       .imagenette, "data/imagenette"⟩),
  ("vit-tiny-randaug",     ⟨vitTinyAblationSpec, vitTinyRandAugConfig,     .imagenette, "data/imagenette"⟩),
  ("vit-tiny-mixup",       ⟨vitTinyAblationSpec, vitTinyMixupConfig,       .imagenette, "data/imagenette"⟩),
  ("vit-tiny-cutmix",      ⟨vitTinyAblationSpec, vitTinyCutmixConfig,      .imagenette, "data/imagenette"⟩),
  ("vit-tiny-knn-mixup",   ⟨vitTinyAblationSpec, vitTinyKnnMixupConfig,    .imagenette, "data/imagenette"⟩),
  ("vit-tiny-focal",       ⟨vitTinyAblationSpec, vitTinyFocalConfig,       .imagenette, "data/imagenette"⟩),
  ("vit-tiny-full",        ⟨vitTinyAblationSpec, vitTinyFullConfig,        .imagenette, "data/imagenette"⟩),
  ("vit-tiny-ema",         ⟨vitTinyAblationSpec, vitTinyEmaConfig,         .imagenette, "data/imagenette"⟩),
  ("vit-tiny-swa",         ⟨vitTinyAblationSpec, vitTinySwaConfig,         .imagenette, "data/imagenette"⟩),
  ("vit-tiny-swag",        ⟨vitTinyAblationSpec, vitTinySwagConfig,        .imagenette, "data/imagenette"⟩),
  ("vit-tiny-tta",         ⟨vitTinyAblationSpec, vitTinyTtaConfig,         .imagenette, "data/imagenette"⟩),
  ("vit-tiny-cutmix-ema",  ⟨vitTinyAblationSpec, vitTinyCutmixEmaConfig,   .imagenette, "data/imagenette"⟩),
  ("vit-tiny-cutmix-swa",  ⟨vitTinyAblationSpec, vitTinyCutmixSwaConfig,   .imagenette, "data/imagenette"⟩),
  ("vit-tiny-kitchensink", ⟨vitTinyAblationSpec, vitTinyKitchenSinkConfig, .imagenette, "data/imagenette"⟩),

  -- Chapter 9: ConvNeXt-Tiny activation ablation (GELU vs ReLU, both
  -- with LN). The full paper recipe on Imagenette (224×224); a CIFAR
  -- "mini" pair at 32×32 doubles as a fast-compile smoke variant.
  ("convnext-tiny-gelu",    ⟨convNextTinyGeluSpec,   convNextTinyConfig, .imagenette, "data/imagenette"⟩),
  ("convnext-tiny-relu",    ⟨convNextTinyReluSpec,   convNextTinyConfig, .imagenette, "data/imagenette"⟩),
  ("convnext-tiny-bn-gelu", ⟨convNextTinyBnGeluSpec, convNextTinyConfig, .imagenette, "data/imagenette"⟩),
  ("convnext-tiny-bn-relu", ⟨convNextTinyBnReluSpec, convNextTinyConfig, .imagenette, "data/imagenette"⟩),
  ("convnext-mini-gelu",    ⟨convNextMiniGeluSpec,   convNextMiniConfig, .cifar10,    "data"⟩),
  ("convnext-mini-relu",    ⟨convNextMiniReluSpec,   convNextMiniConfig, .cifar10,    "data"⟩),
  ("convnext-mini-bn-gelu", ⟨convNextMiniBnGeluSpec, convNextMiniConfig, .cifar10,    "data"⟩),
  ("convnext-mini-bn-relu", ⟨convNextMiniBnReluSpec, convNextMiniConfig, .cifar10,    "data"⟩)
]

def main (args : List String) : IO Unit := do
  let name := args.head?.getD ""
  if name == "" || name == "--list" then
    IO.println "Available ablations:"
    for (n, _) in ablations do
      IO.println s!"  {n}"
    IO.println "\nUsage: ablation <name> [dataDir]"
    return

  match ablations.lookup name with
  | some run =>
    let dataDir := (args.tail.head?).getD run.dataDir
    -- Override the spec name to include the ablation suffix for unique vmfb paths
    let spec := { run.spec with name := run.spec.name ++ "-" ++ name }
    IO.eprintln s!"Ablation: {name}"
    IO.eprintln s!"  spec: {run.spec.name}, optimizer: {if run.config.useAdam then "Adam" else "SGD"}"
    IO.eprintln s!"  lr: {run.config.learningRate}, cosine: {run.config.cosineDecay}, wd: {run.config.weightDecay}"
    IO.eprintln s!"  aug: {run.config.augment}, label_smooth: {run.config.labelSmoothing}"
    spec.train run.config dataDir run.dataset
  | none =>
    IO.eprintln s!"Unknown ablation: {name}"
    IO.eprintln "Use --list to see available ablations."
    IO.Process.exit 1
