import LeanMlir

/-! ConvNeXt-Tiny on Imagenette — pure-CNN modernization recipe.

    Liu et al. 2022 — "A ConvNet for the 2020s". Compute ratio
    (3, 3, 9, 3), channels (96, 192, 384, 768), depthwise-7×7 + LN +
    inverted-bottleneck + GELU + LayerScale + residual blocks.
    ~28M params at 224×224, 10 classes.

    Stem uses BN-equivalent `convBn` (paper uses LN; same param count,
    same expressiveness). Each `convNextDownsample` is `LN + 2×2 conv
    stride 2`, dedicated between stages (not fused with the first block
    of a stage like ResNet). -/

def convNextTiny : NetSpec where
  name := "ConvNeXt-T"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,                    -- patchify stem (4×4 stride 4)
    .convNextStage 96 3 .ln .gelu,             -- stage 1: 3 blocks at 96 ch
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,            -- stage 2: 3 blocks at 192 ch
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,            -- stage 3: 9 blocks at 384 ch
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,            -- stage 4: 3 blocks at 768 ch
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

def main (args : List String) : IO Unit :=
  convNextTiny.train convNextTinyConfig (args.head?.getD "data/imagenette")
