import LeanMlir

/-! EfficientNet V2-S on Imagenette — Fused-MBConv (early stages) + MBConv + Swish + SE.
    224×224, 10 classes. -/

def efficientNetV2S : NetSpec where
  name := "EfficientNet V2-S"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 24 3 2 .same,
    .fusedMbConv  24  24 1 3 1 2 false,
    .fusedMbConv  24  48 4 3 2 4 false,
    .fusedMbConv  48  64 4 3 2 4 false,
    .mbConv  64 128 4 3 2 6 true,
    .mbConv 128 160 6 3 1 9 true,
    .mbConv 160 256 6 3 2 15 true,
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def efficientNetV2SConfig : TrainConfig where
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
  efficientNetV2S.train efficientNetV2SConfig (args.head?.getD "data/imagenette")
