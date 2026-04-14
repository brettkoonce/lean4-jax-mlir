import LeanMlir

/-! MobileNet V4-Medium on Imagenette — UIB (Universal Inverted Bottleneck) blocks.
    Conv-only variant (no attention). 224×224, 10 classes. -/

def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .fusedMbConv 32 48 4 3 2 1 false,
    .uib  48  80 4 2 3 5,
    .uib  80  80 2 1 3 3,
    .uib  80 160 6 2 0 3,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 5,
    .uib 160 160 4 1 5 0,
    .uib 160 160 4 1 0 3,
    .uib 160 160 4 1 3 0,
    .uib 160 160 4 1 0 0,
    .uib 160 160 4 1 3 3,
    .uib 160 256 6 2 5 5,
    .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 0 3,
    .uib 256 256 4 1 3 0,
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV4MediumConfig : TrainConfig where
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
  mobilenetV4Medium.train mobilenetV4MediumConfig (args.head?.getD "data/imagenette")
