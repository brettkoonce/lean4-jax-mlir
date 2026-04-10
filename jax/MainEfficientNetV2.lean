import LeanJax

/-! EfficientNet V2-S on Imagenette
    Fused-MBConv in early stages, MBConv in later stages.
    ~20M params -/

def efficientNetV2S : NetSpec where
  name := "EfficientNet V2-S"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 24 3 2 .same,                            -- 224→112
    .fusedMbConv  24  24 1 3 1 2 false,                 -- 112, Fused-MBConv1
    .fusedMbConv  24  48 4 3 2 4 false,                 -- 112→56, Fused-MBConv4
    .fusedMbConv  48  64 4 3 2 4 false,                 -- 56→28, Fused-MBConv4
    .mbConv  64 128 4 3 2 6 true,                       -- 28→14, MBConv4+SE
    .mbConv 128 160 6 3 1 9 true,                       -- 14, MBConv6+SE
    .mbConv 160 256 6 3 2 15 true,                      -- 14→7, MBConv6+SE
    .convBn 256 1280 1 1 .same,                         -- 1x1 head
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def efficientNetV2Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval efficientNetV2S.validate!

def main (args : List String) : IO Unit :=
  runJax efficientNetV2S efficientNetV2Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_efficientnet_v2s.py"
