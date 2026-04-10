import LeanJax

/-! EfficientNet-B0 on Imagenette
    MBConv blocks with Swish activation and squeeze-excitation.
    ~4.0M params -/

def efficientNetB0 : NetSpec where
  name := "EfficientNet-B0"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                          -- 224→112
    .mbConv  32  16 1 3 1 1 true,                     -- 112
    .mbConv  16  24 6 3 2 2 true,                     -- 112→56
    .mbConv  24  40 6 5 2 2 true,                     -- 56→28
    .mbConv  40  80 6 3 2 3 true,                     -- 28→14
    .mbConv  80 112 6 5 1 3 true,                     -- 14
    .mbConv 112 192 6 5 2 4 true,                     -- 14→7
    .mbConv 192 320 6 3 1 1 true,                     -- 7
    .convBn 320 1280 1 1 .same,                       -- 1x1 head
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def efficientNetConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval efficientNetB0.validate!

def main (args : List String) : IO Unit :=
  runJax efficientNetB0 efficientNetConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_efficientnet_b0.py"
