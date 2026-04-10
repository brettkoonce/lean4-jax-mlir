import LeanJax

/-! MobileNet v1 on Imagenette
    Standard conv → 13 depthwise-separable convs → GAP → Dense
    ~3.2M params (1.0 width multiplier) -/

def mobilenetV1 : NetSpec where
  name := "MobileNet v1"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,          -- 224→112
    .separableConv  32  64 1,         -- 112
    .separableConv  64 128 2,         -- 112→56
    .separableConv 128 128 1,         -- 56
    .separableConv 128 256 2,         -- 56→28
    .separableConv 256 256 1,         -- 28
    .separableConv 256 512 2,         -- 28→14
    .separableConv 512 512 1,         -- 14  (×5)
    .separableConv 512 512 1,
    .separableConv 512 512 1,
    .separableConv 512 512 1,
    .separableConv 512 512 1,
    .separableConv 512 1024 2,        -- 14→7
    .separableConv 1024 1024 1,       -- 7
    .globalAvgPool,
    .dense 1024 10 .identity
  ]

def mobilenetConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval mobilenetV1.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV1 mobilenetConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_mobilenet_v1.py"
