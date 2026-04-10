import LeanJax

/-! MobileNet v3-Large on Imagenette
    Hard-swish, squeeze-excitation with hard-sigmoid, inverted residuals.
    ~4.2M params -/

def mobilenetV3Large : NetSpec where
  name := "MobileNet v3-Large"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 16 3 2 .same,                             -- 224→112, HS
    .mbConvV3  16  16  16  3 1 false false,              -- 112, RE
    .mbConvV3  16  24  64  3 2 false false,              -- 112→56, RE
    .mbConvV3  24  24  72  3 1 false false,              -- 56, RE
    .mbConvV3  24  40  72  5 2 true  false,              -- 56→28, RE, SE
    .mbConvV3  40  40 120  5 1 true  false,              -- 28, RE, SE
    .mbConvV3  40  40 120  5 1 true  false,              -- 28, RE, SE
    .mbConvV3  40  80 240  3 2 false true,               -- 28→14, HS
    .mbConvV3  80  80 200  3 1 false true,               -- 14, HS
    .mbConvV3  80  80 184  3 1 false true,               -- 14, HS
    .mbConvV3  80  80 184  3 1 false true,               -- 14, HS
    .mbConvV3  80 112 480  3 1 true  true,               -- 14, HS, SE
    .mbConvV3 112 112 672  5 1 true  true,               -- 14, HS, SE
    .mbConvV3 112 160 672  5 2 true  true,               -- 14→7, HS, SE
    .mbConvV3 160 160 960  5 1 true  true,               -- 7, HS, SE
    .mbConvV3 160 160 960  5 1 true  true,               -- 7, HS, SE
    .convBn 160 960 1 1 .same,                           -- 1x1 head
    .globalAvgPool,
    .dense 960 10 .identity
  ]

def mobilenetV3Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval mobilenetV3Large.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV3Large mobilenetV3Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_mobilenet_v3.py"
