import LeanMlir

/-! MobileNet v3-Large on Imagenette — IREE training pipeline.
    Inverted residuals with SE, h-swish/h-sigmoid (approximated via swish/sigmoid).
    ~4.2M params, 224×224, 10 classes. -/

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

def mobilenetV3LargeConfig : TrainConfig where
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
  mobilenetV3Large.train mobilenetV3LargeConfig (args.head?.getD "data/imagenette")
