import LeanJax

/-! MobileNet v2 on Imagenette — Inverted residual blocks
    Conv3/2 → 17 inverted residual blocks → Conv1x1 → GAP → Dense
    ~2.2M params -/

def mobilenetV2 : NetSpec where
  name := "MobileNet v2"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                    -- 224→112
    .invertedResidual  32  16 1 1 1,            -- 112, t=1
    .invertedResidual  16  24 6 2 2,            -- 112→56, t=6
    .invertedResidual  24  32 6 2 3,            -- 56→28, t=6
    .invertedResidual  32  64 6 2 4,            -- 28→14, t=6
    .invertedResidual  64  96 6 1 3,            -- 14, t=6
    .invertedResidual  96 160 6 2 3,            -- 14→7, t=6
    .invertedResidual 160 320 6 1 1,            -- 7, t=6
    .convBn 320 1280 1 1 .same,                 -- 1x1 conv to 1280
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV2Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval mobilenetV2.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV2 mobilenetV2Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_mobilenet_v2.py"
