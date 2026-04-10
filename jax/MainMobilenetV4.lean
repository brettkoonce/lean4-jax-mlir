import LeanJax

/-! MobileNet V4-Medium on Imagenette
    Universal Inverted Bottleneck blocks with varied DW configurations.
    Conv-only variant (no attention). -/

def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                    -- 224→112
    .fusedMbConv 32 48 4 3 2 1 false,           -- 112→56, FusedIB
    -- Stage 2: ExtraDW + IB
    .uib  48  80 4 2 3 5,                        -- 56→28, ExtraDW
    .uib  80  80 2 1 3 3,                        -- 28, ExtraDW
    -- Stage 3: mixed UIB blocks
    .uib  80 160 6 2 0 3,                        -- 28→14, IB
    .uib 160 160 4 1 3 3,                        -- 14, ExtraDW
    .uib 160 160 4 1 3 5,                        -- 14, ExtraDW
    .uib 160 160 4 1 5 0,                        -- 14, ConvNext
    .uib 160 160 4 1 0 3,                        -- 14, IB
    .uib 160 160 4 1 3 0,                        -- 14, ConvNext
    .uib 160 160 4 1 0 0,                        -- 14, FFN
    .uib 160 160 4 1 3 3,                        -- 14, ExtraDW
    -- Stage 4
    .uib 160 256 6 2 5 5,                        -- 14→7, ExtraDW
    .uib 256 256 4 1 5 5,                        -- 7, ExtraDW
    .uib 256 256 4 1 0 3,                        -- 7, IB
    .uib 256 256 4 1 3 0,                        -- 7, ConvNext
    .convBn 256 1280 1 1 .same,                 -- 1x1 head
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV4Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval mobilenetV4Medium.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV4Medium mobilenetV4Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_mobilenet_v4.py"
