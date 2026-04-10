import LeanJax

/-! ResNet-50 on Imagenette — Bottleneck blocks
    Conv7/2 → Pool → BN3(64→256) → BN4(256→512) → BN6(512→1024) → BN3(1024→2048) → GAP → 10
    ~23.5M params -/

def resnet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 10 .identity
  ]

-- Adam + zero-init residual for bottleneck training
def resnet50Config : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true

#eval resnet50.validate!

def main (args : List String) : IO Unit :=
  runJax resnet50 resnet50Config .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_resnet50.py"
