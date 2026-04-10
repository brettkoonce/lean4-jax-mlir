import LeanJax

/-! ResNet-34 on Imagenette — S4TF book Ch. 4
    Conv7/2 → Pool → Res3(64) → Res4(128) → Res6(256) → Res3(512) → GAP → 10
    ~21.3M params -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def resnetConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true

#eval resnet34.validate!

def main (args : List String) : IO Unit :=
  runJax resnet34 resnetConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_resnet34.py"
