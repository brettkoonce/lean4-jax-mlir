import LeanMlir

/-! ResNet-34 on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → Adam training loop.
    ~21.3M params, 224×224 input, 10 classes.

    Architecture and training recipe are the only things this file
    defines; everything else (param shapes, MLIR codegen, vmfb compile,
    init, training loop, val eval, save) lives in `LeanMlir/Train.lean`. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                    -- 2/2 instead of 3/2 (IREE compat)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def resnet34Config : TrainConfig where
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
  resnet34.train resnet34Config (args.head?.getD "data/imagenette")
