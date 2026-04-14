import LeanMlir

/-! VGG-16-BN on Imagenette — full IREE training pipeline.
    13 conv layers + GAP + dense, all 3x3 convs with BN.
    ~14.7M params (GAP variant), 224×224 input, 10 classes. -/

def vgg16bn : NetSpec where
  name := "VGG-16-BN"
  imageH := 224
  imageW := 224
  layers := [
    -- Block 1
    .convBn   3  64 3 1 .same,
    .convBn  64  64 3 1 .same,
    .maxPool 2 2,
    -- Block 2
    .convBn  64 128 3 1 .same,
    .convBn 128 128 3 1 .same,
    .maxPool 2 2,
    -- Block 3
    .convBn 128 256 3 1 .same,
    .convBn 256 256 3 1 .same,
    .convBn 256 256 3 1 .same,
    .maxPool 2 2,
    -- Block 4
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .maxPool 2 2,
    -- Block 5
    .convBn 512 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .maxPool 2 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def vgg16bnConfig : TrainConfig where
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
  vgg16bn.train vgg16bnConfig (args.head?.getD "data/imagenette")
