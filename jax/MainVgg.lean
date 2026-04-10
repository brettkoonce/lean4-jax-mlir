import LeanJax

/-! VGG-16 (with BN) on Imagenette
    13 conv layers + 3 dense layers, all 3x3 convs.
    ~134M params (mostly in the dense layers) -/

def vgg16bn : NetSpec where
  name := "VGG-16-BN"
  imageH := 224
  imageW := 224
  layers := [
    -- Block 1
    .convBn   3  64 3 1 .same,         -- 224
    .convBn  64  64 3 1 .same,         -- 224
    .maxPool 2 2,                       -- 224→112
    -- Block 2
    .convBn  64 128 3 1 .same,         -- 112
    .convBn 128 128 3 1 .same,         -- 112
    .maxPool 2 2,                       -- 112→56
    -- Block 3
    .convBn 128 256 3 1 .same,         -- 56
    .convBn 256 256 3 1 .same,         -- 56
    .convBn 256 256 3 1 .same,         -- 56
    .maxPool 2 2,                       -- 56→28
    -- Block 4
    .convBn 256 512 3 1 .same,         -- 28
    .convBn 512 512 3 1 .same,         -- 28
    .convBn 512 512 3 1 .same,         -- 28
    .maxPool 2 2,                       -- 28→14
    -- Block 5
    .convBn 512 512 3 1 .same,         -- 14
    .convBn 512 512 3 1 .same,         -- 14
    .convBn 512 512 3 1 .same,         -- 14
    .maxPool 2 2,                       -- 14→7
    -- Classifier
    .globalAvgPool,                     -- 7→1 (GAP instead of flatten+FC7*7*512)
    .dense 512 10 .identity
  ]

-- Adam works better than SGD for this deep sequential model
def vggConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 50
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true

#eval vgg16bn.validate!

def main (args : List String) : IO Unit :=
  runJax vgg16bn vggConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_vgg16bn.py"
