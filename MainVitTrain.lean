import LeanMlir

/-! ViT-Tiny on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → Adam training loop.
    Patch 16×16 → 192-dim, 12 blocks, 3 heads, MLP 768
    ~5.5M params, 224×224 input, 10 classes. -/

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity               -- classification head
  ]

def vitTinyConfig : TrainConfig where
  learningRate := 0.0003
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true
  labelSmoothing := 0.1

def main (args : List String) : IO Unit :=
  vitTiny.train vitTinyConfig (args.head?.getD "data/imagenette")
