import LeanJax

/-! Vision Transformer (ViT-Tiny) on Imagenette
    Patch 16×16 → 192-dim, 12 blocks, 3 heads, MLP 768
    ~5.5M params -/

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity               -- classification head
  ]

def vitConfig : TrainConfig where
  learningRate := 0.0003
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.01
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval vitTiny.validate!

def main (args : List String) : IO Unit :=
  runJax vitTiny vitConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_vit_tiny.py"
