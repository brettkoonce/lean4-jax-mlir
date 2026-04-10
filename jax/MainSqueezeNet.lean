import LeanJax

/-! SqueezeNet v1.1 on Imagenette
    Fire modules: squeeze 1x1 → expand (1x1 || 3x3) → concat
    ~740K params — extremely lightweight -/

def squeezenet : NetSpec where
  name := "SqueezeNet v1.1"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 3 2 .same,            -- 224→112
    .maxPool 3 2,                       -- 112→56
    .fireModule  64 16  64  64,         -- fire2: 64→128
    .fireModule 128 16  64  64,         -- fire3: 128→128
    .maxPool 3 2,                       -- 56→28
    .fireModule 128 32 128 128,         -- fire4: 128→256
    .fireModule 256 32 128 128,         -- fire5: 256→256
    .maxPool 3 2,                       -- 28→14
    .fireModule 256 48 192 192,         -- fire6: 256→384
    .fireModule 384 48 192 192,         -- fire7: 384→384
    .fireModule 384 64 256 256,         -- fire8: 384→512
    .fireModule 512 64 256 256,         -- fire9: 512→512
    .convBn 512 10 1 1 .same,          -- classifier 1x1
    .globalAvgPool,
    .dense 10 10 .identity
  ]

def squeezenetConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 192
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true

#eval squeezenet.validate!

def main (args : List String) : IO Unit :=
  runJax squeezenet squeezenetConfig .imagenette
    (args.head? |>.getD "data/imagenette")
    "generated_squeezenet.py"
