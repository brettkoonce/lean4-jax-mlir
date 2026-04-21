import LeanMlir

/-! VJP oracle: conv — tests `conv2d_has_vjp` + `flatten_has_vjp`. -/

def convOnly : NetSpec where
  name   := "vjp-oracle-conv"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 4 3 .same .identity,   -- 4 × 28 × 28 = 3136
    .flatten,
    .dense 3136 10 .identity
  ]

def vjpCfg : TrainConfig where
  learningRate := 0.001
  batchSize    := 4
  epochs       := 1
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit :=
  convOnly.train vjpCfg (args.head?.getD "data") .mnist
