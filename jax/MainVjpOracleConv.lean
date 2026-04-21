import Jax

def convOnly : NetSpec where
  name   := "vjp-oracle-conv"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 4 3 .same .identity,
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

#eval convOnly.validate!

def main (args : List String) : IO Unit :=
  runJax convOnly vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_conv.py"
