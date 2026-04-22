import Jax

def mbConvNet : NetSpec where
  name   := "vjp-oracle-mbconv"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,
    .mbConv 4 4 2 3 1 1 true,
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

#eval mbConvNet.validate!

def main (args : List String) : IO Unit :=
  runJax mbConvNet vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_mbconv.py"
