import Jax

def convPool : NetSpec where
  name   := "vjp-oracle-conv-pool"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 4 3 .same .identity,
    .maxPool 2 2,
    .flatten,
    .dense 784 10 .identity
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

#eval convPool.validate!

def main (args : List String) : IO Unit :=
  runJax convPool vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_conv_pool.py"
