import LeanMlir

/-! VJP oracle: conv_pool — tests `maxPool_has_vjp` in a realistic context. -/

def convPool : NetSpec where
  name   := "vjp-oracle-conv-pool"
  imageH := 28
  imageW := 28
  layers := [
    .conv2d 1 4 3 .same .identity,   -- 4 × 28 × 28
    .maxPool 2 2,                     -- 4 × 14 × 14 = 784
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

def main (args : List String) : IO Unit :=
  convPool.train vjpCfg (args.head?.getD "data") .mnist
