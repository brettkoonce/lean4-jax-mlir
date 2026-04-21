import LeanMlir

/-! VJP oracle: convbn — tests `convBn_has_vjp` (conv + BN + ReLU). -/

def convBnOnly : NetSpec where
  name   := "vjp-oracle-convbn"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,          -- conv + BN + ReLU, 4 × 28 × 28
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
  convBnOnly.train vjpCfg (args.head?.getD "data") .mnist
