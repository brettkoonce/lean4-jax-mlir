import LeanMlir

/-! VJP oracle: mbConv — tests `elemwiseProduct_has_vjp` (SE gate) plus
    the MBConv composition (expand + depthwise + SE + project with Swish).
    This is the one axiom family not already covered by the other oracle
    cases. -/

def mbConvNet : NetSpec where
  name   := "vjp-oracle-mbconv"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,                -- stem: 4×28×28
    .mbConv 4 4 2 3 1 1 true,              -- expand=2, kSize=3, stride=1, n=1, SE on
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
  mbConvNet.train vjpCfg (args.head?.getD "data") .mnist
