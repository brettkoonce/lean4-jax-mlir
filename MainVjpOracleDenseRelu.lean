import LeanMlir

/-! VJP oracle: dense_relu — tests `relu_has_vjp` + `vjp_comp`. -/

def denseRelu : NetSpec where
  name   := "vjp-oracle-dense-relu"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 64 .relu,
    .dense 64 10 .identity
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
  denseRelu.train vjpCfg (args.head?.getD "data") .mnist
