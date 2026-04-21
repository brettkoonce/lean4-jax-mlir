import Jax

def attentionNet : NetSpec where
  name   := "vjp-oracle-attention"
  imageH := 28
  imageW := 28
  layers := [
    .patchEmbed 1 16 7 16,
    .transformerEncoder 16 2 32 1,
    .dense 16 10 .identity
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

#eval attentionNet.validate!

def main (args : List String) : IO Unit :=
  runJax attentionNet vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_attention.py"
