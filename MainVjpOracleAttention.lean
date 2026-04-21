import LeanMlir

/-! VJP oracle: attention — smallest ViT-shaped net exercising
    `transformerBlock_has_vjp_mat` (which bundles LN, MHA with
    scaled-dot-product attention, residuals, and the MLP sublayer).
    MNIST 28×28 → 7×7 patches → 1 block → classifier. -/

def attentionNet : NetSpec where
  name   := "vjp-oracle-attention"
  imageH := 28
  imageW := 28
  layers := [
    .patchEmbed 1 16 7 16,                -- 16 patches of 7×7, dim=16
    .transformerEncoder 16 2 32 1,         -- 1 block, 2 heads, mlpDim=32
    .dense 16 10 .identity                  -- classifier off CLS token
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
  attentionNet.train vjpCfg (args.head?.getD "data") .mnist
