import Jax

/-! MNIST CNN — phase-2 mirror of `MainMnistCnnTrain.lean` (phase 3).

    Same NetSpec, same training config, same seed — so `tests/diff_traces.py`
    can compare a phase-2 JAX run against a phase-3 Lean→IREE run step by
    step (see traces/CROSS_BACKEND_RESULTS.md for the MLP precedent).

    Architecture: 4 conv-BN layers (1→32→32→64→64) with two maxPool/2
    downsamples, flatten, then two dense (3136→512→10). ~1.7M params. -/

def mnistCnn : NetSpec where
  name := "MNIST-CNN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

def mnistCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 15
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false

#eval mnistCnn.validate!

def main (args : List String) : IO Unit :=
  runJax mnistCnn mnistCnnConfig .mnist (args.head? |>.getD "data") "generated_mnist_cnn.py"
