import LeanJax

/-! MNIST MLP — S4TF book Ch. 1
    784 → 512 (ReLU) → 512 (ReLU) → 10 -/

def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

def mnistConfig : TrainConfig where
  learningRate := 0.1
  batchSize    := 128
  epochs       := 12

#eval mnistMlp.validate!

def main (args : List String) : IO Unit :=
  runJax mnistMlp mnistConfig .mnist (args.head? |>.getD "data") "generated_mnist_mlp.py"
