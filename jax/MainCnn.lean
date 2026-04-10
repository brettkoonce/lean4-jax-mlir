import LeanJax

/-! MNIST CNN — S4TF book Ch. 2
    Conv(1→32,3×3) → Conv(32→32,3×3) → Pool → 6272→512→512→10 -/

def mnistCnn : NetSpec where
  name := "MNIST CNN"
  layers := [
    .conv2d  1 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 6272 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

def cnnConfig : TrainConfig where
  learningRate := 0.01
  batchSize    := 128
  epochs       := 12

#eval mnistCnn.validate!

def main (args : List String) : IO Unit :=
  runJax mnistCnn cnnConfig .mnist (args.head? |>.getD "data") "generated_mnist_cnn.py"
