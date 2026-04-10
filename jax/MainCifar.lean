import LeanJax

/-! CIFAR-10 CNN — S4TF book Ch. 3
    Conv²→Pool→Conv²→Pool→Dense³, 2.43M params -/

def cifarCnn : NetSpec where
  name := "CIFAR-10 CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

def cifarConfig : TrainConfig where
  learningRate := 0.01
  batchSize    := 128
  epochs       := 25

#eval cifarCnn.validate!

def main (args : List String) : IO Unit :=
  runJax cifarCnn cifarConfig .cifar10 (args.head? |>.getD "data/cifar-10") "generated_cifar_cnn.py"
