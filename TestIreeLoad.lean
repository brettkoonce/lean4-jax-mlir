import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! Progressive test: session load + data load + one step. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def main : IO Unit := do
  IO.eprintln "1: session"
  let sess ← IreeSession.create ".lake/build/resnet34_train_step.vmfb"
  IO.eprintln "2: train data"
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenette "data/imagenette/train.bin"
  IO.eprintln s!"3: {nTrain} images, {F32.size trainImg} floats"
  IO.eprintln s!"4: nParams = {resnet34.totalParams}"
  IO.println "All OK"
