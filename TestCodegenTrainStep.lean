import LeanMlir.Types
import LeanMlir.MlirCodegen

def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

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

def main : IO Unit := do
  -- MLP
  let mlir := MlirCodegen.generateTrainStep mnistMlp 128
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile ".lake/build/gen_train_step.mlir" mlir
  IO.println s!"MLP: {mlir.length} chars"

  -- CNN
  let cnnMlir := MlirCodegen.generateTrainStep mnistCnn 128 "jit_cnn_train_step"
  IO.FS.writeFile ".lake/build/gen_cnn_train_step.mlir" cnnMlir
  IO.println s!"CNN: {cnnMlir.length} chars"

  -- Compile both
  for (name, path) in [("MLP", ".lake/build/gen_train_step.mlir"),
                        ("CNN", ".lake/build/gen_cnn_train_step.mlir")] do
    let compArgs ← ireeCompileArgs path (path.replace ".mlir" ".vmfb")
    let r ← IO.Process.output {
      cmd := ".venv/bin/iree-compile",
      args := compArgs
    }
    if r.exitCode != 0 then
      IO.eprintln s!"{name} compile failed:\n{r.stderr.take 1000}"
    else
      IO.println s!"{name} compiled ✓"

  IO.println "Done."

-- CIFAR generation not in main, just verify it compiles via CLI:
-- MlirCodegen.generateTrainStep cifarCnn 128 "jit_cifar_train_step"
