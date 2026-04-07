import LeanJax.Types
import LeanJax.MlirCodegen

/-! MNIST CNN → MLIR → .vmfb via Lean-generated StableHLO. Tests that the
    new conv/pool/flatten emission in `MlirCodegen.lean` matches the
    hand-written reference. -/

def mnistCnn : NetSpec where
  name   := "MNIST CNN"
  imageH := 28
  imageW := 28
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
  let batchSize := 128
  IO.println s!"Lean 4 → MLIR  {mnistCnn.name}  (batch={batchSize})"
  let mlir := MlirCodegen.generate mnistCnn batchSize
  let mlirPath := ".lake/build/mnist_cnn.mlir"
  let vmfbPath := ".lake/build/mnist_cnn.vmfb"
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"  wrote {mlirPath} ({mlir.length} chars)"

  let args ← ireeCompileArgs mlirPath vmfbPath
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed (exit {r.exitCode})"
    IO.eprintln r.stderr
    IO.Process.exit 1
  IO.println s!"  compiled → {vmfbPath}"
