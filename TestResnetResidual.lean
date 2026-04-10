import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! Test: generate ResNet-34 forward + train step MLIR with real residualBlock layers
    (skip connections, projections) and compile with IREE. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,      -- use 2/2 (IREE lacks select_and_scatter for 3/2)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def compile (src out : String) : IO Bool := do
  let args ← ireeCompileArgs src out
  let r ← IO.Process.output {
    cmd := ".venv/bin/iree-compile"
    args := args
  }
  if r.exitCode != 0 then
    IO.eprintln s!"Compile FAILED:\n{r.stderr.take 3000}"
    return false
  return true

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  IO.println s!"ResNet-34: {resnet34.totalParams} params"

  -- Forward
  let fwd := MlirCodegen.generate resnet34 16
  IO.FS.writeFile ".lake/build/resnet34_fwd.mlir" fwd
  IO.println s!"Forward: {fwd.length} chars"
  if ← compile ".lake/build/resnet34_fwd.mlir" ".lake/build/resnet34_fwd.vmfb"
  then IO.println "✓ Forward compiled"

  -- Train step
  let ts := MlirCodegen.generateTrainStep resnet34 16 "jit_resnet34_train_step"
  IO.FS.writeFile ".lake/build/resnet34_train_step.mlir" ts
  IO.println s!"Train step: {ts.length} chars"
  if ← compile ".lake/build/resnet34_train_step.mlir" ".lake/build/resnet34_train_step.vmfb"
  then IO.println "✓ Train step compiled"
