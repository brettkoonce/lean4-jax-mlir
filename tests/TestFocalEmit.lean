import LeanMlir.MlirCodegen
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.SpecHelpers

/-! Smoke test: emit a train_step with useFocal=true on a tiny MLP
    and verify it iree-compiles cleanly. The focal-loss codegen
    introduces ~12 extra MLIR ops in the loss + backward block; this
    catches obvious syntax / SSA-name regressions. -/

def tinyMlpSpec : NetSpec where
  name := "tiny-mlp"
  imageH := 28
  imageW := 28
  layers := [
    .flatten,
    .dense 784 32 .relu,
    .dense 32 10 .identity
  ]

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

def main : IO Unit := do
  let spec := tinyMlpSpec
  -- Compare CE vs focal — both should iree-compile.
  let mlirCe    := MlirCodegen.generateTrainStep spec 4 (useFocal := false)
  let mlirFocal := MlirCodegen.generateTrainStep spec 4 (useFocal := true) (focalGamma := 2.0)
                                                  (labelSmoothing := 0.0)
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile ".lake/build/tiny_mlp_train_ce.mlir"    mlirCe
  IO.FS.writeFile ".lake/build/tiny_mlp_train_focal.mlir" mlirFocal
  IO.println s!"CE MLIR:    {mlirCe.length} chars"
  IO.println s!"Focal MLIR: {mlirFocal.length} chars (Δ = {mlirFocal.length - mlirCe.length})"

  let compiler ← findIreeCompile
  for (name, path) in [("CE", ".lake/build/tiny_mlp_train_ce.mlir"),
                       ("Focal", ".lake/build/tiny_mlp_train_focal.mlir")] do
    let r ← IO.Process.output {
      cmd := compiler,
      args := #[path, "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
                "-o", s!"{path}.vmfb"]
    }
    if r.exitCode != 0 then
      IO.println s!"  {name}: FAIL"
      IO.println (r.stderr.take 800)
    else
      IO.println s!"  {name}: OK"
