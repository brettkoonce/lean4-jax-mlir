import LeanMlir.Types
import LeanMlir.MlirCodegen

/-! Test: generate ResNet-34 forward MLIR and compile with IREE.
    This validates convBn (instance norm) + globalAvgPool + strided conv
    + residual blocks (emitted as individual convBn layers for now). -/

-- ResNet-34 without residualBlock sugar — expanded to individual layers
-- for testing the convBn emission before adding residualBlock support
def resnet34_expanded : NetSpec where
  name := "ResNet-34-Expanded"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,      -- stem: 224→112
    .maxPool 3 2,                  -- 112→56
    -- Stage 1: 3 blocks at 64 channels, no downsampling
    .convBn 64 64 3 1 .same, .convBn 64 64 3 1 .same,
    .convBn 64 64 3 1 .same, .convBn 64 64 3 1 .same,
    .convBn 64 64 3 1 .same, .convBn 64 64 3 1 .same,
    -- (skip connections omitted for forward-only test)
    .globalAvgPool,
    .dense 64 10 .identity
  ]

def main : IO Unit := do
  let mlir := MlirCodegen.generate resnet34_expanded 128
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile ".lake/build/resnet34_fwd.mlir" mlir
  IO.println s!"Generated {mlir.length} chars"

  let compileArgs ← ireeCompileArgs ".lake/build/resnet34_fwd.mlir" ".lake/build/resnet34_fwd.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"Compile failed:\n{r.stderr.take 1000}"
  else
    IO.println s!"Compiled → .lake/build/resnet34_fwd.vmfb"
