import LeanJax.Types
import LeanJax.MlirCodegen
import LeanJax.IreeRuntime

/-! MNIST MLP → MLIR → IREE (CUDA) — Phase 1, step 2+3+FFI of Lean_MLIR.md

    One binary that:
      1. emits StableHLO from the Lean `NetSpec`
      2. shells out to `iree-compile` to produce a `.vmfb`
      3. loads the `.vmfb` via FFI into a native `IreeSession`
      4. runs a forward pass on GPU, entirely from Lean (no Python at runtime) -/

def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

def main : IO Unit := do
  let batchSize := 128
  IO.println s!"Lean 4 → MLIR  {mnistMlp.name}  (batch={batchSize})"

  let mlir := MlirCodegen.generate mnistMlp batchSize
  let mlirPath := ".lake/build/mnist_mlp.mlir"
  let vmfbPath := ".lake/build/mnist_mlp.vmfb"
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"  wrote {mlirPath} ({mlir.length} chars)"

  -- Invoke iree-compile. sm_86 is a workaround for IREE issue #21122 which
  let ireeCompile := ".venv/bin/iree-compile"
  let args ← ireeCompileArgs mlirPath vmfbPath
  IO.println s!"  $ {ireeCompile} {String.intercalate " " args.toList}"
  let r ← IO.Process.output { cmd := ireeCompile, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed (exit {r.exitCode})"
    IO.eprintln r.stderr
    IO.Process.exit 1
  IO.println s!"  compiled → {vmfbPath}"

  -- Load the .vmfb into an IREE runtime session via FFI. No subprocess here.
  IO.println s!"Loading {vmfbPath} into IREE CUDA session..."
  let sess ← IreeSession.create vmfbPath
  IO.println "  session ready"

  -- Dummy random weights to prove the invocation works.
  let zero (n : Nat) : FloatArray := Id.run do
    let mut a : FloatArray := .empty
    for _ in [:n] do a := a.push 0.0
    return a
  let x  : FloatArray := Id.run do
    let mut a : FloatArray := .empty
    for i in [:batchSize * 784] do a := a.push (((i.toFloat) * 0.0001) - 1.0)
    return a
  let bsz : USize := batchSize.toUSize
  let logits ← IreeSession.mlpForward sess x
    (zero (784*512)) (zero 512) (zero (512*512)) (zero 512) (zero (512*10)) (zero 10) bsz
  IO.println s!"  forward pass returned {logits.size} logits (expected {batchSize*10})"

  -- Benchmark: 78 calls (matches the eval truncation from mlir_poc tests)
  let t0 ← IO.monoMsNow
  for _ in [:78] do
    let _ ← IreeSession.mlpForward sess x
      (zero (784*512)) (zero 512) (zero (512*512)) (zero 512) (zero (512*10)) (zero 10) bsz
    pure ()
  let t1 ← IO.monoMsNow
  IO.println s!"  bench: 78 calls in {t1-t0}ms ({(t1-t0).toFloat / 78.0}ms/call)"
