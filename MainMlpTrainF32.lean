import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.MlirCodegen

/-! MNIST MLP training — f32 ByteArray version.
    Same architecture as MainMlpTrain but with zero f64↔f32 conversion
    at the FFI boundary. All tensor data stays float32 from init to GPU. -/

def mnistMlp : NetSpec where
  name := "MNIST MLP"
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

def main : IO Unit := do
  -- Generate + compile train step vmfb
  IO.FS.createDirAll ".lake/build"
  let mlir := MlirCodegen.generateTrainStep mnistMlp 128
  IO.FS.writeFile ".lake/build/train_step.mlir" mlir
  let compileArgs ← ireeCompileArgs ".lake/build/train_step.mlir" ".lake/build/train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr}"
    IO.Process.exit 1
  IO.println "Train step .vmfb compiled."

  IO.println "Loading MNIST (f32, instant)..."
  let (trainImages, nTrain) ← F32.loadIdxImages "data/train-images-idx3-ubyte"
  let (trainLabels, _)      ← F32.loadIdxLabels "data/train-labels-idx1-ubyte"
  IO.println s!"  train: {nTrain} images"

  IO.println "Loading IREE..."
  let trainSess ← IreeSession.create ".lake/build/train_step.vmfb"
  IO.println "  ready"

  IO.println "Initializing params + velocities (f32, instant)..."
  let stddev0 := Float.sqrt (2.0 / 784.0)
  let stddev1 := Float.sqrt (2.0 / 512.0)
  let weights := F32.concat #[
    ← F32.heInit 1 (784*512).toUSize stddev0,
    ← F32.const 512 0.0,
    ← F32.heInit 2 (512*512).toUSize stddev1,
    ← F32.const 512 0.0,
    ← F32.heInit 3 (512*10).toUSize stddev1,
    ← F32.const 10 0.0
  ]
  -- Velocities initialized to zero (same layout as weights)
  let velocities ← F32.const MlpLayout.nParams.toUSize 0.0
  let params := F32.concat #[weights, velocities]
  IO.println s!"  {F32.size weights} weights + {F32.size velocities} velocities"

  let batch : USize := 128
  let batchN : Nat := 128
  let lr : Float := 0.1
  let epochs := 12
  let bpE := nTrain / batchN
  -- Shapes: params + velocities (same shapes duplicated)
  let shapes := packShapes (MlpLayout.paramShapes ++ MlpLayout.paramShapes)
  let xSh := packXShape #[batchN, 784]
  let nTotal := 2 * MlpLayout.nParams

  let mut p := params
  for epoch in [:epochs] do
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xb := F32.sliceImages trainImages (bi*batchN) batchN 784
      let yb := F32.sliceLabels trainLabels (bi*batchN) batchN
      let out ← IreeSession.trainStepF32 trainSess "jit_train_step.main"
                  p shapes xb xSh yb lr batch
      epochLoss := epochLoss + F32.extractLoss out nTotal
      p := F32.dropLoss out nTotal
    let tTrain ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.println s!"Epoch {epoch+1}: loss={avgLoss} ({tTrain-t0}ms)"
