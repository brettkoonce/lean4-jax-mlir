import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! MNIST CNN training — uses auto-generated train step with BN + momentum.
    Conv-BN-ReLU architecture, compiled to CUDA via IREE. -/

def mnistCnn : NetSpec where
  name := "MNIST-CNN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

namespace MnistCnnLayout
def nParams : Nat := mnistCnn.totalParams

def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in mnistCnn.layers do
    match l with
    | .convBn ic oc k _ _ => shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _      => shapes := shapes.push #[fi, fo] |>.push #[fo]
    | _ => pure ()
  return shapes

def allShapes : Array (Array Nat) := paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 2 * nParams
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 784]
end MnistCnnLayout

def compile (src out : String) : IO Bool := do
  let r ← IO.Process.output {
    cmd := ".venv/bin/iree-compile"
    args := #[src, "--iree-hal-target-backends=cuda", "--iree-cuda-target=sm_86", "-o", out]
  }
  if r.exitCode != 0 then
    IO.eprintln s!"Compile FAILED:\n{r.stderr.take 2000}"
    return false
  return true

def main : IO Unit := do
  IO.eprintln s!"MNIST CNN: {MnistCnnLayout.nParams} params"
  IO.FS.createDirAll ".lake/build"

  -- Generate + compile train step
  let vmfb := ".lake/build/mnist_cnn_train_step.vmfb"
  if !(← System.FilePath.pathExists vmfb) then
    IO.eprintln "Generating train step MLIR..."
    let ts := MlirCodegen.generateTrainStep mnistCnn 128 "jit_mnist_cnn_train_step"
    IO.FS.writeFile ".lake/build/mnist_cnn_train_step.mlir" ts
    IO.eprintln s!"  {ts.length} chars"
    IO.eprintln "Compiling..."
    if !(← compile ".lake/build/mnist_cnn_train_step.mlir" vmfb) then
      IO.Process.exit 1
    IO.eprintln "  compiled"

  -- Load IREE session
  IO.eprintln "Loading IREE..."
  let sess ← IreeSession.create vmfb

  -- Load MNIST data
  IO.eprintln "Loading MNIST..."
  let (trainImg, nTrain) ← F32.loadIdxImages "data/train-images-idx3-ubyte"
  let (trainLbl, _) ← F32.loadIdxLabels "data/train-labels-idx1-ubyte"
  IO.eprintln s!"  {nTrain} images"

  -- Init params
  IO.eprintln "Init params..."
  let mut parts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := MnistCnnLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      parts := parts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          parts := parts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          parts := parts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          parts := parts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else si := si + 1
  let params := F32.concat parts
  let velocity ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params"

  -- Training
  let batchN : Nat := 128
  let batch : USize := 128
  let epochs := 15
  let bpE := nTrain / batchN
  let ppi := 784
  let allShapes := MnistCnnLayout.shapesBA
  let xSh := MnistCnnLayout.xShape batchN
  let nP := MnistCnnLayout.nParams
  let nT := MnistCnnLayout.nTotal

  IO.eprintln s!"Training ({bpE} batches/epoch, batch={batchN})"
  let mut p := params
  let mut v := velocity
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize ppi.toUSize (epoch + 7).toUSize
    curImg := sImg; curLbl := sLbl
    let lr : Float := 0.002 * (1.0 - epoch.toFloat / epochs.toFloat)
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba := F32.sliceImages curImg (bi * batchN) batchN ppi
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := p.append v
      let out ← IreeSession.trainStepF32 sess "jit_mnist_cnn_train_step.main"
                  packed allShapes xba xSh yb lr batch
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      v := F32.slice out nP nP
    let t1 ← IO.monoMsNow
    -- Test accuracy
    let (testImg, nTest) ← F32.loadIdxImages "data/t10k-images-idx3-ubyte"
    let (testLbl, _) ← F32.loadIdxLabels "data/t10k-labels-idx1-ubyte"
    -- Forward-only for accuracy would need separate vmfb; report loss for now
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"
