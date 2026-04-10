import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! CIFAR-10 CNN with batch norm + momentum. Auto-generated train step. -/

def cifarCnn : NetSpec where
  name := "CIFAR-10-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn  3 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

namespace CifarBnLayout
def nParams : Nat := cifarCnn.totalParams
def paramShapes : Array (Array Nat) := Id.run do
  let mut s : Array (Array Nat) := #[]
  for l in cifarCnn.layers do
    match l with
    | .convBn ic oc k _ _ => s := s.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _      => s := s.push #[fi, fo] |>.push #[fo]
    | _ => pure ()
  return s
def allShapes : Array (Array Nat) := paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 2 * nParams
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end CifarBnLayout

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
  IO.eprintln s!"CIFAR-10 BN CNN: {CifarBnLayout.nParams} params"
  IO.FS.createDirAll ".lake/build"

  let vmfb := ".lake/build/cifar_bn_train_step.vmfb"
  if !(← System.FilePath.pathExists vmfb) then
    IO.eprintln "Generating train step MLIR..."
    let ts := MlirCodegen.generateTrainStep cifarCnn 128 "jit_cifar_bn_train_step"
    IO.FS.writeFile ".lake/build/cifar_bn_train_step.mlir" ts
    IO.eprintln s!"  {ts.length} chars"
    IO.eprintln "Compiling..."
    if !(← compile ".lake/build/cifar_bn_train_step.mlir" vmfb) then IO.Process.exit 1
    IO.eprintln "  compiled"

  IO.eprintln "Loading IREE..."
  let sess ← IreeSession.create vmfb

  IO.eprintln "Loading CIFAR-10..."
  let mut trainRaw : ByteArray := .empty
  let mut trainLbl : ByteArray := .empty
  let mut nTrain : Nat := 0
  for i in [1:6] do
    let raw ← IO.FS.readBinFile s!"data/cifar-10/data_batch_{i}.bin"
    let n := raw.size / 3073
    for j in [:n] do
      let off := j * 3073
      trainLbl := trainLbl.push raw[off]!
      trainLbl := trainLbl.push 0; trainLbl := trainLbl.push 0; trainLbl := trainLbl.push 0
    trainRaw := trainRaw.append raw
    nTrain := nTrain + n
  IO.eprintln s!"  {nTrain} images"

  IO.eprintln "Init params..."
  let mut parts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := CifarBnLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      parts := parts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1; si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          parts := parts.push (← F32.const n1.toUSize 1.0); si := si + 1
          parts := parts.push (← F32.const (shapes[si]![0]!).toUSize 0.0); si := si + 1
        else
          parts := parts.push (← F32.const n1.toUSize 0.0); si := si + 1
    else si := si + 1
  let params := F32.concat parts
  let velocity ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params"

  let batchN : Nat := 128
  let batch : USize := 128
  let epochs := 30
  let bpE := nTrain / batchN
  let ppi := 3072
  let xSh := CifarBnLayout.xShape batchN
  let nP := CifarBnLayout.nParams
  let nT := CifarBnLayout.nTotal

  IO.eprintln s!"Training ({bpE} batches/epoch, batch={batchN})"
  let mut p := params
  let mut v := velocity
  let mut curLbl := trainLbl
  for epoch in [:epochs] do
    -- Shuffle labels (images loaded per-batch from raw)
    -- For CIFAR we load from raw per-batch, so shuffle the indices instead
    -- Simple approach: shuffle labels and use same order for raw access
    let lr : Float := 0.002 * (1.0 - epoch.toFloat / epochs.toFloat)
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba ← F32.cifarBatch trainRaw (bi * batchN).toUSize batchN.toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := p.append v
      let out ← IreeSession.trainStepF32 sess "jit_cifar_bn_train_step.main"
                  packed CifarBnLayout.shapesBA xba xSh yb lr batch
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      v := F32.slice out nP nP
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"
