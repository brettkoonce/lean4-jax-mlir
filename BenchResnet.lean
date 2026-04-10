import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! ResNet-34 batch size benchmark. Reads IREE_BATCH (default 32).
    Generates vmfb to a batch-specific path so it doesn't clobber the
    running batch-16 training. Runs 5 epochs for timing. -/

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

namespace ResnetLayout

def nParams : Nat := resnet34.totalParams

def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet34.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[oc, blockIc, 3, 3] |>.push #[oc] |>.push #[oc]
        shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

def nTotal : Nat := 2 * nParams

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

end ResnetLayout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  let batchStr ← (IO.getEnv "IREE_BATCH").map (·.getD "32")
  let batchN := batchStr.toNat!
  let batch : USize := batchN.toUSize
  let mlirPath := s!".lake/build/r34_bench_b{batchN}.mlir"
  let vmfbPath := s!".lake/build/r34_bench_b{batchN}.vmfb"

  IO.eprintln s!"ResNet-34 benchmark: batch={batchN}"

  -- Generate + compile
  IO.FS.createDirAll ".lake/build"
  let ts := MlirCodegen.generateTrainStep resnet34 batchN "jit_resnet34_train_step"
  IO.FS.writeFile mlirPath ts
  IO.eprintln s!"  MLIR: {ts.length} chars"
  let compileArgs ← ireeCompileArgs mlirPath vmfbPath
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 2000}"
    IO.Process.exit 1
  IO.eprintln "  vmfb compiled"

  let sess ← IreeSession.create vmfbPath
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenette (dataDir ++ "/train.bin")
  IO.eprintln s!"  data: {nTrain} images"

  -- Init params + velocity
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := ResnetLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      si := si + 1
  let p := F32.concat paramParts
  let v ← F32.const (F32.size p).toUSize 0.0

  let bpE := nTrain / batchN
  let pixelsPerImage := 3 * 224 * 224
  let allShapes := packShapes (ResnetLayout.paramShapes ++ ResnetLayout.paramShapes)
  let xSh := ResnetLayout.xShape batchN
  let nP := ResnetLayout.nParams
  let nT := ResnetLayout.nTotal

  IO.eprintln s!"  training: {bpE} steps/epoch, batch={batchN}"
  let mut params := p
  let mut vel := v
  for epoch in [:5] do
    let lr : Float := 0.002
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba := F32.sliceImages trainImg (bi * batchN) batchN pixelsPerImage
      let yb := F32.sliceLabels trainLbl (bi * batchN) batchN
      let packed := params.append vel
      let out ← IreeSession.trainStepF32 sess "jit_resnet34_train_step.main"
                  packed allShapes xba xSh yb lr batch
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      params := F32.slice out 0 nP
      vel := F32.slice out nP nP
      if bi < 3 || bi % 50 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss}"
    let t1 ← IO.monoMsNow
    let msPerStep := (t1 - t0).toFloat / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/5: loss={epochLoss / bpE.toFloat} ({t1-t0}ms, {msPerStep}ms/step)"
