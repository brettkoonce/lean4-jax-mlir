import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! ResNet-50 on Imagenette — full training pipeline with bottleneck blocks.
    ~23.5M params, 224×224 input, 10 classes. -/

def resnet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 10 .identity
  ]

namespace R50Layout

def nParams : Nat := resnet50.totalParams

def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet50.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        -- 1x1 reduce
        shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        -- 3x3
        shapes := shapes.push #[mid, mid, 3, 3] |>.push #[mid] |>.push #[mid]
        -- 1x1 expand
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- Full shapes: params ++ m (1st moment) ++ v (2nd moment) for Adam
def allShapes : Array (Array Nat) := paramShapes ++ paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 3 * nParams  -- params + m + v

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

-- BN layer info: (pidx, oc) pairs for each BN layer
def bnLayers : Array (Nat × Nat) := MlirCodegen.collectBnLayers resnet50

-- Pack BN shapes for FFI: [n_bn_layers, oc0, oc1, ...] as int32 LE
def bnShapesBA : ByteArray := Id.run do
  let push := fun (ba : ByteArray) (v : Nat) =>
    let v32 : UInt32 := v.toUInt32
    ba.push (v32 &&& 0xFF).toUInt8
      |>.push ((v32 >>> 8) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 16) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 24) &&& 0xFF).toUInt8
  let mut ba := push .empty bnLayers.size
  for (_, oc) in bnLayers do ba := push ba oc
  return ba

-- Total BN stat floats: sum of oc * 2 (mean + var per layer)
def nBnStats : Nat := bnLayers.foldl (fun acc (_, oc) => acc + oc * 2) 0

-- Param shapes for eval forward (params + bn_mean/var for each BN layer)
def evalShapes : Array (Array Nat) := Id.run do
  let mut shapes := paramShapes
  for (_, oc) in bnLayers do
    shapes := shapes.push #[oc] |>.push #[oc]  -- mean, var
  return shapes
def evalShapesBA : ByteArray := packShapes evalShapes

end R50Layout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"ResNet-50: {R50Layout.nParams} params"

  -- Generate + compile train step
  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let batchN : Nat := 32
  let mlir := MlirCodegen.generateTrainStep resnet50 batchN "jit_resnet50_train_step"
  IO.FS.writeFile ".lake/build/resnet50_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  -- Also generate forward vmfb for eval
  let fwdMlir := MlirCodegen.generate resnet50 batchN
  IO.FS.writeFile ".lake/build/resnet50_fwd.mlir" fwdMlir

  -- Eval forward with fixed BN running stats
  let evalFwdMlir := MlirCodegen.generateEval resnet50 batchN
  IO.FS.writeFile ".lake/build/resnet50_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/resnet50_fwd.mlir" ".lake/build/resnet50_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 1000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/resnet50_fwd_eval.mlir" ".lake/build/resnet50_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward (fixed BN) compile failed: {re.stderr.take 1000}"
  else
    IO.eprintln "  eval forward (fixed BN) compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/resnet50_train_step.mlir" ".lake/build/resnet50_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  compiled"

  let sess ← IreeSession.create ".lake/build/resnet50_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  data: {nTrain} images"

  -- Init params + velocity
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := R50Layout.paramShapes
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
  -- Adam state: m (1st moment) and v (2nd moment), both zero-initialized
  let adamM ← F32.const (F32.size p).toUSize 0.0
  let adamV ← F32.const (F32.size p).toUSize 0.0
  IO.eprintln s!"  {F32.size p} params + m + v ({(p.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let batch : USize := 32
  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := R50Layout.shapesBA
  let xSh := R50Layout.xShape batchN
  let nP := R50Layout.nParams
  let nT := R50Layout.nTotal
  let baseLR : Float := 0.001

  let bnShapes := R50Layout.bnShapesBA
  let nBnStats := R50Layout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, label_smooth=0.1, wd=1e-4"
  IO.eprintln s!"  BN layers: {R50Layout.bnLayers.size}, BN stat floats: {nBnStats}"
  let mut params := p
  let mut m := adamM
  let mut v := adamV
  -- Running BN stats (EMA, momentum=0.1)
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  let mut globalStep : Nat := 0
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
    -- Cosine LR with 3-epoch warmup
    let lr : Float := if epoch < 3 then
      baseLR * (epoch.toFloat + 1.0) / 3.0
    else
      baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - 3.0) / (epochs.toFloat - 3.0)))
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let xba256 := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xbaCropped ← F32.randomCrop xba256 batch 3 256 256 224 224 (epoch * 10000 + bi).toUSize
      let xba ← F32.randomHFlip xbaCropped batch 3 224 224 (epoch * 10000 + bi + 7777).toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := (params.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess "jit_resnet50_train_step.main"
                  packed allShapes xba xSh yb lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      params := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      -- Extract batch BN stats and update running stats via EMA
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
      -- Use momentum 1.0 for first step (initialize), 0.1 thereafter
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    -- Val eval every 10 epochs (using running BN stats)
    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := ".lake/build/resnet50_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := R50Layout.xShape evalBatch
        -- Pack params + running BN stats for eval forward
        let evalParams := params.append runningBnStats
        let evalShapesBA := R50Layout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "resnet_50_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/resnet50_params.bin" params
  IO.FS.writeBinFile ".lake/build/resnet50_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
