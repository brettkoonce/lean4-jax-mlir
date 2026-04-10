import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! MobileNet V4-Medium on Imagenette — UIB (Universal Inverted Bottleneck) blocks.
    Conv-only variant (no attention). 224×224, 10 classes. -/

def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .fusedMbConv 32 48 4 3 2 1 false,
    .uib  48  80 4 2 3 5,
    .uib  80  80 2 1 3 3,
    .uib  80 160 6 2 0 3,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 5,
    .uib 160 160 4 1 5 0,
    .uib 160 160 4 1 0 3,
    .uib 160 160 4 1 3 0,
    .uib 160 160 4 1 0 0,
    .uib 160 160 4 1 3 3,
    .uib 160 256 6 2 5 5,
    .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 0 3,
    .uib 256 256 4 1 3 0,
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

namespace MobilenetV4Layout

def nParams : Nat := mobilenetV4Medium.totalParams

def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in mobilenetV4Medium.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .fusedMbConv ic oc expand kSize _ n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        -- Fused expand: regular k×k convBn
        shapes := shapes.push #[mid, blockIc, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        if expand != 1 then
          shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      -- preDW (if preDWk > 0): W[ic, 1, k, k], gamma/beta [ic]
      if preDWk > 0 then
        shapes := shapes.push #[ic, 1, preDWk, preDWk] |>.push #[ic] |>.push #[ic]
      -- Expand 1×1: W[mid, ic, 1, 1]
      shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      -- postDW (if postDWk > 0): W[mid, 1, k, k]
      if postDWk > 0 then
        shapes := shapes.push #[mid, 1, postDWk, postDWk] |>.push #[mid] |>.push #[mid]
      -- Project 1×1: W[oc, mid, 1, 1]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

def allShapes : Array (Array Nat) := paramShapes ++ paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 3 * nParams

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

def bnLayers : Array (Nat × Nat) := MlirCodegen.collectBnLayers mobilenetV4Medium

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

def nBnStats : Nat := bnLayers.foldl (fun acc (_, oc) => acc + oc * 2) 0

def evalShapes : Array (Array Nat) := Id.run do
  let mut shapes := paramShapes
  for (_, oc) in bnLayers do
    shapes := shapes.push #[oc] |>.push #[oc]
  return shapes
def evalShapesBA : ByteArray := packShapes evalShapes

end MobilenetV4Layout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"MobileNet V4-Medium: {MobilenetV4Layout.nParams} params"

  let batchN : Nat := 32
  let batch : USize := 32

  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep mobilenetV4Medium batchN "jit_mobilenet_v4_train_step"
  IO.FS.writeFile ".lake/build/mobilenet_v4_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate mobilenetV4Medium batchN
  IO.FS.writeFile ".lake/build/mobilenet_v4_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval mobilenetV4Medium batchN
  IO.FS.writeFile ".lake/build/mobilenet_v4_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/mobilenet_v4_fwd.mlir" ".lake/build/mobilenet_v4_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 2000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/mobilenet_v4_fwd_eval.mlir" ".lake/build/mobilenet_v4_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward compile failed: {re.stderr.take 2000}"
  else
    IO.eprintln "  eval forward compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/mobilenet_v4_train_step.mlir" ".lake/build/mobilenet_v4_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"train compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  train step compiled"

  let sess ← IreeSession.create ".lake/build/mobilenet_v4_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := MobilenetV4Layout.paramShapes
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
  let params := F32.concat paramParts
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := MobilenetV4Layout.shapesBA
  let xSh := MobilenetV4Layout.xShape batchN
  let nP := MobilenetV4Layout.nParams
  let nT := MobilenetV4Layout.nTotal
  let baseLR : Float := 0.001

  let bnShapes := MobilenetV4Layout.bnShapesBA
  let nBnStats := MobilenetV4Layout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, wd=1e-4"
  IO.eprintln s!"  BN layers: {MobilenetV4Layout.bnLayers.size}, BN stat floats: {nBnStats}"
  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  let mut globalStep : Nat := 0
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
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
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess "jit_mobilenet_v4_train_step.main"
                  packed allShapes xba xSh yb lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := ".lake/build/mobilenet_v4_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := MobilenetV4Layout.xShape evalBatch
        let evalParams := p.append runningBnStats
        let evalShapesBA := MobilenetV4Layout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "mobilenet_v4_medium_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/mobilenet_v4_params.bin" p
  IO.FS.writeBinFile ".lake/build/mobilenet_v4_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
