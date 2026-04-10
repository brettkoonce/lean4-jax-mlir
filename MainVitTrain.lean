import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! ViT-Tiny on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → Adam training loop.
    Patch 16×16 → 192-dim, 12 blocks, 3 heads, MLP 768
    ~5.5M params, 224×224 input, 10 classes. -/

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity               -- classification head
  ]

namespace VitLayout

def nParams : Nat := vitTiny.totalParams

-- Build param shapes array by walking the spec
def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in vitTiny.layers do
    match l with
    | .patchEmbed ic dim p nP =>
      shapes := shapes.push #[dim, ic, p, p] |>.push #[dim]  -- W, b
      shapes := shapes.push #[dim]                            -- cls
      shapes := shapes.push #[nP + 1, dim]                    -- pos
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        -- LN1
        shapes := shapes.push #[dim] |>.push #[dim]
        -- Wq, bq
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- Wk, bk
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- Wv, bv
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- Wo, bo
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- LN2
        shapes := shapes.push #[dim] |>.push #[dim]
        -- Wfc1, bfc1
        shapes := shapes.push #[dim, mlpDim] |>.push #[mlpDim]
        -- Wfc2, bfc2
        shapes := shapes.push #[mlpDim, dim] |>.push #[dim]
      -- Final LN
      shapes := shapes.push #[dim] |>.push #[dim]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | _ => pure ()
  return shapes

def allShapes : Array (Array Nat) := paramShapes ++ paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 3 * nParams

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

-- ViT has no BN, so bnLayers is empty
def bnLayers : Array (Nat × Nat) := MlirCodegen.collectBnLayers vitTiny
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

-- For ViT eval is identical to train forward (no BN running stats)
def evalShapes : Array (Array Nat) := paramShapes
def evalShapesBA : ByteArray := packShapes evalShapes

end VitLayout

#eval vitTiny.validate!

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"ViT-Tiny: {VitLayout.nParams} params"

  let batchN : Nat := 32
  let batch : USize := 32

  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep vitTiny batchN "jit_vit_tiny_train_step"
  IO.FS.writeFile ".lake/build/vit_tiny_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate vitTiny batchN
  IO.FS.writeFile ".lake/build/vit_tiny_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval vitTiny batchN
  IO.FS.writeFile ".lake/build/vit_tiny_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/vit_tiny_fwd.mlir" ".lake/build/vit_tiny_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 2000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/vit_tiny_fwd_eval.mlir" ".lake/build/vit_tiny_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward compile failed: {re.stderr.take 2000}"
  else
    IO.eprintln "  eval forward compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/vit_tiny_train_step.mlir" ".lake/build/vit_tiny_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"train compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  train step compiled"

  let sess ← IreeSession.create ".lake/build/vit_tiny_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  -- Init params: He init for weights, special init for cls/pos/LN
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := VitLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      -- Next 1D: bias (zero) or LN gamma+beta (1.0, 0.0)
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 && shapes[si + 1]![0]! == n1 then
          -- LN: gamma=1.0, beta=0.0
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          -- bias=0
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      -- 1D-only (e.g., cls token, isolated bias)
      paramParts := paramParts.push (← F32.const n.toUSize 0.0)
      si := si + 1
  let params := F32.concat paramParts
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := VitLayout.shapesBA
  let xSh := VitLayout.xShape batchN
  let nP := VitLayout.nParams
  let nT := VitLayout.nTotal
  let baseLR : Float := 0.0003

  let bnShapes := VitLayout.bnShapesBA
  let nBnStats := VitLayout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, label_smooth=0.1, wd=1e-4"
  IO.eprintln s!"  no BN (LayerNorm only)"
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
    let lr : Float := if epoch < 5 then
      baseLR * (epoch.toFloat + 1.0) / 5.0
    else
      baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - 5.0) / (epochs.toFloat - 5.0)))
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
      let out ← IreeSession.trainStepAdamF32 sess "jit_vit_tiny_train_step.main"
                  packed allShapes xba xSh yb lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      -- ViT has no BN, but extract BN stats output (if any) anyway for safety
      if nBnStats > 0 then
        let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
        let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
        runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := ".lake/build/vit_tiny_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := VitLayout.xShape evalBatch
        let evalParams := p
        let evalShapesBA := VitLayout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "vit_tiny_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy: {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/vit_tiny_params.bin" p
  IO.eprintln "Saved params."
