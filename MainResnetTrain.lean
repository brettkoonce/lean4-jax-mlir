import LeanJax.IreeRuntime
import LeanJax.F32Array
import LeanJax.Types
import LeanJax.Spec
import LeanJax.MlirCodegen

/-! ResNet-34 on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → SGD training loop.
    ~21.3M params, 224×224 input, 10 classes. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                    -- 2/2 instead of 3/2 (IREE compat)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

namespace ResnetLayout

-- convBn params: W (oc*ic*k*k) + gamma (oc) + beta (oc)
-- residualBlock: 2 convBn per block + optional projection convBn
-- See Spec.lean Layer.nParams for the formula.

def nParams : Nat := resnet34.totalParams  -- 21289802

-- Build param shapes array by walking the spec
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

-- Full shapes: all param shapes then all velocity shapes (same shapes repeated)
def allShapes : Array (Array Nat) := paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 2 * nParams  -- params + velocities

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

end ResnetLayout

def main (args : List String) : IO Unit := do
  IO.eprintln "step 1"
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.println s!"ResNet-34: {ResnetLayout.nParams} params"

  -- Pre-compile: run TestResnetResidual first to generate .vmfb
  let vmfbPath := ".lake/build/resnet34_train_step.vmfb"
  let vmfbExists ← System.FilePath.pathExists vmfbPath
  if !vmfbExists then
    IO.eprintln "ERROR: .vmfb not found. Run test-resnet-residual first to generate it."
    IO.Process.exit 1

  IO.eprintln "step 2: load IREE session"
  let t0 ← IO.monoMsNow

  -- Load IREE session
  IO.eprintln "  creating session..."
  let sess ← IreeSession.create ".lake/build/resnet34_train_step.vmfb"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"  session loaded ({t1 - t0}ms)"

  -- Load Imagenette data
  IO.eprintln s!"step 3: load data from {dataDir}"
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenette (dataDir ++ "/train.bin")
  IO.eprintln s!"  train: {nTrain} images ({F32.size trainImg} floats)"
  let t2 ← IO.monoMsNow
  IO.eprintln s!"  loaded ({t2 - t1}ms)"

  -- Initialize params: He init for weights, 1.0 for gamma, 0.0 for beta/bias
  -- paramShapes order: convBn → [W(4D), gamma(1D), beta(1D)], dense → [W(2D), bias(1D)]
  IO.eprintln "step 4: init params"
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := ResnetLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      -- Weight tensor: He init
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      -- Next 1D params: check if it's a convBn triple (gamma + beta) or dense pair (bias)
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          -- convBn: gamma=1.0, beta=0.0
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          -- dense: bias=0.0
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      si := si + 1
  let params := F32.concat paramParts
  -- Initialize velocity buffers to zero (same size as params)
  let velocity ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + {F32.size velocity} velocity ({(params.size + velocity.size) / 1024 / 1024} MB)"

  -- Training loop: pack params++velocity, pass through FFI, unpack
  let batchN : Nat := 16
  let batch : USize := 16
  let epochs := 80
  let bpE := nTrain / batchN
  let pixelsPerImage := 3 * 224 * 224
  let shapes := ResnetLayout.shapesBA
  let xSh := ResnetLayout.xShape batchN
  let nP := ResnetLayout.nParams
  let nT := ResnetLayout.nTotal  -- params + velocities

  IO.eprintln s!"step 5: training ({bpE} batches/epoch, batch={batchN}, BN, momentum=0.9)"
  let mut p := params
  let mut v := velocity
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  for epoch in [:epochs] do
    -- Shuffle training data each epoch
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize pixelsPerImage.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
    let lr : Float := 0.002 * (1.0 - epoch.toFloat / epochs.toFloat)
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba := F32.sliceImages curImg (bi * batchN) batchN pixelsPerImage
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      -- Pack params ++ velocity into one ByteArray
      let packed := p.append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepF32 sess "jit_resnet34_train_step.main"
                  packed shapes xba xSh yb lr batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      -- Unpack: first nP floats = updated params, next nP = updated velocity
      p := F32.slice out 0 nP
      v := F32.slice out nP nP
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"
