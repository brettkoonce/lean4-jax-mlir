import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! Unified training loop for any spec that uses the Adam codegen path.

Each Main*Train.lean used to inline ~250 lines of identical
code: generate MLIR → compile vmfbs → init params → for-each-epoch
{ shuffle, for-each-batch { trainStepAdamF32, EMA, log }, val eval } →
save. The body was the same for ResNet/MobileNet/EfficientNet/ViT/VGG
modulo a handful of name strings.

`NetSpec.train` is the extracted function. A trainer is now:

    def main (args : List String) : IO Unit :=
      resnet34.train resnet34Config (args.head?.getD "data/imagenette")

The 250 lines collapse to one call. Adding a new architecture means
defining the spec + a `TrainConfig` value — no copy-paste plumbing. -/

namespace NetSpec

/-- File-path prefix for the generated MLIR / vmfb / saved-params files
    associated with this spec. Uses the sanitized spec name so adding
    a new spec automatically gets a unique non-colliding prefix without
    the trainer author having to pick one. -/
def buildPrefix (spec : NetSpec) : String :=
  ".lake/build/" ++ spec.sanitizedName

/-- The fully qualified `<module>.<func>` name the train step's main
    function lives at — what `trainStepAdamF32` wants as its `fnName`.
    Mirrors how `MlirCodegen.generateTrainStep` builds the module
    name from the third argument it's given (`jit_<sanitized>_train_step`). -/
def trainFnName (spec : NetSpec) : String :=
  "jit_" ++ spec.sanitizedName ++ "_train_step.main"

/-- Cache key for an MLIR string under the current IREE backend.
    Combines the MLIR content with the backend env var so switching
    `IREE_BACKEND=rocm` ↔ `cuda` invalidates the cache. -/
private def cacheKey (mlir : String) : IO String := do
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "cuda")
  return toString (mlir ++ "::" ++ backend).hash

/-- Find iree-compile: check .venv/bin first (local dev), then PATH. -/
private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  -- Fall back to PATH (Docker, system install)
  return "iree-compile"

private def runIree (mlirPath outPath : String) : IO Bool := do
  let args ← ireeCompileArgs mlirPath outPath
  let compiler ← findIreeCompile
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed for {mlirPath}: {r.stderr.take 3000}"
    return false
  return true

/-- Compile `mlirPath` to `outPath` via iree-compile, but skip the work
    if `outPath` exists and `outPath ++ ".hash"` matches the cache key
    for the current MLIR + IREE backend. Returns (compiledNewly, success). -/
private def runIreeCached (mlirPath outPath mlir : String) : IO (Bool × Bool) := do
  let key ← cacheKey mlir
  let hashPath := outPath ++ ".hash"
  let vmfbExists ← System.FilePath.pathExists outPath
  let hashExists ← System.FilePath.pathExists hashPath
  if vmfbExists && hashExists then
    let cached ← IO.FS.readFile hashPath
    if cached.trim == key then
      return (false, true)  -- cache hit, skip the slow path
  -- Cache miss → compile and write the sidecar.
  let ok ← runIree mlirPath outPath
  if ok then
    IO.FS.writeFile hashPath key
  return (true, ok)

/-- Generate forward / eval-forward / train-step MLIR for `spec`,
    write them to `.lake/build/<sanitized>_*.mlir`, and compile each
    to a `.vmfb`. Cached on the MLIR content + IREE backend so a
    second run with no codegen changes skips iree-compile entirely
    (saves ~10-15 min for ResNet-sized models). Returns the path to
    the train-step vmfb. -/
def compileVmfbs (spec : NetSpec) (cfg : TrainConfig) : IO String := do
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix

  IO.eprintln "Generating train step MLIR..."
  let trainMlir := MlirCodegen.generateTrainStep spec cfg.batchSize ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (labelSmoothing := cfg.labelSmoothing)
    (weightDecay := cfg.weightDecay)
    (useAdam := cfg.useAdam)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.eprintln s!"  {trainMlir.length} chars"

  let fwdMlir := MlirCodegen.generate spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let (fwdNew, fwdOk) ← runIreeCached s!"{pfx}_fwd.mlir" s!"{pfx}_fwd.vmfb" fwdMlir
  if fwdOk then IO.eprintln (if fwdNew then "  forward compiled" else "  forward (cached)")

  let (evalNew, evalOk) ← runIreeCached s!"{pfx}_fwd_eval.mlir" s!"{pfx}_fwd_eval.vmfb" evalFwdMlir
  if evalOk then IO.eprintln (if evalNew then "  eval forward compiled" else "  eval forward (cached)")

  let (trainNew, trainOk) ← runIreeCached s!"{pfx}_train_step.mlir" s!"{pfx}_train_step.vmfb" trainMlir
  if trainOk then
    IO.eprintln (if trainNew then "  train step compiled" else "  train step (cached)")
  else
    IO.Process.exit 1
  return s!"{pfx}_train_step.vmfb"

/-- Dataset-specific I/O: where the training/val data lives, how big each
    image is on disk, and what augmentation to apply per batch. Each
    dataset (Imagenette / CIFAR-10 / MNIST) constructs one of these and
    `runTraining` consumes it. -/
private structure DatasetIO where
  /-- Pixels per training image as stored on disk (before any crop). -/
  trainPixels : Nat
  /-- Pixels per val image as stored on disk. -/
  valPixels : Nat
  /-- Channels (1 for MNIST, 3 for CIFAR/Imagenette). -/
  channels : Nat
  loadTrain : String → IO (ByteArray × ByteArray × Nat)
  loadVal   : String → IO (ByteArray × ByteArray × Nat)
  /-- Apply augmentation to a slice of training data. Receives the raw
      slice, the batch size, and a seed (epoch * 10000 + bi). Must
      return a buffer matching `inputFlatDim spec` floats per image. -/
  augmentBatch : (raw : ByteArray) → (batch : USize) → (seed : Nat) → IO ByteArray

/-- Imagenette: 256×256 train, 224×224 val, random-crop-to-224 + hflip. -/
private def imagenetteIO : DatasetIO where
  trainPixels := 3 * 256 * 256
  valPixels   := 3 * 224 * 224
  channels    := 3
  loadTrain   := fun dir => F32.loadImagenetteSized (dir ++ "/train.bin") 256
  loadVal     := fun dir => F32.loadImagenette (dir ++ "/val.bin")
  augmentBatch := fun raw batch seed => do
    let cropped ← F32.randomCrop raw batch 3 256 256 224 224 seed.toUSize
    F32.randomHFlip cropped batch 3 224 224 (seed + 7777).toUSize

/-- MNIST: 28×28 IDX format, no augmentation. -/
private def mnistIO : DatasetIO where
  trainPixels := 1 * 28 * 28
  valPixels   := 1 * 28 * 28
  channels    := 1
  loadTrain := fun dir => do
    let (imgs, n) ← F32.loadIdxImages (dir ++ "/train-images-idx3-ubyte")
    let (lbls, _) ← F32.loadIdxLabels (dir ++ "/train-labels-idx1-ubyte")
    return (imgs, lbls, n)
  loadVal := fun dir => do
    let (imgs, n) ← F32.loadIdxImages (dir ++ "/t10k-images-idx3-ubyte")
    let (lbls, _) ← F32.loadIdxLabels (dir ++ "/t10k-labels-idx1-ubyte")
    return (imgs, lbls, n)
  augmentBatch := fun raw _ _ => return raw

/-- CIFAR-10: 5 train batches concatenated, 1 test batch, random hflip. -/
private def cifar10IO : DatasetIO where
  trainPixels := 3 * 32 * 32
  valPixels   := 3 * 32 * 32
  channels    := 3
  loadTrain := fun dir => do
    let mut raw : ByteArray := .empty
    let mut labels : ByteArray := .empty
    let mut nTotal : Nat := 0
    for i in [1:6] do
      let batchRaw ← IO.FS.readBinFile s!"{dir}/cifar-10/data_batch_{i}.bin"
      let n := batchRaw.size / 3073
      for j in [:n] do
        labels := labels.push batchRaw[j * 3073]!
        labels := labels.push 0
        labels := labels.push 0
        labels := labels.push 0
      raw := raw.append batchRaw
      nTotal := nTotal + n
    let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
    return (imgs, labels, nTotal)
  loadVal := fun dir => do
    let raw ← IO.FS.readBinFile s!"{dir}/cifar-10/test_batch.bin"
    let n := raw.size / 3073
    let mut labels : ByteArray := .empty
    for j in [:n] do
      labels := labels.push raw[j * 3073]!
      labels := labels.push 0
      labels := labels.push 0
      labels := labels.push 0
    let imgs ← F32.cifarBatch raw 0 n.toUSize
    return (imgs, labels, n)
  augmentBatch := fun raw batch seed =>
    F32.randomHFlip raw batch 3 32 32 seed.toUSize

private def datasetIO : DatasetKind → DatasetIO
  | .imagenette => imagenetteIO
  | .mnist      => mnistIO
  | .cifar10    => cifar10IO

/-- Adam + cosine-LR + running-BN-stats training loop, generic over
    `DatasetKind`. The dataset specifies how to load the train/val
    data and what augmentation to apply per batch; everything else
    (init, optimizer, BN EMA, val eval, save) is identical across
    datasets.

    `spec` must have been compiled via `compileVmfbs` first. -/
def runTraining (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind)
    (dataDir : String) (sess : IreeSession) : IO Unit := do
  let pfx := spec.buildPrefix
  let batchN : Nat := cfg.batchSize
  let batch  : USize := cfg.batchSize.toUSize
  let dio := datasetIO ds

  let (trainImg, trainLbl, nTrain) ← dio.loadTrain dataDir
  IO.eprintln s!"  train: {nTrain} images ({dio.trainPixels} floats/image)"

  let params ← spec.heInitParams
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let bpE := nTrain / batchN
  let trainPixels := dio.trainPixels
  let allShapes := spec.shapesBA
  let xSh := spec.xShape batchN
  let nP := spec.totalParams
  let nT := 3 * nP
  let baseLR : Float := cfg.learningRate
  let warmup : Nat := cfg.warmupEpochs
  let epochs : Nat := cfg.epochs
  let nClasses : USize := spec.numClasses.toUSize

  let bnShapes := spec.bnShapesBA
  let nBnStats := spec.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine warmup={warmup}, label_smooth=0.1, wd={cfg.weightDecay}"
  IO.eprintln s!"  BN layers: {spec.bnLayers.size}, BN stat floats: {nBnStats}"

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

    let lr : Float := if cfg.cosineDecay then
      if epoch < warmup then
        baseLR * (epoch.toFloat + 1.0) / warmup.toFloat
      else
        baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - warmup.toFloat) / (epochs.toFloat - warmup.toFloat)))
    else
      baseLR  -- constant LR

    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let xbaRaw := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xba ← if cfg.augment then dio.augmentBatch xbaRaw batch (epoch * 10000 + bi) else pure xbaRaw
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess spec.trainFnName
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
      let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← dio.loadVal dataDir
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := spec.xShape evalBatch
        let evalParams := p.append runningBnStats
        let evalShapesBA := spec.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch dio.valPixels
          let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * spec.numClasses).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."

/-- End-to-end: compile all three vmfbs, load the train-step session,
    and run the training loop on the chosen dataset. The high-level
    entry point that every Main*Train.lean now calls.

    Defaults to Imagenette so existing trainers don't have to change. -/
def train (spec : NetSpec) (cfg : TrainConfig) (dataDir : String)
    (ds : DatasetKind := .imagenette) : IO Unit := do
  IO.eprintln s!"{spec.name}: {spec.totalParams} params"
  let trainVmfb ← spec.compileVmfbs cfg
  let sess ← IreeSession.create trainVmfb
  IO.eprintln "  session loaded"
  spec.runTraining cfg ds dataDir sess

end NetSpec
