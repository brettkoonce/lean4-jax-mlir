import LeanMlir.IreeRuntime
import LeanMlir.MnistData

/-! MNIST CNN training via hand-written VJPs in IREE. Uses the generic
    trainStepPacked FFI with CnnLayout shape descriptors. -/

def constFA (n : Nat) (v : Float) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for _ in [:n] do a := a.push v
  return a

def randnFA (seed : Nat) (n : Nat) (scale : Float := 1.0) : FloatArray := Id.run do
  let mut s : UInt64 := seed.toUInt64 + 1
  let mut arr : FloatArray := .empty
  for _ in [:n] do
    let mut acc : Float := 0.0
    for _ in [:3] do
      s := s ^^^ (s <<< 13); s := s ^^^ (s >>> 7); s := s ^^^ (s <<< 17)
      acc := acc + s.toFloat / UInt64.size.toFloat - 0.5
    arr := arr.push (acc * 2.0 * scale)
  return arr

/-- He init for conv (fan_in = ic * kH * kW) or dense (fan_in = in_features). -/
def heInit (seed fanIn n : Nat) : FloatArray :=
  randnFA seed n (Float.sqrt (2.0 / fanIn.toFloat))

/-- Concat FloatArrays via toList. -/
def concatFA (arrays : Array FloatArray) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for arr in arrays do
    for v in arr.toList do out := out.push v
  return out

def dropLoss (out : FloatArray) (nParams : Nat) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for i in [:nParams] do a := a.push out[i]!
  return a

def argmax10 (row : FloatArray) (off : Nat) : Nat := Id.run do
  let mut best : Nat := 0; let mut bestv := row[off]!
  for i in [1:10] do
    let v := row[off+i]!
    if v > bestv then best := i; bestv := v
  return best

def floatSlice (src : FloatArray) (start count : Nat) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for i in [:count] do a := a.push src[start+i]!
  return a

def main : IO Unit := do
  IO.println "Loading MNIST..."
  let (trainImages, nTrain) ← MnistData.loadImages "data/train-images-idx3-ubyte"
  let (trainLabelsB, _) ← MnistData.loadLabels "data/train-labels-idx1-ubyte"
  let (testImages, nTest) ← MnistData.loadImages "data/t10k-images-idx3-ubyte"
  let testLabelsRaw ← IO.FS.readBinFile "data/t10k-labels-idx1-ubyte"
  IO.println s!"  train: {nTrain}, test: {nTest}"

  IO.println "Loading IREE modules..."
  let trainSess ← IreeSession.create ".lake/build/cnn_train_step.vmfb"
  let evalSess  ← IreeSession.create ".lake/build/mnist_cnn.vmfb"
  IO.println "  ready"

  -- He-init packed params: conv weights use fan_in = ic * k * k
  IO.println "Initializing 3.5M params..."
  let params := concatFA #[
    heInit 10 (1*3*3) (32*1*3*3),     -- W0
    constFA 32 0.0,                    -- b0
    heInit 11 (32*3*3) (32*32*3*3),   -- W1
    constFA 32 0.0,                    -- b1
    heInit 12 6272 (6272*512),         -- W2
    constFA 512 0.0,                   -- b2
    heInit 13 512 (512*512),           -- W3
    constFA 512 0.0,                   -- b3
    heInit 14 512 (512*10),            -- W4
    constFA 10 0.0                     -- b4
  ]
  IO.println s!"  {params.size} params (expected {CnnLayout.nParams})"

  let batch : USize := 128
  let batchN : Nat := 128
  let lr : Float := 0.01
  let epochs := 12
  let bpE := nTrain / batchN
  let shapes := CnnLayout.shapesBA
  let xSh := CnnLayout.xShape batchN

  let mut p := params
  for epoch in [:epochs] do
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xb := MnistData.sliceImages trainImages (bi*batchN) batchN
      let yb := MnistData.sliceLabels trainLabelsB (bi*batchN) batchN
      let out ← IreeSession.trainStepPacked trainSess "jit_cnn_train_step.main"
                  p shapes xb xSh yb lr batch
      epochLoss := epochLoss + out[CnnLayout.lossIdx]!
      p := dropLoss out CnnLayout.nParams
    let tTrain ← IO.monoMsNow

    -- Eval on test set using forward-only .vmfb
    -- Unpack first 5 param pairs for mlpForward... actually CNN needs
    -- different forward. Use the generic invoke or evalSess.
    -- For now, just report training loss (eval needs separate CNN forward FFI).
    let avgLoss := epochLoss / bpE.toFloat
    IO.println s!"Epoch {epoch+1}: loss={avgLoss} ({tTrain-t0}ms)"
