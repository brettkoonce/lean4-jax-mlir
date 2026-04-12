import LeanMlir.IreeRuntime
import LeanMlir.MnistData

/-! MNIST MLP training in Lean, orchestrating an IREE-backed train_step
    kernel. Params live as a single packed FloatArray; each step ships them
    to GPU, pulls back updated params + loss. Test accuracy via the separate
    forward-only `.vmfb`. -/

def constFA (n : Nat) (v : Float) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for _ in [:n] do a := a.push v
  return a

/-- Pseudo-Gaussian via 3-sum of uniforms (central limit, stddev 1). -/
def randnFA (seed : Nat) (n : Nat) (scale : Float := 1.0) : FloatArray := Id.run do
  let mut s : UInt64 := seed.toUInt64 + 1
  let mut arr : FloatArray := .empty
  for _ in [:n] do
    let mut acc : Float := 0.0
    for _ in [:3] do
      s := s ^^^ (s <<< 13)
      s := s ^^^ (s >>> 7)
      s := s ^^^ (s <<< 17)
      let u : Float := s.toFloat / UInt64.size.toFloat
      acc := acc + u - 0.5
    arr := arr.push (acc * 2.0 * scale)
  return arr

/-- He init for ReLU: W ~ N(0, 2/fanIn). -/
def heInit (seed fanIn fanOut : Nat) : FloatArray :=
  let stddev := Float.sqrt (2.0 / fanIn.toFloat)
  randnFA seed (fanIn * fanOut) stddev

/-- Concatenate 6 FloatArrays. -/
def packParams (W0 b0 W1 b1 W2 b2 : FloatArray) : FloatArray := Id.run do
  let mut p : FloatArray := .empty
  for v in W0.toList do p := p.push v
  for v in b0.toList do p := p.push v
  for v in W1.toList do p := p.push v
  for v in b1.toList do p := p.push v
  for v in W2.toList do p := p.push v
  for v in b2.toList do p := p.push v
  return p

/-- Drop loss from train_step output (keep just params). -/
def dropLoss (out : FloatArray) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for i in [:MlpLayout.nParams] do a := a.push out[i]!
  return a

/-- Extract a slice of a FloatArray. -/
def floatSlice (src : FloatArray) (start count : Nat) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for i in [:count] do a := a.push src[start+i]!
  return a

/-- argmax over a single row of size 10. -/
def argmax10 (row : FloatArray) (off : Nat) : Nat := Id.run do
  let mut best : Nat := 0
  let mut bestv : Float := row[off]!
  for i in [1:10] do
    let v := row[off+i]!
    if v > bestv then best := i; bestv := v
  return best

def main : IO Unit := do
  IO.println "Loading MNIST..."
  let (trainImages, nTrain) ← MnistData.loadImages "data/train-images-idx3-ubyte"
  let (trainLabelsB, _)    ← MnistData.loadLabels "data/train-labels-idx1-ubyte"
  let (testImages,  nTest) ← MnistData.loadImages "data/t10k-images-idx3-ubyte"
  let (testLabelsB, _)     ← MnistData.loadLabels "data/t10k-labels-idx1-ubyte"
  -- Extract raw test labels as Nat for accuracy computation
  let testLabelsRaw ← IO.FS.readBinFile "data/t10k-labels-idx1-ubyte"
  IO.println s!"  train: {nTrain} images, test: {nTest} images"

  IO.println "Loading IREE modules..."
  let trainSess ← IreeSession.create ".lake/build/train_step.vmfb"
  let evalSess  ← IreeSession.create ".lake/build/mnist_mlp.vmfb"
  IO.println "  sessions ready"

  -- Init params via He
  IO.println "Initializing params..."
  let W0 := heInit 1 784 512
  let b0 := constFA 512 0.0
  let W1 := heInit 2 512 512
  let b1 := constFA 512 0.0
  let W2 := heInit 3 512 10
  let b2 := constFA 10 0.0
  let mut params := packParams W0 b0 W1 b1 W2 b2
  IO.println s!"  {params.size} params"

  let batch : USize := 128
  let batchSize : Nat := 128
  let lr : Float := 0.1
  let epochs := 12
  let batchesPerEpoch := nTrain / batchSize  -- 468

  for epoch in [:epochs] do
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:batchesPerEpoch] do
      let xb := MnistData.sliceImages trainImages (bi*batchSize) batchSize
      let yb := MnistData.sliceLabels trainLabelsB (bi*batchSize) batchSize
      let out ← IreeSession.mlpTrainStep trainSess params xb yb lr batch
      epochLoss := epochLoss + out[MlpLayout.lossIdx]!
      params := dropLoss out
    let tTrain ← IO.monoMsNow
    let avgLoss := epochLoss / batchesPerEpoch.toFloat

    -- Test accuracy: unpack params, run forward on test batches
    let W0' := floatSlice params 0 MlpLayout.nW0
    let b0' := floatSlice params MlpLayout.nW0 MlpLayout.nb0
    let W1' := floatSlice params (MlpLayout.nW0+MlpLayout.nb0) MlpLayout.nW1
    let b1' := floatSlice params (MlpLayout.nW0+MlpLayout.nb0+MlpLayout.nW1) MlpLayout.nb1
    let W2' := floatSlice params (MlpLayout.nW0+MlpLayout.nb0+MlpLayout.nW1+MlpLayout.nb1) MlpLayout.nW2
    let b2' := floatSlice params (MlpLayout.nW0+MlpLayout.nb0+MlpLayout.nW1+MlpLayout.nb1+MlpLayout.nW2) MlpLayout.nb2
    let testBatches := nTest / batchSize  -- 78
    let mut correct : Nat := 0
    for bi in [:testBatches] do
      let xb := MnistData.sliceImages testImages (bi*batchSize) batchSize
      let logits ← IreeSession.mlpForward evalSess xb W0' b0' W1' b1' W2' b2' batch
      for i in [:batchSize] do
        let pred := argmax10 logits (i*10)
        let gt := testLabelsRaw[8 + bi*batchSize + i]!.toNat
        if pred == gt then correct := correct + 1
    let t1 ← IO.monoMsNow
    let acc := correct.toFloat / (testBatches*batchSize).toFloat * 100.0
    IO.println s!"Epoch {epoch+1}: loss={avgLoss} acc={acc}% ({tTrain-t0}ms train + {t1-tTrain}ms eval)"
