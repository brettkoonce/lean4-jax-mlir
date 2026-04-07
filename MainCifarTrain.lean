import LeanJax.IreeRuntime
import LeanJax.MnistData  -- also has CifarData
import LeanJax.Types
import LeanJax.MlirCodegen

/-! CIFAR-10 CNN training via hand-written VJPs in IREE.
    Conv²→Pool→Conv²→Pool→Dense³, 2.4M params, SGD lr=0.01, 25 epochs. -/

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

def heInit (seed fanIn n : Nat) : FloatArray :=
  randnFA seed n (Float.sqrt (2.0 / fanIn.toFloat))

def concatFA (arrays : Array FloatArray) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for arr in arrays do
    for i in [:arr.size] do out := out.push arr[i]!
  return out

def dropLoss (out : FloatArray) (nParams : Nat) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for i in [:nParams] do a := a.push out[i]!
  return a

def cifarCnn : NetSpec where
  name := "CIFAR-10 CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense  512 512 .relu,
    .dense  512  10 .identity
  ]

def main : IO Unit := do
  -- Generate forward .vmfb (for eval if needed later)
  let fwdMlir := MlirCodegen.generate cifarCnn 128
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile ".lake/build/cifar_cnn.mlir" fwdMlir
  let compileArgs ← ireeCompileArgs ".lake/build/cifar_cnn.mlir" ".lake/build/cifar_cnn.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {r.stderr}"
    IO.Process.exit 1
  IO.println "Forward .vmfb compiled."

  IO.println "Loading CIFAR-10 (raw)..."
  -- Load raw bytes to avoid 150M FloatArray pushes at init time
  let mut trainRaw : ByteArray := .empty
  let mut trainLbl : ByteArray := .empty
  let mut nTrain : Nat := 0
  for i in [1:6] do
    let raw ← IO.FS.readBinFile ("data/cifar-10/data_batch_" ++ toString i ++ ".bin")
    let n := raw.size / 3073
    for j in [:n] do
      let off := j * 3073
      trainLbl := trainLbl.push raw[off]!
      trainLbl := trainLbl.push 0
      trainLbl := trainLbl.push 0
      trainLbl := trainLbl.push 0
    trainRaw := trainRaw.append (raw)
    nTrain := nTrain + n
  IO.println s!"  train: {nTrain} images ({trainRaw.size} raw bytes)"

  IO.println "Loading IREE train_step..."
  let trainSess ← IreeSession.create ".lake/build/cifar_train_step.vmfb"
  IO.println "  ready"

  IO.println "Initializing 2.4M params..."
  let params := concatFA #[
    heInit 20 (3*3*3) (32*3*3*3),       -- W0: conv 3→32
    constFA 32 0.0,                      -- b0
    heInit 21 (32*3*3) (32*32*3*3),     -- W1: conv 32→32
    constFA 32 0.0,                      -- b1
    heInit 22 (32*3*3) (64*32*3*3),     -- W2: conv 32→64
    constFA 64 0.0,                      -- b2
    heInit 23 (64*3*3) (64*64*3*3),     -- W3: conv 64→64
    constFA 64 0.0,                      -- b3
    heInit 24 4096 (4096*512),           -- W4: dense
    constFA 512 0.0,                     -- b4
    heInit 25 512 (512*512),             -- W5: dense
    constFA 512 0.0,                     -- b5
    heInit 26 512 (512*10),              -- W6: dense
    constFA 10 0.0                       -- b6
  ]
  IO.println s!"  {params.size} params (expected {CifarLayout.nParams})"

  let batch : USize := 128
  let batchN : Nat := 128
  let lr : Float := 0.01
  let epochs := 25
  let bpE := nTrain / batchN  -- 390 (50000/128)
  let shapes := CifarLayout.shapesBA
  let xSh := CifarLayout.xShape batchN

  let mut p := params
  for epoch in [:epochs] do
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      -- Build batch from raw bytes (each record: 1 byte label + 3072 bytes pixels)
      let mut xb : FloatArray := .empty
      let yb := MnistData.sliceLabels trainLbl (bi*batchN) batchN
      for si in [:batchN] do
        let recOff := (bi*batchN + si) * 3073  -- record offset in raw
        for pi in [:3072] do
          xb := xb.push (trainRaw[recOff + 1 + pi]!.toNat.toFloat / 255.0)
      let out ← IreeSession.trainStepPacked trainSess "jit_cifar_train_step.main"
                  p shapes xb xSh yb lr batch
      epochLoss := epochLoss + out[CifarLayout.lossIdx]!
      p := dropLoss out CifarLayout.nParams
    let t1 ← IO.monoMsNow
    IO.println s!"Epoch {epoch+1}: loss={epochLoss / bpE.toFloat} ({t1-t0}ms)"
