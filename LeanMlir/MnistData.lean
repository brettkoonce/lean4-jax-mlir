/-! MNIST IDX format loader.
    Reads raw binary IDX files as ByteArrays, produces FloatArrays for
    training. Same format the existing JAX codegen uses. -/

namespace MnistData

/-- Read a big-endian u32 from a ByteArray at offset. -/
private def readU32BE (ba : ByteArray) (off : Nat) : UInt32 :=
  (ba[off]!.toUInt32 <<< 24) ||| (ba[off+1]!.toUInt32 <<< 16) |||
  (ba[off+2]!.toUInt32 <<< 8) ||| ba[off+3]!.toUInt32

/-- Load MNIST images. File layout: 16-byte header (magic=2051, n, rows, cols)
    then `n * rows * cols` u8 bytes. Returns pixels normalized to [0,1] as
    a flat FloatArray of size `n * 784`. -/
def loadImages (path : String) : IO (FloatArray × Nat) := do
  let raw ← IO.FS.readBinFile path
  let magic := readU32BE raw 0
  if magic != 2051 then
    throw (.userError s!"bad IDX magic for images: {magic}")
  let n := (readU32BE raw 4).toNat
  let rows := (readU32BE raw 8).toNat
  let cols := (readU32BE raw 12).toNat
  let total := n * rows * cols
  let mut arr : FloatArray := .empty
  for i in [:total] do
    arr := arr.push (raw[16+i]!.toNat.toFloat / 255.0)
  return (arr, n)

/-- Load MNIST labels. File layout: 8-byte header (magic=2049, n) then n u8.
    Returns labels as a ByteArray packed int32 LE (4 bytes per label), matching
    the train_step FFI expectation. Also returns raw label bytes for argmax
    comparisons. -/
def loadLabels (path : String) : IO (ByteArray × Nat) := do
  let raw ← IO.FS.readBinFile path
  let magic := readU32BE raw 0
  if magic != 2049 then
    throw (.userError s!"bad IDX magic for labels: {magic}")
  let n := (readU32BE raw 4).toNat
  -- Pack labels as int32 LE for the FFI
  let mut ba : ByteArray := .empty
  for i in [:n] do
    let b := raw[8+i]!
    ba := ba.push b
    ba := ba.push 0
    ba := ba.push 0
    ba := ba.push 0
  return (ba, n)

/-- Extract a `batch * 784` slice from a flat pixel FloatArray. -/
def sliceImages (images : FloatArray) (start count : Nat) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for i in [:count*784] do
    out := out.push images[start*784 + i]!
  return out

/-- Extract `batch * 4` bytes from the int32-packed label ByteArray. -/
def sliceLabels (labels : ByteArray) (start count : Nat) : ByteArray := Id.run do
  let mut out : ByteArray := .empty
  for i in [:count*4] do
    out := out.push labels[start*4 + i]!
  return out

end MnistData

/-! CIFAR-10 binary loader.
    Each batch file: 10000 records × (1 byte label + 3072 bytes pixels).
    Pixels are CHW (R×1024, G×1024, B×1024). We normalize to [0,1]. -/

namespace CifarData

/-- Load one CIFAR-10 binary batch file. Returns (images: FloatArray, labels: ByteArray int32 LE, count). -/
def loadBatch (path : String) : IO (FloatArray × ByteArray × Nat) := do
  let raw ← IO.FS.readBinFile path
  let recordSize := 3073  -- 1 label + 3072 pixels
  let n := raw.size / recordSize
  let mut images : FloatArray := .empty
  let mut labels : ByteArray := .empty
  for i in [:n] do
    let off := i * recordSize
    -- label: 1 byte → int32 LE
    let lbl := raw[off]!
    labels := labels.push lbl
    labels := labels.push 0
    labels := labels.push 0
    labels := labels.push 0
    -- pixels: 3072 bytes → float32 normalized
    for j in [:3072] do
      images := images.push (raw[off + 1 + j]!.toNat.toFloat / 255.0)
  return (images, labels, n)

/-- Load all 5 training batches + 1 test batch. Returns (trainImg, trainLbl, nTrain, testImg, testLbl, nTest). -/
def loadAll (dir : String) : IO (FloatArray × ByteArray × Nat × FloatArray × ByteArray × Nat) := do
  let mut trainImg : FloatArray := .empty
  let mut trainLbl : ByteArray := .empty
  let mut nTrain : Nat := 0
  for i in [1:6] do
    let (img, lbl, n) ← loadBatch (dir ++ "/data_batch_" ++ toString i ++ ".bin")
    -- Push element by element (no toList to avoid GC pressure on large arrays)
    for j in [:img.size] do trainImg := trainImg.push img[j]!
    trainLbl := trainLbl.append lbl
    nTrain := nTrain + n
  let (testImg, testLbl, nTest) ← loadBatch (dir ++ "/test_batch.bin")
  return (trainImg, trainLbl, nTrain, testImg, testLbl, nTest)

/-- Slice images: batch of `count` images each 3072 floats. -/
def sliceImages (images : FloatArray) (start count : Nat) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for i in [:count * 3072] do
    out := out.push images[start * 3072 + i]!
  return out

/-- Slice labels from int32 LE ByteArray. -/
def sliceLabels (labels : ByteArray) (start count : Nat) : ByteArray := Id.run do
  let mut out : ByteArray := .empty
  for i in [:count * 4] do
    out := out.push labels[start * 4 + i]!
  return out

end CifarData
