/-!
# CIFAR-10 CNN in Lean 4 (multi-core, im2col + DGEMM)

Architecture matching Swift for TensorFlow book (Chapter 3):
  Conv3×3(3→32) → Conv3×3(32→32) → MaxPool2×2
  → Conv3×3(32→64) → Conv3×3(64→64) → MaxPool2×2
  → Dense 4096→512 → Dense 512→512 → Dense 512→10

Input: 3×32×32 RGB images, 10 classes.
Total parameters: 2,431,018

Optimized following the MNIST-2D im2col+DGEMM approach:
  • All Conv2D replaced by im2col + cblas_dgemm
  • Dense layers batched across the whole chunk (one DGEMM per weight matrix)
  • Tree-based gradient merge (log₂ depth)
  • Batch size scales with worker count (nWorkers × 32, min 128)
  • Fixed lr=0.003 (conservative; 0.01 diverged in MNIST-2D experiments)
-/

-- ===========================================================================
--  BLAS FFI
--  C := alpha * op(A) * op(B) + beta * C  (row-major, in-place on C)
--  order: 101=RowMajor  transX: 111=NoTrans 112=Trans
-- ===========================================================================

@[extern "lean_cblas_dgemm"]
opaque cblasDgemm
    (order transA transB : UInt32)
    (M N K : UInt64) (alpha : Float)
    (A : @& FloatArray) (lda : UInt64)
    (B : @& FloatArray) (ldb : UInt64)
    (beta : Float) (C : FloatArray) (ldc : UInt64) : FloatArray

-- ===========================================================================
--  RNG
-- ===========================================================================

structure Rng where
  state : UInt64

def Rng.new (seed : UInt64 := 42) : Rng :=
  ⟨if seed == 0 then 1 else seed⟩

def Rng.next (self : Rng) : Rng × Float :=
  let s := self.state
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 7)
  let s := s ^^^ (s <<< 17)
  (⟨s⟩, s.toNat.toFloat / 18446744073709551616.0 * 2.0 - 1.0)

def Rng.nextNat (self : Rng) (n : Nat) : Rng × Nat :=
  let (rng', f) := self.next
  (rng', ((f + 1.0) / 2.0 * n.toFloat).toUInt64.toNat % n)

def Rng.shuffle (self : Rng) (arr : Array Nat) : Rng × Array Nat := Id.run do
  let mut rng := self
  let mut a := arr
  for i in [0:a.size] do
    let ri := a.size - 1 - i
    if ri > 0 then
      let (r', j) := rng.nextNat (ri + 1)
      rng := r'
      let tmp := a[ri]!
      a := a.set! ri a[j]!
      a := a.set! j tmp
  (rng, a)

-- ===========================================================================
--  FloatArray helpers
-- ===========================================================================

def fazeros (n : Nat) : FloatArray :=
  ⟨Array.replicate n 0.0⟩

def faadd (a b : FloatArray) : FloatArray := Id.run do
  let n := a.size
  let mut c := fazeros n
  for i in [0:n] do
    c := c.set! i (a[i]! + b[i]!)
  c

-- ===========================================================================
--  Softmax / argmax
-- ===========================================================================

def softmax (v : FloatArray) : FloatArray := Id.run do
  let n := v.size
  let mut mx := v[0]!
  for i in [1:n] do
    let vi := v[i]!
    mx := if vi > mx then vi else mx
  let mut exps := fazeros n
  let mut total : Float := 0.0
  for i in [0:n] do
    let e := Float.exp (v[i]! - mx)
    exps := exps.set! i e
    total := total + e
  let mut out := fazeros n
  for i in [0:n] do
    out := out.set! i (exps[i]! / total)
  out

def argmax (v : FloatArray) : Nat := Id.run do
  let mut best := 0
  let mut bv := v[0]!
  for i in [1:v.size] do
    let vi := v[i]!
    if vi > bv then
      best := i
      bv := vi
  best

-- ===========================================================================
--  CIFAR-10 loading
--  Binary format: each record is 1 byte label + 3072 bytes pixel data
--  Pixel data: 1024 R, 1024 G, 1024 B (row-major 32×32) — channel-first [3,32,32]
-- ===========================================================================

private def extractCIFARImage (buf : ByteArray) (recordIdx : Nat) : Nat × FloatArray := Id.run do
  let base := recordIdx * 3073
  let label := buf[base]!.toNat
  let mut v := fazeros 3072
  for p in [0:3072] do
    v := v.set! p (buf[base + 1 + p]!.toNat.toFloat / 255.0)
  (label, v)

structure Dataset where
  images : Array FloatArray
  labels : Array Nat
  count  : Nat

def Dataset.loadCIFARTrain (dir : String) : IO Dataset := do
  let mut imgs : Array FloatArray := #[]
  let mut lbls : Array Nat := #[]
  for batch in [1, 2, 3, 4, 5] do
    let path := dir ++ s!"/data_batch_{batch}.bin"
    let buf ← IO.FS.readBinFile path
    let nRecords := buf.size / 3073
    IO.println s!"  batch {batch}: {nRecords} images"
    for i in [0:nRecords] do
      let (label, img) := extractCIFARImage buf i
      imgs := imgs.push img
      lbls := lbls.push label
  IO.println s!"  Total training: {imgs.size} images, 3×32×32"
  return ⟨imgs, lbls, imgs.size⟩

def Dataset.loadCIFARTest (dir : String) : IO Dataset := do
  let path := dir ++ "/test_batch.bin"
  let buf ← IO.FS.readBinFile path
  let nRecords := buf.size / 3073
  let mut imgs : Array FloatArray := #[]
  let mut lbls : Array Nat := #[]
  for i in [0:nRecords] do
    let (label, img) := extractCIFARImage buf i
    imgs := imgs.push img
    lbls := lbls.push label
  IO.println s!"  Test: {imgs.size} images, 3×32×32"
  return ⟨imgs, lbls, imgs.size⟩

-- ===========================================================================
--  ReLU helpers
-- ===========================================================================

def faRelu (v : FloatArray) : FloatArray := Id.run do
  let n := v.size
  let mut out := fazeros n
  for i in [0:n] do
    let x := v[i]!
    out := out.set! i (if x > 0.0 then x else 0.0)
  out

def faReluBwd (dOut z : FloatArray) : FloatArray := Id.run do
  let n := z.size
  let mut out := fazeros n
  for i in [0:n] do
    out := out.set! i (if z[i]! > 0.0 then dOut[i]! else 0.0)
  out

-- ===========================================================================
--  im2col / col2im for Conv2D (3×3, same padding pad=1, channel-first)
--  im2col: input[IC×H×W] → patches[(H×W) × (IC×9)]
--  col2im: patches[(H×W) × (IC×9)] → dInput[IC×H×W]  (scatter-add)
-- ===========================================================================

def im2col (input : FloatArray) (ic h w : Nat) : FloatArray := Id.run do
  let hw  := h * w
  let ic9 := ic * 9
  let mut out := fazeros (hw * ic9)
  for r in [0:h] do
    for c in [0:w] do
      let row := r * w + c
      for i in [0:ic] do
        for kr in [0:3] do
          let ir := r + kr   -- actual input row = ir - 1  (pad offset)
          for kc in [0:3] do
            let jc := c + kc -- actual input col = jc - 1
            let col := i * 9 + kr * 3 + kc
            let v := if ir >= 1 && ir <= h && jc >= 1 && jc <= w then
                       input[i * hw + (ir - 1) * w + (jc - 1)]!
                     else 0.0
            out := out.set! (row * ic9 + col) v
  out

def col2im (patches : FloatArray) (ic h w : Nat) : FloatArray := Id.run do
  let hw  := h * w
  let ic9 := ic * 9
  let mut dI := fazeros (ic * hw)
  for r in [0:h] do
    for c in [0:w] do
      let row := r * w + c
      for i in [0:ic] do
        for kr in [0:3] do
          let ir := r + kr
          for kc in [0:3] do
            let jc := c + kc
            if ir >= 1 && ir <= h && jc >= 1 && jc <= w then
              let col  := i * 9 + kr * 3 + kc
              let iIdx := i * hw + (ir - 1) * w + (jc - 1)
              dI := dI.set! iIdx (dI[iIdx]! + patches[row * ic9 + col]!)
  dI

-- Broadcast bias[OC] across output[OC×HW].
private def convAddBias (out bias : FloatArray) (oc hw : Nat) : FloatArray := Id.run do
  let mut out := out
  for o in [0:oc] do
    let b := bias[o]!
    for j in [0:hw] do
      let idx := o * hw + j
      out := out.set! idx (out[idx]! + b)
  out

/-- Conv2D forward via im2col + DGEMM.
    output[OC×HW] = kernel[OC×IC9] × im2col^T[IC9×HW] + bias
    Returns (output, im2col_mat) — im2col_mat saved for backward pass. -/
def conv2dFwdGemm (input : FloatArray) (ic oc h w : Nat)
    (kernel bias : FloatArray) : FloatArray × FloatArray :=
  let hw  := h * w
  let ic9 := ic * 9
  let mat := im2col input ic h w                       -- [HW × IC9]
  let z := cblasDgemm 101 111 112 oc.toUInt64 hw.toUInt64 ic9.toUInt64 1.0
              kernel ic9.toUInt64 mat ic9.toUInt64 0.0 (fazeros (oc * hw)) hw.toUInt64
  (convAddBias z bias oc hw, mat)

/-- Conv2D backward via DGEMM + col2im.
    dKernel[OC×IC9] = dOut[OC×HW]  × mat[HW×IC9]
    d_mat[HW×IC9]   = dOut^T[HW×OC] × kernel[OC×IC9]  → col2im → dInput
    Returns (dKernel, dBias, dInput). -/
def conv2dBwdGemm (mat : FloatArray) (ic oc h w : Nat)
    (kernel dOut : FloatArray) : FloatArray × FloatArray × FloatArray :=
  let hw  := h * w
  let ic9 := ic * 9
  let dK := cblasDgemm 101 111 111 oc.toUInt64 ic9.toUInt64 hw.toUInt64 1.0
               dOut hw.toUInt64 mat ic9.toUInt64 0.0 (fazeros (oc * ic9)) ic9.toUInt64
  let dB := Id.run do
    let mut dB := fazeros oc
    for o in [0:oc] do
      let mut s : Float := 0.0
      for j in [0:hw] do s := s + dOut[o * hw + j]!
      dB := dB.set! o s
    dB
  let d_mat := cblasDgemm 101 112 111 hw.toUInt64 ic9.toUInt64 oc.toUInt64 1.0
                 dOut hw.toUInt64 kernel ic9.toUInt64 0.0 (fazeros (hw * ic9)) ic9.toUInt64
  (dK, dB, col2im d_mat ic h w)

-- ===========================================================================
--  MaxPool 2×2 stride 2
-- ===========================================================================

/-- MaxPool forward. input: C×H×W → output: C×(H/2)×(W/2), maxIndices. -/
def maxpool2dFwd (input : FloatArray) (c h w : Nat) : FloatArray × FloatArray := Id.run do
  let oh := h / 2
  let ow := w / 2
  let outSize := c * oh * ow
  let mut out := fazeros outSize
  let mut idx := fazeros outSize
  let mut k := 0
  for ch in [0:c] do
    let chOff := ch * h * w
    for r in [0:oh] do
      for col in [0:ow] do
        let sr := r * 2
        let sc := col * 2
        let i00 := chOff + sr * w + sc
        let i01 := i00 + 1
        let i10 := chOff + (sr + 1) * w + sc
        let i11 := i10 + 1
        let v00 := input[i00]!
        let v01 := input[i01]!
        let v10 := input[i10]!
        let v11 := input[i11]!
        let mut maxV := v00
        let mut maxI := i00
        if v01 > maxV then maxV := v01; maxI := i01
        if v10 > maxV then maxV := v10; maxI := i10
        if v11 > maxV then maxV := v11; maxI := i11
        out := out.set! k maxV
        idx := idx.set! k maxI.toFloat
        k := k + 1
  (out, idx)

/-- MaxPool backward. Routes gradient to max positions. -/
def maxpool2dBwd (dOut maxIdx : FloatArray) (inputSize : Nat) : FloatArray := Id.run do
  let mut dI := fazeros inputSize
  for i in [0:dOut.size] do
    let flatIdx := maxIdx[i]!.toUInt64.toNat
    dI := dI.set! flatIdx (dI[flatIdx]! + dOut[i]!)
  dI

-- ===========================================================================
--  Network structure
--  Conv1a: 3×3, 3→32    (864 kernel + 32 bias)
--  Conv1b: 3×3, 32→32   (9216 kernel + 32 bias)
--  MaxPool 2×2                                     → 32×16×16 = 8192
--  Conv2a: 3×3, 32→64   (18432 kernel + 64 bias)
--  Conv2b: 3×3, 64→64   (36864 kernel + 64 bias)
--  MaxPool 2×2                                     → 64×8×8 = 4096
--  Dense1: 4096→512      (2097152 + 512)
--  Dense2: 512→512       (262144 + 512)
--  Dense3: 512→10        (5120 + 10)
--  Total: 2,431,018 params
-- ===========================================================================

structure Net where
  k1a    : FloatArray   -- 32×3×9 = 864
  bias1a : FloatArray   -- 32
  k1b    : FloatArray   -- 32×32×9 = 9216
  bias1b : FloatArray   -- 32
  k2a    : FloatArray   -- 64×32×9 = 18432
  bias2a : FloatArray   -- 64
  k2b    : FloatArray   -- 64×64×9 = 36864
  bias2b : FloatArray   -- 64
  w1     : FloatArray   -- 512×4096 = 2097152
  b1     : FloatArray   -- 512
  w2     : FloatArray   -- 512×512 = 262144
  b2     : FloatArray   -- 512
  w3     : FloatArray   -- 10×512 = 5120
  b3     : FloatArray   -- 10

def Net.init (rng : Rng) : Net × Rng := Id.run do
  let mut g := rng
  -- conv1a: 3→32, fan_in=3*9=27, fan_out=32*9=288
  let s1a := Float.sqrt (6.0 / (27.0 + 288.0))
  let mut k1a := fazeros 864
  for i in [0:864] do
    let (g', v) := g.next; g := g'; k1a := k1a.set! i (v * s1a)
  -- conv1b: 32→32, fan_in=288, fan_out=288
  let s1b := Float.sqrt (6.0 / 576.0)
  let mut k1b := fazeros 9216
  for i in [0:9216] do
    let (g', v) := g.next; g := g'; k1b := k1b.set! i (v * s1b)
  -- conv2a: 32→64, fan_in=288, fan_out=576
  let s2a := Float.sqrt (6.0 / (288.0 + 576.0))
  let mut k2a := fazeros 18432
  for i in [0:18432] do
    let (g', v) := g.next; g := g'; k2a := k2a.set! i (v * s2a)
  -- conv2b: 64→64, fan_in=576, fan_out=576
  let s2b := Float.sqrt (6.0 / 1152.0)
  let mut k2b := fazeros 36864
  for i in [0:36864] do
    let (g', v) := g.next; g := g'; k2b := k2b.set! i (v * s2b)
  -- dense1: 4096→512
  let s3 := Float.sqrt (6.0 / (4096.0 + 512.0))
  let mut w1 := fazeros 2097152
  for i in [0:2097152] do
    let (g', v) := g.next; g := g'; w1 := w1.set! i (v * s3)
  -- dense2: 512→512
  let s4 := Float.sqrt (6.0 / 1024.0)
  let mut w2 := fazeros 262144
  for i in [0:262144] do
    let (g', v) := g.next; g := g'; w2 := w2.set! i (v * s4)
  -- dense3: 512→10
  let s5 := Float.sqrt (6.0 / 522.0)
  let mut w3 := fazeros 5120
  for i in [0:5120] do
    let (g', v) := g.next; g := g'; w3 := w3.set! i (v * s5)
  (⟨k1a, fazeros 32, k1b, fazeros 32,
    k2a, fazeros 64, k2b, fazeros 64,
    w1, fazeros 512, w2, fazeros 512, w3, fazeros 10⟩, g)

-- ===========================================================================
--  Dense layer primitives via DGEMM (row-major)
-- ===========================================================================

-- y[M] = W[M×K] * x[K] + b[M]  (single-sample inference)
private def denseForward (W b x : FloatArray) (M K : Nat) : FloatArray :=
  let y := cblasDgemm 101 111 111 M.toUInt64 1 K.toUInt64 1.0
              W K.toUInt64 x 1 0.0 (fazeros M) 1
  Id.run do
    let mut y := y
    for i in [0:M] do y := y.set! i (y[i]! + b[i]!)
    y

-- Z[N×M] = X[N×K] × W[M×K]^T + b[M]  (bias broadcast)
private def denseBatchForward (W b X : FloatArray) (N M K : Nat) : FloatArray :=
  let Z := cblasDgemm 101 111 112 N.toUInt64 M.toUInt64 K.toUInt64 1.0
              X K.toUInt64 W K.toUInt64 0.0 (fazeros (N * M)) M.toUInt64
  Id.run do
    let mut Z := Z
    for i in [0:N] do
      for j in [0:M] do
        Z := Z.set! (i * M + j) (Z[i * M + j]! + b[j]!)
    Z

-- accW[M×K] += dZ[N×M]^T × X[N×K]
private def denseBatchAccumW (accW dZ X : FloatArray) (N M K : Nat) : FloatArray :=
  cblasDgemm 101 112 111 M.toUInt64 K.toUInt64 N.toUInt64 1.0
    dZ M.toUInt64 X K.toUInt64 1.0 accW K.toUInt64

-- dX[N×K] = dZ[N×M] × W[M×K]
private def denseBatchInputGrad (W dZ : FloatArray) (N M K : Nat) : FloatArray :=
  cblasDgemm 101 111 111 N.toUInt64 K.toUInt64 M.toUInt64 1.0
    dZ M.toUInt64 W K.toUInt64 0.0 (fazeros (N * K)) K.toUInt64

-- accB[M] += column sums of dZ[N×M]
private def batchAccumBias (accB dZ : FloatArray) (N M : Nat) : FloatArray := Id.run do
  let mut accB := accB
  for i in [0:N] do
    for j in [0:M] do
      accB := accB.set! j (accB[j]! + dZ[i * M + j]!)
  accB

-- ===========================================================================
--  Inference forward pass
-- ===========================================================================

def Net.forward (net : Net) (x : FloatArray) : FloatArray := Id.run do
  -- Conv block 1: 3×32×32 → 32×32×32 → pool → 32×16×16
  let (z1a, _) := conv2dFwdGemm x     3  32 32 32 net.k1a net.bias1a
  let a1a := faRelu z1a
  let (z1b, _) := conv2dFwdGemm a1a  32  32 32 32 net.k1b net.bias1b
  let a1b := faRelu z1b
  let (pool1, _) := maxpool2dFwd a1b 32 32 32          -- 32×16×16 = 8192
  -- Conv block 2: 32×16×16 → 64×16×16 → pool → 64×8×8
  let (z2a, _) := conv2dFwdGemm pool1 32 64 16 16 net.k2a net.bias2a
  let a2a := faRelu z2a
  let (z2b, _) := conv2dFwdGemm a2a  64  64 16 16 net.k2b net.bias2b
  let a2b := faRelu z2b
  let (pool2, _) := maxpool2dFwd a2b 64 16 16          -- 64×8×8 = 4096
  -- Dense layers
  let a1 := faRelu (denseForward net.w1 net.b1 pool2 512 4096)
  let a2 := faRelu (denseForward net.w2 net.b2 a1   512  512)
  softmax (denseForward net.w3 net.b3 a2 10 512)

-- ===========================================================================
--  Training: parallel chunk-based gradient accumulation
-- ===========================================================================

structure ChunkResult where
  dK1a    : FloatArray
  dBias1a : FloatArray
  dK1b    : FloatArray
  dBias1b : FloatArray
  dK2a    : FloatArray
  dBias2a : FloatArray
  dK2b    : FloatArray
  dBias2b : FloatArray
  dW1     : FloatArray
  dB1     : FloatArray
  dW2     : FloatArray
  dB2     : FloatArray
  dW3     : FloatArray
  dB3     : FloatArray
  loss    : Float
  correct : Nat

def ChunkResult.zeros : ChunkResult :=
  ⟨fazeros 864,     fazeros 32,  fazeros 9216,    fazeros 32,
   fazeros 18432,   fazeros 64,  fazeros 36864,   fazeros 64,
   fazeros 2097152, fazeros 512, fazeros 262144,  fazeros 512,
   fazeros 5120,    fazeros 10,  0.0, 0⟩

instance : Inhabited ChunkResult := ⟨ChunkResult.zeros⟩

def ChunkResult.merge (a b : ChunkResult) : ChunkResult :=
  ⟨faadd a.dK1a b.dK1a,     faadd a.dBias1a b.dBias1a,
   faadd a.dK1b b.dK1b,     faadd a.dBias1b b.dBias1b,
   faadd a.dK2a b.dK2a,     faadd a.dBias2a b.dBias2a,
   faadd a.dK2b b.dK2b,     faadd a.dBias2b b.dBias2b,
   faadd a.dW1 b.dW1,       faadd a.dB1 b.dB1,
   faadd a.dW2 b.dW2,       faadd a.dB2 b.dB2,
   faadd a.dW3 b.dW3,       faadd a.dB3 b.dB3,
   a.loss + b.loss, a.correct + b.correct⟩

/-- Tree-based parallel merge: O(log n) depth instead of O(n). -/
def treeMerge (results : Array ChunkResult) : ChunkResult := Id.run do
  if results.size == 0 then return ChunkResult.zeros
  if results.size == 1 then return results[0]!
  let mut current := results
  while current.size > 1 do
    let mut next : Array ChunkResult := #[]
    let pairs := current.size / 2
    let mut tasks : Array (Task ChunkResult) := #[]
    for i in [0:pairs] do
      let a := current[i * 2]!
      let b := current[i * 2 + 1]!
      let t := Task.spawn fun _ => ChunkResult.merge a b
      tasks := tasks.push t
    for t in tasks do
      next := next.push t.get
    if current.size % 2 == 1 then
      next := next.push current[current.size - 1]!
    current := next
  current[0]!

/-- Forward + backward for N samples — batched DGEMM version.

  im2col sizes per sample:
    ic1a: (32×32) × (3×9)   = 1024 × 27    = 27,648
    ic1b: (32×32) × (32×9)  = 1024 × 288   = 294,912
    ic2a: (16×16) × (32×9)  = 256  × 288   = 73,728
    ic2b: (16×16) × (64×9)  = 256  × 576   = 147,456

  Five phases:
    1. Conv block 1 + 2 forward per sample → poolBatch[N×4096]
    2. Batched dense forward  (one DGEMM per weight matrix)
    3. Per-sample softmax, loss, dZ3 assembly
    4. Batched dense backward (one DGEMM per weight matrix)
    5. Per-sample conv backward (im2col + DGEMM)
-/
def computeChunk (net : Net) (ds : Dataset) (indices : Array Nat)
    (start stop : Nat) : ChunkResult := Id.run do
  let N := stop - start

  -- ====================================================================
  -- Phase 1: Conv forward for all N samples.
  -- Save im2col mats and pre-relu activations for backward.
  -- ====================================================================
  let mut sampIc1a     : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampIc1b     : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampIc2a     : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampIc2b     : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampZ1a      : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampZ1b      : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampZ2a      : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampZ2b      : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampPool1Idx : Array FloatArray := Array.replicate N (fazeros 0)
  let mut sampPool2Idx : Array FloatArray := Array.replicate N (fazeros 0)
  let mut poolBatch    := fazeros (N * 4096)

  for k in [0:N] do
    let idx := indices[start + k]!
    let x   := ds.images[idx]!
    -- Conv block 1
    let (z1a, ic1a) := conv2dFwdGemm x    3  32 32 32 net.k1a net.bias1a
    let a1a := faRelu z1a
    let (z1b, ic1b) := conv2dFwdGemm a1a 32  32 32 32 net.k1b net.bias1b
    let a1b := faRelu z1b
    let (pool1, pool1Idx) := maxpool2dFwd a1b 32 32 32   -- 32×16×16 = 8192
    -- Conv block 2
    let (z2a, ic2a) := conv2dFwdGemm pool1 32 64 16 16 net.k2a net.bias2a
    let a2a := faRelu z2a
    let (z2b, ic2b) := conv2dFwdGemm a2a  64  64 16 16 net.k2b net.bias2b
    let a2b := faRelu z2b
    let (pool2, pool2Idx) := maxpool2dFwd a2b 64 16 16   -- 64×8×8 = 4096
    -- Store intermediates
    sampIc1a     := sampIc1a.set!     k ic1a
    sampIc1b     := sampIc1b.set!     k ic1b
    sampIc2a     := sampIc2a.set!     k ic2a
    sampIc2b     := sampIc2b.set!     k ic2b
    sampZ1a      := sampZ1a.set!      k z1a
    sampZ1b      := sampZ1b.set!      k z1b
    sampZ2a      := sampZ2a.set!      k z2a
    sampZ2b      := sampZ2b.set!      k z2b
    sampPool1Idx := sampPool1Idx.set! k pool1Idx
    sampPool2Idx := sampPool2Idx.set! k pool2Idx
    -- Append pool2 into poolBatch
    let base := k * 4096
    for j in [0:4096] do
      poolBatch := poolBatch.set! (base + j) pool2[j]!

  -- ====================================================================
  -- Phase 2: Batched dense forward.
  -- ====================================================================
  let Z1Batch := denseBatchForward net.w1 net.b1 poolBatch N 512 4096  -- [N×512]
  let A1Batch := faRelu Z1Batch
  let Z2Batch := denseBatchForward net.w2 net.b2 A1Batch  N 512  512  -- [N×512]
  let A2Batch := faRelu Z2Batch
  let Z3Batch := denseBatchForward net.w3 net.b3 A2Batch  N  10  512  -- [N×10]

  -- ====================================================================
  -- Phase 3: Per-sample softmax, loss, dZ3 assembly.
  -- ====================================================================
  let mut dZ3Batch := fazeros (N * 10)
  let mut lossSum  : Float := 0.0
  let mut correct  : Nat   := 0

  for k in [0:N] do
    let idx   := indices[start + k]!
    let label := ds.labels[idx]!
    let base  := k * 10
    let mut z3k := fazeros 10
    for j in [0:10] do z3k := z3k.set! j Z3Batch[base + j]!
    let prob := softmax z3k
    lossSum := lossSum - Float.log (prob[label]! + 1.0e-8)
    if argmax prob == label then correct := correct + 1
    for j in [0:10] do
      dZ3Batch := dZ3Batch.set! (base + j)
        (prob[j]! - if j == label then 1.0 else 0.0)

  -- ====================================================================
  -- Phase 4: Batched dense backward.
  -- ====================================================================
  -- Dense3 backward
  let mut accW3 := fazeros 5120
  let mut accB3 := fazeros 10
  accW3 := denseBatchAccumW accW3 dZ3Batch A2Batch N 10 512
  accB3 := batchAccumBias   accB3 dZ3Batch N 10
  let dA2Batch := denseBatchInputGrad net.w3 dZ3Batch N 10 512         -- [N×512]
  -- Dense2 backward
  let dZ2Batch := faReluBwd dA2Batch Z2Batch
  let mut accW2 := fazeros 262144
  let mut accB2 := fazeros 512
  accW2 := denseBatchAccumW accW2 dZ2Batch A1Batch N 512 512
  accB2 := batchAccumBias   accB2 dZ2Batch N 512
  let dA1Batch := denseBatchInputGrad net.w2 dZ2Batch N 512 512        -- [N×512]
  -- Dense1 backward
  let dZ1Batch := faReluBwd dA1Batch Z1Batch
  let mut accW1 := fazeros 2097152
  let mut accB1 := fazeros 512
  accW1 := denseBatchAccumW accW1 dZ1Batch poolBatch N 512 4096
  accB1 := batchAccumBias   accB1 dZ1Batch N 512
  let dPool2Batch := denseBatchInputGrad net.w1 dZ1Batch N 512 4096    -- [N×4096]

  -- ====================================================================
  -- Phase 5: Per-sample conv backward.
  -- ====================================================================
  let mut accK1a    := fazeros 864
  let mut accBias1a := fazeros 32
  let mut accK1b    := fazeros 9216
  let mut accBias1b := fazeros 32
  let mut accK2a    := fazeros 18432
  let mut accBias2a := fazeros 64
  let mut accK2b    := fazeros 36864
  let mut accBias2b := fazeros 64

  for k in [0:N] do
    -- Extract dPool2[k] from dPool2Batch
    let mut dPool2 := fazeros 4096
    let base := k * 4096
    for j in [0:4096] do dPool2 := dPool2.set! j dPool2Batch[base + j]!

    -- MaxPool2 backward: dPool2 → dA2b (64×16×16 = 16384)
    let dA2b := maxpool2dBwd dPool2 sampPool2Idx[k]! (64 * 16 * 16)

    -- ReLU2b + Conv2b backward
    let dZ2b := faReluBwd dA2b sampZ2b[k]!
    let (dK2b, dBias2b, dA2a) := conv2dBwdGemm sampIc2b[k]! 64 64 16 16 net.k2b dZ2b
    for i in [0:36864] do
      accK2b    := accK2b.set!    i (accK2b[i]!    + dK2b[i]!)
    for i in [0:64] do
      accBias2b := accBias2b.set! i (accBias2b[i]! + dBias2b[i]!)

    -- ReLU2a + Conv2a backward → dPool1 (32×16×16 = 8192)
    let dZ2a := faReluBwd dA2a sampZ2a[k]!
    let (dK2a, dBias2a, dPool1) := conv2dBwdGemm sampIc2a[k]! 32 64 16 16 net.k2a dZ2a
    for i in [0:18432] do
      accK2a    := accK2a.set!    i (accK2a[i]!    + dK2a[i]!)
    for i in [0:64] do
      accBias2a := accBias2a.set! i (accBias2a[i]! + dBias2a[i]!)

    -- MaxPool1 backward: dPool1 → dA1b (32×32×32 = 32768)
    let dA1b := maxpool2dBwd dPool1 sampPool1Idx[k]! (32 * 32 * 32)

    -- ReLU1b + Conv1b backward
    let dZ1b := faReluBwd dA1b sampZ1b[k]!
    let (dK1b, dBias1b, dA1a) := conv2dBwdGemm sampIc1b[k]! 32 32 32 32 net.k1b dZ1b
    for i in [0:9216] do
      accK1b    := accK1b.set!    i (accK1b[i]!    + dK1b[i]!)
    for i in [0:32] do
      accBias1b := accBias1b.set! i (accBias1b[i]! + dBias1b[i]!)

    -- ReLU1a + Conv1a backward (dInput discarded — it's the network input)
    let dZ1a := faReluBwd dA1a sampZ1a[k]!
    let (dK1a, dBias1a, _) := conv2dBwdGemm sampIc1a[k]! 3 32 32 32 net.k1a dZ1a
    for i in [0:864] do
      accK1a    := accK1a.set!    i (accK1a[i]!    + dK1a[i]!)
    for i in [0:32] do
      accBias1a := accBias1a.set! i (accBias1a[i]! + dBias1a[i]!)

  ⟨accK1a, accBias1a, accK1b, accBias1b,
   accK2a, accBias2a, accK2b, accBias2b,
   accW1, accB1, accW2, accB2, accW3, accB3, lossSum, correct⟩

-- ===========================================================================
--  Apply averaged gradients
-- ===========================================================================

def Net.applyGrad (net : Net) (cr : ChunkResult) (bLen : Nat) (lr : Float) : Net := Id.run do
  let s := lr / bLen.toFloat
  let mut nk1a := fazeros 864
  for i in [0:864]     do nk1a := nk1a.set! i (net.k1a[i]!   - s * cr.dK1a[i]!)
  let mut nb1a := fazeros 32
  for i in [0:32]      do nb1a := nb1a.set! i (net.bias1a[i]! - s * cr.dBias1a[i]!)
  let mut nk1b := fazeros 9216
  for i in [0:9216]    do nk1b := nk1b.set! i (net.k1b[i]!   - s * cr.dK1b[i]!)
  let mut nb1b := fazeros 32
  for i in [0:32]      do nb1b := nb1b.set! i (net.bias1b[i]! - s * cr.dBias1b[i]!)
  let mut nk2a := fazeros 18432
  for i in [0:18432]   do nk2a := nk2a.set! i (net.k2a[i]!   - s * cr.dK2a[i]!)
  let mut nb2a := fazeros 64
  for i in [0:64]      do nb2a := nb2a.set! i (net.bias2a[i]! - s * cr.dBias2a[i]!)
  let mut nk2b := fazeros 36864
  for i in [0:36864]   do nk2b := nk2b.set! i (net.k2b[i]!   - s * cr.dK2b[i]!)
  let mut nb2b := fazeros 64
  for i in [0:64]      do nb2b := nb2b.set! i (net.bias2b[i]! - s * cr.dBias2b[i]!)
  let mut nw1 := fazeros 2097152
  for i in [0:2097152] do nw1  := nw1.set!  i (net.w1[i]!    - s * cr.dW1[i]!)
  let mut nb1 := fazeros 512
  for i in [0:512]     do nb1  := nb1.set!  i (net.b1[i]!    - s * cr.dB1[i]!)
  let mut nw2 := fazeros 262144
  for i in [0:262144]  do nw2  := nw2.set!  i (net.w2[i]!    - s * cr.dW2[i]!)
  let mut nb2 := fazeros 512
  for i in [0:512]     do nb2  := nb2.set!  i (net.b2[i]!    - s * cr.dB2[i]!)
  let mut nw3 := fazeros 5120
  for i in [0:5120]    do nw3  := nw3.set!  i (net.w3[i]!    - s * cr.dW3[i]!)
  let mut nb3 := fazeros 10
  for i in [0:10]      do nb3  := nb3.set!  i (net.b3[i]!    - s * cr.dB3[i]!)
  ⟨nk1a, nb1a, nk1b, nb1b, nk2a, nb2a, nk2b, nb2b,
   nw1, nb1, nw2, nb2, nw3, nb3⟩

-- ===========================================================================
--  Training loop + eval
-- ===========================================================================

def trainEpoch (net : Net) (ds : Dataset) (lr : Float)
    (batchSize : Nat) (nWorkers : Nat) (minSamplesPerWorker : Nat)
    (epoch : Nat) (rng : Rng) : IO (Net × Rng) := do
  let indices := Array.range ds.count
  let (rng, indices) := rng.shuffle indices
  let nBatches := (ds.count + batchSize - 1) / batchSize
  let mut net := net
  let mut rng := rng
  let mut totalCorrect : Nat := 0
  let mut totalLoss : Float := 0.0
  let mut seen : Nat := 0

  for b in [0:nBatches] do
    let bStart := b * batchSize
    let bEnd := if bStart + batchSize > ds.count then ds.count else bStart + batchSize
    let bLen := bEnd - bStart

    -- Cap effective workers so each gets at least minSamplesPerWorker
    let effectiveWorkers := Nat.min nWorkers (bLen / minSamplesPerWorker |>.max 1)
    let chunkSize := (bLen + effectiveWorkers - 1) / effectiveWorkers
    let mut tasks : Array (Task ChunkResult) := #[]
    for w in [0:effectiveWorkers] do
      let wStart := bStart + w * chunkSize
      let wEnd := if wStart + chunkSize > bEnd then bEnd else wStart + chunkSize
      if wStart < bEnd then
        let netSnap := net
        let t := Task.spawn fun _ => computeChunk netSnap ds indices wStart wEnd
        tasks := tasks.push t

    let mut results : Array ChunkResult := #[]
    for t in tasks do
      results := results.push t.get
    let merged := treeMerge results

    totalLoss    := totalLoss    + merged.loss
    totalCorrect := totalCorrect + merged.correct
    seen         := seen         + bLen
    net          := net.applyGrad merged bLen lr

    if (b + 1) % 20 == 0 || b + 1 == nBatches then
      let pct := totalCorrect.toFloat / seen.toFloat * 100.0
      let avg := totalLoss / seen.toFloat
      IO.println s!"  epoch {epoch}  [batch {b+1}/{nBatches}]  loss={avg}  acc={pct}%"

  return (net, rng)

structure EvalResult where
  correct : Nat
  total   : Nat
  loss    : Float

def evaluate (net : Net) (ds : Dataset) (nWorkers : Nat) : IO EvalResult := do
  let chunkSize := (ds.count + nWorkers - 1) / nWorkers
  let mut tasks : Array (Task (Nat × Float)) := #[]
  for w in [0:nWorkers] do
    let wStart := w * chunkSize
    let wEnd := if wStart + chunkSize > ds.count then ds.count else wStart + chunkSize
    if wStart < ds.count then
      let t := Task.spawn fun _ => Id.run do
        let mut c : Nat := 0
        let mut loss : Float := 0.0
        for i in [wStart:wEnd] do
          let prob := net.forward ds.images[i]!
          let label := ds.labels[i]!
          loss := loss - Float.log (prob[label]! + 1.0e-8)
          if argmax prob == label then c := c + 1
        (c, loss)
      tasks := tasks.push t
  let mut correct : Nat := 0
  let mut lossSum : Float := 0.0
  for t in tasks do
    let (c, l) := t.get
    correct := correct + c
    lossSum := lossSum + l
  return ⟨correct, ds.count, lossSum / ds.count.toFloat⟩

-- ===========================================================================
--  main
-- ===========================================================================

def main (args : List String) : IO Unit := do
  let dir := args.head? |>.getD "./data"
  IO.println "╔══════════════════════════════════════════════════════════════════════╗"
  IO.println "║  CIFAR-10 CNN · Conv²×32→Pool→Conv²×64→Pool→512→512→10 · DGEMM opt ║"
  IO.println "╚══════════════════════════════════════════════════════════════════════╝"

  IO.println "Loading training set …"
  let train ← Dataset.loadCIFARTrain dir
  IO.println "Loading test set …"
  let test  ← Dataset.loadCIFARTest dir

  let nWorkers ← do
    let result ← IO.Process.output { cmd := "nproc", args := #[] }
    match result.stdout.trim.toNat? with
    | some k => pure (if k > 1 then k else 2)
    | none   => pure 4

  let minSamplesPerWorker := 32
  let batchSize := Nat.max 128 (nWorkers * minSamplesPerWorker)
  -- Fixed LR: conservative based on MNIST-2D findings (0.01 diverged at epoch 2)
  let baseLr := 0.003
  let lr := baseLr
  let epochs := 12

  IO.println s!"workers={nWorkers}  batchSize={batchSize}  lr={lr}  minPerWorker={minSamplesPerWorker}  epochs={epochs}  params=2431018"
  IO.println "Starting training..."

  let mut rng := Rng.new 314159
  let (net₀, rng') := Net.init rng
  rng := rng'
  let mut net := net₀

  for e in [0:epochs] do
    -- LR warmup only kicks in when lr != baseLr (currently a no-op since both are 0.003)
    let epochLr := if batchSize > 256 && e < 3
                   then baseLr + (lr - baseLr) * ((e + 1).toFloat / 3.0)
                   else lr
    let (net', rng') ← trainEpoch net train epochLr batchSize nWorkers minSamplesPerWorker (e + 1) rng
    net := net'
    rng := rng'
    let result ← evaluate net test nWorkers
    let accuracy := result.correct.toFloat / result.total.toFloat
    IO.println s!"[Epoch {e + 1}] Accuracy: {result.correct}/{result.total} ({accuracy}) Loss: {result.loss}"

  IO.println "\nDone."
