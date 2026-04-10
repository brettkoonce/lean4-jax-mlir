/-!
# CIFAR-10 CNN in Lean 4

Architecture matching Swift for TensorFlow book (Chapter 3):
  Conv3×3(3→32) → Conv3×3(32→32) → MaxPool2×2
  → Conv3×3(32→64) → Conv3×3(64→64) → MaxPool2×2
  → Dense 4096→512 → Dense 512→512 → Dense 512→10

Input: 3×32×32 RGB images, 10 classes.
Total parameters: 2,431,018
-/

-- ===========================================================================
-- RNG
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
-- FloatArray helpers
-- ===========================================================================

def fazeros (n : Nat) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for _ in [0:n] do
    a := a.push 0.0
  a

def faadd (a b : FloatArray) : FloatArray := Id.run do
  let n := a.size
  let mut c : FloatArray := .empty
  for i in [0:n] do
    c := c.push (a[i]! + b[i]!)
  c

-- ===========================================================================
-- Softmax / argmax
-- ===========================================================================

def softmax (v : FloatArray) : FloatArray := Id.run do
  let n := v.size
  let mut mx := v[0]!
  for i in [1:n] do
    let vi := v[i]!
    mx := if vi > mx then vi else mx
  let mut exps : FloatArray := .empty
  let mut total : Float := 0.0
  for i in [0:n] do
    let e := Float.exp (v[i]! - mx)
    exps := exps.push e
    total := total + e
  let mut out : FloatArray := .empty
  for i in [0:n] do
    out := out.push (exps[i]! / total)
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
-- CIFAR-10 loading
-- CIFAR-10 binary format: each record is 1 byte label + 3072 bytes pixel data
-- Pixel data is 1024 R values, 1024 G values, 1024 B values (row-major 32×32)
-- This is already in channel-first order: [3, 32, 32]
-- ===========================================================================

private def extractCIFARImage (buf : ByteArray) (recordIdx : Nat) : Nat × FloatArray := Id.run do
  let base := recordIdx * 3073
  let label := buf[base]!.toNat
  let mut v : FloatArray := .empty
  for p in [0:3072] do
    v := v.push (buf[base + 1 + p]!.toNat.toFloat / 255.0)
  (label, v)

structure Dataset where
  images : Array FloatArray
  labels : Array Nat
  count  : Nat

/-- Load CIFAR-10 training data from 5 batch files. -/
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

/-- Load CIFAR-10 test data from single file. -/
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
-- ReLU helpers
-- ===========================================================================

def faRelu (v : FloatArray) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for i in [0:v.size] do
    let x := v[i]!
    out := out.push (if x > 0.0 then x else 0.0)
  out

def faReluBwd (dOut z : FloatArray) : FloatArray := Id.run do
  let mut out : FloatArray := .empty
  for i in [0:z.size] do
    out := out.push (if z[i]! > 0.0 then dOut[i]! else 0.0)
  out

-- ===========================================================================
-- Conv2D (3×3, same padding)
-- Layout: channel-first. Tensor [C,H,W] at index c*H*W + r*W + c_col
-- Kernel [OC,IC,3,3] at index oc*(IC*9) + ic*9 + kr*3 + kc
-- Same padding with 3×3: pad=1 on each side.
-- ===========================================================================

/-- Conv2D forward. input: IC×H×W, kernel: OC×IC×3×3, bias: OC → output: OC×H×W -/
def conv2dFwd (input : FloatArray) (ic oc h w : Nat)
    (kernel bias : FloatArray) : FloatArray := Id.run do
  let hw := h * w
  let ks := ic * 9
  let mut out : FloatArray := .empty
  for o in [0:oc] do
    for r in [0:h] do
      for c in [0:w] do
        let mut s := bias[o]!
        for i in [0:ic] do
          for kr in [0:3] do
            let ir := r + kr  -- ir - 1 is the actual input row (pad=1)
            if ir >= 1 && ir <= h then
              for kc in [0:3] do
                let jc := c + kc
                if jc >= 1 && jc <= w then
                  s := s + kernel[o * ks + i * 9 + kr * 3 + kc]! *
                           input[i * hw + (ir - 1) * w + (jc - 1)]!
        out := out.push s
  out

/-- Conv2D backward. Returns (dKernel, dBias, dInput). -/
def conv2dBwd (input : FloatArray) (ic oc h w : Nat)
    (kernel : FloatArray) (dOut : FloatArray) : FloatArray × FloatArray × FloatArray := Id.run do
  let hw := h * w
  let ks := ic * 9
  let mut dK := fazeros (oc * ks)
  let mut dB := fazeros oc
  let mut dI := fazeros (ic * hw)
  for o in [0:oc] do
    for r in [0:h] do
      for c in [0:w] do
        let d := dOut[o * hw + r * w + c]!
        dB := dB.set! o (dB[o]! + d)
        for i in [0:ic] do
          for kr in [0:3] do
            let ir := r + kr
            if ir >= 1 && ir <= h then
              for kc in [0:3] do
                let jc := c + kc
                if jc >= 1 && jc <= w then
                  let kIdx := o * ks + i * 9 + kr * 3 + kc
                  let iIdx := i * hw + (ir - 1) * w + (jc - 1)
                  dK := dK.set! kIdx (dK[kIdx]! + d * input[iIdx]!)
                  dI := dI.set! iIdx (dI[iIdx]! + d * kernel[kIdx]!)
  (dK, dB, dI)

-- ===========================================================================
-- MaxPool 2×2 stride 2
-- ===========================================================================

/-- MaxPool forward. input: C×H×W → output: C×(H/2)×(W/2), maxIndices (as Float). -/
def maxpool2dFwd (input : FloatArray) (c h w : Nat) : FloatArray × FloatArray := Id.run do
  let oh := h / 2
  let ow := w / 2
  let mut out : FloatArray := .empty
  let mut idx : FloatArray := .empty
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
        out := out.push maxV
        idx := idx.push maxI.toFloat
  (out, idx)

/-- MaxPool backward. dOut: C×(H/2)×(W/2), routes gradient to max positions. -/
def maxpool2dBwd (dOut maxIdx : FloatArray) (inputSize : Nat) : FloatArray := Id.run do
  let mut dI := fazeros inputSize
  for i in [0:dOut.size] do
    let flatIdx := maxIdx[i]!.toUInt64.toNat
    dI := dI.set! flatIdx (dI[flatIdx]! + dOut[i]!)
  dI

-- ===========================================================================
-- Network structure
-- Conv1a: 3×3, 3→32   (864 kernel + 32 bias)
-- Conv1b: 3×3, 32→32  (9,216 kernel + 32 bias)
-- MaxPool 2×2          (no params)              → 32×16×16
-- Conv2a: 3×3, 32→64  (18,432 kernel + 64 bias)
-- Conv2b: 3×3, 64→64  (36,864 kernel + 64 bias)
-- MaxPool 2×2          (no params)              → 64×8×8
-- Dense1: 4096→512     (2,097,152 + 512)
-- Dense2: 512→512      (262,144 + 512)
-- Dense3: 512→10       (5,120 + 10)
-- Total: 2,431,018 params
-- ===========================================================================

structure Net where
  k1a    : FloatArray   -- 32×3×3×3 = 864
  bias1a : FloatArray   -- 32
  k1b    : FloatArray   -- 32×32×3×3 = 9216
  bias1b : FloatArray   -- 32
  k2a    : FloatArray   -- 64×32×3×3 = 18432
  bias2a : FloatArray   -- 64
  k2b    : FloatArray   -- 64×64×3×3 = 36864
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
  let mut k1a : FloatArray := .empty
  let s1a := Float.sqrt (6.0 / (27.0 + 288.0))
  for _ in [0:864] do
    let (g', v) := g.next; g := g'; k1a := k1a.push (v * s1a)

  -- conv1b: 32→32, fan_in=32*9=288, fan_out=32*9=288
  let mut k1b : FloatArray := .empty
  let s1b := Float.sqrt (6.0 / 576.0)
  for _ in [0:9216] do
    let (g', v) := g.next; g := g'; k1b := k1b.push (v * s1b)

  -- conv2a: 32→64, fan_in=32*9=288, fan_out=64*9=576
  let mut k2a : FloatArray := .empty
  let s2a := Float.sqrt (6.0 / (288.0 + 576.0))
  for _ in [0:18432] do
    let (g', v) := g.next; g := g'; k2a := k2a.push (v * s2a)

  -- conv2b: 64→64, fan_in=64*9=576, fan_out=64*9=576
  let mut k2b : FloatArray := .empty
  let s2b := Float.sqrt (6.0 / 1152.0)
  for _ in [0:36864] do
    let (g', v) := g.next; g := g'; k2b := k2b.push (v * s2b)

  -- dense1: 4096→512
  let mut w1 : FloatArray := .empty
  let s3 := Float.sqrt (6.0 / (4096.0 + 512.0))
  for _ in [0:2097152] do
    let (g', v) := g.next; g := g'; w1 := w1.push (v * s3)

  -- dense2: 512→512
  let mut w2 : FloatArray := .empty
  let s4 := Float.sqrt (6.0 / 1024.0)
  for _ in [0:262144] do
    let (g', v) := g.next; g := g'; w2 := w2.push (v * s4)

  -- dense3: 512→10
  let mut w3 : FloatArray := .empty
  let s5 := Float.sqrt (6.0 / 522.0)
  for _ in [0:5120] do
    let (g', v) := g.next; g := g'; w3 := w3.push (v * s5)

  (⟨k1a, fazeros 32, k1b, fazeros 32,
    k2a, fazeros 64, k2b, fazeros 64,
    w1, fazeros 512, w2, fazeros 512, w3, fazeros 10⟩, g)

/-- Forward pass (inference — no intermediates saved). -/
def Net.forward (net : Net) (x : FloatArray) : FloatArray := Id.run do
  -- conv block 1: conv1a(3→32) → relu → conv1b(32→32) → relu → pool(32×32→16×16)
  let z1a := conv2dFwd x 3 32 32 32 net.k1a net.bias1a
  let a1a := faRelu z1a
  let z1b := conv2dFwd a1a 32 32 32 32 net.k1b net.bias1b
  let a1b := faRelu z1b
  let (pool1, _) := maxpool2dFwd a1b 32 32 32  -- → 32×16×16 = 8192

  -- conv block 2: conv2a(32→64) → relu → conv2b(64→64) → relu → pool(16×16→8×8)
  let z2a := conv2dFwd pool1 32 64 16 16 net.k2a net.bias2a
  let a2a := faRelu z2a
  let z2b := conv2dFwd a2a 64 64 16 16 net.k2b net.bias2b
  let a2b := faRelu z2b
  let (pool2, _) := maxpool2dFwd a2b 64 16 16  -- → 64×8×8 = 4096

  -- dense 4096→512 relu
  let mut a1 : FloatArray := .empty
  for i in [0:512] do
    let base := i * 4096
    let mut s := net.b1[i]!
    for j in [0:4096] do
      s := s + net.w1[base + j]! * pool2[j]!
    a1 := a1.push (if s > 0.0 then s else 0.0)

  -- dense 512→512 relu
  let mut a2 : FloatArray := .empty
  for i in [0:512] do
    let base := i * 512
    let mut s := net.b2[i]!
    for j in [0:512] do
      s := s + net.w2[base + j]! * a1[j]!
    a2 := a2.push (if s > 0.0 then s else 0.0)

  -- dense 512→10
  let mut z3 : FloatArray := .empty
  for i in [0:10] do
    let base := i * 512
    let mut s := net.b3[i]!
    for j in [0:512] do
      s := s + net.w3[base + j]! * a2[j]!
    z3 := z3.push s

  softmax z3

-- ===========================================================================
-- Training: parallel chunk-based gradient accumulation
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
  ⟨fazeros 864, fazeros 32, fazeros 9216, fazeros 32,
   fazeros 18432, fazeros 64, fazeros 36864, fazeros 64,
   fazeros 2097152, fazeros 512, fazeros 262144, fazeros 512,
   fazeros 5120, fazeros 10, 0.0, 0⟩

def ChunkResult.merge (a b : ChunkResult) : ChunkResult :=
  ⟨faadd a.dK1a b.dK1a, faadd a.dBias1a b.dBias1a,
   faadd a.dK1b b.dK1b, faadd a.dBias1b b.dBias1b,
   faadd a.dK2a b.dK2a, faadd a.dBias2a b.dBias2a,
   faadd a.dK2b b.dK2b, faadd a.dBias2b b.dBias2b,
   faadd a.dW1 b.dW1, faadd a.dB1 b.dB1,
   faadd a.dW2 b.dW2, faadd a.dB2 b.dB2,
   faadd a.dW3 b.dW3, faadd a.dB3 b.dB3,
   a.loss + b.loss, a.correct + b.correct⟩

/-- Forward + backward for a slice of samples. -/
def computeChunk (net : Net) (ds : Dataset) (indices : Array Nat)
    (start stop : Nat) : ChunkResult := Id.run do
  let mut accK1a    := fazeros 864
  let mut accBias1a := fazeros 32
  let mut accK1b    := fazeros 9216
  let mut accBias1b := fazeros 32
  let mut accK2a    := fazeros 18432
  let mut accBias2a := fazeros 64
  let mut accK2b    := fazeros 36864
  let mut accBias2b := fazeros 64
  let mut accW1     := fazeros 2097152
  let mut accB1     := fazeros 512
  let mut accW2     := fazeros 262144
  let mut accB2     := fazeros 512
  let mut accW3     := fazeros 5120
  let mut accB3     := fazeros 10
  let mut lossSum   : Float := 0.0
  let mut correct   : Nat := 0

  for k in [start:stop] do
    let idx := indices[k]!
    let x     := ds.images[idx]!   -- 3072 = 3×32×32
    let label := ds.labels[idx]!

    -- ======== FORWARD (saving intermediates) ========

    -- Conv block 1
    let z1a := conv2dFwd x 3 32 32 32 net.k1a net.bias1a          -- 32×32×32 = 32768
    let a1a := faRelu z1a
    let z1b := conv2dFwd a1a 32 32 32 32 net.k1b net.bias1b       -- 32×32×32 = 32768
    let a1b := faRelu z1b
    let (pool1, pool1Idx) := maxpool2dFwd a1b 32 32 32             -- 32×16×16 = 8192

    -- Conv block 2
    let z2a := conv2dFwd pool1 32 64 16 16 net.k2a net.bias2a     -- 64×16×16 = 16384
    let a2a := faRelu z2a
    let z2b := conv2dFwd a2a 64 64 16 16 net.k2b net.bias2b       -- 64×16×16 = 16384
    let a2b := faRelu z2b
    let (pool2, pool2Idx) := maxpool2dFwd a2b 64 16 16             -- 64×8×8 = 4096

    -- dense1: pool2 → 512
    let mut denseZ1 : FloatArray := .empty
    let mut denseA1 : FloatArray := .empty
    for i in [0:512] do
      let base := i * 4096
      let mut s := net.b1[i]!
      for j in [0:4096] do
        s := s + net.w1[base + j]! * pool2[j]!
      denseZ1 := denseZ1.push s
      denseA1 := denseA1.push (if s > 0.0 then s else 0.0)

    -- dense2: 512 → 512
    let mut denseZ2 : FloatArray := .empty
    let mut denseA2 : FloatArray := .empty
    for i in [0:512] do
      let base := i * 512
      let mut s := net.b2[i]!
      for j in [0:512] do
        s := s + net.w2[base + j]! * denseA1[j]!
      denseZ2 := denseZ2.push s
      denseA2 := denseA2.push (if s > 0.0 then s else 0.0)

    -- dense3: 512 → 10
    let mut z3 : FloatArray := .empty
    for i in [0:10] do
      let base := i * 512
      let mut s := net.b3[i]!
      for j in [0:512] do
        s := s + net.w3[base + j]! * denseA2[j]!
      z3 := z3.push s

    let prob := softmax z3

    -- stats
    lossSum := lossSum - Float.log (prob[label]! + 1.0e-8)
    if argmax prob == label then correct := correct + 1

    -- ======== BACKWARD ========

    -- dense3: dz3 = prob - one_hot → accW3, accB3, da2
    let mut da2 := fazeros 512
    for i in [0:10] do
      let target := if i == label then 1.0 else 0.0
      let dz3i := prob[i]! - target
      accB3 := accB3.set! i (accB3[i]! + dz3i)
      let base := i * 512
      for j in [0:512] do
        let wIdx := base + j
        accW3 := accW3.set! wIdx (accW3[wIdx]! + dz3i * denseA2[j]!)
        da2 := da2.set! j (da2[j]! + net.w3[wIdx]! * dz3i)

    -- dense2: dz2 = da2 ⊙ relu'(z2) → accW2, accB2, da1
    let mut da1 := fazeros 512
    for i in [0:512] do
      let dz2i := if denseZ2[i]! > 0.0 then da2[i]! else 0.0
      accB2 := accB2.set! i (accB2[i]! + dz2i)
      let base := i * 512
      for j in [0:512] do
        let wIdx := base + j
        accW2 := accW2.set! wIdx (accW2[wIdx]! + dz2i * denseA1[j]!)
        da1 := da1.set! j (da1[j]! + net.w2[wIdx]! * dz2i)

    -- dense1: dz1 = da1 ⊙ relu'(z1) → accW1, accB1, dPool2
    let mut dPool2 := fazeros 4096
    for i in [0:512] do
      let dz1i := if denseZ1[i]! > 0.0 then da1[i]! else 0.0
      accB1 := accB1.set! i (accB1[i]! + dz1i)
      let base := i * 4096
      for j in [0:4096] do
        let wIdx := base + j
        accW1 := accW1.set! wIdx (accW1[wIdx]! + dz1i * pool2[j]!)
        dPool2 := dPool2.set! j (dPool2[j]! + net.w1[wIdx]! * dz1i)

    -- ======== BACKWARD through conv block 2 ========

    -- maxpool2 backward: route dPool2 through max positions → dA2b
    let dA2b := maxpool2dBwd dPool2 pool2Idx (64 * 16 * 16)

    -- relu2b backward
    let dZ2b := faReluBwd dA2b z2b

    -- conv2b backward → dK2b, dBias2b, dA2a
    let (dK2b, dBias2b, dA2a) := conv2dBwd a2a 64 64 16 16 net.k2b dZ2b

    for i in [0:36864] do
      accK2b := accK2b.set! i (accK2b[i]! + dK2b[i]!)
    for i in [0:64] do
      accBias2b := accBias2b.set! i (accBias2b[i]! + dBias2b[i]!)

    -- relu2a backward
    let dZ2a := faReluBwd dA2a z2a

    -- conv2a backward → dK2a, dBias2a, dPool1
    let (dK2a, dBias2a, dPool1) := conv2dBwd pool1 32 64 16 16 net.k2a dZ2a

    for i in [0:18432] do
      accK2a := accK2a.set! i (accK2a[i]! + dK2a[i]!)
    for i in [0:64] do
      accBias2a := accBias2a.set! i (accBias2a[i]! + dBias2a[i]!)

    -- ======== BACKWARD through conv block 1 ========

    -- maxpool1 backward: route dPool1 through max positions → dA1b
    let dA1b := maxpool2dBwd dPool1 pool1Idx (32 * 32 * 32)

    -- relu1b backward
    let dZ1b := faReluBwd dA1b z1b

    -- conv1b backward → dK1b, dBias1b, dA1a
    let (dK1b, dBias1b, dA1a) := conv2dBwd a1a 32 32 32 32 net.k1b dZ1b

    for i in [0:9216] do
      accK1b := accK1b.set! i (accK1b[i]! + dK1b[i]!)
    for i in [0:32] do
      accBias1b := accBias1b.set! i (accBias1b[i]! + dBias1b[i]!)

    -- relu1a backward
    let dZ1a := faReluBwd dA1a z1a

    -- conv1a backward → dK1a, dBias1a (don't need dInput)
    let (dK1a, dBias1a, _) := conv2dBwd x 3 32 32 32 net.k1a dZ1a

    for i in [0:864] do
      accK1a := accK1a.set! i (accK1a[i]! + dK1a[i]!)
    for i in [0:32] do
      accBias1a := accBias1a.set! i (accBias1a[i]! + dBias1a[i]!)

  ⟨accK1a, accBias1a, accK1b, accBias1b,
   accK2a, accBias2a, accK2b, accBias2b,
   accW1, accB1, accW2, accB2, accW3, accB3, lossSum, correct⟩

/-- Apply averaged gradients. -/
def Net.applyGrad (net : Net) (cr : ChunkResult) (bLen : Nat) (lr : Float) : Net := Id.run do
  let s := lr / bLen.toFloat

  let mut nk1a : FloatArray := .empty
  for i in [0:864] do nk1a := nk1a.push (net.k1a[i]! - s * cr.dK1a[i]!)
  let mut nb1a : FloatArray := .empty
  for i in [0:32] do nb1a := nb1a.push (net.bias1a[i]! - s * cr.dBias1a[i]!)

  let mut nk1b : FloatArray := .empty
  for i in [0:9216] do nk1b := nk1b.push (net.k1b[i]! - s * cr.dK1b[i]!)
  let mut nb1b : FloatArray := .empty
  for i in [0:32] do nb1b := nb1b.push (net.bias1b[i]! - s * cr.dBias1b[i]!)

  let mut nk2a : FloatArray := .empty
  for i in [0:18432] do nk2a := nk2a.push (net.k2a[i]! - s * cr.dK2a[i]!)
  let mut nb2a : FloatArray := .empty
  for i in [0:64] do nb2a := nb2a.push (net.bias2a[i]! - s * cr.dBias2a[i]!)

  let mut nk2b : FloatArray := .empty
  for i in [0:36864] do nk2b := nk2b.push (net.k2b[i]! - s * cr.dK2b[i]!)
  let mut nb2b : FloatArray := .empty
  for i in [0:64] do nb2b := nb2b.push (net.bias2b[i]! - s * cr.dBias2b[i]!)

  let mut nw1 : FloatArray := .empty
  for i in [0:2097152] do nw1 := nw1.push (net.w1[i]! - s * cr.dW1[i]!)
  let mut nb1 : FloatArray := .empty
  for i in [0:512] do nb1 := nb1.push (net.b1[i]! - s * cr.dB1[i]!)

  let mut nw2 : FloatArray := .empty
  for i in [0:262144] do nw2 := nw2.push (net.w2[i]! - s * cr.dW2[i]!)
  let mut nb2 : FloatArray := .empty
  for i in [0:512] do nb2 := nb2.push (net.b2[i]! - s * cr.dB2[i]!)

  let mut nw3 : FloatArray := .empty
  for i in [0:5120] do nw3 := nw3.push (net.w3[i]! - s * cr.dW3[i]!)
  let mut nb3 : FloatArray := .empty
  for i in [0:10] do nb3 := nb3.push (net.b3[i]! - s * cr.dB3[i]!)

  ⟨nk1a, nb1a, nk1b, nb1b, nk2a, nb2a, nk2b, nb2b,
   nw1, nb1, nw2, nb2, nw3, nb3⟩

-- ===========================================================================
-- Training loop + eval
-- ===========================================================================

def trainEpoch (net : Net) (ds : Dataset) (lr : Float)
    (batchSize : Nat) (nWorkers : Nat)
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
    let chunkSize := (bLen + nWorkers - 1) / nWorkers

    let mut tasks : Array (Task ChunkResult) := #[]
    for w in [0:nWorkers] do
      let wStart := bStart + w * chunkSize
      let wEnd := if wStart + chunkSize > bEnd then bEnd else wStart + chunkSize
      if wStart < bEnd then
        let netSnap := net
        let t := Task.spawn fun _ => computeChunk netSnap ds indices wStart wEnd
        tasks := tasks.push t

    let mut merged := ChunkResult.zeros
    for t in tasks do
      merged := ChunkResult.merge merged t.get

    totalLoss := totalLoss + merged.loss
    totalCorrect := totalCorrect + merged.correct
    seen := seen + bLen
    net := net.applyGrad merged bLen lr

    -- progress every 50 batches
    if (b + 1) % 50 == 0 || b + 1 == nBatches then
      let pct := totalCorrect.toFloat / seen.toFloat * 100.0
      let avg := totalLoss / seen.toFloat
      IO.println s!"  epoch {epoch} [batch {b+1}/{nBatches}] loss={avg} acc={pct}%"

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
-- main
-- ===========================================================================

def main (args : List String) : IO Unit := do
  let dir := args.head? |>.getD "./data"
  IO.println "╔════════════════════════════════════════════════════════════════════╗"
  IO.println "║ CIFAR-10 CNN · Conv²×32→Pool→Conv²×64→Pool→512→512→10 · S4TF cfg ║"
  IO.println "╚════════════════════════════════════════════════════════════════════╝"

  IO.println "Loading training set …"
  let train ← Dataset.loadCIFARTrain dir
  IO.println "Loading test set …"
  let test  ← Dataset.loadCIFARTest dir

  let nWorkers ← do
    let result ← IO.Process.output { cmd := "nproc", args := #[] }
    match result.stdout.trim.toNat? with
    | some k => pure (if k > 1 then k else 2)
    | none   => pure 4

  let lr := 0.1
  let batchSize := 128
  let epochs := 12

  IO.println s!"workers={nWorkers}  lr={lr}  batch={batchSize}  epochs={epochs}  params=2431018"
  IO.println "Starting training..."

  let mut rng := Rng.new 314159
  let (net₀, rng') := Net.init rng
  rng := rng'
  let mut net := net₀

  for e in [0:epochs] do
    let (net', rng') ← trainEpoch net train lr batchSize nWorkers (e + 1) rng
    net := net'
    rng := rng'
    let result ← evaluate net test nWorkers
    let accuracy := result.correct.toFloat / result.total.toFloat
    IO.println s!"[Epoch {e + 1}] Accuracy: {result.correct}/{result.total} ({accuracy}) Loss: {result.loss}"

  IO.println "\nDone."
