/-!
# MNIST CNN in Lean 4

Architecture: Conv3×3(1→32) → Conv3×3(32→32) → MaxPool2×2 → Dense 6272→512 → Dense 512→512 → Dense 512→10
Matching Swift for TensorFlow MNIST-2D config.
-/

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
--  Softmax / argmax
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
--  MNIST loading
-- ===========================================================================

private def rd32 (b : ByteArray) (off : Nat) : Nat :=
  b[off]!.toNat * 16777216 + b[off+1]!.toNat * 65536 +
  b[off+2]!.toNat * 256    + b[off+3]!.toNat

private def extractImage28x28 (imgBuf : ByteArray) (imgIdx : Nat) : FloatArray := Id.run do
  let base := 16 + imgIdx * 784
  let mut v : FloatArray := .empty
  for p in [0:784] do
    v := v.push (imgBuf[base + p]!.toNat.toFloat / 255.0)
  v

structure Dataset where
  images : Array FloatArray
  labels : Array Nat
  count  : Nat

def Dataset.load (imgPath lblPath : String) (maxN : Nat) : IO Dataset := do
  let ib ← IO.FS.readBinFile imgPath
  let lb ← IO.FS.readBinFile lblPath
  let total := rd32 ib 4
  let n := if maxN < total then maxN else total
  IO.println s!"  {total} in file, using {n}, 28×28"
  let mut imgs : Array FloatArray := #[]
  let mut lbls : Array Nat := #[]
  for i in [0:n] do
    imgs := imgs.push (extractImage28x28 ib i)
    lbls := lbls.push lb[8 + i]!.toNat
  return ⟨imgs, lbls, n⟩


-- ===========================================================================
--  ReLU helpers
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
--  Conv2D (3×3, same padding)
--  Layout: channel-first.  Tensor [C,H,W] at index c*H*W + r*W + c_col
--  Kernel [OC,IC,3,3] at index oc*(IC*9) + ic*9 + kr*3 + kc
--  Same padding with 3×3: pad=1 on each side.
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
            let ir := r + kr    -- ir - 1 is the actual input row (pad=1)
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
--  MaxPool 2×2 stride 2
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
--  Network structure
--  Conv1a: 3×3, 1→32    (288 kernel + 32 bias)
--  Conv1b: 3×3, 32→32   (9216 kernel + 32 bias)
--  MaxPool 2×2           (no params)
--  Dense1: 6272→512      (3,211,264 + 512)
--  Dense2: 512→512       (262,144 + 512)
--  Dense3: 512→10        (5,120 + 10)
--  Total: ~3.49M params
-- ===========================================================================

structure Net where
  k1a    : FloatArray     -- 32×1×3×3 = 288
  bias1a : FloatArray     -- 32
  k1b    : FloatArray     -- 32×32×3×3 = 9216
  bias1b : FloatArray     -- 32
  w1     : FloatArray     -- 512×6272 = 3211264
  b1     : FloatArray     -- 512
  w2     : FloatArray     -- 512×512 = 262144
  b2     : FloatArray     -- 512
  w3     : FloatArray     -- 10×512 = 5120
  b3     : FloatArray     -- 10

def Net.init (rng : Rng) : Net × Rng := Id.run do
  let mut g := rng
  let mut k1a : FloatArray := .empty
  let s1 := Float.sqrt (6.0 / 10.0)     -- sqrt(6/(1*9 + 32*9)) ≈ crude Xavier
  for _ in [0:288] do
    let (g', v) := g.next; g := g'; k1a := k1a.push (v * s1)
  let mut k1b : FloatArray := .empty
  let s2 := Float.sqrt (6.0 / 576.0)    -- sqrt(6/(32*9 + 32*9))
  for _ in [0:9216] do
    let (g', v) := g.next; g := g'; k1b := k1b.push (v * s2)
  let mut w1 : FloatArray := .empty
  let s3 := Float.sqrt (6.0 / 6784.0)   -- sqrt(6/(6272+512))
  for _ in [0:3211264] do
    let (g', v) := g.next; g := g'; w1 := w1.push (v * s3)
  let mut w2 : FloatArray := .empty
  let s4 := Float.sqrt (6.0 / 1024.0)
  for _ in [0:262144] do
    let (g', v) := g.next; g := g'; w2 := w2.push (v * s4)
  let mut w3 : FloatArray := .empty
  let s5 := Float.sqrt (6.0 / 522.0)
  for _ in [0:5120] do
    let (g', v) := g.next; g := g'; w3 := w3.push (v * s5)
  (⟨k1a, fazeros 32, k1b, fazeros 32,
    w1, fazeros 512, w2, fazeros 512, w3, fazeros 10⟩, g)

/-- Forward pass (inference — no intermediates saved). -/
def Net.forward (net : Net) (x : FloatArray) : FloatArray := Id.run do
  let z1a := conv2dFwd x 1 32 28 28 net.k1a net.bias1a
  let a1a := faRelu z1a
  let z1b := conv2dFwd a1a 32 32 28 28 net.k1b net.bias1b
  let a1b := faRelu z1b
  let (pool, _) := maxpool2dFwd a1b 32 28 28
  -- dense 6272→512 relu
  let mut a1 : FloatArray := .empty
  for i in [0:512] do
    let base := i * 6272
    let mut s := net.b1[i]!
    for j in [0:6272] do
      s := s + net.w1[base + j]! * pool[j]!
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
--  Training: parallel chunk-based gradient accumulation
-- ===========================================================================

structure ChunkResult where
  dK1a   : FloatArray
  dBias1a : FloatArray
  dK1b   : FloatArray
  dBias1b : FloatArray
  dW1    : FloatArray
  dB1    : FloatArray
  dW2    : FloatArray
  dB2    : FloatArray
  dW3    : FloatArray
  dB3    : FloatArray
  loss   : Float
  correct : Nat

def ChunkResult.zeros : ChunkResult :=
  ⟨fazeros 288, fazeros 32, fazeros 9216, fazeros 32,
   fazeros 3211264, fazeros 512, fazeros 262144, fazeros 512,
   fazeros 5120, fazeros 10, 0.0, 0⟩

def ChunkResult.merge (a b : ChunkResult) : ChunkResult :=
  ⟨faadd a.dK1a b.dK1a, faadd a.dBias1a b.dBias1a,
   faadd a.dK1b b.dK1b, faadd a.dBias1b b.dBias1b,
   faadd a.dW1 b.dW1, faadd a.dB1 b.dB1,
   faadd a.dW2 b.dW2, faadd a.dB2 b.dB2,
   faadd a.dW3 b.dW3, faadd a.dB3 b.dB3,
   a.loss + b.loss, a.correct + b.correct⟩

/-- Forward + backward for a slice of samples. -/
def computeChunk (net : Net) (ds : Dataset) (indices : Array Nat)
    (start stop : Nat) : ChunkResult := Id.run do
  let mut accK1a := fazeros 288
  let mut accBias1a := fazeros 32
  let mut accK1b := fazeros 9216
  let mut accBias1b := fazeros 32
  let mut accW1 := fazeros 3211264
  let mut accB1 := fazeros 512
  let mut accW2 := fazeros 262144
  let mut accB2 := fazeros 512
  let mut accW3 := fazeros 5120
  let mut accB3 := fazeros 10
  let mut lossSum : Float := 0.0
  let mut correct : Nat := 0

  for k in [start:stop] do
    let idx := indices[k]!
    let x := ds.images[idx]!          -- 784 = 1×28×28
    let label := ds.labels[idx]!

    -- ======== FORWARD (saving intermediates) ========
    let z1a := conv2dFwd x 1 32 28 28 net.k1a net.bias1a         -- 25088
    let a1a := faRelu z1a                                          -- 25088
    let z1b := conv2dFwd a1a 32 32 28 28 net.k1b net.bias1b      -- 25088
    let a1b := faRelu z1b                                          -- 25088
    let (pool, poolIdx) := maxpool2dFwd a1b 32 28 28              -- 6272, 6272

    -- dense1: pool → 512
    let mut denseZ1 : FloatArray := .empty
    let mut denseA1 : FloatArray := .empty
    for i in [0:512] do
      let base := i * 6272
      let mut s := net.b1[i]!
      for j in [0:6272] do
        s := s + net.w1[base + j]! * pool[j]!
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

    -- dense1: dz1 = da1 ⊙ relu'(z1) → accW1, accB1, dPool
    let mut dPool := fazeros 6272
    for i in [0:512] do
      let dz1i := if denseZ1[i]! > 0.0 then da1[i]! else 0.0
      accB1 := accB1.set! i (accB1[i]! + dz1i)
      let base := i * 6272
      for j in [0:6272] do
        let wIdx := base + j
        accW1 := accW1.set! wIdx (accW1[wIdx]! + dz1i * pool[j]!)
        dPool := dPool.set! j (dPool[j]! + net.w1[wIdx]! * dz1i)

    -- maxpool backward: route dPool through max positions
    let dA1b := maxpool2dBwd dPool poolIdx (32 * 28 * 28)

    -- relu1b backward
    let dZ1b := faReluBwd dA1b z1b

    -- conv1b backward → dK1b, dBias1b, dA1a
    let (dK1b, dBias1b, dA1a) := conv2dBwd a1a 32 32 28 28 net.k1b dZ1b
    -- accumulate conv1b kernel/bias gradients
    for i in [0:9216] do
      accK1b := accK1b.set! i (accK1b[i]! + dK1b[i]!)
    for i in [0:32] do
      accBias1b := accBias1b.set! i (accBias1b[i]! + dBias1b[i]!)

    -- relu1a backward
    let dZ1a := faReluBwd dA1a z1a

    -- conv1a backward → dK1a, dBias1a
    let (dK1a, dBias1a, _) := conv2dBwd x 1 32 28 28 net.k1a dZ1a
    for i in [0:288] do
      accK1a := accK1a.set! i (accK1a[i]! + dK1a[i]!)
    for i in [0:32] do
      accBias1a := accBias1a.set! i (accBias1a[i]! + dBias1a[i]!)

  ⟨accK1a, accBias1a, accK1b, accBias1b,
   accW1, accB1, accW2, accB2, accW3, accB3, lossSum, correct⟩

/-- Apply averaged gradients. -/
def Net.applyGrad (net : Net) (cr : ChunkResult) (bLen : Nat) (lr : Float) : Net := Id.run do
  let s := lr / bLen.toFloat
  let mut nk1a : FloatArray := .empty
  for i in [0:288] do nk1a := nk1a.push (net.k1a[i]! - s * cr.dK1a[i]!)
  let mut nb1a : FloatArray := .empty
  for i in [0:32] do nb1a := nb1a.push (net.bias1a[i]! - s * cr.dBias1a[i]!)
  let mut nk1b : FloatArray := .empty
  for i in [0:9216] do nk1b := nk1b.push (net.k1b[i]! - s * cr.dK1b[i]!)
  let mut nb1b : FloatArray := .empty
  for i in [0:32] do nb1b := nb1b.push (net.bias1b[i]! - s * cr.dBias1b[i]!)
  let mut nw1 : FloatArray := .empty
  for i in [0:3211264] do nw1 := nw1.push (net.w1[i]! - s * cr.dW1[i]!)
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
  ⟨nk1a, nb1a, nk1b, nb1b, nw1, nb1, nw2, nb2, nw3, nb3⟩

-- ===========================================================================
--  Training loop + eval
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
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║  MNIST CNN · Conv3×3→Conv3×3→Pool→512→512→10 · S4TF cfg  ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"

  IO.println "Loading training set …"
  let train ← Dataset.load (dir ++ "/train-images-idx3-ubyte")
                            (dir ++ "/train-labels-idx1-ubyte") 60000

  IO.println "Loading test set …"
  let test ← Dataset.load (dir ++ "/t10k-images-idx3-ubyte")
                           (dir ++ "/t10k-labels-idx1-ubyte") 10000

  let nWorkers ← do
    let result ← IO.Process.output { cmd := "nproc", args := #[] }
    match result.stdout.trim.toNat? with
    | some k => pure (if k > 1 then k else 2)
    | none   => pure 4

  let lr        := 0.01
  let batchSize := 128
  let epochs    := 12

  IO.println s!"workers={nWorkers}  lr={lr}  batch={batchSize}  epochs={epochs}  params=3489130"
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
