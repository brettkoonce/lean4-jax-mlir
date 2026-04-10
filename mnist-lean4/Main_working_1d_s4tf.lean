/-!
# MNIST in Lean 4 — 3-layer MLP

Architecture: 784 → 512 (ReLU) → 512 (ReLU) → 10 (softmax)  [28×28 raw pixels]
Lean v4.28+ compatible.
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
--  Network: 784 → 512 (ReLU) → 512 (ReLU) → 10 (Softmax)
--
--  W1: 512×784 = 401408    b1: 512
--  W2: 512×512 = 262144    b2: 512
--  W3: 10×512  = 5120      b3: 10
--  Total: 669,706 parameters
-- ===========================================================================

structure Net where
  w1 : FloatArray     -- 512 × 784 = 401408
  b1 : FloatArray     -- 512
  w2 : FloatArray     -- 512 × 512 = 262144
  b2 : FloatArray     -- 512
  w3 : FloatArray     -- 10 × 512 = 5120
  b3 : FloatArray     -- 10

def Net.init (rng : Rng) : Net × Rng := Id.run do
  let mut g := rng
  let scale1 := Float.sqrt (6.0 / 1296.0)   -- sqrt(6/(512+784))
  let mut w1 : FloatArray := .empty
  for _ in [0:401408] do
    let (g', v) := g.next; g := g'
    w1 := w1.push (v * scale1)
  let scale2 := Float.sqrt (6.0 / 1024.0)   -- sqrt(6/(512+512))
  let mut w2 : FloatArray := .empty
  for _ in [0:262144] do
    let (g', v) := g.next; g := g'
    w2 := w2.push (v * scale2)
  let scale3 := Float.sqrt (6.0 / 522.0)    -- sqrt(6/(10+512))
  let mut w3 : FloatArray := .empty
  for _ in [0:5120] do
    let (g', v) := g.next; g := g'
    w3 := w3.push (v * scale3)
  (⟨w1, fazeros 512, w2, fazeros 512, w3, fazeros 10⟩, g)

/-- Forward pass for inference. -/
def Net.forward (net : Net) (x : FloatArray) : FloatArray := Id.run do
  -- layer 1: a1 = relu(W1·x + b1)
  let mut a1 : FloatArray := .empty
  for i in [0:512] do
    let base := i * 784
    let mut s := net.b1[i]!
    for j in [0:784] do
      s := s + net.w1[base + j]! * x[j]!
    a1 := a1.push (if s > 0.0 then s else 0.0)
  -- layer 2: a2 = relu(W2·a1 + b2)
  let mut a2 : FloatArray := .empty
  for i in [0:512] do
    let base := i * 512
    let mut s := net.b2[i]!
    for j in [0:512] do
      s := s + net.w2[base + j]! * a1[j]!
    a2 := a2.push (if s > 0.0 then s else 0.0)
  -- layer 3: softmax(W3·a2 + b3)
  let mut z3 : FloatArray := .empty
  for i in [0:10] do
    let base := i * 512
    let mut s := net.b3[i]!
    for j in [0:512] do
      s := s + net.w3[base + j]! * a2[j]!
    z3 := z3.push s
  softmax z3

-- ===========================================================================
--  FloatArray addition (for merging gradient chunks)
-- ===========================================================================

def faadd (a b : FloatArray) : FloatArray := Id.run do
  let n := a.size
  let mut c : FloatArray := .empty
  for i in [0:n] do
    c := c.push (a[i]! + b[i]!)
  c

-- ===========================================================================
--  Parallel training — chunk-based gradient accumulation
-- ===========================================================================

structure ChunkResult where
  accW1 : FloatArray
  accB1 : FloatArray
  accW2 : FloatArray
  accB2 : FloatArray
  accW3 : FloatArray
  accB3 : FloatArray
  loss  : Float
  correct : Nat

def ChunkResult.zeros : ChunkResult :=
  ⟨fazeros 401408, fazeros 512, fazeros 262144, fazeros 512,
   fazeros 5120, fazeros 10, 0.0, 0⟩

def ChunkResult.merge (a b : ChunkResult) : ChunkResult :=
  ⟨faadd a.accW1 b.accW1, faadd a.accB1 b.accB1,
   faadd a.accW2 b.accW2, faadd a.accB2 b.accB2,
   faadd a.accW3 b.accW3, faadd a.accB3 b.accB3,
   a.loss + b.loss, a.correct + b.correct⟩

/-- Pure function: forward+backward over a slice of samples.
    Each Task gets its own gradient accumulators. -/
def computeChunk (net : Net) (ds : Dataset) (indices : Array Nat)
    (start stop : Nat) : ChunkResult := Id.run do
  let mut accW1 := fazeros 401408
  let mut accB1 := fazeros 512
  let mut accW2 := fazeros 262144
  let mut accB2 := fazeros 512
  let mut accW3 := fazeros 5120
  let mut accB3 := fazeros 10
  let mut lossSum : Float := 0.0
  let mut correct : Nat := 0

  for k in [start:stop] do
    let idx := indices[k]!
    let x := ds.images[idx]!
    let label := ds.labels[idx]!

    -- forward (save z1, a1, z2, a2)
    let mut z1 : FloatArray := .empty
    let mut a1 : FloatArray := .empty
    for i in [0:512] do
      let base := i * 784
      let mut s := net.b1[i]!
      for j in [0:784] do
        s := s + net.w1[base + j]! * x[j]!
      z1 := z1.push s
      a1 := a1.push (if s > 0.0 then s else 0.0)

    let mut z2 : FloatArray := .empty
    let mut a2 : FloatArray := .empty
    for i in [0:512] do
      let base := i * 512
      let mut s := net.b2[i]!
      for j in [0:512] do
        s := s + net.w2[base + j]! * a1[j]!
      z2 := z2.push s
      a2 := a2.push (if s > 0.0 then s else 0.0)

    let mut z3 : FloatArray := .empty
    for i in [0:10] do
      let base := i * 512
      let mut s := net.b3[i]!
      for j in [0:512] do
        s := s + net.w3[base + j]! * a2[j]!
      z3 := z3.push s

    let prob := softmax z3

    lossSum := lossSum - Float.log (prob[label]! + 1.0e-8)
    if argmax prob == label then correct := correct + 1

    -- backward: layer 3
    let mut da2 := fazeros 512
    for i in [0:10] do
      let target := if i == label then 1.0 else 0.0
      let dz3i := prob[i]! - target
      accB3 := accB3.set! i (accB3[i]! + dz3i)
      let base := i * 512
      for j in [0:512] do
        let wIdx := base + j
        accW3 := accW3.set! wIdx (accW3[wIdx]! + dz3i * a2[j]!)
        da2 := da2.set! j (da2[j]! + net.w3[wIdx]! * dz3i)

    -- backward: layer 2
    let mut da1 := fazeros 512
    for i in [0:512] do
      let dz2i := if z2[i]! > 0.0 then da2[i]! else 0.0
      accB2 := accB2.set! i (accB2[i]! + dz2i)
      let base := i * 512
      for j in [0:512] do
        let wIdx := base + j
        accW2 := accW2.set! wIdx (accW2[wIdx]! + dz2i * a1[j]!)
        da1 := da1.set! j (da1[j]! + net.w2[wIdx]! * dz2i)

    -- backward: layer 1
    for j in [0:512] do
      let dz1j := if z1[j]! > 0.0 then da1[j]! else 0.0
      accB1 := accB1.set! j (accB1[j]! + dz1j)
      let base := j * 784
      for p in [0:784] do
        let wIdx := base + p
        accW1 := accW1.set! wIdx (accW1[wIdx]! + dz1j * x[p]!)

  ⟨accW1, accB1, accW2, accB2, accW3, accB3, lossSum, correct⟩

/-- Apply averaged gradients to produce new weights. -/
def Net.applyGrad (net : Net) (cr : ChunkResult) (bLen : Nat) (lr : Float) : Net := Id.run do
  let scale := lr / bLen.toFloat
  let mut newW1 : FloatArray := .empty
  for i in [0:401408] do
    newW1 := newW1.push (net.w1[i]! - scale * cr.accW1[i]!)
  let mut newB1 : FloatArray := .empty
  for i in [0:512] do
    newB1 := newB1.push (net.b1[i]! - scale * cr.accB1[i]!)
  let mut newW2 : FloatArray := .empty
  for i in [0:262144] do
    newW2 := newW2.push (net.w2[i]! - scale * cr.accW2[i]!)
  let mut newB2 : FloatArray := .empty
  for i in [0:512] do
    newB2 := newB2.push (net.b2[i]! - scale * cr.accB2[i]!)
  let mut newW3 : FloatArray := .empty
  for i in [0:5120] do
    newW3 := newW3.push (net.w3[i]! - scale * cr.accW3[i]!)
  let mut newB3 : FloatArray := .empty
  for i in [0:10] do
    newB3 := newB3.push (net.b3[i]! - scale * cr.accB3[i]!)
  ⟨newW1, newB1, newW2, newB2, newW3, newB3⟩

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

    -- split batch across workers
    let chunkSize := (bLen + nWorkers - 1) / nWorkers
    let mut tasks : Array (Task ChunkResult) := #[]
    for w in [0:nWorkers] do
      let wStart := bStart + w * chunkSize
      let wEnd := if wStart + chunkSize > bEnd then bEnd else wStart + chunkSize
      if wStart < bEnd then
        let netSnap := net
        let t := Task.spawn fun _ => computeChunk netSnap ds indices wStart wEnd
        tasks := tasks.push t

    -- collect and merge
    let mut merged := ChunkResult.zeros
    for t in tasks do
      merged := ChunkResult.merge merged t.get

    totalLoss := totalLoss + merged.loss
    totalCorrect := totalCorrect + merged.correct
    seen := seen + bLen

    net := net.applyGrad merged bLen lr

  let pct := totalCorrect.toFloat / seen.toFloat * 100.0
  let avg := totalLoss / seen.toFloat
  IO.println s!"  epoch {epoch}  loss={avg}  train_acc={pct}%"
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
  IO.println "╔═══════════════════════════════════════════════════════╗"
  IO.println "║  MNIST · 784→512→512→10 · S4TF-matched config       ║"
  IO.println "╚═══════════════════════════════════════════════════════╝"

  IO.println "Loading training set …"
  let train ← Dataset.load (dir ++ "/train-images-idx3-ubyte")
                            (dir ++ "/train-labels-idx1-ubyte") 60000

  IO.println "Loading test set …"
  let test ← Dataset.load (dir ++ "/t10k-images-idx3-ubyte")
                           (dir ++ "/t10k-labels-idx1-ubyte") 10000

  -- detect cores
  let nWorkers ← do
    let result ← IO.Process.output { cmd := "nproc", args := #[] }
    match result.stdout.trim.toNat? with
    | some k => pure (if k > 1 then k else 2)
    | none   => pure 4

  let lr        := 0.1
  let batchSize := 128
  let epochs    := 12

  IO.println s!"workers={nWorkers}  lr={lr}  batch={batchSize}  epochs={epochs}  params=669706"
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
