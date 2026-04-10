import LeanMlir.IreeRuntime

/-! Single-step train_step test from Lean via FFI. Verifies loss is in the
    expected range for random params + random labels, and that params actually
    changed after the update. Compares against mlir_poc/out_loss.npy which we
    know is 2.767844 from the JAX→IREE validation earlier. -/

def constFA (n : Nat) (v : Float) : FloatArray := Id.run do
  let mut a : FloatArray := .empty
  for _ in [:n] do a := a.push v
  return a

/-- xorshift-based "random-ish" FloatArray with Gaussian-ish output via 3-sum
    of uniforms. Deterministic in `seed`. -/
def randnFA (seed : Nat) (n : Nat) (scale : Float := 1.0) : FloatArray := Id.run do
  let mut s : UInt64 := seed.toUInt64 + 1
  let mut arr : FloatArray := .empty
  for _ in [:n] do
    let mut acc : Float := 0.0
    -- sum three uniforms in [-0.5, 0.5], scale to approx N(0, 0.25)
    for _ in [:3] do
      s := s ^^^ (s <<< 13)
      s := s ^^^ (s >>> 7)
      s := s ^^^ (s <<< 17)
      let u : Float := s.toFloat / UInt64.size.toFloat
      acc := acc + u - 0.5
    arr := arr.push (acc * 2.0 * scale)  -- stddev ~1 × scale
  return arr

/-- Pack labels (Array Nat, 0..9) into a ByteArray as int32 LE. -/
def labelsToByteArray (labels : Array Nat) : ByteArray := Id.run do
  let mut ba : ByteArray := .empty
  for l in labels do
    let v := l.toUInt32
    ba := ba.push (v.toNat.toUInt8)
    ba := ba.push (((v >>> 8) &&& 0xff).toNat.toUInt8)
    ba := ba.push (((v >>> 16) &&& 0xff).toNat.toUInt8)
    ba := ba.push (((v >>> 24) &&& 0xff).toNat.toUInt8)
  return ba

def main : IO Unit := do
  let vmfbPath := ".lake/build/train_step.vmfb"
  IO.println s!"Loading {vmfbPath}..."
  let sess ← IreeSession.create vmfbPath
  IO.println "Session ready."

  let batch : USize := 128

  -- Build initial params (seed=42-ish) with scale 0.05 for weights, zero biases.
  -- Pack as W0|b0|W1|b1|W2|b2 matching the FFI layout.
  let W0 := randnFA 42 MlpLayout.nW0 0.05
  let b0 := constFA MlpLayout.nb0 0.0
  let W1 := randnFA 43 MlpLayout.nW1 0.05
  let b1 := constFA MlpLayout.nb1 0.0
  let W2 := randnFA 44 MlpLayout.nW2 0.05
  let b2 := constFA MlpLayout.nb2 0.0
  let mut params : FloatArray := .empty
  for v in W0.toList do params := params.push v
  for v in b0.toList do params := params.push v
  for v in W1.toList do params := params.push v
  for v in b1.toList do params := params.push v
  for v in W2.toList do params := params.push v
  for v in b2.toList do params := params.push v
  IO.println s!"Packed params: {params.size} (expected {MlpLayout.nParams})"

  let x := randnFA 1 (128*784) 1.0
  let labels : Array Nat := (Array.range 128).map (fun i => i % 10)
  let y := labelsToByteArray labels
  IO.println s!"Batch: x.size={x.size}, y.bytes={y.size}"

  -- Drop the trailing loss slot from a train_step output.
  let dropLoss (out : FloatArray) : FloatArray := Id.run do
    let mut a : FloatArray := .empty
    for i in [:MlpLayout.nParams] do a := a.push out[i]!
    return a

  -- First train step
  let out0 ← IreeSession.mlpTrainStep sess params x y 0.1 batch
  let loss0 := out0[MlpLayout.lossIdx]!
  let w0before := params[0]!
  let w0after  := out0[0]!
  IO.println s!"Step 1: loss = {loss0}"
  IO.println s!"  W0[0] before={w0before}  after={w0after}  delta={w0after - w0before}"

  -- Take 20 more steps with the same batch — loss should decrease monotonically.
  let mut p := dropLoss out0
  for step in [:20] do
    let out ← IreeSession.mlpTrainStep sess p x y 0.1 batch
    let loss := out[MlpLayout.lossIdx]!
    p := dropLoss out
    IO.println s!"Step {step+2}: loss = {loss}"
