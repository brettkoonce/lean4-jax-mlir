import LeanMlir.IreeRuntime

/-! Smoke test for the Lean→IREE FFI: load .vmfb, run forward once,
    print first 10 logits. Inputs are random but deterministic to match
    `mlir_poc/validate_mnist_mlp.py` (numpy seed=42). -/

/-- Fill a FloatArray of size `n` with numpy-like `standard_normal` from a
    fixed-seed RNG. Not quite numpy's output (different RNG), but stable. -/
def mkFloatArray (seed : Nat) (n : Nat) (scale : Float := 1.0) : FloatArray := Id.run do
  let mut arr : FloatArray := .empty
  let mut s := seed
  for _ in [:n] do
    -- cheap xorshift + uniform→normal via Box-Muller approximation
    s := s.xor (s <<< 13) |>.land 0xFFFFFFFFFFFFFFFF
    s := s.xor (s >>> 7)  |>.land 0xFFFFFFFFFFFFFFFF
    s := s.xor (s <<< 17) |>.land 0xFFFFFFFFFFFFFFFF
    let u := (s.toFloat / (UInt64.size.toFloat)) - 0.5
    arr := arr.push (u * 3.0 * scale)
  return arr

def main : IO Unit := do
  let vmfbPath := ".lake/build/mnist_mlp.vmfb"
  IO.println s!"Loading IREE session from {vmfbPath}..."
  let sess ← IreeSession.create vmfbPath
  IO.println "Session ready."

  let batch : USize := 128
  let x  := mkFloatArray  1 (128*784)
  let W0 := mkFloatArray  2 (784*512) 0.05
  let b0 := mkFloatArray  3 512 0.0
  let W1 := mkFloatArray  4 (512*512) 0.05
  let b1 := mkFloatArray  5 512 0.0
  let W2 := mkFloatArray  6 (512*10) 0.05
  let b2 := mkFloatArray  7 10 0.0

  IO.println "Invoking forward..."
  let logits ← IreeSession.mlpForward sess x W0 b0 W1 b1 W2 b2 batch
  IO.println s!"Got {logits.size} logits."
  let row0 := (Array.range 10).map (fun i => logits[i]!)
  IO.println s!"logits[0,:10] = {row0}"

  -- Simple benchmark: 78 forward calls
  let t0 ← IO.monoMsNow
  for _ in [:78] do
    let _ ← IreeSession.mlpForward sess x W0 b0 W1 b1 W2 b2 batch
    pure ()
  let t1 ← IO.monoMsNow
  IO.println s!"FFI from Lean: 78 calls in {t1-t0}ms = {(t1-t0).toFloat / 78.0}ms/call"
