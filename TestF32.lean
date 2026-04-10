import LeanMlir.F32Array

def main : IO Unit := do
  IO.println "=== F32 ByteArray Tests ==="

  let zeros ← F32.const 10 0.0
  IO.println s!"const: {F32.size zeros} floats, val[0]={F32.read zeros 0}"

  let w ← F32.heInit 42 1000 0.05
  IO.println s!"heInit: {F32.size w} floats, val[0]={F32.read w 0}"

  let ab := F32.concat #[zeros, w]
  IO.println s!"concat: {F32.size ab} floats"

  let sl := F32.slice ab 5 10
  IO.println s!"slice(5,10): {F32.size sl} floats"

  let (imgs, n) ← F32.loadIdxImages "data/train-images-idx3-ubyte"
  IO.println s!"MNIST images: {n}, {F32.size imgs} floats, pixel[0]={F32.read imgs 0}"

  let (lbls, nl) ← F32.loadIdxLabels "data/train-labels-idx1-ubyte"
  IO.println s!"MNIST labels: {nl}, {lbls.size} bytes"

  let xb := F32.sliceImages imgs 0 128 784
  IO.println s!"batch slice: {F32.size xb} floats"

  IO.println "=== PASS ==="
