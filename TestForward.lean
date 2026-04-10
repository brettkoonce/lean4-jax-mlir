import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen

/-! Quick test: verify forwardF32 FFI works on ResNet-34. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

namespace ResnetLayout
def nParams : Nat := resnet34.totalParams
def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet34.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[oc, blockIc, 3, 3] |>.push #[oc] |>.push #[oc]
        shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes
end ResnetLayout

def main : IO Unit := do
  IO.println "Testing forwardF32 on ResNet-34..."

  -- Check forward vmfb exists
  let vmfbPath := ".lake/build/resnet34_fwd.vmfb"
  unless ← System.FilePath.pathExists vmfbPath do
    IO.eprintln "No forward vmfb — run test-resnet-residual first"
    IO.Process.exit 1

  let sess ← IreeSession.create vmfbPath
  IO.println "  session loaded"

  -- Init random params (untrained — just testing plumbing)
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := ResnetLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      si := si + 1
  let params := F32.concat paramParts
  IO.println s!"  {F32.size params} params"

  -- Load val data
  let (valImg, valLbl, nVal) ← F32.loadImagenette "data/imagenette/val.bin"
  IO.println s!"  val: {nVal} images"

  -- Forward 1 batch
  let batch : USize := 16
  let pixelsPerImage := 3 * 224 * 224
  let paramShapesBA := packShapes ResnetLayout.paramShapes
  let xSh := packXShape #[16, pixelsPerImage]

  let xba := F32.sliceImages valImg 0 16 pixelsPerImage
  let t0 ← IO.monoMsNow
  let logits ← IreeSession.forwardF32 sess "resnet_34.forward"
                  params paramShapesBA xba xSh batch 10
  let t1 ← IO.monoMsNow
  IO.println s!"  forward: {t1-t0}ms, logits size={logits.size} bytes"

  -- Print predictions vs labels (untrained so expect ~10% random)
  let mut correct : Nat := 0
  for i in [:16] do
    let pred := F32.argmax10 logits (i * 10).toUSize
    let label := valLbl.data[i * 4]!.toNat
    if pred.toNat == label then correct := correct + 1
  IO.println s!"  batch accuracy: {correct}/16 (untrained, expect ~10%)"
  IO.println "Forward FFI works!"
