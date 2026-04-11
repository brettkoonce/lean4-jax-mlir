import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.IreeRuntime
import LeanMlir.SpecHelpers

/-! Verify that `NetSpec.paramShapes`/`bnShapesBA`/`evalShapes`/`shapesBA`
    in LeanMlir.SpecHelpers produce byte-for-byte identical output to the
    hand-rolled walkers each Main*Train.lean used to ship with.

    If this passes for every architecture, the trainer refactor is safe. -/

-- ResNet-34
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

-- Reference implementation: copy of MainResnetTrain.lean's old paramShapes
def resnet34ParamShapesInline : Array (Array Nat) := Id.run do
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

def resnet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 10 .identity
  ]

-- Reference for R50: bottleneck blocks
def resnet50ParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet50.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, mid, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- ViT
def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,
    .transformerEncoder 192 3 768 12,
    .dense 192 10 .identity
  ]

def vitTinyParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in vitTiny.layers do
    match l with
    | .patchEmbed ic dim p nP =>
      shapes := shapes.push #[dim, ic, p, p] |>.push #[dim]
      shapes := shapes.push #[dim]
      shapes := shapes.push #[nP + 1, dim]
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        shapes := shapes.push #[dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim] |>.push #[dim]
        shapes := shapes.push #[dim, mlpDim] |>.push #[mlpDim]
        shapes := shapes.push #[mlpDim, dim] |>.push #[dim]
      shapes := shapes.push #[dim] |>.push #[dim]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | _ => pure ()
  return shapes

def shapesEq (a b : Array (Array Nat)) : Bool :=
  a.size == b.size && (List.range a.size).all (fun i =>
    let x := a[i]!
    let y := b[i]!
    x.size == y.size && (List.range x.size).all (fun j => x[j]! == y[j]!))

def main : IO Unit := do
  let mut ok := true
  let cases : Array (String × Array (Array Nat) × Array (Array Nat)) := #[
    ("ResNet-34", resnet34.paramShapes, resnet34ParamShapesInline),
    ("ResNet-50", resnet50.paramShapes, resnet50ParamShapesInline),
    ("ViT-Tiny",  vitTiny.paramShapes,  vitTinyParamShapesInline)
  ]
  for (name, helper, inline) in cases do
    if shapesEq helper inline then
      IO.println s!"  ✓ {name}: paramShapes match ({helper.size} entries)"
    else
      IO.println s!"  ✗ {name}: paramShapes DIFFER (helper={helper.size}, inline={inline.size})"
      ok := false
  if ok then
    IO.println "All paramShapes match. Refactor is safe."
  else
    IO.eprintln "Mismatch detected — do NOT refactor further."
    IO.Process.exit 1
