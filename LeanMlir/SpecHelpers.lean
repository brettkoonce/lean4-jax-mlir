import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.F32Array

/-! Spec → tensor-shape / packed-bytes / parameter-init helpers.

These were copy-pasted into every `Main*Train.lean` file. Centralizing
them here means each trainer just imports `LeanMlir` and asks for
`spec.paramShapes`, `spec.bnShapesBA`, etc.

Anything new added here should be a pure function of `NetSpec` — no IO,
no architecture-specific specialization. Adding a new layer type means
adding a case here once and every trainer picks it up. -/

namespace NetSpec

/-- Per-parameter tensor shapes for the entire spec, in the same order
    `MlirCodegen.emitTrainStepSig` walks them. Used to pack/unpack the
    `params ++ m ++ v` ByteArray that goes into and out of every
    `trainStepAdamF32` call. -/
def paramShapes (spec : NetSpec) : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in spec.layers do
    match l with
    | .conv2d ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc]
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
    | .invertedResidual ic oc expand _stride n =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .mbConv ic oc expand kSize _stride n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .mbConvV3 ic oc expandCh kSize _stride useSE _useHSwish =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      if expandCh != ic then
        shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      shapes := shapes.push #[mid, 1, kSize, kSize] |>.push #[mid] |>.push #[mid]
      if useSE then
        shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
        shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .fusedMbConv ic oc expand kSize _stride n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        shapes := shapes.push #[mid, blockIc, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        if expand != 1 then
          shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      if preDWk > 0 then
        shapes := shapes.push #[ic, 1, preDWk, preDWk] |>.push #[ic] |>.push #[ic]
      shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      if postDWk > 0 then
        shapes := shapes.push #[mid, 1, postDWk, postDWk] |>.push #[mid] |>.push #[mid]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .patchEmbed ic dim p nP =>
      shapes := shapes.push #[dim, ic, p, p] |>.push #[dim]    -- W, b
      shapes := shapes.push #[dim]                              -- cls token
      shapes := shapes.push #[nP + 1, dim]                      -- positional embedding
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        -- LN1 (gamma, beta)
        shapes := shapes.push #[dim] |>.push #[dim]
        -- Q, K, V projections (W, b each)
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- Output projection
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- LN2
        shapes := shapes.push #[dim] |>.push #[dim]
        -- MLP fc1, fc2
        shapes := shapes.push #[dim, mlpDim] |>.push #[mlpDim]
        shapes := shapes.push #[mlpDim, dim] |>.push #[dim]
      -- Final LN after all blocks
      shapes := shapes.push #[dim] |>.push #[dim]
    | _ => pure ()
  return shapes

/-- Packed `params ++ m ++ v` shape array (3× the param shapes), as int32 LE
    `ByteArray`. This is what gets passed to `trainStepAdamF32`. -/
def shapesBA (spec : NetSpec) : ByteArray :=
  packShapes (spec.paramShapes ++ spec.paramShapes ++ spec.paramShapes)

/-- BN-layer (pidx, oc) pairs as discovered by the codegen. ViT-style
    transformer specs return an empty array. -/
def bnLayers (spec : NetSpec) : Array (Nat × Nat) :=
  MlirCodegen.collectBnLayers spec

/-- Total float count needed to store running BN stats (mean + var per
    BN layer). 0 for ViT and any non-BN architecture. -/
def nBnStats (spec : NetSpec) : Nat :=
  spec.bnLayers.foldl (fun acc (_, oc) => acc + oc * 2) 0

/-- BN shapes packed for the FFI: `[n_bn_layers, oc0, oc1, ...]` as int32 LE.
    The `trainStepAdamF32` FFI uses this to know how many BN-stat outputs
    to pop after the params/loss. -/
def bnShapesBA (spec : NetSpec) : ByteArray := Id.run do
  let push := fun (ba : ByteArray) (v : Nat) =>
    let v32 : UInt32 := v.toUInt32
    ba.push (v32 &&& 0xFF).toUInt8
      |>.push ((v32 >>> 8) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 16) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 24) &&& 0xFF).toUInt8
  let bn := spec.bnLayers
  let mut ba := push .empty bn.size
  for (_, oc) in bn do ba := push ba oc
  return ba

/-- Param shapes for the eval forward pass: regular params followed by
    one `[oc]` mean and `[oc]` var per BN layer. ViT-style specs collapse
    this to just `paramShapes` because there are no BN layers. -/
def evalShapes (spec : NetSpec) : Array (Array Nat) := Id.run do
  let mut shapes := spec.paramShapes
  for (_, oc) in spec.bnLayers do
    shapes := shapes.push #[oc] |>.push #[oc]
  return shapes

/-- Packed `evalShapes` as int32 LE for the eval forward FFI. -/
def evalShapesBA (spec : NetSpec) : ByteArray :=
  packShapes spec.evalShapes

/-- Packed input-tensor shape for a flat-image batch (NCHW collapsed
    to `[batch, 3*H*W]`). -/
def xShape (spec : NetSpec) (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * spec.imageH * spec.imageW]

/-- He-initialize all parameters for a spec. Walks `paramShapes`:
    - Rank ≥ 2 tensors → He-normal init using fan-in
    - Rank-1 tensors that come in pairs (gamma, beta) → 1.0 / 0.0 (BN/LN)
    - Lone rank-1 tensors → 0.0 (biases)

    The pair-vs-lone heuristic mirrors the inline logic that was in every
    `Main*Train.lean` and matches the order `paramShapes` emits.
    Returns the concatenated parameter buffer. -/
def heInitParams (spec : NetSpec) : IO ByteArray := do
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := spec.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      -- Look at the next 1D entries: pair → BN/LN gamma+beta (1.0, 0.0),
      --                              single → bias (0.0)
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 && shapes[si + 1]![0]! == n1 then
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)  -- gamma
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)  -- beta
          si := si + 1
        else
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)  -- bias
          si := si + 1
    else
      -- Lone rank-1 (e.g. ViT cls token) → zeros
      paramParts := paramParts.push (← F32.const n.toUSize 0.0)
      si := si + 1
  return F32.concat paramParts

end NetSpec
