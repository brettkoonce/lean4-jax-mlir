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
    to `[batch, channels*H*W]`). Channel count comes from the first
    conv-style layer; defaults to 1 for pure-MLP specs. -/
def xShape (spec : NetSpec) (batch : Nat) : ByteArray :=
  packXShape #[batch, MlirCodegen.inputFlatDim spec]

/-- Sanitized base name for the spec — same transformation the codegen
    applies when generating MLIR module names. -/
def sanitizedName (spec : NetSpec) : String :=
  MlirCodegen.sanitize spec.name

/-- The eval forward function name to pass to `forwardF32`. The codegen
    emits modules of the form `@<sanitized_name>_eval` containing
    `func.func @forward_eval`, so the FFI call wants the qualified
    `"<sanitized_name>_eval.forward_eval"`. Computing this from the
    spec means trainers can never get the spelling wrong by hand. -/
def evalFnName (spec : NetSpec) : String :=
  spec.sanitizedName ++ "_eval.forward_eval"

/-- Emit He-initialized (W) + zero-bias for one conv+bias group. -/
private def heConvB (oc ic k : Nat) (seed : USize) : IO (ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (oc*ic*k*k).toUSize (Float.sqrt (2.0 / (ic*k*k).toFloat))
  let b ← F32.const oc.toUSize 0.0
  return (W, b, seed + 1)

/-- Emit He-initialized (W) + γ=1 + β=0 for one convBn group. -/
private def heConvBn (oc ic k : Nat) (seed : USize) : IO (ByteArray × ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (oc*ic*k*k).toUSize (Float.sqrt (2.0 / (ic*k*k).toFloat))
  let g ← F32.const oc.toUSize 1.0
  let b ← F32.const oc.toUSize 0.0
  return (W, g, b, seed + 1)

/-- Emit He-initialized (W) + zero-bias for one dense layer. -/
private def heDense (fi fo : Nat) (seed : USize) : IO (ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (fi*fo).toUSize (Float.sqrt (2.0 / fi.toFloat))
  let b ← F32.const fo.toUSize 0.0
  return (W, b, seed + 1)

/-- Emit γ=1, β=0 for one LayerNorm. -/
private def heLN (dim : Nat) : IO (ByteArray × ByteArray) := do
  let g ← F32.const dim.toUSize 1.0
  let b ← F32.const dim.toUSize 0.0
  return (g, b)

/-- He-initialize one layer's parameters. Dispatches per layer constructor
    so semantics (bias vs γ, cls token vs LN start, etc.) are unambiguous —
    no more shape-peek heuristic that misfires at patchEmbed or
    transformer-block boundaries. -/
private def heInitLayer (l : Layer) (seed : USize) : IO (Array ByteArray × USize) := do
  match l with
  | .dense fi fo _ =>
    let (W, b, s') ← heDense fi fo seed
    return (#[W, b], s')
  | .conv2d ic oc k _ _ =>
    let (W, b, s') ← heConvB oc ic k seed
    return (#[W, b], s')
  | .convBn ic oc k _ _ =>
    let (W, g, b, s') ← heConvBn oc ic k seed
    return (#[W, g, b], s')
  | .residualBlock ic oc nBlocks firstStride =>
    let needsProj := !(ic == oc && firstStride == 1)
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:nBlocks] do
      let blockIc := if bi == 0 then ic else oc
      let (W1, g1, b1, s') ← heConvBn oc blockIc 3 s
      parts := parts.push W1 |>.push g1 |>.push b1
      let (W2, g2, b2, s'') ← heConvBn oc oc 3 s'
      parts := parts.push W2 |>.push g2 |>.push b2
      s := s''
      if bi == 0 && needsProj then
        let (Wp, gp, bp, s''') ← heConvBn oc ic 1 s
        parts := parts.push Wp |>.push gp |>.push bp
        s := s'''
    return (parts, s)
  | .bottleneckBlock ic oc nBlocks firstStride =>
    let mid := oc / 4
    let needsProj := !(ic == oc && firstStride == 1)
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:nBlocks] do
      let blockIc := if bi == 0 then ic else oc
      let (W1, g1, b1, s') ← heConvBn mid blockIc 1 s
      parts := parts.push W1 |>.push g1 |>.push b1
      let (W2, g2, b2, s'') ← heConvBn mid mid 3 s'
      parts := parts.push W2 |>.push g2 |>.push b2
      let (W3, g3, b3, s''') ← heConvBn oc mid 1 s''
      parts := parts.push W3 |>.push g3 |>.push b3
      s := s'''
      if bi == 0 && needsProj then
        let (Wp, gp, bp, sp) ← heConvBn oc ic 1 s
        parts := parts.push Wp |>.push gp |>.push bp
        s := sp
    return (parts, s)
  | .invertedResidual ic oc expand _stride n =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:n] do
      let blockIc := if bi == 0 then ic else oc
      let mid := blockIc * expand
      if expand != 1 then
        let (W, g, b, s') ← heConvBn mid blockIc 1 s
        parts := parts.push W |>.push g |>.push b
        s := s'
      let (Wdw, gdw, bdw, s') ← heConvBn mid 1 3 s
      parts := parts.push Wdw |>.push gdw |>.push bdw
      let (Wp, gp, bp, s'') ← heConvBn oc mid 1 s'
      parts := parts.push Wp |>.push gp |>.push bp
      s := s''
    return (parts, s)
  | .mbConv ic oc expand kSize _stride n useSE =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:n] do
      let blockIc := if bi == 0 then ic else oc
      let mid := blockIc * expand
      let seMid := Nat.max 1 (mid / 4)
      if expand != 1 then
        let (W, g, b, s') ← heConvBn mid blockIc 1 s
        parts := parts.push W |>.push g |>.push b
        s := s'
      let (Wdw, gdw, bdw, s') ← heConvBn mid 1 kSize s
      parts := parts.push Wdw |>.push gdw |>.push bdw
      s := s'
      if useSE then
        let (Wsq, bsq, s'') ← heConvB seMid mid 1 s
        parts := parts.push Wsq |>.push bsq
        s := s''
        let (Wex, bex, s''') ← heConvB mid seMid 1 s
        parts := parts.push Wex |>.push bex
        s := s'''
      let (Wp, gp, bp, sp) ← heConvBn oc mid 1 s
      parts := parts.push Wp |>.push gp |>.push bp
      s := sp
    return (parts, s)
  | .patchEmbed ic dim p nP =>
    -- Conv W, bias=0, cls=0, pos=He init (fan-in is nP+1).
    let W ← F32.heInit seed (dim*ic*p*p).toUSize (Float.sqrt (2.0 / (ic*p*p).toFloat))
    let b ← F32.const dim.toUSize 0.0
    let cls ← F32.const dim.toUSize 0.0
    let pos ← F32.heInit (seed + 1) ((nP+1)*dim).toUSize (Float.sqrt (2.0 / (nP+1).toFloat))
    return (#[W, b, cls, pos], seed + 2)
  | .transformerEncoder dim _heads mlpDim nBlocks =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for _bi in [:nBlocks] do
      let (g1, b1) ← heLN dim
      parts := parts.push g1 |>.push b1
      let (Wq, bq, s') ← heDense dim dim s
      parts := parts.push Wq |>.push bq
      let (Wk, bk, s'') ← heDense dim dim s'
      parts := parts.push Wk |>.push bk
      let (Wv, bv, s''') ← heDense dim dim s''
      parts := parts.push Wv |>.push bv
      let (Wo, bo, s4) ← heDense dim dim s'''
      parts := parts.push Wo |>.push bo
      let (g2, b2) ← heLN dim
      parts := parts.push g2 |>.push b2
      let (Wfc1, bfc1, s5) ← heDense dim mlpDim s4
      parts := parts.push Wfc1 |>.push bfc1
      let (Wfc2, bfc2, s6) ← heDense mlpDim dim s5
      parts := parts.push Wfc2 |>.push bfc2
      s := s6
    -- Final LN after all blocks
    let (gf, bf) ← heLN dim
    parts := parts.push gf |>.push bf
    return (parts, s)
  | _ =>
    -- Unsupported layer (no trainable params OR not in paramShapes).
    -- flatten / maxPool / globalAvgPool / reshape have no params → empty.
    return (#[], seed)

/-- He-initialize all parameters for a spec, walking layer-by-layer. -/
def heInitParams (spec : NetSpec) : IO ByteArray := do
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  for l in spec.layers do
    let (parts, s') ← heInitLayer l seed
    paramParts := paramParts ++ parts
    seed := s'
  return F32.concat paramParts

end NetSpec
