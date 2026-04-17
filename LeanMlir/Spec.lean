import LeanMlir.Types
/-! Spec helpers: param counting, feature queries, arch display, validation. -/

def Layer.nParams : Layer → Nat
  | .conv2d ic oc k _ _     => oc * ic * k * k + oc
  | .convBn ic oc k _ _     => oc * ic * k * k + 2 * oc
  | .dense fi fo _           => fi * fo + fo
  | .residualBlock ic oc n fs =>
      let needsProj := !(ic == oc && fs == 1)
      let idBlock := 2 * (oc * oc * 3 * 3 + 2 * oc)
      let firstBlock := if needsProj
        then (oc * ic * 3 * 3 + 2 * oc) + (oc * oc * 3 * 3 + 2 * oc) + (oc * ic * 1 * 1 + 2 * oc)
        else idBlock
      firstBlock + (n - 1) * idBlock
  | .bottleneckBlock ic oc n fs =>
      let mid := oc / 4
      let needsProj := !(ic == oc && fs == 1)
      let idBlock := (mid * oc * 1 + 2 * mid) + (mid * mid * 9 + 2 * mid) + (oc * mid * 1 + 2 * oc)
      let firstBlock := if needsProj
        then (mid * ic * 1 + 2 * mid) + (mid * mid * 9 + 2 * mid) + (oc * mid * 1 + 2 * oc) +
             (oc * ic * 1 + 2 * oc)
        else idBlock
      firstBlock + (n - 1) * idBlock
  | .separableConv ic oc _ =>
      (ic * 9 + 2 * ic) + (oc * ic + 2 * oc)
  | .invertedResidual ic oc expand stride n =>
      let mid := ic * expand
      let expandP := if expand == 1 then 0 else (mid * ic + 2 * mid)
      let dwP := mid * 9 + 2 * mid
      let projP := oc * mid + 2 * oc
      let firstBlock := expandP + dwP + projP
      let midR := oc * expand
      let expandR := if expand == 1 then 0 else (midR * oc + 2 * midR)
      let dwR := midR * 9 + 2 * midR
      let projR := oc * midR + 2 * oc
      let restBlock := expandR + dwR + projR
      firstBlock + (n - 1) * restBlock
  | .mbConv ic oc expand k stride n useSE =>
      let mid := ic * expand
      let expandP := if expand == 1 then 0 else (mid * ic + 2 * mid)
      let dwP := mid * k * k + 2 * mid
      let seMid := Nat.max 1 (mid / 4)
      let seP := if useSE then (seMid * mid + seMid) + (mid * seMid + mid) else 0
      let projP := oc * mid + 2 * oc
      let firstBlock := expandP + dwP + seP + projP
      let midR := oc * expand
      let expandR := if expand == 1 then 0 else (midR * oc + 2 * midR)
      let dwR := midR * k * k + 2 * midR
      let seMidR := Nat.max 1 (midR / 4)
      let seR := if useSE then (seMidR * midR + seMidR) + (midR * seMidR + midR) else 0
      let projR := oc * midR + 2 * oc
      let restBlock := expandR + dwR + seR + projR
      firstBlock + (n - 1) * restBlock
  | .mbConvV3 ic oc expandCh k _ useSE _ =>
      let expandP := if expandCh == ic then 0 else (expandCh * ic + 2 * expandCh)
      let dwP := expandCh * k * k + 2 * expandCh
      let seMid := Nat.max 1 (expandCh / 4)
      let seP := if useSE then (seMid * expandCh + seMid) + (expandCh * seMid + expandCh) else 0
      let projP := oc * expandCh + 2 * oc
      expandP + dwP + seP + projP
  | .fusedMbConv ic oc expand k stride n useSE =>
      let mid := if expand == 1 then oc else ic * expand
      let expandP := mid * ic * k * k + 2 * mid
      let seMid := Nat.max 1 (mid / 4)
      let seP := if useSE then (seMid * mid + seMid) + (mid * seMid + mid) else 0
      let projP := if expand == 1 then 0 else (oc * mid + 2 * oc)
      let firstBlock := expandP + seP + projP
      let midR := if expand == 1 then oc else oc * expand
      let expandR := midR * oc * k * k + 2 * midR
      let seMidR := Nat.max 1 (midR / 4)
      let seR := if useSE then (seMidR * midR + seMidR) + (midR * seMidR + midR) else 0
      let projR := if expand == 1 then 0 else (oc * midR + 2 * oc)
      let restBlock := expandR + seR + projR
      firstBlock + (n - 1) * restBlock
  | .uib ic oc expand stride preDWk postDWk =>
      let mid := ic * expand
      let preDW := if preDWk > 0 then ic * preDWk * preDWk + 2 * ic else 0
      let expandP := mid * ic + 2 * mid
      let postDW := if postDWk > 0 then mid * postDWk * postDWk + 2 * mid else 0
      let projP := oc * mid + 2 * oc
      preDW + expandP + postDW + projP
  | .fireModule ic sq e1 e3 =>
      (sq * ic + 2 * sq) + (e1 * sq + 2 * e1) + (e3 * sq * 9 + 2 * e3)
  | .patchEmbed ic dim p nP =>
      dim * ic * p * p + dim + dim + (nP + 1) * dim
  | .transformerEncoder dim _heads mlpDim nBlocks =>
      let perBlock := 2 * dim
                    + 3 * (dim * dim + dim)
                    + (dim * dim + dim)
                    + 2 * dim
                    + (dim * mlpDim + mlpDim)
                    + (mlpDim * dim + dim)
      nBlocks * perBlock + 2 * dim
  | .mambaBlock dim stateSize expand nBlocks =>
      -- Approximate per-block count (Gu & Dao 2023 structure):
      --   in_proj (D → 2·E·D) + out_proj (E·D → D): ~3·E·D²
      --   SSM dt/B/C projections: ~3·E·D·N
      --   depthwise 1D conv (kernel 4) + bias: ~5·E·D
      --   RMSNorm γ: D
      let perBlock := 3 * expand * dim * dim
                    + 3 * expand * dim * stateSize
                    + 5 * expand * dim
                    + dim
      nBlocks * perBlock
  | _                        => 0

def NetSpec.totalParams (s : NetSpec) : Nat :=
  s.layers.foldl (fun acc l => acc + l.nParams) 0

-- Feature queries (used by codegen to gate helper emission)
def NetSpec.hasConv (s : NetSpec) : Bool :=
  s.layers.any fun | .conv2d .. => true | .convBn .. => true | .residualBlock .. => true | .bottleneckBlock .. => true | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | .fusedMbConv .. => true | .uib .. => true | .fireModule .. => true | .patchEmbed .. => true | _ => false

def NetSpec.hasPool (s : NetSpec) : Bool :=
  s.layers.any fun | .maxPool .. => true | _ => false

def NetSpec.hasBn (s : NetSpec) : Bool :=
  s.layers.any fun | .convBn .. => true | .residualBlock .. => true | .bottleneckBlock .. => true | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | .fusedMbConv .. => true | .uib .. => true | .fireModule .. => true | _ => false

def NetSpec.hasResidual (s : NetSpec) : Bool :=
  s.layers.any fun | .residualBlock .. => true | .bottleneckBlock .. => true | _ => false

def NetSpec.hasBottleneck (s : NetSpec) : Bool :=
  s.layers.any fun | .bottleneckBlock .. => true | _ => false

def NetSpec.hasSeparable (s : NetSpec) : Bool :=
  s.layers.any fun | .separableConv .. => true | .invertedResidual .. => true | .mbConv .. => true | .mbConvV3 .. => true | .uib .. => true | _ => false

def NetSpec.hasInvertedResidual (s : NetSpec) : Bool :=
  s.layers.any fun | .invertedResidual .. => true | _ => false

def NetSpec.hasMbConv (s : NetSpec) : Bool :=
  s.layers.any fun | .mbConv .. => true | .mbConvV3 .. => true | .fusedMbConv .. => true | _ => false

def NetSpec.hasGlobalAvgPool (s : NetSpec) : Bool :=
  s.layers.any fun | .globalAvgPool => true | _ => false

def NetSpec.hasTransformer (s : NetSpec) : Bool :=
  s.layers.any fun | .transformerEncoder .. => true | _ => false

def NetSpec.numClasses (s : NetSpec) : Nat :=
  match s.layers.getLast? with
  | some (.dense _ fo _) => fo
  | _                     => 0

def NetSpec.archStr (s : NetSpec) : String :=
  " → ".intercalate (s.layers.map fun l =>
    match l with
    | .conv2d ic oc k _ _       => s!"Conv({ic}→{oc},{k}x{k})"
    | .convBn ic oc k st _      => s!"Conv({ic}→{oc},{k}x{k}/{st})+BN"
    | .maxPool sz _              => s!"Pool({sz}x{sz})"
    | .globalAvgPool             => "GAP"
    | .flatten                   => "Flatten"
    | .dense fi fo act           =>
      let a := match act with | .relu => ",ReLU" | .relu6 => ",ReLU6" | .identity => ""
      s!"{fi}→{fo}{a}"
    | .residualBlock ic oc n fs   => s!"Res{n}({ic}→{oc},s{fs})"
    | .bottleneckBlock ic oc n fs => s!"BN{n}({ic}→{oc},s{fs})"
    | .separableConv ic oc st        => s!"Sep({ic}→{oc},s{st})"
    | .invertedResidual ic oc e s n     => s!"IR{n}({ic}→{oc},e{e},s{s})"
    | .mbConv ic oc e k s n useSE      => s!"MB{n}({ic}→{oc},e{e},k{k},s{s}" ++ (if useSE then ",SE" else "") ++ ")"
    | .mbConvV3 ic oc exp k s useSE hs => s!"V3({ic}→{oc},{exp},k{k},s{s}" ++
        (if useSE then ",SE" else "") ++ (if hs then ",HS" else ",RE") ++ ")"
    | .fusedMbConv ic oc e k s n useSE => s!"FMB{n}({ic}→{oc},e{e},k{k},s{s}" ++ (if useSE then ",SE" else "") ++ ")"
    | .uib ic oc e s pDW poDW => s!"UIB({ic}→{oc},e{e},s{s},dw{pDW}/{poDW})"
    | .fireModule ic sq e1 e3 => s!"Fire({ic}→{e1 + e3},sq{sq})"
    | .patchEmbed ic dim p _     => s!"Patch({ic}→{dim},{p}x{p})"
    | .transformerEncoder dim h _ n => s!"Trans({n}x[{h}h,{dim}])"
    | .mambaBlock dim st exp n   => s!"Mamba{n}(dim={dim},state={st},exp={exp})")

-- ===========================================================================
-- Validation: catch channel/dimension mismatches at `lake build` time
-- ===========================================================================

/-- Output channels of a layer. Returns 0 for structural layers (pool, flatten, GAP). -/
def Layer.outChannels : Layer → Nat
  | .conv2d _ oc _ _ _              => oc
  | .convBn _ oc _ _ _              => oc
  | .dense _ fo _                   => fo
  | .residualBlock _ oc _ _         => oc
  | .bottleneckBlock _ oc _ _       => oc
  | .separableConv _ oc _           => oc
  | .invertedResidual _ oc _ _ _    => oc
  | .mbConv _ oc _ _ _ _ _          => oc
  | .mbConvV3 _ oc _ _ _ _ _        => oc
  | .fusedMbConv _ oc _ _ _ _ _     => oc
  | .uib _ oc _ _ _ _               => oc
  | .fireModule _ _ e1 e3           => e1 + e3
  | .patchEmbed _ dim _ _           => dim
  | .transformerEncoder dim _ _ _   => dim
  | .mambaBlock dim _ _ _           => dim
  | _                               => 0  -- pool/flatten/GAP: pass-through

/-- Input channels expected by a layer. Returns 0 for layers that accept any input. -/
def Layer.inChannels : Layer → Nat
  | .conv2d ic _ _ _ _              => ic
  | .convBn ic _ _ _ _              => ic
  | .dense fi _ _                   => fi
  | .residualBlock ic _ _ _         => ic
  | .bottleneckBlock ic _ _ _       => ic
  | .separableConv ic _ _           => ic
  | .invertedResidual ic _ _ _ _    => ic
  | .mbConv ic _ _ _ _ _ _          => ic
  | .mbConvV3 ic _ _ _ _ _ _        => ic
  | .fusedMbConv ic _ _ _ _ _ _     => ic
  | .uib ic _ _ _ _ _               => ic
  | .fireModule ic _ _ _            => ic
  | .patchEmbed ic _ _ _            => ic
  | .transformerEncoder dim _ _ _   => dim
  | .mambaBlock dim _ _ _           => dim
  | _                               => 0  -- pool/flatten/GAP: accept anything

/-- Validate that channel dimensions chain correctly through the spec.
    Returns `none` if valid, or `some errorMessage` describing the first mismatch. -/
def NetSpec.validate (s : NetSpec) : Option String := Id.run do
  let mut prevOc : Nat := 0
  let mut idx : Nat := 0
  let mut afterGAP := false
  let mut afterFlatten := false
  let mut afterTransformer := false
  for l in s.layers do
    let ic := l.inChannels
    -- Check channel match (skip structural layers and first layer)
    if ic > 0 && prevOc > 0 then
      if afterGAP || afterTransformer then
        if ic != prevOc then
          return some s!"Layer {idx}: expected fanIn={prevOc} after GAP/transformer, got {ic}"
      else if afterFlatten then
        pure ()
      else if ic != prevOc then
        return some s!"Layer {idx}: input channels {ic} ≠ previous output channels {prevOc}"
    -- Update state
    match l with
    | .globalAvgPool => afterGAP := true
    | .flatten       => afterFlatten := true
    | .transformerEncoder .. => afterTransformer := true
    -- Mamba behaves like transformer for shape-chaining: (L, D) in, (L, D) out.
    | .mambaBlock .. => afterTransformer := true
    | _ =>
      let oc := l.outChannels
      if oc > 0 then
        prevOc := oc
        afterGAP := false
        afterFlatten := false
        afterTransformer := false
    idx := idx + 1
  return none

/-- Validate at build time. Use as: `#eval mySpec.validate!` -/
def NetSpec.validate! (s : NetSpec) : IO Unit := do
  match s.validate with
  | none     => pure ()
  | some err => throw (IO.userError s!"{s.name}: {err}")
