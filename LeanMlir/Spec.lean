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
  | .swinStage dim heads mlpDim windowSize nBlocks =>
      -- Per-block (Liu et al. 2021 Swin):
      --   2 LayerNorms (γ+β per): 2·(2·dim) = 4·dim
      --   W-MSA: Q/K/V/O projections: 4·(dim² + dim)
      --   Relative position bias: (2·ws-1)² · heads
      --   MLP: dim·mlpDim + mlpDim + mlpDim·dim + dim
      let biasTerms := (2 * windowSize - 1) * (2 * windowSize - 1) * heads
      let perBlock := 4 * dim
                    + 4 * (dim * dim + dim)
                    + biasTerms
                    + (dim * mlpDim + mlpDim)
                    + (mlpDim * dim + dim)
      nBlocks * perBlock
  | .patchMerging inDim outDim =>
      -- LN on concatenated 4·inDim + linear (4·inDim → outDim) + bias.
      (2 * 4 * inDim) + (4 * inDim * outDim + outDim)
  | .unetDown ic oc =>
      -- 2 × (conv3x3 + BN): ic→oc then oc→oc; maxPool adds zero params.
      (9 * ic * oc + 2 * oc) + (9 * oc * oc + 2 * oc)
  | .unetUp ic oc =>
      -- Transposed-conv 2×2 upsample (ic→oc) + concat(2·oc) + 2 × (conv3x3 + BN)
      -- 2oc→oc, then oc→oc.
      (4 * ic * oc + oc) + (9 * 2 * oc * oc + 2 * oc) + (9 * oc * oc + 2 * oc)
  | .transformerDecoder dim _heads mlpDim nBlocks nQueries =>
      -- Per block: 3 LayerNorms, self-attn Q/K/V/O (4·dim²+4·dim),
      -- cross-attn Q/K/V/O (4·dim²+4·dim), FFN (2·dim·mlpDim + dim + mlpDim).
      let perBlock := 3 * 2 * dim
                    + 2 * (4 * (dim * dim + dim))
                    + (dim * mlpDim + mlpDim) + (mlpDim * dim + dim)
      nBlocks * perBlock + nQueries * dim  -- + object queries embedding
  | .detrHeads dim nClasses =>
      -- Class head: linear dim → (nClasses + 1) [+1 is "no object"]
      -- Box head: 3-layer MLP dim → dim → dim → 4, shared across queries.
      (dim * (nClasses + 1) + (nClasses + 1))
      + (dim * dim + dim) + (dim * dim + dim) + (dim * 4 + 4)
  | .shuffleBlock ic oc groups nUnits =>
      -- Bottleneck width is oc/4; grouped 1×1 convs reduce params by `groups`.
      -- First unit downsamples (stride-2 DWConv, avg-pool skip path).
      let g := Nat.max 1 groups
      let mid := Nat.max 1 (oc / 4)
      -- 1×1 grouped conv ic → mid + BN: (ic*mid)/g + 2*mid
      -- 3×3 depthwise mid → mid + BN: 9*mid + 2*mid
      -- 1×1 grouped conv mid → oc + BN: (mid*oc)/g + 2*oc
      let firstUnit := ((ic * mid) / g + 2 * mid)
                     + (9 * mid + 2 * mid)
                     + ((mid * oc) / g + 2 * oc)
      -- Identity-skip units: ic = oc for the inner block.
      let restUnit  := ((oc * mid) / g + 2 * mid)
                     + (9 * mid + 2 * mid)
                     + ((mid * oc) / g + 2 * oc)
      firstUnit + (nUnits - 1) * restUnit
  | .evoformerBlock msaChannels pairChannels nBlocks =>
      -- Per-block breakdown (approx, matching AlphaFold 2 supplementary):
      --   MSA row-attn w/ pair bias:   ~ 4·cm²          (Q/K/V/O on MSA channels)
      --   MSA col-attn:                ~ 4·cm²
      --   MSA transition (4×):         ~ 8·cm²          (two denses)
      --   Outer product mean → pair:   ~ cm²·cz / 32    (bottleneck to pair)
      --   Triangle multiplicative ×2:  ~ 4·cz²          (outgoing + incoming)
      --   Triangle attention      ×2:  ~ 4·cz²          (starting + ending node)
      --   Pair transition (4×):        ~ 8·cz²
      let cm := msaChannels
      let cz := pairChannels
      let perBlock := 16 * cm * cm
                    + cm * cm * cz / 32
                    + 16 * cz * cz
      nBlocks * perBlock
  | .structureModule singleChannels pairChannels nBlocks =>
      -- Shared-weights recurrent IPA. Per (single) round: IPA attention
      -- (~ 4·cs² + cs·cz/4 for the pair-bias path) + backbone update
      -- (~ cs²) + per-residue χ-angle head (~ cs² for MLPs).
      -- Weights SHARED across `nBlocks` rounds, so params don't multiply.
      let cs := singleChannels
      let cz := pairChannels
      let _ := nBlocks  -- recurrence, not stacking
      6 * cs * cs + cs * cz / 4
  | .mobileVitBlock ic dim _heads mlpDim nTxBlocks =>
      -- Local conv 3×3 + BN: 9·ic² + 2·ic
      -- 1×1 projection in (ic → dim) + BN: ic·dim + 2·dim
      -- Inner transformer (standard): nTxBlocks × ~(12·dim² + 2·dim·mlp)
      -- 1×1 projection out (dim → ic) + BN: dim·ic + 2·ic
      -- Fusion conv 3×3 on concat(ic + ic = 2·ic) → ic + BN: 18·ic² + 2·ic
      let localConv := 9 * ic * ic + 2 * ic
      let projIn    := ic * dim + 2 * dim
      let perTx     := 4 * (dim * dim + dim) + 2 * (dim * mlpDim + mlpDim)
                     + 4 * dim
      let txParams  := nTxBlocks * perTx + 2 * dim
      let projOut   := dim * ic + 2 * ic
      let fusion    := 18 * ic * ic + 2 * ic
      localConv + projIn + txParams + projOut + fusion
  | .convNextStage channels nBlocks =>
      -- Per block:
      --   DWConv 7×7 (depthwise):     49·c + c       — 50·c
      --   LayerNorm (γ, β):           2·c
      --   1×1 expand (c → 4c) + bias: 4·c² + 4·c
      --   1×1 project (4c → c) + bias: 4·c² + c
      --   LayerScale (γ per channel): c
      -- Total per block ≈ 8·c² + 58·c
      let c := channels
      nBlocks * (8 * c * c + 58 * c)
  | .convNextDownsample ic oc =>
      -- LayerNorm on ic channels + 2×2 conv stride-2 (ic → oc) with bias
      2 * ic + 4 * ic * oc + oc
  | .waveNetBlock residualCh skipCh nLayers =>
      -- Per dilated residual block:
      --   Dilated causal conv (kernel 2), res → 2·res (filter + gate):
      --     2·(2·res)·res + 2·res = 4·res² + 2·res
      --   1×1 residual projection (res → res) + bias: res² + res
      --   1×1 skip projection (res → skip) + bias: res·skip + skip
      let res := residualCh
      let skip := skipCh
      let perBlock := 4 * res * res + 2 * res + res * res + res + res * skip + skip
      nLayers * perBlock
  | .positionalEncoding _inputDim _numFrequencies =>
      -- Deterministic sinusoidal basis — zero trainable parameters.
      0
  | .nerfMLP encodedPosDim encodedDirDim hiddenDim =>
      let eP := encodedPosDim
      let eD := encodedDirDim
      let h  := hiddenDim
      let l1 := eP * h + h
      let l2to4 := 3 * (h * h + h)
      let l5 := (h + eP) * h + h
      let l6to8 := 3 * (h * h + h)
      let densHead := h * 1 + 1
      let featProj := h * h + h
      let dirLayer := (h + eD) * 128 + 128
      let rgbHead  := 128 * 3 + 3
      l1 + l2to4 + l5 + l6to8 + densHead + featProj + dirLayer + rgbHead
  | .darknetBlock channels nBlocks =>
      -- Per residual block: 1×1 (c → c/2) + BN + 3×3 (c/2 → c) + BN + residual
      --   = (c · c/2 + 2·c/2) + (9 · c/2 · c + 2·c)
      --   = c²/2 + c + 9·c²/2 + 2·c = 5·c² + 3·c
      let c := channels
      nBlocks * (5 * c * c + 3 * c)
  | .cspBlock ic oc nBlocks =>
      -- Split + process + concat CSP. Approximation:
      --   Input split: 1×1 (ic → ic/2) + BN, twice (one per branch): 2·(ic²/2 + ic)
      --   Bottleneck residual × nBlocks on one branch at ic/2 channels:
      --     each block ≈ 5·(ic/2)² + 3·(ic/2)
      --   Concat + 1×1 out (ic → oc) + BN: ic·oc + 2·oc
      let half := Nat.max 1 (ic / 2)
      let splitPart := 2 * (ic * half + 2 * half)
      let bottleneckPart := nBlocks * (5 * half * half + 3 * half)
      let outPart := ic * oc + 2 * oc
      splitPart + bottleneckPart + outPart
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
    | .mambaBlock dim st exp n   => s!"Mamba{n}(dim={dim},state={st},exp={exp})"
    | .swinStage dim h _ ws n    => s!"Swin{n}(dim={dim},heads={h},win={ws})"
    | .patchMerging i o          => s!"PatchMerge({i}→{o})"
    | .unetDown ic oc            => s!"UNetDown({ic}→{oc})"
    | .unetUp ic oc              => s!"UNetUp({ic}→{oc})"
    | .transformerDecoder dim h _ n nq => s!"Dec{n}x[{h}h,{dim}],{nq}q"
    | .detrHeads dim c           => s!"DETR-heads({dim}→cls{c+1}+box4)"
    | .shuffleBlock ic oc g n    => s!"Shuffle{n}({ic}→{oc},g{g})"
    | .evoformerBlock cm cz n    => s!"Evoformer{n}(msa={cm},pair={cz})"
    | .structureModule cs cz n   => s!"StructMod{n}(s={cs},z={cz})"
    | .mobileVitBlock ic d h m n => s!"MobileViT(ic={ic},d={d},h={h},mlp={m},L={n})"
    | .convNextStage c n         => s!"ConvNeXt{n}(c={c})"
    | .convNextDownsample i o    => s!"CNXDown({i}→{o})"
    | .waveNetBlock r s n        => s!"WaveNet{n}(res={r},skip={s})"
    | .positionalEncoding d L    => s!"PE({d}→{d * 2 * L},L={L})"
    | .nerfMLP eP eD h           => s!"NeRF-MLP(p={eP},d={eD},h={h})"
    | .darknetBlock c n          => s!"Dark{n}(c={c})"
    | .cspBlock i o n            => s!"CSP{n}({i}→{o})")

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
  | .swinStage dim _ _ _ _          => dim
  | .patchMerging _ outDim          => outDim
  | .unetDown _ oc                  => oc
  | .unetUp _ oc                    => oc
  | .transformerDecoder dim _ _ _ _ => dim
  | .detrHeads _ nClasses           => nClasses + 1  -- class-head output width (informational)
  | .shuffleBlock _ oc _ _          => oc
  | .evoformerBlock msaCh _ _       => msaCh  -- MSA channels as the "main" dim
  | .structureModule sCh _ _        => sCh    -- single-repr channels
  | .mobileVitBlock ic _ _ _ _      => ic     -- block is ic → ic
  | .convNextStage c _              => c      -- stage preserves channels
  | .convNextDownsample _ oc        => oc
  | .waveNetBlock _ skipCh _        => skipCh     -- skip-sum is what flows to the head
  | .positionalEncoding inputDim numFreq => inputDim * 2 * numFreq
  | .nerfMLP _ _ _                  => 4    -- 1-dim density + 3-dim RGB (flattened)
  | .darknetBlock c _               => c    -- preserves channels
  | .cspBlock _ oc _                => oc
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
  | .swinStage dim _ _ _ _          => dim
  | .patchMerging inDim _           => inDim
  | .unetDown ic _                  => ic
  | .unetUp ic _                    => ic
  | .transformerDecoder dim _ _ _ _ => dim
  | .detrHeads dim _                => dim
  | .shuffleBlock ic _ _ _          => ic
  | .evoformerBlock msaCh _ _       => msaCh
  | .structureModule sCh _ _        => sCh
  | .mobileVitBlock ic _ _ _ _      => ic
  | .convNextStage c _              => c
  | .convNextDownsample ic _        => ic
  | .waveNetBlock residualCh _ _    => residualCh
  | .positionalEncoding inputDim _  => inputDim
  | .nerfMLP encodedPosDim _ _      => encodedPosDim
  | .darknetBlock c _               => c
  | .cspBlock ic _ _                => ic
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
    -- Swin stage: (H·W, D) in, same out. Patch merging updates prevOc to outDim.
    | .swinStage .. => pure ()  -- dim unchanged; prevOc already = dim
    | .patchMerging _ outDim =>
        prevOc := outDim
        afterGAP := false; afterFlatten := false; afterTransformer := false
    -- Transformer decoder: same dim in / dim out, same afterTransformer treatment.
    | .transformerDecoder .. => afterTransformer := true
    -- Evoformer / Structure Module: abstract "channels" carried through.
    | .evoformerBlock .. => afterTransformer := true
    | .structureModule .. => afterTransformer := true
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
