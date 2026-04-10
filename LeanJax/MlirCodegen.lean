import LeanJax.Types
import LeanJax.Spec
/-! MLIR (StableHLO) code generator: emit MLIR modules from `NetSpec`.

    Supports MLPs (`.dense`) and CNNs (`.conv2d`, `.maxPool`, `.flatten`).
    Walks layers tracking the current activation shape. For CNNs with a flat
    input, emits a reshape to (batch, ic, imageH, imageW) at the head.

    Layout: NCHW tensors, OIHW kernels (matches the JAX codegen convention).
    Params are interleaved as (W0, b0, W1, b1, ...) in the function signature,
    with `idx` advancing for every conv or dense layer. -/

namespace MlirCodegen

/-- Total flat input size: for CNN, ic * imageH * imageW; for MLP, first
    dense layer's fanIn. -/
private def inputFlatDim (spec : NetSpec) : Nat :=
  match spec.layers.head? with
  | some (.dense fanIn _ _) => fanIn
  | some (.conv2d ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.convBn ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.invertedResidual ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.mbConv ic _ _ _ _ _ _) => ic * spec.imageH * spec.imageW
  | _ => spec.imageH * spec.imageW

/-- If the first layer is conv/convBn, returns the NCHW input channels. -/
private def inputChannels (spec : NetSpec) : Option Nat :=
  match spec.layers.head? with
  | some (.conv2d ic _ _ _ _) => some ic
  | some (.convBn ic _ _ _ _) => some ic
  | some (.invertedResidual ic _ _ _ _) => some ic
  | some (.mbConv ic _ _ _ _ _ _) => some ic
  | _ => none

/-- Render tensor type: `tensor<128x1x28x28xf32>`. -/
private def tensorTy (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xf32>"

/-- Sanitize a string for use as an MLIR identifier. -/
private def sanitize (s : String) : String :=
  s.toLower.map (fun c => if c.isAlphanum then c else '_')

/-- Compute SAME padding. Returns (padH0, padH1, padW0, padW1). -/
private def samePad (h w kSize stride : Nat) : Nat × Nat × Nat × Nat :=
  let oH := (h + stride - 1) / stride
  let oW := (w + stride - 1) / stride
  let tH := if (oH - 1) * stride + kSize > h then (oH - 1) * stride + kSize - h else 0
  let tW := if (oW - 1) * stride + kSize > w then (oW - 1) * stride + kSize - w else 0
  (tH / 2, tH - tH / 2, tW / 2, tW - tW / 2)

/-- Standard conv dimension numbers block (NCHW/OIHW). -/
private def convDimNumbers : String :=
  "        dimension_numbers = #stablehlo.conv<raw\n" ++
  "          input_batch_dimension = 0, input_feature_dimension = 1,\n" ++
  "          input_spatial_dimensions = [2, 3],\n" ++
  "          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,\n" ++
  "          kernel_spatial_dimensions = [2, 3],\n" ++
  "          output_batch_dimension = 0, output_feature_dimension = 1,\n" ++
  "          output_spatial_dimensions = [2, 3]>,\n"

/-- Standard conv attribute block (NCHW/OIHW), symmetric padding, stride 1. -/
private def convAttrBlock (pad : Nat) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  "        feature_group_count = 1 : i64,\n" ++
  s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n" ++
  "        rhs_dilation = array<i64: 1, 1>,\n" ++
  "        window_strides = array<i64: 1, 1>\n"

/-- General conv attribute block with explicit asymmetric padding and optional dilations. -/
private def convAttrBlockFull (pH0 pH1 pW0 pW1 : Nat)
    (sH sW : Nat := 1) (lhsH lhsW : Nat := 1) (rhsH rhsW : Nat := 1) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  "        feature_group_count = 1 : i64,\n" ++
  s!"        lhs_dilation = array<i64: {lhsH}, {lhsW}>,\n" ++
  s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n" ++
  s!"        rhs_dilation = array<i64: {rhsH}, {rhsW}>,\n" ++
  s!"        window_strides = array<i64: {sH}, {sW}>\n"

/-- Depthwise conv attribute block: feature_group_count = channels. -/
private def dwConvAttrBlock (pad : Nat) (channels : Nat) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  s!"        feature_group_count = {channels} : i64,\n" ++
  s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n" ++
  "        rhs_dilation = array<i64: 1, 1>,\n" ++
  "        window_strides = array<i64: 1, 1>\n"

/-- General depthwise conv attribute block with asymmetric padding, strides, dilations. -/
private def dwConvAttrBlockFull (pH0 pH1 pW0 pW1 : Nat) (channels : Nat)
    (sH sW : Nat := 1) (lhsH lhsW : Nat := 1) (rhsH rhsW : Nat := 1)
    (batchGroupCount : Nat := 1) : String :=
  s!"        batch_group_count = {batchGroupCount} : i64,\n" ++
  convDimNumbers ++
  s!"        feature_group_count = {channels} : i64,\n" ++
  s!"        lhs_dilation = array<i64: {lhsH}, {lhsW}>,\n" ++
  s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n" ++
  s!"        rhs_dilation = array<i64: {rhsH}, {rhsW}>,\n" ++
  s!"        window_strides = array<i64: {sH}, {sW}>\n"

/-- Emit a conv2d + bias + activation block. Returns (code, newSSA, newShape).
    Assumes NCHW input [b, ic, h, w], produces [b, oc, h, w] for SAME padding,
    stride 1. -/
private def emitConv2d (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize : Nat) (act : Activation) : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let newShape := [b, oc, h, w]
    let pad := (kSize - 1) / 2
    let kShape := tensorTy [oc, ic, kSize, kSize]
    let mut s := ""
    s := s ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ "        dimension_numbers = #stablehlo.conv<raw\n"
    s := s ++ "          input_batch_dimension = 0,\n"
    s := s ++ "          input_feature_dimension = 1,\n"
    s := s ++ "          input_spatial_dimensions = [2, 3],\n"
    s := s ++ "          kernel_output_feature_dimension = 0,\n"
    s := s ++ "          kernel_input_feature_dimension = 1,\n"
    s := s ++ "          kernel_spatial_dimensions = [2, 3],\n"
    s := s ++ "          output_batch_dimension = 0,\n"
    s := s ++ "          output_feature_dimension = 1,\n"
    s := s ++ "          output_spatial_dimensions = [2, 3]\n"
    s := s ++ "        >,\n"
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {kShape}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]" ++
         s!" : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cva{pidx} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy newShape}\n"
    match act with
    | .relu =>
      s := s ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cvr{pidx} = stablehlo.maximum %cva{pidx}, %cvz{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cvr{pidx}", newShape)
    | .identity => return (s, s!"%cva{pidx}", newShape)
    | .relu6 => return (s, s!"%cva{pidx}", newShape)
  | _ => return ("    // conv2d error: expected rank-4 NCHW shape\n", curSSA, curShape)

/-- Emit convBn: conv (possibly strided) + instance norm + optional ReLU.
    Params: W{pidx} (kernel), g{pidx} (gamma/scale), bt{pidx} (beta/shift).
    Instance norm: per-sample, per-channel spatial statistics. -/
private def emitConvBn (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) (relu : Bool := true) (fixedBN : Bool := false)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let newShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy newShape}\n"
    if fixedBN then
      -- Use fixed running mean/var from inputs %bn_mean{pidx}, %bn_var{pidx}
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %bn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %bn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    else
      -- Batch norm: mean/var via 2-step reduction [2,3] then [0]
      let bnN := b * oH * oW
      s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
      s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
      s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    -- Normalize + affine (shared path)
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy newShape}\n"
    if relu then
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cbn_out{pidx}", newShape)
    else
      return (s, preSSA, newShape)
  | _ => return ("    // convBn error\n", curSSA, curShape)

/-- Emit depthwise convBn: depthwise conv + BN + optional ReLU6.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Output channels = input channels.
    If `noAct := true`, emits only convBn (no activation), returning pre-activation
    so the caller can attach its own (e.g., Swish for MBConv). -/
private def emitDepthwiseConvBn (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) (fixedBN : Bool := false) (noAct : Bool := false)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let newShape := [b, channels, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Depthwise conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy newShape}\n"
    if fixedBN then
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %bn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %bn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    else
      let bnN := b * oH * oW
      s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
      s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
      s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    -- Normalize + affine
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy newShape}\n"
    if noAct then
      return (s, preSSA, newShape)
    -- ReLU6
    s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_six{pidx} = stablehlo.constant dense<6.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r{pidx}, %cbn_six{pidx} : {tensorTy newShape}\n"
    return (s, s!"%cbn_out{pidx}", newShape)
  | _ => return ("    // depthwiseConvBn error\n", curSSA, curShape)

/-- Emit an inverted residual block stage for inference.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitInvertedResidual (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand firstStride nBlocks : Nat) (fixedBN : Bool := false)
    : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := blockIc * expand
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Expand: 1×1 convBn + ReLU6 (skip if expand == 1)
    if expand != 1 then
      let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 false (fixedBN := fixedBN)
      code := code ++ s1
      -- Manual ReLU6 after convBn (which was called with relu=false)
      code := code ++ s!"    %ir_r6z{p} = stablehlo.constant dense<0.0> : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6r{p} = stablehlo.maximum {out1}, %ir_r6z{p} : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6s{p} = stablehlo.constant dense<6.0> : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6o{p} = stablehlo.minimum %ir_r6r{p}, %ir_r6s{p} : {tensorTy sh1}\n"
      ssa := s!"%ir_r6o{p}"; shape := sh1; p := p + 1
    -- 2. Depthwise: 3×3 depthwise convBn + ReLU6
    let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid 3 stride (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; shape := sh2; p := p + 1
    -- 3. Project: 1×1 convBn, NO activation
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 4. Skip connection: if stride==1 AND blockIc==oc, add (no ReLU)
    if useSkip then
      code := code ++ s!"    %ir_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%ir_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit an MBConv (EfficientNet) block stage for inference.
    Same structure as InvertedResidual but:
      - uses the layer's `kSize` (not hardcoded 3)
      - uses Swish activation (x * sigmoid(x)) instead of ReLU6
      - squeeze-and-excitation is NOT implemented yet (skipped, with warning).
    Returns (code, newSSA, newShape, newPidx). -/
private def emitMbConv (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand kSize firstStride nBlocks : Nat) (useSE : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  if useSE then
    code := code ++ "    // WARNING: MBConv useSE=true requested but SE is not implemented (phase 2); emitting without SE.\n"
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := blockIc * expand
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Expand: 1×1 convBn + Swish (skip if expand == 1)
    if expand != 1 then
      let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 false (fixedBN := fixedBN)
      code := code ++ s1
      -- Manual Swish (logistic * input) after convBn
      code := code ++ s!"    %mb_sig{p} = stablehlo.logistic {out1} : {tensorTy sh1}\n"
      code := code ++ s!"    %mb_sw{p} = stablehlo.multiply {out1}, %mb_sig{p} : {tensorTy sh1}\n"
      ssa := s!"%mb_sw{p}"; shape := sh1; p := p + 1
    -- 2. Depthwise: k×k depthwise convBn + Swish
    let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid kSize stride
                             (fixedBN := fixedBN) (noAct := true)
    code := code ++ s2
    code := code ++ s!"    %mb_dw_sig{p} = stablehlo.logistic {out2} : {tensorTy sh2}\n"
    code := code ++ s!"    %mb_dw_sw{p} = stablehlo.multiply {out2}, %mb_dw_sig{p} : {tensorTy sh2}\n"
    ssa := s!"%mb_dw_sw{p}"; shape := sh2; p := p + 1
    -- 3. (SE block would go here — phase 2)
    -- 4. Project: 1×1 convBn, NO activation
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 5. Skip connection: if stride==1 AND blockIc==oc, add (no post-add activation)
    if useSkip then
      code := code ++ s!"    %mb_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%mb_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit global average pool: (b, c, h, w) -> (b, c). -/
private def emitGlobalAvgPool (pos : Nat) (curSSA : String) (curShape : List Nat)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let newShape := [b, c]
    let spatialN := h * w
    let mut s := ""
    s := s ++ s!"    %gap_zf{pos} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %gap_sum{pos} = stablehlo.reduce({curSSA} init: %gap_zf{pos}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy curShape}, tensor<f32>) -> {tensorTy newShape}\n"
    s := s ++ s!"    %gap_N{pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %gap{pos} = stablehlo.divide %gap_sum{pos}, %gap_N{pos} : {tensorTy newShape}\n"
    return (s, s!"%gap{pos}", newShape)
  | _ => return ("    // globalAvgPool error\n", curSSA, curShape)

/-- Emit max pool. Handles kSize != stride (e.g., 3/2 in ResNet). -/
private def emitMaxPool (pos : Nat) (curSSA : String) (curShape : List Nat)
    (size stride : Nat) : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let oH := (h + stride - 1) / stride  -- ceil div for SAME
    let oW := (w + stride - 1) / stride
    let newShape := [b, c, oH, oW]
    -- Padding for SAME pooling
    let padNeeded := (oH - 1) * stride + size
    let padH := if padNeeded > h then padNeeded - h else 0
    let padW := if padNeeded > w then padNeeded - w else 0
    let padH0 := padH / 2; let padH1 := padH - padH0
    let padW0 := padW / 2; let padW1 := padW - padW0
    let mut s := ""
    s := s ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
    s := s ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({curSSA}, %pli{pos}) (" ++ "{\n"
    s := s ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
    s := s ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
    s := s ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
    s := s ++ "      }) " ++ "{\n"
    s := s ++ s!"        window_dimensions = array<i64: 1, 1, {size}, {size}>,\n"
    s := s ++ s!"        window_strides    = array<i64: 1, 1, {stride}, {stride}>"
    -- Add padding if needed (for pool kSize 3 stride 2)
    if padH > 0 || padW > 0 then
      s := s ++ s!",\n        padding = dense<[[0, 0], [0, 0], [{padH0}, {padH1}], [{padW0}, {padW1}]]> : tensor<4x2xi64>"
    s := s ++ "\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, tensor<f32>) -> {tensorTy newShape}\n"
    return (s, s!"%pl{pos}", newShape)
  | _ => return ("    // maxPool error: expected rank-4 input\n", curSSA, curShape)

/-- Emit flatten: (b, c, h, w) -> (b, c*h*w). -/
private def emitFlatten (pos : Nat) (curSSA : String) (curShape : List Nat)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let flat := c * h * w
    let newShape := [b, flat]
    let s := s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy newShape}\n"
    return (s, s!"%fl{pos}", newShape)
  | _ => return ("    // flatten: already flat or unknown rank\n", curSSA, curShape)

/-- Emit a residual block stage: nBlocks basic blocks, first may downsample.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitResidualBlock (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc nBlocks firstStride : Nat) (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let needsProj := !(ic == oc && firstStride == 1)
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let blockInShape := shape
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    -- conv1: 3×3, stride, relu
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc oc 3 stride true (fixedBN := fixedBN)
    code := code ++ s1; ssa := out1; shape := sh1; p := p + 1
    -- conv2: 3×3, stride 1, NO relu
    let (s2, out2, _sh2) := emitConvBn p ssa shape oc oc 3 1 false (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; p := p + 1
    -- Skip connection
    let mut skipSSA := blockIn
    if bi == 0 && needsProj then
      -- 1×1 projection convBn (no relu)
      let (sp, outp, _) := emitConvBn p blockIn blockInShape ic oc 1 firstStride false (fixedBN := fixedBN)
      code := code ++ sp; skipSSA := outp; p := p + 1
    -- Add + ReLU
    code := code ++ s!"    %rb_add{startPidx}_{bi} = stablehlo.add {ssa}, {skipSSA} : {tensorTy shape}\n"
    code := code ++ s!"    %rb_rz{startPidx}_{bi} = stablehlo.constant dense<0.0> : {tensorTy shape}\n"
    code := code ++ s!"    %rb_out{startPidx}_{bi} = stablehlo.maximum %rb_add{startPidx}_{bi}, %rb_rz{startPidx}_{bi} : {tensorTy shape}\n"
    ssa := s!"%rb_out{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a bottleneck block stage: nBlocks bottleneck blocks (1x1→3x3→1x1).
    mid = oc / 4 is the bottleneck channels. First block may downsample. -/
private def emitBottleneckBlock (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc nBlocks firstStride : Nat) (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mid := oc / 4
  let needsProj := !(ic == oc && firstStride == 1)
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let blockInShape := shape
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    -- conv1: 1×1, stride 1, relu (reduce channels)
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 true (fixedBN := fixedBN)
    code := code ++ s1; ssa := out1; shape := sh1; p := p + 1
    -- conv2: 3×3, stride, relu
    let (s2, out2, sh2) := emitConvBn p ssa shape mid mid 3 stride true (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; shape := sh2; p := p + 1
    -- conv3: 1×1, stride 1, NO relu (expand channels)
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- Skip connection
    let mut skipSSA := blockIn
    if bi == 0 && needsProj then
      let (sp, outp, _) := emitConvBn p blockIn blockInShape ic oc 1 firstStride false (fixedBN := fixedBN)
      code := code ++ sp; skipSSA := outp; p := p + 1
    -- Add + ReLU
    code := code ++ s!"    %bn_add{startPidx}_{bi} = stablehlo.add {ssa}, {skipSSA} : {tensorTy shape}\n"
    code := code ++ s!"    %bn_rz{startPidx}_{bi} = stablehlo.constant dense<0.0> : {tensorTy shape}\n"
    code := code ++ s!"    %bn_out{startPidx}_{bi} = stablehlo.maximum %bn_add{startPidx}_{bi}, %bn_rz{startPidx}_{bi} : {tensorTy shape}\n"
    ssa := s!"%bn_out{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a dense layer. Assumes (batch, fanIn) input. -/
private def emitDense (pidx : Nat) (curSSA : String) (batchSize fanIn fanOut : Nat)
    (act : Activation) : String × String := Id.run do
  let bTy := tensorTy [batchSize, fanOut]
  let mut s := ""
  s := s ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({tensorTy [batchSize, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]\n"
  s := s ++ s!"           : ({tensorTy [fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %ab{pidx} = stablehlo.add %mm{pidx}, %bb{pidx} : {bTy}\n"
  match act with
  | .relu =>
    s := s ++ s!"    %z{pidx} = stablehlo.constant dense<0.0> : {bTy}\n"
    s := s ++ s!"    %h{pidx} = stablehlo.maximum %ab{pidx}, %z{pidx} : {bTy}\n"
    return (s, s!"%h{pidx}")
  | .identity => return (s, s!"%ab{pidx}")
  | .relu6 => return (s, s!"%ab{pidx}")

/-- Emit the `@forward` function body walking layers in order. -/
private def emitForwardBody (spec : NetSpec) (batchSize : Nat) (fixedBN : Bool := false) : String := Id.run do
  let mut code : String := ""
  let mut curSSA : String := "%x"
  let mut curShape : List Nat := [batchSize, inputFlatDim spec]
  -- If first layer is conv, reshape flat input → NCHW
  match inputChannels spec with
  | some ic =>
    let nchw := [batchSize, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %xr = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%xr"
    curShape := nchw
  | none => pure ()
  let mut pidx : Nat := 0   -- param index (conv + dense)
  let mut pos  : Nat := 0   -- layer position (for unique SSA names on pool/flatten)
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut act =>
      let (snip, newSSA) := emitDense pidx curSSA batchSize fanIn fanOut act
      code := code ++ snip
      curSSA := newSSA
      curShape := [batchSize, fanOut]
      pidx := pidx + 1
    | .conv2d ic oc kSize _ act =>
      let (snip, newSSA, newShape) := emitConv2d pidx curSSA curShape ic oc kSize act
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 1
    | .maxPool size stride =>
      let (snip, newSSA, newShape) := emitMaxPool pos curSSA curShape size stride
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .flatten =>
      let (snip, newSSA, newShape) := emitFlatten pos curSSA curShape
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .convBn ic oc kSize stride _ =>
      let (snip, newSSA, newShape) := emitConvBn pidx curSSA curShape ic oc kSize stride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 1
    | .globalAvgPool =>
      let (snip, newSSA, newShape) := emitGlobalAvgPool pos curSSA curShape
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .residualBlock ic oc nBlocks firstStride =>
      let (snip, newSSA, newShape, newPidx) := emitResidualBlock pidx curSSA curShape ic oc nBlocks firstStride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let (snip, newSSA, newShape, newPidx) := emitBottleneckBlock pidx curSSA curShape ic oc nBlocks firstStride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .invertedResidual ic oc expand stride nBlocks =>
      let (snip, newSSA, newShape, newPidx) := emitInvertedResidual pidx curSSA curShape ic oc expand stride nBlocks (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .mbConv ic oc expand kSize stride nBlocks useSE =>
      let (snip, newSSA, newShape, newPidx) := emitMbConv pidx curSSA curShape ic oc expand kSize stride nBlocks useSE (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | _ =>
      code := code ++ "    // UNSUPPORTED LAYER\n"
    pos := pos + 1
  code := code ++ s!"    return {curSSA} : {tensorTy curShape}\n"
  pure code

/-- Emit function signature: interleaved (W, b) per param-bearing layer. -/
private def emitForwardSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let inDim := inputFlatDim spec
  let mut params : String := s!"%x: {tensorTy [batchSize, inDim]}"
  let mut outShape : List Nat := [batchSize, inDim]
  let mut curShape : List Nat := [batchSize, inDim]
  -- Initial reshape if CNN
  match inputChannels spec with
  | some ic => curShape := [batchSize, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]}"
      curShape := [batchSize, fanOut]
      outShape := curShape
      pidx := pidx + 1
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .convBn ic oc kSize stride _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, blockIc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          -- 1x1 reduce
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 3x3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, mid, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 1x1 expand
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .invertedResidual ic oc expand stride nBlocks =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          -- Expand 1×1 (skip if expand==1)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          -- Depthwise 3×3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- Project 1×1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConv ic oc expand kSize stride nBlocks _useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .maxPool size stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, c, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
      outShape := curShape
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
      outShape := curShape
    | _ => pure ()
  pure s!"func.func @forward(\n    {params}\n  ) -> {tensorTy outShape}"

/-- Generate a StableHLO MLIR module for forward-only inference. -/
def generate (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}\n\n" ++
  s!"module @{sanitize spec.name} " ++ "{\n" ++
  "  " ++ emitForwardSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize ++
  "  }\n" ++
  "}\n"

/-- Collect (pidx, oc) pairs for every convBn layer (including those inside residual/bottleneck blocks). -/
def collectBnLayers (spec : NetSpec) : Array (Nat × Nat) := Id.run do
  let mut result : Array (Nat × Nat) := #[]
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense _ _ _ => pidx := pidx + 1
    | .conv2d _ _ _ _ _ => pidx := pidx + 1
    | .convBn _ oc _ _ _ =>
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        -- conv1
        result := result.push (pidx, oc)
        pidx := pidx + 1
        -- conv2
        result := result.push (pidx, oc)
        pidx := pidx + 1
        if bi == 0 && needsProj then
          -- projection
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        -- 1x1 reduce
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- 3x3
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- 1x1 expand
        result := result.push (pidx, oc)
        pidx := pidx + 1
        if bi == 0 && needsProj then
          -- projection
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        -- Expand 1×1 (skip if expand==1)
        if expand != 1 then
          result := result.push (pidx, mid)
          pidx := pidx + 1
        -- Depthwise 3×3
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- Project 1×1
        result := result.push (pidx, oc)
        pidx := pidx + 1
    | .mbConv ic oc expand _kSize _ nBlocks _useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          result := result.push (pidx, mid)
          pidx := pidx + 1
        result := result.push (pidx, mid)
        pidx := pidx + 1
        result := result.push (pidx, oc)
        pidx := pidx + 1
    | _ => pure ()
  return result

/-- Emit function signature for forward_eval: same as forward but with
    bn_mean/bn_var params appended for each BN layer. -/
private def emitForwardEvalSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let inDim := inputFlatDim spec
  let mut params : String := s!"%x: {tensorTy [batchSize, inDim]}"
  let mut outShape : List Nat := [batchSize, inDim]
  let mut curShape : List Nat := [batchSize, inDim]
  -- Initial reshape if CNN
  match inputChannels spec with
  | some ic => curShape := [batchSize, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]}"
      curShape := [batchSize, fanOut]
      outShape := curShape
      pidx := pidx + 1
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .convBn ic oc kSize stride _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, blockIc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          -- 1x1 reduce
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 3x3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, mid, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 1x1 expand
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .invertedResidual ic oc expand stride nBlocks =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConv ic oc expand kSize stride nBlocks _useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .maxPool _ stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, c, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
      outShape := curShape
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
      outShape := curShape
    | _ => pure ()
  -- Append bn_mean / bn_var params for each BN layer
  let bnLayers := collectBnLayers spec
  for (p, oc) in bnLayers do
    params := params ++ s!",\n    %bn_mean{p}: {tensorTy [oc]}, %bn_var{p}: {tensorTy [oc]}"
  pure s!"func.func @forward_eval(\n    {params}\n  ) -> {tensorTy outShape}"

/-- Generate a StableHLO MLIR module for eval (fixed BN using running mean/var). -/
def generateEval (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} eval — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}, fixedBN: true\n\n" ++
  s!"module @{sanitize spec.name}_eval " ++ "{\n" ++
  "  " ++ emitForwardEvalSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize (fixedBN := true) ++
  "  }\n" ++
  "}\n"

-- ════════════════════════════════════════════════════════════════════
-- Train step generation: forward + softmax-CE loss + VJPs + SGD
-- ════════════════════════════════════════════════════════════════════

/-- Record saved during forward pass for use by backward. -/
private structure FwdRec where
  layer     : Layer
  pidx      : Option Nat   -- param index (conv/dense only)
  pos       : Nat          -- layer position
  inputSSA  : String       -- input to this layer
  preActSSA : String       -- pre-activation (before relu), "" if no activation
  outputSSA : String       -- output of this layer
  inShape   : List Nat     -- input shape
  outShape  : List Nat     -- output shape
  -- convBn intermediates for backward
  convOutSSA : String := ""   -- raw conv output before inst norm
  normSSA    : String := ""   -- (x - mean) * istd
  meanBcSSA  : String := ""   -- broadcast mean (B, OC, oH, oW)
  istdBcSSA  : String := ""   -- broadcast istd (B, OC, oH, oW)
  hasRelu    : Bool := true
  hasRelu6   : Bool := false  -- ReLU6 activation (MobileNet)
  hasSwish   : Bool := false  -- Swish activation (x * sigmoid(x)) (EfficientNet)
  isDepthwise : Bool := false -- depthwise conv (feature_group_count = channels)
  ic         : Nat := 0      -- input channels (for backward kernel shapes)
  kSize      : Nat := 3      -- kernel size
  stride     : Nat := 1      -- conv stride
  -- Residual skip: after this layer's backward, add this gradient SSA
  addSkipGrad : String := ""
instance : Inhabited FwdRec where
  default := { layer := .flatten, pidx := none, pos := 0,
               inputSSA := "", preActSSA := "", outputSSA := "",
               inShape := [], outShape := [] }

/-- Emit convBn forward for train step. Records intermediates for backward.
    Returns (code, FwdRec). -/
private def emitConvBnTrain (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) (relu : Bool) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm: 2-step reduction [2,3] then [0] (IREE can't distribute [0,2,3])
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    let outSSA := if relu then s!"%cbn_out{pidx}" else preSSA
    if relu then
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := outSSA
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := relu
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBn error\n", default)

/-- Emit depthwise convBn forward for train step. Records intermediates for backward.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Activation: ReLU6 by default, or Swish if `useSwish := true`. -/
private def emitDepthwiseConvBnTrain (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) (useSwish : Bool := false) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, channels, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Depthwise conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    if useSwish then
      -- Swish: x * sigmoid(x)
      s := s ++ s!"    %cbn_sig{pidx} = stablehlo.logistic {preSSA} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_sig{pidx} : {tensorTy outShape}\n"
    else
      -- ReLU6
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_r6r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_r6s{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r6r{pidx}, %cbn_r6s{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn channels channels kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false
      hasRelu6 := !useSwish
      hasSwish := useSwish
      isDepthwise := true
      ic := channels, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // depthwiseConvBn error\n", default)

/-- Emit convBn forward for train step with ReLU6 (used for inverted residual expand).
    Same as emitConvBnTrain but with ReLU6 activation instead of ReLU. -/
private def emitConvBnTrainRelu6 (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    -- ReLU6
    s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_r6r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_r6s{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r6r{pidx}, %cbn_r6s{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := true, isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnRelu6 error\n", default)

/-- Emit convBn forward for train step with Swish (x * sigmoid(x)) activation.
    Used by EfficientNet's MBConv. -/
private def emitConvBnTrainSwish (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    -- Swish: x * sigmoid(x)
    s := s ++ s!"    %cbn_sig{pidx} = stablehlo.logistic {preSSA} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_sig{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := false, hasSwish := true, isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnSwish error\n", default)

/-- Emit convBn backward: inst norm VJP + conv backward.
    Returns (code, gradient SSA name, gradient shape). -/
private def emitConvBnBackward (r : FwdRec) (gradSSA : String) : String × String × String := Id.run do
  let p := r.pidx.getD 0
  let oc := r.outShape[1]!
  let b := r.outShape[0]!
  let oH := r.outShape[2]!
  let oW := r.outShape[3]!
  let outTy := tensorTy r.outShape
  let spatialN := oH * oW
  let mut s := ""
  -- ReLU / ReLU6 / Swish backward
  let effGrad := if r.hasRelu then s!"%cbg_relu{p}"
                 else if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else gradSSA
  if r.hasRelu then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_cmp{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_cmp{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasRelu6 then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_gt0{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_lt6{p} = stablehlo.compare LT, {r.preActSSA}, %cbn_r6s{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_mask{p} = stablehlo.and %cbg_gt0{p}, %cbg_lt6{p} : {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_mask{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasSwish then
    -- Swish backward: d/dx[x*σ(x)] = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x)))
    -- Recompute sigmoid from pre-activation.
    s := s ++ s!"    %cbg_sig{p} = stablehlo.logistic {r.preActSSA} : {outTy}\n"
    s := s ++ s!"    %cbg_one{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_1ms{p} = stablehlo.subtract %cbg_one{p}, %cbg_sig{p} : {outTy}\n"
    s := s ++ s!"    %cbg_xt{p} = stablehlo.multiply {r.preActSSA}, %cbg_1ms{p} : {outTy}\n"
    s := s ++ s!"    %cbg_1px{p} = stablehlo.add %cbg_one{p}, %cbg_xt{p} : {outTy}\n"
    s := s ++ s!"    %cbg_dsw{p} = stablehlo.multiply %cbg_sig{p}, %cbg_1px{p} : {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_dsw{p} : {outTy}\n"
  -- d_gamma = reduce_sum(grad * norm, dims=[0,2,3])
  s := s ++ s!"    %cbg_gn{p} = stablehlo.multiply {effGrad}, {r.normSSA} : {outTy}\n"
  s := s ++ s!"    %d_g{p} = stablehlo.reduce(%cbg_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [oc]}\n"
  -- d_beta = reduce_sum(grad, dims=[0,2,3])
  s := s ++ s!"    %d_bt{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [oc]}\n"
  -- d_norm = grad * gamma_broadcast
  s := s ++ s!"    %cbg_dnorm{p} = stablehlo.multiply {effGrad}, %cbn_g_bc{p} : {outTy}\n"
  -- Batch norm backward: d_conv_out
  -- d_xhat = d_norm; xhat = (x - mean) * istd; N = B * oH * oW
  -- d_x = (1/N) * istd * (N * d_xhat - sum(d_xhat, [0,2,3]) - xhat * sum(d_xhat * xhat, [0,2,3]))
  let bnN := b * oH * oW
  let Nf := s!"{bnN}.0"
  -- 2-step reduction: [2,3] then [0]
  s := s ++ s!"    %cbg_sdn_sp{p} = stablehlo.reduce(%cbg_dnorm{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, oc]}\n"
  s := s ++ s!"    %cbg_sdn{p} = stablehlo.reduce(%cbg_sdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
  s := s ++ s!"    %cbg_sdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sdn{p}, dims = [1] : ({tensorTy [oc]}) -> {outTy}\n"
  -- xhat * d_xhat
  s := s ++ s!"    %cbg_xdn{p} = stablehlo.multiply {r.normSSA}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_sxdn_sp{p} = stablehlo.reduce(%cbg_xdn{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, oc]}\n"
  s := s ++ s!"    %cbg_sxdn{p} = stablehlo.reduce(%cbg_sxdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
  s := s ++ s!"    %cbg_sxdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sxdn{p}, dims = [1] : ({tensorTy [oc]}) -> {outTy}\n"
  -- N * d_xhat - sum(d_xhat) - xhat * sum(d_xhat * xhat)
  s := s ++ s!"    %cbg_Nc{p} = stablehlo.constant dense<{Nf}> : {outTy}\n"
  s := s ++ s!"    %cbg_t1{p} = stablehlo.multiply %cbg_Nc{p}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t2{p} = stablehlo.subtract %cbg_t1{p}, %cbg_sdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t3{p} = stablehlo.multiply {r.normSSA}, %cbg_sxdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t4{p} = stablehlo.subtract %cbg_t2{p}, %cbg_t3{p} : {outTy}\n"
  -- d_conv_out = (1/N) * istd * result
  s := s ++ s!"    %cbg_t5{p} = stablehlo.multiply {r.istdBcSSA}, %cbg_t4{p} : {outTy}\n"
  s := s ++ s!"    %cbg_invN{p} = stablehlo.constant dense<{1.0 / bnN.toFloat}> : {outTy}\n"
  s := s ++ s!"    %cbg_dconv{p} = stablehlo.multiply %cbg_invN{p}, %cbg_t5{p} : {outTy}\n"
  -- Conv backward: dW via transpose trick
  let ic := r.ic
  let kSize := r.kSize
  let stride := r.stride
  let h := r.inShape[2]!
  let w := r.inShape[3]!
  let inC := r.inShape[1]!
  if stride == 1 then
    -- Stride-1 case: simple transpose trick
    let pad := (kSize - 1) / 2
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ convAttrBlock pad
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, kSize, kSize]}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
    -- dx via reverse+transpose
    s := s ++ s!"    %cbg_Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %cbg_Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ convAttrBlock pad
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  else
    -- Strided conv backward: dW computation
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    if kSize == 1 then
      -- 1×1 strided: subsample input at stride positions, then conv → 1×1
      s := s ++ s!"    %cbg_xsub{p} = \"stablehlo.slice\"({r.inputSSA}) " ++ "{"
      s := s ++ s!" start_indices = array<i64: 0, 0, 0, 0>,"
      s := s ++ s!" limit_indices = array<i64: {b}, {inC}, {h}, {w}>,"
      s := s ++ s!" strides = array<i64: 1, 1, {stride}, {stride}>"
      s := s ++ "}" ++ s!" : ({tensorTy r.inShape}) -> {tensorTy [b, inC, oH, oW]}\n"
      s := s ++ s!"    %cbg_bt_xs{p} = stablehlo.transpose %cbg_xsub{p}, dims = [1, 0, 2, 3] : ({tensorTy [b, inC, oH, oW]}) -> {tensorTy [inC, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_xs{p}, %cbg_bt_g{p}) " ++ "{\n"
      s := s ++ convAttrBlock 0
      s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, oH, oW]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, 1, 1]}\n"
      s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, 1, 1]}) -> {tensorTy [oc, inC, 1, 1]}\n"
    else
      -- kSize > 1 strided: use rhs_dilation
      s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
      s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
      s := s ++ convAttrBlockFull pH0 pH1 pW0 pW1 1 1 1 1 stride stride
      s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, kSize, kSize]}\n"
      s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, kSize, kSize]}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
    -- dx: conv(dy, W_rev, stride=1, lhs_dilation=stride, padding computed)
    let dxPH0 := kSize - 1 - pH0
    let dxPH1 := kSize - 1 - pH1 + (h + pH0 + pH1 - kSize) % stride
    let dxPW0 := kSize - 1 - pW0
    let dxPW1 := kSize - 1 - pW1 + (w + pW0 + pW1 - kSize) % stride
    s := s ++ s!"    %cbg_Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %cbg_Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ convAttrBlockFull dxPH0 dxPH1 dxPW0 dxPW1 1 1 stride stride
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  return (s, s!"%cbg_dx{p}", s!"%d_W{p}")

/-- Emit depthwise convBn backward: BN VJP + depthwise conv backward.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Returns (code, gradient SSA name, dW SSA name). -/
private def emitDepthwiseConvBnBackward (r : FwdRec) (gradSSA : String) : String × String × String := Id.run do
  let p := r.pidx.getD 0
  let channels := r.outShape[1]!
  let b := r.outShape[0]!
  let oH := r.outShape[2]!
  let oW := r.outShape[3]!
  let outTy := tensorTy r.outShape
  let mut s := ""
  -- ReLU6 / Swish backward
  let effGrad := if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else gradSSA
  if r.hasRelu6 then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_gt0{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_lt6{p} = stablehlo.compare LT, {r.preActSSA}, %cbn_r6s{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_mask{p} = stablehlo.and %cbg_gt0{p}, %cbg_lt6{p} : {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_mask{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasSwish then
    -- Swish backward: see emitConvBnBackward comment.
    s := s ++ s!"    %cbg_sig{p} = stablehlo.logistic {r.preActSSA} : {outTy}\n"
    s := s ++ s!"    %cbg_one{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_1ms{p} = stablehlo.subtract %cbg_one{p}, %cbg_sig{p} : {outTy}\n"
    s := s ++ s!"    %cbg_xt{p} = stablehlo.multiply {r.preActSSA}, %cbg_1ms{p} : {outTy}\n"
    s := s ++ s!"    %cbg_1px{p} = stablehlo.add %cbg_one{p}, %cbg_xt{p} : {outTy}\n"
    s := s ++ s!"    %cbg_dsw{p} = stablehlo.multiply %cbg_sig{p}, %cbg_1px{p} : {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_dsw{p} : {outTy}\n"
  -- d_gamma, d_beta (same as regular convBn)
  s := s ++ s!"    %cbg_gn{p} = stablehlo.multiply {effGrad}, {r.normSSA} : {outTy}\n"
  s := s ++ s!"    %d_g{p} = stablehlo.reduce(%cbg_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %d_bt{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_dnorm{p} = stablehlo.multiply {effGrad}, %cbn_g_bc{p} : {outTy}\n"
  -- BN backward (same formula)
  let bnN := b * oH * oW
  let Nf := s!"{bnN}.0"
  s := s ++ s!"    %cbg_sdn_sp{p} = stablehlo.reduce(%cbg_dnorm{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, channels]}\n"
  s := s ++ s!"    %cbg_sdn{p} = stablehlo.reduce(%cbg_sdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_sdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sdn{p}, dims = [1] : ({tensorTy [channels]}) -> {outTy}\n"
  s := s ++ s!"    %cbg_xdn{p} = stablehlo.multiply {r.normSSA}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_sxdn_sp{p} = stablehlo.reduce(%cbg_xdn{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, channels]}\n"
  s := s ++ s!"    %cbg_sxdn{p} = stablehlo.reduce(%cbg_sxdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_sxdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sxdn{p}, dims = [1] : ({tensorTy [channels]}) -> {outTy}\n"
  s := s ++ s!"    %cbg_Nc{p} = stablehlo.constant dense<{Nf}> : {outTy}\n"
  s := s ++ s!"    %cbg_t1{p} = stablehlo.multiply %cbg_Nc{p}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t2{p} = stablehlo.subtract %cbg_t1{p}, %cbg_sdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t3{p} = stablehlo.multiply {r.normSSA}, %cbg_sxdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t4{p} = stablehlo.subtract %cbg_t2{p}, %cbg_t3{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t5{p} = stablehlo.multiply {r.istdBcSSA}, %cbg_t4{p} : {outTy}\n"
  s := s ++ s!"    %cbg_invN{p} = stablehlo.constant dense<{1.0 / bnN.toFloat}> : {outTy}\n"
  s := s ++ s!"    %cbg_dconv{p} = stablehlo.multiply %cbg_invN{p}, %cbg_t5{p} : {outTy}\n"
  -- Depthwise conv backward
  let kSize := r.kSize
  let stride := r.stride
  let h := r.inShape[2]!
  let w := r.inShape[3]!
  -- dW: transpose input [B,C,H,W]->[C,B,H,W], transpose grad [B,C,oH,oW]->[C,B,oH,oW]
  --     conv with batch_group_count=channels (each channel independently)
  --     For depthwise: dW[c,1,kH,kW] = sum_b input[b,c,:,:] conv grad[b,c,:,:]
  if stride == 1 then
    let pad := (kSize - 1) / 2
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [channels, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [channels, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ s!"        batch_group_count = {channels} : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [channels, b, h, w]}, {tensorTy [channels, b, oH, oW]}) -> {tensorTy [1, channels, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [1, channels, kSize, kSize]}) -> {tensorTy [channels, 1, kSize, kSize]}\n"
    -- dx: reverse kernel, conv with feature_group_count=channels
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %W{p}, dims = [2, 3] : {tensorTy [channels, 1, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  else
    -- Strided depthwise backward
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    -- dW: transpose + conv with batch_group_count=channels, rhs_dilation=stride
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [channels, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [channels, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ s!"        batch_group_count = {channels} : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        lhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ s!"        rhs_dilation = array<i64: {stride}, {stride}>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [channels, b, h, w]}, {tensorTy [channels, b, oH, oW]}) -> {tensorTy [1, channels, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [1, channels, kSize, kSize]}) -> {tensorTy [channels, 1, kSize, kSize]}\n"
    -- dx: reverse kernel, conv with feature_group_count=channels, lhs_dilation=stride
    let dxPH0 := kSize - 1 - pH0
    let dxPH1 := kSize - 1 - pH1 + (h + pH0 + pH1 - kSize) % stride
    let dxPW0 := kSize - 1 - pW0
    let dxPW1 := kSize - 1 - pW1 + (w + pW0 + pW1 - kSize) % stride
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %W{p}, dims = [2, 3] : {tensorTy [channels, 1, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        lhs_dilation = array<i64: {stride}, {stride}>,\n"
    s := s ++ s!"        padding = dense<[[{dxPH0}, {dxPH1}], [{dxPW0}, {dxPW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  return (s, s!"%cbg_dx{p}", s!"%d_W{p}")

/-- Emit Adam update for one param.
    m_new = β1*m + (1-β1)*grad
    v_new = β2*v + (1-β2)*grad²
    m_hat = m_new / bc1    (bc1 = 1 - β1^t, precomputed)
    v_hat = v_new / bc2    (bc2 = 1 - β2^t, precomputed)
    w_new = w - lr * m_hat / (sqrt(v_hat) + ε)
    + optional decoupled weight decay: w_new = w_new - wd*lr*w -/
private def emitAdamUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) : String × String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  -- m_new = β1*m + (1-β1)*grad
  s := s ++ s!"    %b1_{tag} = stablehlo.broadcast_in_dim %beta1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ob1_{tag} = stablehlo.broadcast_in_dim %one_minus_b1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ms_{tag} = stablehlo.multiply %b1_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %mg_{tag} = stablehlo.multiply %ob1_{tag}, {gradSSA} : {ty}\n"
  s := s ++ s!"    %mn_{tag} = stablehlo.add %ms_{tag}, %mg_{tag} : {ty}\n"
  -- v_new = β2*v + (1-β2)*grad²
  s := s ++ s!"    %b2_{tag} = stablehlo.broadcast_in_dim %beta2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ob2_{tag} = stablehlo.broadcast_in_dim %one_minus_b2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %vsc_{tag} = stablehlo.multiply %b2_{tag}, {vSSA} : {ty}\n"
  s := s ++ s!"    %g2_{tag} = stablehlo.multiply {gradSSA}, {gradSSA} : {ty}\n"
  s := s ++ s!"    %vg_{tag} = stablehlo.multiply %ob2_{tag}, %g2_{tag} : {ty}\n"
  s := s ++ s!"    %vn_{tag} = stablehlo.add %vsc_{tag}, %vg_{tag} : {ty}\n"
  -- Bias correction: m_hat = m_new / bc1, v_hat = v_new / bc2
  s := s ++ s!"    %bc1_{tag} = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %bc2_{tag} = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %mh_{tag} = stablehlo.divide %mn_{tag}, %bc1_{tag} : {ty}\n"
  s := s ++ s!"    %vh_{tag} = stablehlo.divide %vn_{tag}, %bc2_{tag} : {ty}\n"
  -- w_new = w - lr * m_hat / (sqrt(v_hat) + ε)
  s := s ++ s!"    %lr_{tag} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %eps_{tag} = stablehlo.broadcast_in_dim %adam_eps, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %sqv_{tag} = stablehlo.sqrt %vh_{tag} : {ty}\n"
  s := s ++ s!"    %den_{tag} = stablehlo.add %sqv_{tag}, %eps_{tag} : {ty}\n"
  s := s ++ s!"    %rat_{tag} = stablehlo.divide %mh_{tag}, %den_{tag} : {ty}\n"
  s := s ++ s!"    %upd_{tag} = stablehlo.multiply %lr_{tag}, %rat_{tag} : {ty}\n"
  s := s ++ s!"    %sub_{tag} = stablehlo.subtract {paramSSA}, %upd_{tag} : {ty}\n"
  if applyWeightDecay then
    -- Decoupled weight decay: w = w - lr*m_hat/(sqrt(v_hat)+ε) - wd*lr*w
    s := s ++ s!"    %wd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wdlr_{tag} = stablehlo.multiply %wd_{tag}, %lr_{tag} : {ty}\n"
    s := s ++ s!"    %wdp_{tag} = stablehlo.multiply %wdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %new_{tag} = stablehlo.subtract %sub_{tag}, %wdp_{tag} : {ty}\n"
  return (s, if applyWeightDecay then s!"%new_{tag}" else s!"%sub_{tag}", s!"%mn_{tag}", s!"%vn_{tag}")

/-- Emit Adam update for a convBn layer (W, gamma, beta). -/
private def emitConvBnAdam (p ic oc kSize : Nat) : String × Array String × Array String := Id.run do
  let wShape := [oc, ic, kSize, kSize]
  let bShape := [oc]
  let mut s := ""
  let (s1, wNew, mwNew, vwNew) := emitAdamUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"W{p}" (applyWeightDecay := true)
  s := s ++ s1
  let (s2, gNew, mgNew, vgNew) := emitAdamUpdate s!"%g{p}" s!"%d_g{p}" s!"%m_g{p}" s!"%v_g{p}" bShape s!"g{p}"
  s := s ++ s2
  let (s3, btNew, mbtNew, vbtNew) := emitAdamUpdate s!"%bt{p}" s!"%d_bt{p}" s!"%m_bt{p}" s!"%v_bt{p}" bShape s!"bt{p}"
  s := s ++ s3
  let retNames := #[wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
  let retTypes := #[tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape]
  return (s, retNames, retTypes)

/-- Emit the full train step (forward + loss + backward + SGD). -/
private def emitTrainStepBody (spec : NetSpec) (batchSize : Nat) (_moduleName : String) : String := Id.run do
  let B := batchSize
  let nClasses := spec.numClasses
  let mut code : String := ""
  code := code ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  code := code ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  code := code ++ "    // Adam constants\n"
  code := code ++ "    %beta1 = stablehlo.constant dense<0.9> : tensor<f32>\n"
  code := code ++ "    %beta2 = stablehlo.constant dense<0.999> : tensor<f32>\n"
  code := code ++ "    %one_minus_b1 = stablehlo.constant dense<0.1> : tensor<f32>\n"
  code := code ++ "    %one_minus_b2 = stablehlo.constant dense<0.001> : tensor<f32>\n"
  code := code ++ "    %adam_eps = stablehlo.constant dense<1.0e-08> : tensor<f32>\n"
  code := code ++ "    %wdecay = stablehlo.constant dense<1.0e-04> : tensor<f32>\n"
  code := code ++ "    // Bias correction: bc1 = 1 - β1^t, bc2 = 1 - β2^t\n"
  code := code ++ "    %one_scalar = stablehlo.constant dense<1.0> : tensor<f32>\n"
  code := code ++ "    %b1t = stablehlo.power %beta1, %t : tensor<f32>\n"
  code := code ++ "    %bc1 = stablehlo.subtract %one_scalar, %b1t : tensor<f32>\n"
  code := code ++ "    %b2t = stablehlo.power %beta2, %t : tensor<f32>\n"
  code := code ++ "    %bc2 = stablehlo.subtract %one_scalar, %b2t : tensor<f32>\n\n"

  -- ═══════════════ FORWARD PASS ═══════════════
  code := code ++ "    // ======================== FORWARD ========================\n"
  let mut curSSA : String := "%x_flat"
  let mut curShape : List Nat := [B, inputFlatDim spec]

  -- Reshape if CNN
  match inputChannels spec with
  | some ic =>
    let nchw := [B, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %x = stablehlo.reshape %x_flat : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%x"
    curShape := nchw
  | none => curSSA := "%x_flat"

  let mut pidx : Nat := 0
  let mut pos : Nat := 0
  let mut records : Array FwdRec := #[]

  for l in spec.layers do
    let inSSA := curSSA
    let inShape := curShape
    match l with
    | .conv2d ic oc kSize _pad act =>
      match curShape with
      | [b, _, h, w] =>
        let outShape := [b, oc, h, w]
        let pad := (kSize - 1) / 2
        code := code ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
        code := code ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
        let preSSA := s!"%cva{pidx}"
        code := code ++ s!"    {preSSA} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy outShape}\n"
        if act == .relu then
          code := code ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
          code := code ++ s!"    %cvr{pidx} = stablehlo.maximum {preSSA}, %cvz{pidx} : {tensorTy outShape}\n"
          curSSA := s!"%cvr{pidx}"
        else
          curSSA := preSSA
        records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
        curShape := outShape
        pidx := pidx + 1
      | _ => pure ()

    | .dense fanIn fanOut act =>
      let outShape := [B, fanOut]
      code := code ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx}, contracting_dims = [1] x [0],\n"
      code := code ++ s!"             precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"           : ({tensorTy [B, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {tensorTy outShape}\n"
      code := code ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [fanOut]}) -> {tensorTy outShape}\n"
      let preSSA := s!"%ab{pidx}"
      code := code ++ s!"    {preSSA} = stablehlo.add %mm{pidx}, %bb{pidx} : {tensorTy outShape}\n"
      if act == .relu then
        code := code ++ s!"    %dz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
        code := code ++ s!"    %dr{pidx} = stablehlo.maximum {preSSA}, %dz{pidx} : {tensorTy outShape}\n"
        curSSA := s!"%dr{pidx}"
      else
        curSSA := preSSA
      records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
      curShape := outShape
      pidx := pidx + 1

    | .maxPool size stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        let outShape := [b, c, oH, oW]
        let padNeeded := (oH - 1) * stride + size
        let padH := if padNeeded > h then padNeeded - h else 0
        let padW := if padNeeded > w then padNeeded - w else 0
        let padH0 := padH / 2; let padH1 := padH - padH0
        let padW0 := padW / 2; let padW1 := padW - padW0
        code := code ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
        -- Explicit pad for SAME pooling (IREE doesn't support padded select_and_scatter)
        let hasPad := padH > 0 || padW > 0
        let paddedShape := [b, c, h + padH, w + padW]
        let mut poolInput := curSSA
        if hasPad then
          code := code ++ s!"    %plpad{pos} = stablehlo.pad {curSSA}, %pli{pos}, low = [0, 0, {padH0}, {padW0}], high = [0, 0, {padH1}, {padW1}], interior = [0, 0, 0, 0] : ({tensorTy curShape}, tensor<f32>) -> {tensorTy paddedShape}\n"
          poolInput := s!"%plpad{pos}"
        let poolInShape := if hasPad then paddedShape else curShape
        code := code ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({poolInput}, %pli{pos}) (" ++ "{\n"
        code := code ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
        code := code ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
        code := code ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
        code := code ++ "      }) " ++ "{" ++ s!"window_dimensions = array<i64: 1, 1, {size}, {size}>, "
        code := code ++ s!"window_strides = array<i64: 1, 1, {stride}, {stride}>" ++ "}\n"
        code := code ++ s!"      : ({tensorTy poolInShape}, tensor<f32>) -> {tensorTy outShape}\n"
        curSSA := s!"%pl{pos}"
        let savedInput := if hasPad then poolInput else inSSA
        let padInfo := s!"{padH0},{padH1},{padW0},{padW1}"
        let poolLayer := Layer.maxPool size stride
        let mut poolRec : FwdRec := default
        poolRec := { poolRec with layer := poolLayer, pidx := none, pos := pos }
        poolRec := { poolRec with inputSSA := savedInput, preActSSA := padInfo }
        poolRec := { poolRec with outputSSA := curSSA }
        poolRec := { poolRec with inShape := inShape, outShape := outShape }
        records := records.push poolRec
        curShape := outShape
      | _ => pure ()

    | .flatten =>
      match curShape with
      | [b, c, h, w] =>
        let flat := c * h * w
        let outShape := [b, flat]
        code := code ++ s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy outShape}\n"
        curSSA := s!"%fl{pos}"
        records := records.push { layer := .flatten, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | .convBn ic oc kSize stride _ =>
      let (snip, rec) := emitConvBnTrain pidx pos curSSA curShape ic oc kSize stride true
      code := code ++ snip
      curSSA := rec.outputSSA
      curShape := rec.outShape
      records := records.push rec
      pidx := pidx + 1

    | .globalAvgPool =>
      match curShape with
      | [b, c, h, w] =>
        let outShape := [b, c]
        let spatialN := h * w
        code := code ++ s!"    %gap_zf{pos} = stablehlo.constant dense<0.0> : tensor<f32>\n"
        code := code ++ s!"    %gap_sum{pos} = stablehlo.reduce({curSSA} init: %gap_zf{pos}) applies stablehlo.add across dimensions = [2, 3]\n"
        code := code ++ s!"          : ({tensorTy curShape}, tensor<f32>) -> {tensorTy outShape}\n"
        code := code ++ s!"    %gap_N{pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy outShape}\n"
        code := code ++ s!"    %gap{pos} = stablehlo.divide %gap_sum{pos}, %gap_N{pos} : {tensorTy outShape}\n"
        curSSA := s!"%gap{pos}"
        records := records.push { layer := .globalAvgPool, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        -- conv1: 3×3, stride, relu
        let (s1, rec1) := emitConvBnTrain pidx pos curSSA curShape blockIc oc 3 stride true
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- conv2: 3×3, stride 1, NO relu
        let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape oc oc 3 1 false
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- Projection (if needed)
        let mut skipSSA := blockIn
        if bi == 0 && needsProj then
          let (sp, recp) := emitConvBnTrain pidx pos blockIn blockInShape ic oc 1 firstStride false
          code := code ++ sp; skipSSA := recp.outputSSA
          records := records.push recp; pidx := pidx + 1
        -- Add + ReLU
        let addId := s!"{pidx}_{bi}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {skipSSA} : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_rz{addId} = stablehlo.constant dense<0.0> : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_out{addId} = stablehlo.maximum %rb_add{addId}, %rb_rz{addId} : {tensorTy curShape}\n"
        curSSA := s!"%rb_out{addId}"
        -- Record the skip-add-relu as a special FwdRec
        -- The first convBn of this block (at records.size - 2 or - 3) gets the skip grad
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Mark the first convBn of this block to accumulate skip grad
        let firstIdx := if bi == 0 && needsProj then records.size - 3 else records.size - 2
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        -- Record the skip-add for backward
        records := records.push {
          layer := .globalAvgPool  -- reuse as marker (no params)
          pidx := none, pos
          inputSSA := blockIn      -- block input (for skip backward)
          preActSSA := s!"%rb_add{addId}"  -- pre-relu sum
          outputSSA := curSSA
          inShape := blockInShape  -- block input shape
          outShape := curShape
          hasRelu := true
          -- Store projection info in addSkipGrad field: "proj:{projPidx}" or "identity"
          addSkipGrad := if bi == 0 && needsProj then s!"proj:{pidx - 1}" else "identity"
        }

    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        -- conv1: 1×1 reduce, relu
        let (s1, rec1) := emitConvBnTrain pidx pos curSSA curShape blockIc mid 1 1 true
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- conv2: 3×3, stride, relu
        let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape mid mid 3 stride true
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- conv3: 1×1 expand, NO relu
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Projection (if needed)
        let mut skipSSA := blockIn
        if bi == 0 && needsProj then
          let (sp, recp) := emitConvBnTrain pidx pos blockIn blockInShape ic oc 1 firstStride false
          code := code ++ sp; skipSSA := recp.outputSSA
          records := records.push recp; pidx := pidx + 1
        -- Add + ReLU
        let addId := s!"bn{pidx}_{bi}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {skipSSA} : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_rz{addId} = stablehlo.constant dense<0.0> : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_out{addId} = stablehlo.maximum %rb_add{addId}, %rb_rz{addId} : {tensorTy curShape}\n"
        curSSA := s!"%rb_out{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Mark the first convBn of this block to accumulate skip grad
        -- bottleneck: 3 convs + optional proj
        let firstIdx := if bi == 0 && needsProj then records.size - 4 else records.size - 3
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        records := records.push {
          layer := .globalAvgPool
          pidx := none, pos
          inputSSA := blockIn
          preActSSA := s!"%rb_add{addId}"
          outputSSA := curSSA
          inShape := blockInShape
          outShape := curShape
          hasRelu := true
          addSkipGrad := if bi == 0 && needsProj then s!"proj:{pidx - 1}" else "identity"
        }

    | .invertedResidual ic oc expand firstStride nBlocks =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let useSkip := stride == 1 && blockIc == oc
        -- Expand: 1×1, ReLU6 (skip if expand==1)
        if expand != 1 then
          let (s1, rec1) := emitConvBnTrainRelu6 pidx pos curSSA curShape blockIc mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1; pidx := pidx + 1
        -- Depthwise: 3×3, ReLU6
        let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid 3 stride
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- Project: 1×1, NO activation
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Skip connection: stride==1 AND blockIc==oc, NO ReLU after add
        if useSkip then
          let addId := s!"ir{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          -- Mark the first convBn of this block to accumulate skip grad
          -- invertedResidual with expand: expand + dw + proj = 3 records
          -- invertedResidual without expand: dw + proj = 2 records
          let nLayersInBlock := if expand != 1 then 3 else 2
          let firstIdx := records.size - nLayersInBlock
          let firstRec := records[firstIdx]!
          records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
          -- Record the skip-add for backward (NO ReLU, hasRelu := false)
          records := records.push {
            layer := .globalAvgPool  -- reuse as marker
            pidx := none, pos
            inputSSA := blockIn
            preActSSA := s!"%rb_add{addId}"  -- the add result (pre-act is same since no relu)
            outputSSA := curSSA
            inShape := blockInShape
            outShape := curShape
            hasRelu := false  -- NO ReLU after skip add in MobileNetV2
            addSkipGrad := "identity"
          }

    | .mbConv ic oc expand kSize firstStride nBlocks useSE =>
      if useSE then
        code := code ++ "    // WARNING: MBConv useSE=true requested but SE is not implemented (phase 2); training without SE.\n"
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let useSkip := stride == 1 && blockIc == oc
        -- Expand: 1×1, Swish (skip if expand==1)
        if expand != 1 then
          let (s1, rec1) := emitConvBnTrainSwish pidx pos curSSA curShape blockIc mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1; pidx := pidx + 1
        -- Depthwise: k×k, Swish
        let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid kSize stride
                            (useSwish := true)
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- Project: 1×1, NO activation
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Skip connection: stride==1 AND blockIc==oc, NO post-add activation
        if useSkip then
          let addId := s!"mb{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          let nLayersInBlock := if expand != 1 then 3 else 2
          let firstIdx := records.size - nLayersInBlock
          let firstRec := records[firstIdx]!
          records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
          records := records.push {
            layer := .globalAvgPool
            pidx := none, pos
            inputSSA := blockIn
            preActSSA := s!"%rb_add{addId}"
            outputSSA := curSSA
            inShape := blockInShape
            outShape := curShape
            hasRelu := false
            addSkipGrad := "identity"
          }

    | _ => code := code ++ "    // UNSUPPORTED\n"
    pos := pos + 1

  let logitsSSA := curSSA
  let NC := nClasses

  -- ═══════════════ SOFTMAX CE LOSS ═══════════════
  code := code ++ "\n    // ================ SOFTMAX CROSS-ENTROPY ================\n"
  code := code ++ s!"    %maxv = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
  code := code ++ s!"          : ({tensorTy [B, NC]}, tensor<f32>) -> {tensorTy [B]}\n"
  code := code ++ s!"    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
  code := code ++ s!"    %shifted = stablehlo.subtract {logitsSSA}, %maxv_b : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %exp_s = stablehlo.exponential %shifted : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]\n"
  code := code ++ s!"           : ({tensorTy [B, NC]}, tensor<f32>) -> {tensorTy [B]}\n"
  code := code ++ s!"    %log_s = stablehlo.log %sum_e : {tensorTy [B]}\n"
  code := code ++ s!"    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
  code := code ++ s!"    %log_p = stablehlo.subtract %shifted, %log_s_b : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %iota = stablehlo.iota dim = 1 : {tensorTy [B, NC]}".replace "xf32>" "xi32>"  ++ "\n"
  code := code ++ s!"    %y_b = stablehlo.broadcast_in_dim %y, dims = [0] : ({tensorTy [B]}".replace "xf32>" "xi32>" ++ s!") -> {tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ "\n"
  let i1Ty := s!"tensor<{B}x{NC}xi1>"
  code := code ++ s!"    %mask = stablehlo.compare EQ, %iota, %y_b : ({tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ s!", {tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ s!") -> {i1Ty}\n"
  let epsilon := 0.1
  let smoothOn := 1.0 - epsilon + epsilon / nClasses.toFloat
  let smoothOff := epsilon / nClasses.toFloat
  code := code ++ s!"    %onef = stablehlo.constant dense<{smoothOn}> : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %zerof = stablehlo.constant dense<{smoothOff}> : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %onehot = stablehlo.select %mask, %onef, %zerof : {i1Ty}, {tensorTy [B, NC]}\n"
  code := code ++ s!"    %weighted = stablehlo.multiply %log_p, %onehot : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
  code := code ++ s!"           : ({tensorTy [B, NC]}, tensor<f32>) -> tensor<f32>\n"
  code := code ++ s!"    %Bc = stablehlo.constant dense<{B}.0> : tensor<f32>\n"
  code := code ++ s!"    %mean = stablehlo.divide %total, %Bc : tensor<f32>\n"
  code := code ++ s!"    %loss = stablehlo.negate %mean : tensor<f32>\n"

  -- ═══════════════ BACKWARD ═══════════════
  code := code ++ "\n    // ==================== BACKWARD ====================\n"
  code := code ++ s!"    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
  code := code ++ s!"    %softmax = stablehlo.divide %exp_s, %sum_e_b : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %sm_moh = stablehlo.subtract %softmax, %onehot : {tensorTy [B, NC]}\n"
  code := code ++ s!"    %Bc_nc = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> {tensorTy [B, NC]}\n"
  code := code ++ s!"    %d_logits = stablehlo.divide %sm_moh, %Bc_nc : {tensorTy [B, NC]}\n"

  let mut gradSSA := "%d_logits"
  let mut gradShape : List Nat := [B, NC]
  let nRec := records.size
  -- Track projection pidx values whose backward was emitted inline during skip-add handling
  let mut bwdDone : Array Nat := #[]

  for ri in [:nRec] do
    let r : FwdRec := records[nRec - 1 - ri]!
    let p := r.pidx.getD 0
    -- Skip projection records whose backward was already emitted inline
    if r.pidx.isSome && bwdDone.contains p then
      pure ()
    else
    match r.layer with
    | .dense _fanIn fanOut act =>
      let effGrad := if act == .relu then s!"%gp{p}" else gradSSA
      if act == .relu then
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        code := code ++ s!"    %rm{p} = stablehlo.compare GT, {r.preActSSA}, %dz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
        code := code ++ s!"    {effGrad} = stablehlo.select %rm{p}, {gradSSA}, %dz{p} : {i1Ty}, {oTy}\n"
      code := code ++ s!"    %d_W{p} = stablehlo.dot_general {r.inputSSA}, {effGrad}, contracting_dims = [0] x [0],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.inShape}, {tensorTy r.outShape}) -> {tensorTy [r.inShape[1]!, fanOut]}\n"
      code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0]\n"
      code := code ++ s!"          : ({tensorTy r.outShape}, tensor<f32>) -> {tensorTy [fanOut]}\n"
      code := code ++ s!"    %d_in{p} = stablehlo.dot_general {effGrad}, %W{p}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.outShape}, {tensorTy [r.inShape[1]!, fanOut]}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%d_in{p}"
      gradShape := r.inShape

    | .conv2d ic oc kSize _pad act =>
      match r.outShape with
      | [b, _oc, h, w] =>
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        let effGrad := if act == .relu then s!"%gpc{p}" else gradSSA
        if act == .relu then
          code := code ++ s!"    %rmc{p} = stablehlo.compare GT, {r.preActSSA}, %cvz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
          code := code ++ s!"    {effGrad} = stablehlo.select %rmc{p}, {gradSSA}, %cvz{p} : {i1Ty}, {oTy}\n"
        let inC := r.inShape[1]!
        code := code ++ s!"    %bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
        code := code ++ s!"    %bt_g{p} = stablehlo.transpose {effGrad}, dims = [1, 0, 2, 3] : ({oTy}) -> {tensorTy [oc, b, h, w]}\n"
        let pad := (kSize - 1) / 2
        let dWrawShape := [inC, oc, kSize, kSize]
        code := code ++ s!"    %dWr{p} = \"stablehlo.convolution\"(%bt_in{p}, %bt_g{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, h, w]}) -> {tensorTy dWrawShape}\n"
        code := code ++ s!"    %d_W{p} = stablehlo.transpose %dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy dWrawShape}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
        code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
        code := code ++ s!"          : ({oTy}, tensor<f32>) -> {tensorTy [oc]}\n"
        code := code ++ s!"    %Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %Wrev{p} = stablehlo.reverse %Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %d_x{p} = \"stablehlo.convolution\"({effGrad}, %Wrev{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({oTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%d_x{p}"
        gradShape := r.inShape
      | _ => pure ()

    | .convBn _ic _oc _kSize _stride _ =>
      if r.isDepthwise then
        let (bwdCode, dxSSA, _dWSSA) := emitDepthwiseConvBnBackward r gradSSA
        code := code ++ bwdCode
        gradSSA := dxSSA
        gradShape := r.inShape
      else
        let (bwdCode, dxSSA, _dWSSA) := emitConvBnBackward r gradSSA
        code := code ++ bwdCode
        gradSSA := dxSSA
        gradShape := r.inShape
      -- Accumulate skip gradient if tagged
      if r.addSkipGrad != "" then
        code := code ++ s!"    %cbg_skip_acc{p} = stablehlo.add {gradSSA}, {r.addSkipGrad} : {tensorTy r.inShape}\n"
        gradSSA := s!"%cbg_skip_acc{p}"

    | .globalAvgPool =>
      -- Check if this is a skip-add record (with or without ReLU)
      if r.addSkipGrad.startsWith "proj:" || r.addSkipGrad == "identity" then
        -- Residual skip-add backward
        let oTy := tensorTy r.outShape
        let addId := r.preActSSA.replace "%rb_add" ""
        if r.hasRelu then
          -- ReLU backward (ResNet-style)
          let i1Ty := oTy.replace "xf32>" "xi1>"
          code := code ++ s!"    %rb_rcmp{addId} = stablehlo.compare GT, {r.preActSSA}, %rb_rz{addId} : ({oTy}, {oTy}) -> {i1Ty}\n"
          code := code ++ s!"    %rb_dsum{addId} = stablehlo.select %rb_rcmp{addId}, {gradSSA}, %rb_rz{addId} : {i1Ty}, {oTy}\n"
          gradSSA := s!"%rb_dsum{addId}"
        else
          -- No ReLU after skip add (MobileNetV2 inverted residual)
          -- Gradient passes through the add directly
          pure ()
        let skipGradSrc := gradSSA  -- gradient flows to both branches
        if r.addSkipGrad.startsWith "proj:" then
          -- Projected skip: emit projection backward inline, mark as done
          let projPidx := (r.addSkipGrad.drop 5).toNat!
          bwdDone := bwdDone.push projPidx
          let mut projRec : FwdRec := default
          for rr in records do
            if rr.pidx == some projPidx then projRec := rr
          let (projBwd, projDx, _) := emitConvBnBackward projRec skipGradSrc
          code := code ++ projBwd
          code := code ++ s!"    %rb_dskip{addId} = stablehlo.reshape {projDx} : ({tensorTy r.inShape}) -> {tensorTy r.inShape}\n"
        else
          -- Identity skip: gradient passes through directly
          code := code ++ s!"    %rb_dskip{addId} = stablehlo.reshape {skipGradSrc} : ({oTy}) -> {oTy}\n"
        gradShape := r.outShape
      else
        -- Normal globalAvgPool backward
        match r.inShape with
        | [_b, _c, h, w] =>
          let spatialN := h * w
          code := code ++ s!"    %dgap_bc{r.pos} = stablehlo.broadcast_in_dim {gradSSA}, dims = [0, 1] : ({tensorTy r.outShape}) -> {tensorTy r.inShape}\n"
          code := code ++ s!"    %dgap_N{r.pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy r.inShape}\n"
          code := code ++ s!"    %dgap{r.pos} = stablehlo.divide %dgap_bc{r.pos}, %dgap_N{r.pos} : {tensorTy r.inShape}\n"
          gradSSA := s!"%dgap{r.pos}"
          gradShape := r.inShape
        | _ => pure ()

    | .maxPool _size stride =>
      -- MaxPool backward via tile-compare-select (avoids select_and_scatter which
      -- IREE doesn't support). Works correctly for stride==size (non-overlapping).
      -- For overlapping pools (size>stride), use stride==size pooling instead.
      match r.inShape with
      | [b, c, h, w] =>
        let oH := r.outShape[2]!
        let oW := r.outShape[3]!
        -- Tile gradient: (B,C,oH,oW) → (B,C,oH,1,oW,1) → (B,C,oH,S,oW,S) → (B,C,oH*S,oW*S)
        let tileShape := [b, c, oH, stride, oW, stride]
        let expandH := oH * stride
        let expandW := oW * stride
        code := code ++ s!"    %mp_gr{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy r.outShape}) -> {tensorTy [b, c, oH, 1, oW, 1]}\n"
        code := code ++ s!"    %mp_gt{r.pos} = stablehlo.broadcast_in_dim %mp_gr{r.pos}, dims = [0, 1, 2, 3, 4, 5] : ({tensorTy [b, c, oH, 1, oW, 1]}) -> {tensorTy tileShape}\n"
        code := code ++ s!"    %mp_ge{r.pos} = stablehlo.reshape %mp_gt{r.pos} : ({tensorTy tileShape}) -> {tensorTy [b, c, expandH, expandW]}\n"
        -- Tile pooled output similarly (for max comparison mask)
        code := code ++ s!"    %mp_pr{r.pos} = stablehlo.reshape {r.outputSSA} : ({tensorTy r.outShape}) -> {tensorTy [b, c, oH, 1, oW, 1]}\n"
        code := code ++ s!"    %mp_pt{r.pos} = stablehlo.broadcast_in_dim %mp_pr{r.pos}, dims = [0, 1, 2, 3, 4, 5] : ({tensorTy [b, c, oH, 1, oW, 1]}) -> {tensorTy tileShape}\n"
        code := code ++ s!"    %mp_pe{r.pos} = stablehlo.reshape %mp_pt{r.pos} : ({tensorTy tileShape}) -> {tensorTy [b, c, expandH, expandW]}\n"
        -- Slice to input size if expanded > input (can happen with non-exact division)
        let mut gradTile := s!"%mp_ge{r.pos}"
        let mut poolTile := s!"%mp_pe{r.pos}"
        if expandH > h || expandW > w then
          let eTy := tensorTy [b, c, expandH, expandW]
          let inTy := tensorTy [b, c, h, w]
          code := code ++ s!"    %mp_gs{r.pos} = \"stablehlo.slice\"({gradTile}) " ++ "{" ++ s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {b}, {c}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({eTy}) -> {inTy}\n"
          code := code ++ s!"    %mp_ps{r.pos} = \"stablehlo.slice\"({poolTile}) " ++ "{" ++ s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {b}, {c}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({eTy}) -> {inTy}\n"
          gradTile := s!"%mp_gs{r.pos}"
          poolTile := s!"%mp_ps{r.pos}"
        -- Compare original input with tiled max → mask where max was
        let inTy := tensorTy [b, c, h, w]
        let maskTy := inTy.replace "xf32>" "xi1>"
        code := code ++ s!"    %mp_mask{r.pos} = stablehlo.compare EQ, {poolTile}, {r.inputSSA} : ({inTy}, {inTy}) -> {maskTy}\n"
        code := code ++ s!"    %mp_zg{r.pos} = stablehlo.constant dense<0.0> : {inTy}\n"
        code := code ++ s!"    %mp_dx{r.pos} = stablehlo.select %mp_mask{r.pos}, {gradTile}, %mp_zg{r.pos} : {maskTy}, {inTy}\n"
        gradSSA := s!"%mp_dx{r.pos}"
        gradShape := r.inShape
      | _ => pure ()

    | .flatten =>
      code := code ++ s!"    %ufl{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy gradShape}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%ufl{r.pos}"
      gradShape := r.inShape

    | _ => pure ()

  -- ═══════════════ ADAM UPDATES ═══════════════
  code := code ++ "\n    // ================ ADAM UPDATES ================\n"
  let mut paramRetNames : Array String := #[]
  let mut paramRetTypes : Array String := #[]
  let mut mRetNames : Array String := #[]
  let mut mRetTypes : Array String := #[]
  let mut vRetNames : Array String := #[]
  let mut vRetTypes : Array String := #[]
  let mut processedPidx : Array Nat := #[]
  for r in records do
    match r.pidx with
    | some p =>
      if processedPidx.contains p then
        pure ()
      else
        processedPidx := processedPidx.push p
        match r.layer with
        | .conv2d ic oc kSize _ _ =>
          let wShape := [oc, ic, kSize, kSize]; let bShape := [oc]
          let (s1, wN, mwN, vwN) := emitAdamUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"cW{p}" (applyWeightDecay := true)
          let (s2, bN, mbN, vbN) := emitAdamUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"cb{p}"
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | .dense fanIn fanOut _ =>
          let wShape := [fanIn, fanOut]; let bShape := [fanOut]
          let (s1, wN, mwN, vwN) := emitAdamUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"dW{p}" (applyWeightDecay := true)
          let (s2, bN, mbN, vbN) := emitAdamUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"db{p}"
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | .convBn ic oc kSize _ _ =>
          -- For depthwise convBn, weight shape is [channels, 1, kSize, kSize]
          let effIc := if r.isDepthwise then 1 else ic
          let (adamCode, adamNames, adamTypes) := emitConvBnAdam p effIc oc kSize
          code := code ++ adamCode
          -- emitConvBnAdam returns [wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
          paramRetNames := paramRetNames ++ adamNames[:3]
          paramRetTypes := paramRetTypes ++ adamTypes[:3]
          mRetNames := mRetNames ++ adamNames[3:6]
          mRetTypes := mRetTypes ++ adamTypes[3:6]
          vRetNames := vRetNames ++ adamNames[6:]
          vRetTypes := vRetTypes ++ adamTypes[6:]
        | _ => pure ()
    | none => pure ()
  -- Return order: params, m, v, loss, then BN stats (mean0, var0, mean1, var1, ...)
  let mut retNames := paramRetNames ++ mRetNames ++ vRetNames |>.push "%loss"
  let mut retTypes := paramRetTypes ++ mRetTypes ++ vRetTypes |>.push "tensor<f32>"

  -- Append BN mean/var for each BN layer (computed during forward)
  let bnLayers := collectBnLayers spec
  for (p, oc) in bnLayers do
    retNames := retNames.push s!"%cbn_mean{p}" |>.push s!"%cbn_var{p}"
    retTypes := retTypes.push (tensorTy [oc]) |>.push (tensorTy [oc])

  code := code ++ s!"    return {String.intercalate ", " retNames.toList}\n"
  code := code ++ s!"      : {String.intercalate ", " retTypes.toList}\n"
  pure code

/-- Emit the train_step function signature. -/
private def emitTrainStepSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let B := batchSize
  let NC := spec.numClasses
  let inDim := inputFlatDim spec
  let mut params : String := ""
  let mut paramRetTypes : Array String := #[]
  let mut mRetTypes : Array String := #[]
  let mut vRetTypes : Array String := #[]
  let mut pidx : Nat := 0
  let mut curShape : List Nat := [B, inDim]
  match inputChannels spec with
  | some ic => curShape := [B, ic, spec.imageH, spec.imageW]
  | none => pure ()
  -- Helper: emit param block for one layer (used for params, m_, and v_)
  let emitLayerParams := fun (pfx : String) (l : Layer) (idx : Nat) (retTypes : Array String) =>
    match l with
    | .conv2d ic oc kSize _ _ =>
      let wTy := tensorTy [oc, ic, kSize, kSize]; let bTy := tensorTy [oc]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}b{idx}: {bTy},\n",
       retTypes.push wTy |>.push bTy)
    | .dense fanIn fanOut _ =>
      let wTy := tensorTy [fanIn, fanOut]; let bTy := tensorTy [fanOut]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}b{idx}: {bTy},\n",
       retTypes.push wTy |>.push bTy)
    | .convBn ic oc kSize _ _ =>
      let wTy := tensorTy [oc, ic, kSize, kSize]; let gTy := tensorTy [oc]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}g{idx}: {gTy}, %{pfx}bt{idx}: {gTy},\n",
       retTypes.push wTy |>.push gTy |>.push gTy)
    | _ => ("", retTypes)
  -- Walk layers to build param/m/v argument lists
  for l in spec.layers do
    match l with
    | .conv2d ic oc kSize _ _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      pidx := pidx + 1
    | .dense fanIn fanOut _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      curShape := [B, fanOut]
      pidx := pidx + 1
    | .convBn ic oc kSize stride _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let wTy1 := tensorTy [oc, blockIc, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy1}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
        paramRetTypes := paramRetTypes.push wTy1 |>.push gTy |>.push gTy
        mRetTypes := mRetTypes.push wTy1 |>.push gTy |>.push gTy
        vRetTypes := vRetTypes.push wTy1 |>.push gTy |>.push gTy
        pidx := pidx + 1
        let wTy2 := tensorTy [oc, oc, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy2}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
        paramRetTypes := paramRetTypes.push wTy2 |>.push gTy |>.push gTy
        mRetTypes := mRetTypes.push wTy2 |>.push gTy |>.push gTy
        vRetTypes := vRetTypes.push wTy2 |>.push gTy |>.push gTy
        pidx := pidx + 1
        if bi == 0 && needsProj then
          let pTy := tensorTy [oc, ic, 1, 1]
          params := params ++ s!"      %W{pidx}: {pTy}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
          paramRetTypes := paramRetTypes.push pTy |>.push gTy |>.push gTy
          mRetTypes := mRetTypes.push pTy |>.push gTy |>.push gTy
          vRetTypes := vRetTypes.push pTy |>.push gTy |>.push gTy
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let wTy1 := tensorTy [mid, blockIc, 1, 1]
        params := params ++ s!"      %W{pidx}: {wTy1}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        let wTy2 := tensorTy [mid, mid, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy2}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        let wTy3 := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {wTy3}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        pidx := pidx + 1
        if bi == 0 && needsProj then
          let pTy := tensorTy [oc, ic, 1, 1]
          params := params ++ s!"      %W{pidx}: {pTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
          paramRetTypes := paramRetTypes.push pTy |>.push gTyO |>.push gTyO
          mRetTypes := mRetTypes.push pTy |>.push gTyO |>.push gTyO
          vRetTypes := vRetTypes.push pTy |>.push gTyO |>.push gTyO
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .invertedResidual ic oc expand stride nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        -- Expand 1×1 (skip if expand==1)
        if expand != 1 then
          let wTy := tensorTy [mid, blockIc, 1, 1]
          params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
          paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
          mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
          vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
          pidx := pidx + 1
        -- Depthwise 3×3
        let dwTy := tensorTy [mid, 1, 3, 3]
        params := params ++ s!"      %W{pidx}: {dwTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push dwTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push dwTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push dwTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        -- Project 1×1
        let pjTy := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
        pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConv ic oc expand kSize stride nBlocks _useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          let wTy := tensorTy [mid, blockIc, 1, 1]
          params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
          paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
          mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
          vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
          pidx := pidx + 1
        let dwTy := tensorTy [mid, 1, kSize, kSize]
        params := params ++ s!"      %W{pidx}: {dwTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push dwTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push dwTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push dwTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        let pjTy := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
        pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .maxPool _size stride =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
    | _ => pure ()
  -- m_ params (1st moment, same shapes)
  let mut mpidx : Nat := 0
  for l in spec.layers do
    let (pStr, _) := emitLayerParams "m_" l mpidx #[]
    if pStr != "" then
      params := params ++ pStr; mpidx := mpidx + 1
    -- Handle residual/bottleneck/invertedResidual blocks manually
    match l with
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, blockIc, 3, 3]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, oc, 3, 3]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
        mpidx := mpidx + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic, 1, 1]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
          mpidx := mpidx + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4; let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, mid, 3, 3]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
          mpidx := mpidx + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, 3, 3]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
    | .mbConv ic oc expand kSize _ nBlocks _useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
    | _ => pure ()
  -- v_ params (2nd moment, same shapes)
  let mut vpidx2 : Nat := 0
  for l in spec.layers do
    let (pStr, _) := emitLayerParams "v_" l vpidx2 #[]
    if pStr != "" then
      params := params ++ pStr; vpidx2 := vpidx2 + 1
    match l with
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, blockIc, 3, 3]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, oc, 3, 3]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
        vpidx2 := vpidx2 + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic, 1, 1]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
          vpidx2 := vpidx2 + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4; let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, mid, 3, 3]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
          vpidx2 := vpidx2 + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, 3, 3]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
    | .mbConv ic oc expand kSize _ nBlocks _useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
    | _ => pure ()
  params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y: tensor<{B}xi32>,\n"
  params := params ++ "      %lr: tensor<f32>, %t: tensor<f32>"
  let mut retTypes := paramRetTypes ++ mRetTypes ++ vRetTypes |>.push "tensor<f32>"
  -- BN stats return types (mean + var per BN layer)
  let bnLayers := collectBnLayers spec
  for (_, oc) in bnLayers do
    retTypes := retTypes.push (tensorTy [oc]) |>.push (tensorTy [oc])
  pure s!"  func.func @main(\n{params}\n    ) -> ({String.intercalate ", " retTypes.toList})"

/-- Generate a full train_step MLIR module with VJPs. -/
def generateTrainStep (spec : NetSpec) (batchSize : Nat) (moduleName : String := "jit_train_step") : String :=
  s!"// {spec.name} train_step — Generated by Lean 4 → MLIR (StableHLO + VJPs)\n" ++
  s!"// Batch size: {batchSize}\n\n" ++
  s!"module @{moduleName} " ++ "{\n" ++
  emitTrainStepSig spec batchSize ++ " {\n" ++
  emitTrainStepBody spec batchSize moduleName ++
  "  }\n" ++
  "}\n"

end MlirCodegen
