import LeanMlir.Types
import LeanMlir.Spec

set_option maxRecDepth 2000

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
def inputFlatDim (spec : NetSpec) : Nat :=
  match spec.layers.head? with
  | some (.dense fanIn _ _) => fanIn
  | some (.conv2d ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.convBn ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.invertedResidual ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.mbConv ic _ _ _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.patchEmbed ic _ _ _) => ic * spec.imageH * spec.imageW
  | _ => spec.imageH * spec.imageW

/-- If the first layer is conv/convBn, returns the NCHW input channels. -/
def inputChannels (spec : NetSpec) : Option Nat :=
  match spec.layers.head? with
  | some (.conv2d ic _ _ _ _) => some ic
  | some (.convBn ic _ _ _ _) => some ic
  | some (.invertedResidual ic _ _ _ _) => some ic
  | some (.mbConv ic _ _ _ _ _ _) => some ic
  | some (.patchEmbed ic _ _ _) => some ic
  | _ => none

/-- Render tensor type: `tensor<128x1x28x28xf32>`. -/
private def tensorTy (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xf32>"

/-- Sanitize a string for use as an MLIR identifier. Public so trainers
    can derive eval-call names from `spec.name` instead of hardcoding
    them (and getting them wrong — see e.g. the EffNetV2 eval-name bug). -/
def sanitize (s : String) : String :=
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
    (useRelu : Bool := false)
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
    if useRelu then
      -- Plain ReLU (used by UIB / MobileNet V4)
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cbn_out{pidx}", newShape)
    -- ReLU6 (default)
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

/-- Emit Squeeze-and-Excitation block.
    Input: x : (B, mid, H, W). Output: x * σ(W_exp · swish(W_red · GAP(x) + b_red) + b_exp).
    Uses dense (dot_general) in place of 1×1 convs on (B, mid) / (B, seMid) tensors.
    Params: reduce at `pRed` (W: [mid, seMid], b: [seMid]),
            expand at `pExp` (W: [seMid, mid], b: [mid]).
    NOTE: weight shapes in MLIR params are stored [seMid, mid, 1, 1] / [mid, seMid, 1, 1]
    (conv style) — we reshape them to dense layout here.
    Returns (code, outSSA, shape) where shape == input shape. -/
private def emitSEBlock (tag : String) (xSSA : String) (xShape : List Nat)
    (mid seMid pRed pExp : Nat) : String × String := Id.run do
  match xShape with
  | [b, _, h, w] =>
    let xTy := tensorTy xShape
    let mut s := ""
    -- 1. Squeeze: GAP over spatial dims → (B, mid)
    s := s ++ s!"    %se_gs{tag} = stablehlo.reduce({xSSA} init: %se_zf{tag}_init) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
    -- 2. Reduce (dense): W_red reshape [seMid, mid, 1, 1] → [seMid, mid], then matmul.
    --    out[b,s] = sum_m g[b,m] * W_red[s,m] + b_red[s]
    s := s ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
    s := s ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag},\n"
    s := s ++ "              contracting_dims = [1] x [1],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
    -- 3. Swish: rb * σ(rb)
    s := s ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
    -- 4. Expand (dense): W_exp reshape [mid, seMid, 1, 1] → [mid, seMid], then matmul.
    s := s ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
    s := s ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag},\n"
    s := s ++ "              contracting_dims = [1] x [1],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
    -- 5. Sigmoid gate
    s := s ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
    -- 6. Excite: broadcast multiply. Reshape (B, mid) → (B, mid, 1, 1) → (B, mid, H, W)
    s := s ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
    s := s ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
    s := s ++ s!"    %se_out{tag} = stablehlo.multiply {xSSA}, %se_sigb{tag} : {xTy}\n"
    return (s, s!"%se_out{tag}")
  | _ => return ("    // SE error: expected rank-4 input\n", xSSA)

/-- Emit an MBConv (EfficientNet) block stage for inference.
    Same structure as InvertedResidual but:
      - uses the layer's `kSize` (not hardcoded 3)
      - uses Swish activation (x * sigmoid(x)) instead of ReLU6
      - optional Squeeze-and-Excitation between depthwise and project.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitMbConv (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand kSize firstStride nBlocks : Nat) (useSE : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := blockIc * expand
    let seMid := Nat.max 1 (mid / 4)
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
    -- 3. SE block (optional)
    if useSE then
      let tag := s!"_i{startPidx}_{bi}"
      code := code ++ s!"    %se_zf{tag}_init = stablehlo.constant dense<0.0> : tensor<f32>\n"
      let (sse, outSE) := emitSEBlock tag ssa shape mid seMid p (p + 1)
      code := code ++ sse
      ssa := outSE
      p := p + 2
    -- 4. Project: 1×1 convBn, NO activation
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 5. Skip connection: if stride==1 AND blockIc==oc, add (no post-add activation)
    if useSkip then
      code := code ++ s!"    %mb_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%mb_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a Fused-MBConv (EfficientNet V2) block stage for inference.
    Unlike regular MBConv, fused-MBConv replaces the 1×1 expand + k×k depthwise pair
    with a single regular k×k convolution.
    Structure per block:
      1. Fused expand: k×k regular convBn (blockIc → mid) + Swish
      2. Optional SE
      3. Project: 1×1 convBn (mid → oc), linear (SKIPPED if expand == 1)
      4. Skip if blockIc == oc && stride == 1
    Returns (code, newSSA, newShape, newPidx). -/
private def emitFusedMbConv (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand kSize firstStride nBlocks : Nat) (useSE : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := if expand == 1 then oc else blockIc * expand
    let seMid := Nat.max 1 (mid / 4)
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Fused expand: regular k×k convBn, Swish activation.
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid kSize stride false (fixedBN := fixedBN)
    code := code ++ s1
    code := code ++ s!"    %fmb_sig{p} = stablehlo.logistic {out1} : {tensorTy sh1}\n"
    code := code ++ s!"    %fmb_sw{p} = stablehlo.multiply {out1}, %fmb_sig{p} : {tensorTy sh1}\n"
    ssa := s!"%fmb_sw{p}"; shape := sh1; p := p + 1
    -- 2. SE block (optional) — same as MBConv (swish reduce, sigmoid gate).
    if useSE then
      let tag := s!"_f{startPidx}_{bi}"
      code := code ++ s!"    %se_zf{tag}_init = stablehlo.constant dense<0.0> : tensor<f32>\n"
      let (sse, outSE) := emitSEBlock tag ssa shape mid seMid p (p + 1)
      code := code ++ sse
      ssa := outSE
      p := p + 2
    -- 3. Project: 1×1 convBn, NO activation — only when expand != 1.
    if expand != 1 then
      let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
      code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 4. Skip connection: if stride==1 AND blockIc==oc, add (no post-add activation)
    if useSkip then
      code := code ++ s!"    %fmb_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%fmb_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit h-swish forward: x * ReLU6(x + 3) / 6
    Returns (code, output_ssa). Caller provides input_ssa and shape. -/
private def emitHSwishForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %hs_three{tag} = stablehlo.constant dense<3.0> : {ty}\n"
  s := s ++ s!"    %hs_six{tag} = stablehlo.constant dense<6.0> : {ty}\n"
  s := s ++ s!"    %hs_zero{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %hs_xp3{tag} = stablehlo.add {inSSA}, %hs_three{tag} : {ty}\n"
  s := s ++ s!"    %hs_clip{tag} = stablehlo.minimum %hs_xp3{tag}, %hs_six{tag} : {ty}\n"
  s := s ++ s!"    %hs_r6{tag} = stablehlo.maximum %hs_clip{tag}, %hs_zero{tag} : {ty}\n"
  s := s ++ s!"    %hs_div{tag} = stablehlo.divide %hs_r6{tag}, %hs_six{tag} : {ty}\n"
  s := s ++ s!"    %hs_out{tag} = stablehlo.multiply {inSSA}, %hs_div{tag} : {ty}\n"
  return (s, s!"%hs_out{tag}")

/-- Emit h-sigmoid forward: ReLU6(x + 3) / 6 -/
private def emitHSigmoidForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %hsg_three{tag} = stablehlo.constant dense<3.0> : {ty}\n"
  s := s ++ s!"    %hsg_six{tag} = stablehlo.constant dense<6.0> : {ty}\n"
  s := s ++ s!"    %hsg_zero{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %hsg_xp3{tag} = stablehlo.add {inSSA}, %hsg_three{tag} : {ty}\n"
  s := s ++ s!"    %hsg_clip{tag} = stablehlo.minimum %hsg_xp3{tag}, %hsg_six{tag} : {ty}\n"
  s := s ++ s!"    %hsg_r6{tag} = stablehlo.maximum %hsg_clip{tag}, %hsg_zero{tag} : {ty}\n"
  s := s ++ s!"    %hsg_out{tag} = stablehlo.divide %hsg_r6{tag}, %hsg_six{tag} : {ty}\n"
  return (s, s!"%hsg_out{tag}")

/-- Emit ReLU forward (plain, not ReLU6) -/
private def emitReluForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %rel_z{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %rel_o{tag} = stablehlo.maximum {inSSA}, %rel_z{tag} : {ty}\n"
  return (s, s!"%rel_o{tag}")

/-- Emit an MBConvV3 (MobileNetV3) block for inference. Single block.
    - `expandCh` is the absolute mid channel count.
    - If `expandCh == ic`, skip the expand 1×1 convBn.
    - Activation: h-swish if `useHSwish`, otherwise plain ReLU.
    - SE: uses ReLU on reduce, h-sigmoid on gate.
    - Skip connection iff `stride == 1 && ic == oc`. -/
private def emitMbConvV3 (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expandCh kSize stride : Nat) (useSE useHSwish : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  let blockIn := ssa
  let mid := expandCh
  let seMid := Nat.max 1 (mid / 4)
  let useSkip := stride == 1 && ic == oc
  -- 1. Expand: 1×1 convBn + activation (skip if expandCh == ic)
  if expandCh != ic then
    let (s1, out1, sh1) := emitConvBn p ssa shape ic mid 1 1 false (fixedBN := fixedBN)
    code := code ++ s1
    if useHSwish then
      let (sa, oa) := emitHSwishForward s!"_v3e{p}" out1 sh1
      code := code ++ sa; ssa := oa
    else
      let (sa, oa) := emitReluForward s!"_v3e{p}" out1 sh1
      code := code ++ sa; ssa := oa
    shape := sh1; p := p + 1
  -- 2. Depthwise: k×k depthwise convBn + activation
  let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid kSize stride
                           (fixedBN := fixedBN) (noAct := true)
  code := code ++ s2
  if useHSwish then
    let (sa, oa) := emitHSwishForward s!"_v3d{p}" out2 sh2
    code := code ++ sa; ssa := oa
  else
    let (sa, oa) := emitReluForward s!"_v3d{p}" out2 sh2
    code := code ++ sa; ssa := oa
  shape := sh2; p := p + 1
  -- 3. SE block (optional) - uses ReLU on reduce, h-sigmoid on gate
  if useSE then
    -- Inline SE for V3 (different activations than emitSEBlock):
    -- Squeeze: GAP over spatial → (B, mid)
    -- Reduce: dense mid→seMid + ReLU
    -- Expand: dense seMid→mid + h-sigmoid
    -- Excite: broadcast multiply
    match shape with
    | [b, _, h, w] =>
      let xTy := tensorTy shape
      let bcTy := tensorTy [b, mid]
      let bsTy := tensorTy [b, seMid]
      let tag := s!"_v3se{p}"
      -- GAP
      code := code ++ s!"    %sez{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      code := code ++ s!"    %ses{tag} = stablehlo.reduce({ssa} init: %sez{tag}) applies stablehlo.add across dimensions = [2, 3]\n"
      code := code ++ s!"          : ({xTy}, tensor<f32>) -> {bcTy}\n"
      code := code ++ s!"    %sen{tag} = stablehlo.constant dense<{(h * w).toFloat}> : {bcTy}\n"
      code := code ++ s!"    %segap{tag} = stablehlo.divide %ses{tag}, %sen{tag} : {bcTy}\n"
      -- Reduce: dense mid → seMid (W stored as [seMid, mid, 1, 1], reshape to [seMid, mid])
      code := code ++ s!"    %sewr{tag} = stablehlo.reshape %W{p} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
      code := code ++ s!"    %sered{tag} = stablehlo.dot_general %segap{tag}, %sewr{tag}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT] : ({bcTy}, {tensorTy [seMid, mid]}) -> {bsTy}\n"
      code := code ++ s!"    %sebr_bc{tag} = stablehlo.broadcast_in_dim %b{p}, dims = [1] : ({tensorTy [seMid]}) -> {bsTy}\n"
      code := code ++ s!"    %sered_b{tag} = stablehlo.add %sered{tag}, %sebr_bc{tag} : {bsTy}\n"
      -- ReLU on reduce
      code := code ++ s!"    %seredz{tag} = stablehlo.constant dense<0.0> : {bsTy}\n"
      code := code ++ s!"    %sered_a{tag} = stablehlo.maximum %sered_b{tag}, %seredz{tag} : {bsTy}\n"
      -- Expand: dense seMid → mid
      let pe := p + 1
      code := code ++ s!"    %sewe{tag} = stablehlo.reshape %W{pe} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
      code := code ++ s!"    %seexp{tag} = stablehlo.dot_general %sered_a{tag}, %sewe{tag}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT] : ({bsTy}, {tensorTy [mid, seMid]}) -> {bcTy}\n"
      code := code ++ s!"    %sebe_bc{tag} = stablehlo.broadcast_in_dim %b{pe}, dims = [1] : ({tensorTy [mid]}) -> {bcTy}\n"
      code := code ++ s!"    %seexp_b{tag} = stablehlo.add %seexp{tag}, %sebe_bc{tag} : {bcTy}\n"
      -- h-sigmoid on expand
      let (sgcode, sgout) := emitHSigmoidForward s!"hsg{tag}" s!"%seexp_b{tag}" [b, mid]
      code := code ++ sgcode
      -- Broadcast multiply with input
      code := code ++ s!"    %sebc{tag} = stablehlo.reshape {sgout} : ({bcTy}) -> {tensorTy [b, mid, 1, 1]}\n"
      code := code ++ s!"    %sebcb{tag} = stablehlo.broadcast_in_dim %sebc{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
      code := code ++ s!"    %seout{tag} = stablehlo.multiply {ssa}, %sebcb{tag} : {xTy}\n"
      ssa := s!"%seout{tag}"
    | _ => pure ()
    p := p + 2
  -- 4. Project: 1×1 convBn, NO activation
  let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
  code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
  -- 5. Skip connection
  if useSkip then
    code := code ++ s!"    %mb3_add{startPidx} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
    ssa := s!"%mb3_add{startPidx}"
  return (code, ssa, shape, p)

/-- Emit a Universal Inverted Bottleneck (UIB) block for inference. Single block.
    MobileNet V4 building block. Structure:
      1. Optional preDW (depthwise convBn + plain ReLU), consumes stride if present
      2. Expand 1×1 convBn + plain ReLU
      3. Optional postDW (depthwise convBn + plain ReLU), uses remaining stride
      4. Project 1×1 convBn, NO activation
      5. Skip if stride == 1 && ic == oc
    All activations are plain ReLU (NOT ReLU6, NOT Swish, NOT h-swish). -/
private def emitUib (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand stride preDWk postDWk : Nat)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  let blockIn := ssa
  let mid := ic * expand
  let useSkip := stride == 1 && ic == oc
  -- 1. Optional preDW: consumes stride, then effective stride becomes 1
  let mut effectiveStride := stride
  if preDWk > 0 then
    let (s1, out1, sh1) := emitDepthwiseConvBn p ssa shape ic preDWk stride
                             (fixedBN := fixedBN) (useRelu := true)
    code := code ++ s1
    ssa := out1; shape := sh1; p := p + 1
    effectiveStride := 1
  -- 2. Expand 1×1 convBn + plain ReLU
  let (s2, out2, sh2) := emitConvBn p ssa shape ic mid 1 1 true (fixedBN := fixedBN)
  code := code ++ s2
  ssa := out2; shape := sh2; p := p + 1
  -- 3. Optional postDW: uses remaining (effective) stride
  if postDWk > 0 then
    let (s3, out3, sh3) := emitDepthwiseConvBn p ssa shape mid postDWk effectiveStride
                             (fixedBN := fixedBN) (useRelu := true)
    code := code ++ s3
    ssa := out3; shape := sh3; p := p + 1
  -- 4. Project 1×1 convBn, NO activation
  let (s4, out4, sh4) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
  code := code ++ s4
  ssa := out4; shape := sh4; p := p + 1
  -- 5. Skip connection
  if useSkip then
    code := code ++ s!"    %uib_add{startPidx} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
    ssa := s!"%uib_add{startPidx}"
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

/-- Emit patch embedding forward. Input (B, ic, H, W) → (B, N+1, dim).
    Three params: W (dim, ic, p, p), b (dim,), cls (dim,), pos (N+1, dim).
    Returns (code, outSSA, outShape). -/
private def emitPatchEmbedForward (tag : String) (curSSA : String) (curShape : List Nat)
    (ic dim pSize nP : Nat) (pWSSA pBSSA pClsSSA pPosSSA : String)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := h / pSize
    let oW := w / pSize
    let nT := nP + 1
    let convOut := [b, dim, oH, oW]
    let patchFlat := [b, dim, nP]
    let patchT := [b, nP, dim]
    let outShape := [b, nT, dim]
    let mut s := ""
    -- Conv stride=pSize, no padding
    s := s ++ s!"    %pe_cv{tag} = \"stablehlo.convolution\"({curSSA}, {pWSSA}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {pSize}, {pSize}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [dim, ic, pSize, pSize]}) -> {tensorTy convOut}\n"
    -- Add bias
    s := s ++ s!"    %pe_bb{tag} = stablehlo.broadcast_in_dim {pBSSA}, dims = [1] : ({tensorTy [dim]}) -> {tensorTy convOut}\n"
    s := s ++ s!"    %pe_cvb{tag} = stablehlo.add %pe_cv{tag}, %pe_bb{tag} : {tensorTy convOut}\n"
    -- Reshape (B, dim, oH, oW) → (B, dim, nP)
    s := s ++ s!"    %pe_rs{tag} = stablehlo.reshape %pe_cvb{tag} : ({tensorTy convOut}) -> {tensorTy patchFlat}\n"
    -- Transpose (B, dim, nP) → (B, nP, dim)
    s := s ++ s!"    %pe_tr{tag} = stablehlo.transpose %pe_rs{tag}, dims = [0, 2, 1] : ({tensorTy patchFlat}) -> {tensorTy patchT}\n"
    -- CLS token: broadcast (dim,) → (B, 1, dim)
    s := s ++ s!"    %pe_cls_bc{tag} = stablehlo.broadcast_in_dim {pClsSSA}, dims = [2] : ({tensorTy [dim]}) -> {tensorTy [b, 1, dim]}\n"
    -- Concat along axis=1
    s := s ++ s!"    %pe_cc{tag} = stablehlo.concatenate %pe_cls_bc{tag}, %pe_tr{tag}, dim = 1 : ({tensorTy [b, 1, dim]}, {tensorTy patchT}) -> {tensorTy outShape}\n"
    -- Positional embedding: broadcast (nT, dim) → (B, nT, dim), add
    s := s ++ s!"    %pe_pos_bc{tag} = stablehlo.broadcast_in_dim {pPosSSA}, dims = [1, 2] : ({tensorTy [nT, dim]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %pe_out{tag} = stablehlo.add %pe_cc{tag}, %pe_pos_bc{tag} : {tensorTy outShape}\n"
    return (s, s!"%pe_out{tag}", outShape)
  | _ => return ("    // patchEmbed error\n", curSSA, curShape)

/-- Emit LayerNorm forward for a (B, N, D) input, normalizing across D.
    Returns (code, outSSA, saved records: normSSA, istdSSA, meanSSA).
    gamma, beta are (D,). -/
private def emitLayerNormForward (tag : String) (xSSA : String) (shape : List Nat)
    (gammaSSA betaSSA : String)
    : String × String × String × String × String := Id.run do
  match shape with
  | [b, n, d] =>
    let ty := tensorTy shape
    let bnTy := tensorTy [b, n]
    let dTy := tensorTy [d]
    let mut s := ""
    -- sum over d
    s := s ++ s!"    %ln_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %ln_sum{tag} = stablehlo.reduce({xSSA} init: %ln_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %ln_Nc{tag} = stablehlo.constant dense<{d.toFloat}> : {bnTy}\n"
    s := s ++ s!"    %ln_mean{tag} = stablehlo.divide %ln_sum{tag}, %ln_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_mean_bc{tag} = stablehlo.broadcast_in_dim %ln_mean{tag}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
    s := s ++ s!"    %ln_diff{tag} = stablehlo.subtract {xSSA}, %ln_mean_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_sq{tag} = stablehlo.multiply %ln_diff{tag}, %ln_diff{tag} : {ty}\n"
    s := s ++ s!"    %ln_vsum{tag} = stablehlo.reduce(%ln_sq{tag} init: %ln_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %ln_var{tag} = stablehlo.divide %ln_vsum{tag}, %ln_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_eps{tag} = stablehlo.constant dense<1.0e-5> : {bnTy}\n"
    s := s ++ s!"    %ln_ve{tag} = stablehlo.add %ln_var{tag}, %ln_eps{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_istd{tag} = stablehlo.rsqrt %ln_ve{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_istd_bc{tag} = stablehlo.broadcast_in_dim %ln_istd{tag}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
    s := s ++ s!"    %ln_norm{tag} = stablehlo.multiply %ln_diff{tag}, %ln_istd_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_g_bc{tag} = stablehlo.broadcast_in_dim {gammaSSA}, dims = [2] : ({dTy}) -> {ty}\n"
    s := s ++ s!"    %ln_gn{tag} = stablehlo.multiply %ln_norm{tag}, %ln_g_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_b_bc{tag} = stablehlo.broadcast_in_dim {betaSSA}, dims = [2] : ({dTy}) -> {ty}\n"
    s := s ++ s!"    %ln_out{tag} = stablehlo.add %ln_gn{tag}, %ln_b_bc{tag} : {ty}\n"
    return (s, s!"%ln_out{tag}", s!"%ln_norm{tag}", s!"%ln_istd{tag}", s!"%ln_mean{tag}")
  | _ => return ("    // layerNorm error\n", xSSA, "", "", "")

/-- Emit GELU forward (tanh form, exact) for a tensor of given shape.
    Returns (code, outSSA, tanhSSA) where tanhSSA is saved for backward. -/
private def emitGeluForward (tag : String) (xSSA : String) (shape : List Nat)
    : String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %ge_c05{tag} = stablehlo.constant dense<0.5> : {ty}\n"
  s := s ++ s!"    %ge_ck{tag} = stablehlo.constant dense<0.044715> : {ty}\n"
  s := s ++ s!"    %ge_csqrt{tag} = stablehlo.constant dense<0.7978845608028654> : {ty}\n"
  s := s ++ s!"    %ge_x2{tag} = stablehlo.multiply {xSSA}, {xSSA} : {ty}\n"
  s := s ++ s!"    %ge_x3{tag} = stablehlo.multiply %ge_x2{tag}, {xSSA} : {ty}\n"
  s := s ++ s!"    %ge_kx3{tag} = stablehlo.multiply %ge_x3{tag}, %ge_ck{tag} : {ty}\n"
  s := s ++ s!"    %ge_xp{tag} = stablehlo.add {xSSA}, %ge_kx3{tag} : {ty}\n"
  s := s ++ s!"    %ge_u{tag} = stablehlo.multiply %ge_xp{tag}, %ge_csqrt{tag} : {ty}\n"
  s := s ++ s!"    %ge_t{tag} = stablehlo.tanh %ge_u{tag} : {ty}\n"
  s := s ++ s!"    %ge_one{tag} = stablehlo.constant dense<1.0> : {ty}\n"
  s := s ++ s!"    %ge_1pt{tag} = stablehlo.add %ge_one{tag}, %ge_t{tag} : {ty}\n"
  s := s ++ s!"    %ge_hx{tag} = stablehlo.multiply {xSSA}, %ge_c05{tag} : {ty}\n"
  s := s ++ s!"    %ge_out{tag} = stablehlo.multiply %ge_hx{tag}, %ge_1pt{tag} : {ty}\n"
  return (s, s!"%ge_out{tag}", s!"%ge_t{tag}")

/-- Emit dense layer for rank-3 (B, N, D) input → (B, N, fanOut).
    W shape (D, fanOut), b shape (fanOut,). -/
private def emitDense3D (tag : String) (xSSA : String) (shape : List Nat)
    (wSSA bSSA : String) (fanOut : Nat) : String × String × List Nat := Id.run do
  match shape with
  | [b, n, d] =>
    let outShape := [b, n, fanOut]
    let ty := tensorTy shape
    let outTy := tensorTy outShape
    let mut s := ""
    s := s ++ s!"    %d3_mm{tag} = stablehlo.dot_general {xSSA}, {wSSA},\n"
    s := s ++ "              contracting_dims = [2] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {tensorTy [d, fanOut]}) -> {outTy}\n"
    s := s ++ s!"    %d3_bb{tag} = stablehlo.broadcast_in_dim {bSSA}, dims = [2] : ({tensorTy [fanOut]}) -> {outTy}\n"
    s := s ++ s!"    %d3_out{tag} = stablehlo.add %d3_mm{tag}, %d3_bb{tag} : {outTy}\n"
    return (s, s!"%d3_out{tag}", outShape)
  | _ => return ("    // dense3D error\n", xSSA, shape)

/-- Emit multi-head self-attention forward.
    Input x (B, N, D). Params: Wq, bq, Wk, bk, Wv, bv, Wo, bo.
    Returns (code, outSSA, saved Q,K,V,softmax,preProj SSAs). -/
private def emitMHSAForward (tag : String) (xSSA : String) (shape : List Nat)
    (heads : Nat) (wqSSA bqSSA wkSSA bkSSA wvSSA bvSSA woSSA boSSA : String)
    : String × String × String × String × String × String × String := Id.run do
  match shape with
  | [b, n, d] =>
    let dh := d / heads
    let ty := tensorTy shape
    let heTy := tensorTy [b, n, heads, dh]
    let hTy := tensorTy [b, heads, n, dh]
    let sTy := tensorTy [b, heads, n, n]
    let mut s := ""
    -- Q, K, V projections (B, N, D) → (B, N, D)
    let (qCode, qSSA, _) := emitDense3D s!"{tag}_q" xSSA shape wqSSA bqSSA d
    s := s ++ qCode
    let (kCode, kSSA, _) := emitDense3D s!"{tag}_k" xSSA shape wkSSA bkSSA d
    s := s ++ kCode
    let (vCode, vSSA, _) := emitDense3D s!"{tag}_v" xSSA shape wvSSA bvSSA d
    s := s ++ vCode
    -- Reshape to (B, N, heads, dh), transpose to (B, heads, N, dh)
    s := s ++ s!"    %mh_qr{tag} = stablehlo.reshape {qSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_q{tag} = stablehlo.transpose %mh_qr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    s := s ++ s!"    %mh_kr{tag} = stablehlo.reshape {kSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_k{tag} = stablehlo.transpose %mh_kr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    s := s ++ s!"    %mh_vr{tag} = stablehlo.reshape {vSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_v{tag} = stablehlo.transpose %mh_vr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    -- Scores = Q @ K^T : (B, heads, N, dh) x (B, heads, N, dh) → (B, heads, N, N)
    s := s ++ s!"    %mh_sc{tag} = stablehlo.dot_general %mh_q{tag}, %mh_k{tag},\n"
    s := s ++ "              batching_dims = [0, 1] x [0, 1],\n"
    s := s ++ "              contracting_dims = [3] x [3],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({hTy}, {hTy}) -> {sTy}\n"
    -- Scale by 1/sqrt(dh)
    let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
    s := s ++ s!"    %mh_scale{tag} = stablehlo.constant dense<{invSqrtDh}> : {sTy}\n"
    s := s ++ s!"    %mh_ss{tag} = stablehlo.multiply %mh_sc{tag}, %mh_scale{tag} : {sTy}\n"
    -- Softmax: max, shift, exp, sum, divide
    let bhnTy := tensorTy [b, heads, n]
    s := s ++ s!"    %mh_neginf{tag} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
    s := s ++ s!"    %mh_max{tag} = stablehlo.reduce(%mh_ss{tag} init: %mh_neginf{tag}) applies stablehlo.maximum across dimensions = [3]\n"
    s := s ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
    s := s ++ s!"    %mh_max_bc{tag} = stablehlo.broadcast_in_dim %mh_max{tag}, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
    s := s ++ s!"    %mh_shift{tag} = stablehlo.subtract %mh_ss{tag}, %mh_max_bc{tag} : {sTy}\n"
    s := s ++ s!"    %mh_exp{tag} = stablehlo.exponential %mh_shift{tag} : {sTy}\n"
    s := s ++ s!"    %mh_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %mh_sumE{tag} = stablehlo.reduce(%mh_exp{tag} init: %mh_zf{tag}) applies stablehlo.add across dimensions = [3]\n"
    s := s ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
    s := s ++ s!"    %mh_sumE_bc{tag} = stablehlo.broadcast_in_dim %mh_sumE{tag}, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
    s := s ++ s!"    %mh_sm{tag} = stablehlo.divide %mh_exp{tag}, %mh_sumE_bc{tag} : {sTy}\n"
    -- out = softmax @ V : (B, heads, N, N) x (B, heads, N, dh) → (B, heads, N, dh)
    s := s ++ s!"    %mh_av{tag} = stablehlo.dot_general %mh_sm{tag}, %mh_v{tag},\n"
    s := s ++ "              batching_dims = [0, 1] x [0, 1],\n"
    s := s ++ "              contracting_dims = [3] x [2],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
    -- Transpose back (B, heads, N, dh) → (B, N, heads, dh), reshape → (B, N, D)
    s := s ++ s!"    %mh_avT{tag} = stablehlo.transpose %mh_av{tag}, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
    s := s ++ s!"    %mh_pp{tag} = stablehlo.reshape %mh_avT{tag} : ({heTy}) -> {ty}\n"
    -- Output projection
    let (oCode, oSSA, _) := emitDense3D s!"{tag}_o" s!"%mh_pp{tag}" shape woSSA boSSA d
    s := s ++ oCode
    return (s, oSSA, s!"%mh_q{tag}", s!"%mh_k{tag}", s!"%mh_v{tag}", s!"%mh_sm{tag}", s!"%mh_pp{tag}")
  | _ => return ("    // mhsa error\n", xSSA, "", "", "", "", "")

/-- Emit a transformer encoder block forward pass.
    Layout: LN1 → MHSA → + → LN2 → fc1 → GELU → fc2 → +
    Uses pidx starting at `startP` and 8 param pairs per block:
    LN1(g,b), Wq,bq, Wk,bk, Wv,bv, Wo,bo, LN2(g,b), Wfc1,bfc1, Wfc2,bfc2.
    Returns (code, outSSA, newPidx). -/
private def emitTransformerBlockForward (tag : String) (startP : Nat) (xSSA : String) (shape : List Nat)
    (heads mlpDim : Nat) : String × String × Nat := Id.run do
  match shape with
  | [_b, _n, d] =>
    let ty := tensorTy shape
    let mut s := ""
    let mut p := startP
    -- LN1
    let g1 := s!"%W{p}"; let b1 := s!"%b{p}"; p := p + 1
    let (ln1Code, ln1Out, _, _, _) := emitLayerNormForward s!"{tag}_ln1" xSSA shape g1 b1
    s := s ++ ln1Code
    -- MHSA
    let wq := s!"%W{p}"; let bq := s!"%b{p}"; p := p + 1
    let wk := s!"%W{p}"; let bk := s!"%b{p}"; p := p + 1
    let wv := s!"%W{p}"; let bv := s!"%b{p}"; p := p + 1
    let wo := s!"%W{p}"; let bo := s!"%b{p}"; p := p + 1
    let (mhsaCode, mhsaOut, _, _, _, _, _) := emitMHSAForward s!"{tag}_mh" ln1Out shape heads wq bq wk bk wv bv wo bo
    s := s ++ mhsaCode
    -- Residual
    s := s ++ s!"    %tb_r1{tag} = stablehlo.add {xSSA}, {mhsaOut} : {ty}\n"
    let r1 := s!"%tb_r1{tag}"
    -- LN2
    let g2 := s!"%W{p}"; let b2 := s!"%b{p}"; p := p + 1
    let (ln2Code, ln2Out, _, _, _) := emitLayerNormForward s!"{tag}_ln2" r1 shape g2 b2
    s := s ++ ln2Code
    -- fc1 (D → mlpDim)
    let wf1 := s!"%W{p}"; let bf1 := s!"%b{p}"; p := p + 1
    let (fc1Code, fc1Out, fc1Shape) := emitDense3D s!"{tag}_fc1" ln2Out shape wf1 bf1 mlpDim
    s := s ++ fc1Code
    -- GELU
    let (geCode, geOut, _) := emitGeluForward s!"{tag}_ge" fc1Out fc1Shape
    s := s ++ geCode
    -- fc2 (mlpDim → D)
    let wf2 := s!"%W{p}"; let bf2 := s!"%b{p}"; p := p + 1
    let (fc2Code, fc2Out, _) := emitDense3D s!"{tag}_fc2" geOut fc1Shape wf2 bf2 d
    s := s ++ fc2Code
    -- Residual
    s := s ++ s!"    %tb_r2{tag} = stablehlo.add {r1}, {fc2Out} : {ty}\n"
    return (s, s!"%tb_r2{tag}", p)
  | _ => return ("    // transformerBlock error\n", xSSA, startP)

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
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      let (snip, newSSA, newShape, newPidx) := emitFusedMbConv pidx curSSA curShape ic oc expand kSize stride nBlocks useSE (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .mbConvV3 ic oc expandCh kSize stride useSE useHSwish =>
      let (snip, newSSA, newShape, newPidx) := emitMbConvV3 pidx curSSA curShape ic oc expandCh kSize stride useSE useHSwish (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .uib ic oc expand stride preDWk postDWk =>
      let (snip, newSSA, newShape, newPidx) := emitUib pidx curSSA curShape ic oc expand stride preDWk postDWk (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .patchEmbed ic dim pSize nP =>
      let pW := s!"%W{pidx}"
      let pB := s!"%b{pidx}"; pidx := pidx + 1
      let pCls := s!"%W{pidx}"; pidx := pidx + 1
      let pPos := s!"%W{pidx}"; pidx := pidx + 1
      let (snip, newSSA, newShape) := emitPatchEmbedForward s!"{pos}" curSSA curShape ic dim pSize nP pW pB pCls pPos
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .transformerEncoder dim heads mlpDim nBlocks =>
      let mut cs := curSSA
      let mut p := pidx
      for bi in [:nBlocks] do
        let (snip, newSSA, newP) := emitTransformerBlockForward s!"{pos}_{bi}" p cs curShape heads mlpDim
        code := code ++ snip
        cs := newSSA
        p := newP
      -- Final LN
      let g := s!"%W{p}"; let bS := s!"%b{p}"; p := p + 1
      let (lnCode, lnOut, _, _, _) := emitLayerNormForward s!"{pos}_finalln" cs curShape g bS
      code := code ++ lnCode
      cs := lnOut
      -- Slice CLS token: x[:, 0, :] → (B, D)
      match curShape with
      | [b, _, _] =>
        let clsShape := [b, 1, dim]
        let outShape := [b, dim]
        code := code ++ s!"    %te_cls{pos} = \"stablehlo.slice\"({cs}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({tensorTy curShape}) -> {tensorTy clsShape}\n"
        code := code ++ s!"    %te_out{pos} = stablehlo.reshape %te_cls{pos} : ({tensorTy clsShape}) -> {tensorTy outShape}\n"
        curSSA := s!"%te_out{pos}"
        curShape := outShape
      | _ => pure ()
      let _ := dim
      pidx := p
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
    | .mbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            -- SE reduce (conv2d-style: W + b)
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            -- SE expand
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := if expand == 1 then oc else blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          -- Fused expand: regular k×k conv
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConvV3 ic oc expandCh kSize stride useSE _useHSwish =>
      match curShape with
      | [b, _, h, w] =>
        let mid := expandCh
        let seMid := Nat.max 1 (mid / 4)
        if expandCh != ic then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if useSE then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .uib ic oc expand stride preDWk postDWk =>
      match curShape with
      | [b, _, h, w] =>
        let mid := ic * expand
        -- preDW (if preDWk > 0)
        if preDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %g{pidx}: {tensorTy [ic]}, %bt{pidx}: {tensorTy [ic]}"
          pidx := pidx + 1
        -- Expand 1×1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        -- postDW (if postDWk > 0)
        if postDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        -- Project 1×1
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
    | .patchEmbed ic dim pSize nP =>
      -- 3 parameter slots: W (dim, ic, p, p), b (dim,) with name %b{pidx},
      -- then cls as %W{pidx+1} (dim,), and pos as %W{pidx+2} ((nP+1), dim)
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, ic, pSize, pSize]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [nP + 1, dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
      outShape := curShape
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        -- LN1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wq, bq
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wk, bk
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wv, bv
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wo, bo
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- LN2
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wfc1 (dim, mlpDim), bfc1 (mlpDim,)
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, mlpDim]}, %b{pidx}: {tensorTy [mlpDim]}"
        pidx := pidx + 1
        -- Wfc2 (mlpDim, dim), bfc2 (dim,)
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mlpDim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
      -- Final LN
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, dim]
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
    | .mbConv ic oc expand _kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          result := result.push (pidx, mid)
          pidx := pidx + 1
        result := result.push (pidx, mid)
        pidx := pidx + 1
        if useSE then
          -- SE params (reduce, expand): 2 conv2d-style params, no BN.
          pidx := pidx + 2
        result := result.push (pidx, oc)
        pidx := pidx + 1
    | .fusedMbConv ic oc expand _kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        -- Fused expand convBn
        result := result.push (pidx, mid)
        pidx := pidx + 1
        if useSE then
          pidx := pidx + 2
        if expand != 1 then
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .mbConvV3 ic oc expandCh _kSize _ useSE _ =>
      let mid := expandCh
      if expandCh != ic then
        result := result.push (pidx, mid)
        pidx := pidx + 1
      result := result.push (pidx, mid)
      pidx := pidx + 1
      if useSE then
        pidx := pidx + 2
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      -- preDW BN (only if preDWk > 0): oc = ic
      if preDWk > 0 then
        result := result.push (pidx, ic)
        pidx := pidx + 1
      -- Expand BN: oc = mid
      result := result.push (pidx, mid)
      pidx := pidx + 1
      -- postDW BN (only if postDWk > 0): oc = mid
      if postDWk > 0 then
        result := result.push (pidx, mid)
        pidx := pidx + 1
      -- Project BN: oc = oc
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
    | .mbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := if expand == 1 then oc else blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          -- Fused expand: regular k×k conv
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConvV3 ic oc expandCh kSize stride useSE _useHSwish =>
      match curShape with
      | [b, _, h, w] =>
        let mid := expandCh
        let seMid := Nat.max 1 (mid / 4)
        if expandCh != ic then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if useSE then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .uib ic oc expand stride preDWk postDWk =>
      match curShape with
      | [b, _, h, w] =>
        let mid := ic * expand
        if preDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %g{pidx}: {tensorTy [ic]}, %bt{pidx}: {tensorTy [ic]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if postDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
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
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, ic, pSize, pSize]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [nP + 1, dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
      outShape := curShape
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, mlpDim]}, %b{pidx}: {tensorTy [mlpDim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mlpDim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, dim]
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
  hasHSwish  : Bool := false  -- h-swish activation: x * ReLU6(x+3) / 6 (MobileNetV3)
  isDepthwise : Bool := false -- depthwise conv (feature_group_count = channels)
  ic         : Nat := 0      -- input channels (for backward kernel shapes)
  kSize      : Nat := 3      -- kernel size
  stride     : Nat := 1      -- conv stride
  -- Residual skip: after this layer's backward, add this gradient SSA
  addSkipGrad : String := ""
  -- SE block (Squeeze-and-Excitation) intermediates
  isSE       : Bool := false
  sePidxRed  : Nat := 0      -- pidx of SE reduce conv (W, b)
  sePidxExp  : Nat := 0      -- pidx of SE expand conv (W, b)
  seMid      : Nat := 0      -- reduced channels in SE
  seMidFull  : Nat := 0      -- full channels (mid) in SE
  seInputSSA : String := ""  -- SSA of input x to SE (B, mid, H, W)
  seGapSSA   : String := ""  -- GAP result (B, mid) — post squeeze, pre reduce
  seRbSSA    : String := ""  -- reduce-conv + bias (B, seMid) — pre-swish
  seRswSSA   : String := ""  -- reduce swish out (B, seMid)
  seEbSSA    : String := ""  -- expand-conv + bias (B, mid) — pre-sigmoid
  seSigSSA   : String := ""  -- sigmoid result (B, mid)
  seOutSSA   : String := ""  -- x * broadcast(sig) (B, mid, H, W)
  -- SE variant: if true, SE block uses ReLU + h-sigmoid (MobileNetV3);
  -- otherwise swish + sigmoid (EfficientNet).
  seVariant  : Bool := false
  -- ═════════ ViT/Transformer intermediates ═════════
  -- patchEmbed fields (layer marked with isPatchEmbed)
  isPatchEmbed : Bool := false
  pePWPidx     : Nat := 0   -- base pidx for W (conv weight)
  pePBPidx     : Nat := 0   -- pidx for b (conv bias)
  peClsPidx    : Nat := 0   -- pidx for cls token (stored as W{pidx})
  pePosPidx    : Nat := 0   -- pidx for positional embedding
  pePSize      : Nat := 0
  pePIc        : Nat := 0
  pePDim       : Nat := 0
  pePNp        : Nat := 0
  -- Transformer block fields (layer marked with isTransformerBlock)
  isTransformerBlock : Bool := false
  tbBasePidx     : Nat := 0  -- starting pidx for this block's 8 param pairs
  tbHeads        : Nat := 0
  tbMlpDim       : Nat := 0
  tbDim          : Nat := 0
  -- Saved intermediates for each block's backward:
  tbLn1XSSA      : String := ""  -- input to LN1 (block input)
  tbLn1OutSSA    : String := ""
  tbLn1NormSSA   : String := ""
  tbLn1IstdSSA   : String := ""
  tbMhsaOutSSA   : String := ""
  tbR1SSA        : String := ""  -- after first residual (block input)
  tbLn2OutSSA    : String := ""
  tbLn2NormSSA   : String := ""
  tbLn2IstdSSA   : String := ""
  tbMhQSSA       : String := ""
  tbMhKSSA       : String := ""
  tbMhVSSA       : String := ""
  tbMhSmSSA      : String := ""
  tbMhPpSSA      : String := ""  -- pre-projection MHSA output (B, N, D)
  tbFc1OutSSA    : String := ""  -- pre-GELU (B, N, mlpDim)
  tbGeluTSSA     : String := ""  -- saved tanh value for GELU backward
  tbGeluOutSSA   : String := ""  -- post-GELU (B, N, mlpDim)
  -- Final LN (after all blocks) record
  isFinalLn       : Bool := false
  finalLnPidx     : Nat := 0
  finalLnNormSSA  : String := ""
  finalLnIstdSSA  : String := ""
  finalLnInSSA    : String := ""
  finalLnOutSSA   : String := ""
  -- CLS slice record (between final LN and classifier dense)
  isClsSlice      : Bool := false
  clsInShape      : List Nat := []
  clsInSSA        : String := ""
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
    Activation: ReLU6 by default, Swish if `useSwish := true`, or h-swish if
    `useHSwish := true`. -/
private def emitDepthwiseConvBnTrain (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) (useSwish : Bool := false)
    (useHSwish : Bool := false) (useRelu : Bool := false) : String × FwdRec := Id.run do
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
    if useHSwish then
      -- h-swish: x * ReLU6(x + 3) / 6
      s := s ++ s!"    %cbn_hs3{pidx} = stablehlo.constant dense<3.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hs6{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsxp3{pidx} = stablehlo.add {preSSA}, %cbn_hs3{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsclip{pidx} = stablehlo.minimum %cbn_hsxp3{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsr6{pidx} = stablehlo.maximum %cbn_hsclip{pidx}, %cbn_hsz{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsdiv{pidx} = stablehlo.divide %cbn_hsr6{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_hsdiv{pidx} : {tensorTy outShape}\n"
    else if useSwish then
      -- Swish: x * sigmoid(x)
      s := s ++ s!"    %cbn_sig{pidx} = stablehlo.logistic {preSSA} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_sig{pidx} : {tensorTy outShape}\n"
    else if useRelu then
      -- Plain ReLU (used by UIB / MobileNet V4)
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    else
      -- ReLU6 (default)
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
      hasRelu := useRelu
      hasRelu6 := !useSwish && !useHSwish && !useRelu
      hasSwish := useSwish
      hasHSwish := useHSwish
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

/-- Emit convBn forward for train step with h-swish activation:
    h_swish(x) = x * ReLU6(x + 3) / 6. Used by MobileNetV3. -/
private def emitConvBnTrainHSwish (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
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
    -- h-swish: x * ReLU6(x + 3) / 6
    s := s ++ s!"    %cbn_hs3{pidx} = stablehlo.constant dense<3.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hs6{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsxp3{pidx} = stablehlo.add {preSSA}, %cbn_hs3{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsclip{pidx} = stablehlo.minimum %cbn_hsxp3{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsr6{pidx} = stablehlo.maximum %cbn_hsclip{pidx}, %cbn_hsz{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsdiv{pidx} = stablehlo.divide %cbn_hsr6{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_hsdiv{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := false, hasSwish := false, hasHSwish := true
      isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnHSwish error\n", default)

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
  -- ReLU / ReLU6 / Swish / h-swish backward
  -- NOTE: ReLU prefix is `cbg_rel` (not `cbg_relu`) to avoid collision with
  -- `cbg_relu6{p}` when a ReLU layer's pidx equals `6` followed by a ReLU6 layer's pidx.
  let effGrad := if r.hasRelu then s!"%cbg_rel{p}"
                 else if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else if r.hasHSwish then s!"%cbg_hswish{p}"
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
  else if r.hasHSwish then
    -- h-swish backward: d/dx[x * ReLU6(x+3)/6]
    --   = 0            if x ≤ -3
    --   = (2x + 3)/6   if -3 < x < 3
    --   = 1            if x ≥ 3
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_hsn3{p} = stablehlo.constant dense<-3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs3{p} = stablehlo.constant dense<3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs2{p} = stablehlo.constant dense<2.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs6{p} = stablehlo.constant dense<6.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs1{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs0{p} = stablehlo.constant dense<0.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hslt{p} = stablehlo.compare LT, {r.preActSSA}, %cbg_hsn3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hsgt{p} = stablehlo.compare GT, {r.preActSSA}, %cbg_hs3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hs2x{p} = stablehlo.multiply {r.preActSSA}, %cbg_hs2{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hs2xp3{p} = stablehlo.add %cbg_hs2x{p}, %cbg_hs3{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsmid{p} = stablehlo.divide %cbg_hs2xp3{p}, %cbg_hs6{p} : {outTy}\n"
    -- If x < -3: 0 else mid. Then if x > 3: 1 else previous.
    s := s ++ s!"    %cbg_hsw1{p} = stablehlo.select %cbg_hslt{p}, %cbg_hs0{p}, %cbg_hsmid{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    %cbg_hsgrad{p} = stablehlo.select %cbg_hsgt{p}, %cbg_hs1{p}, %cbg_hsw1{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_hsgrad{p} : {outTy}\n"
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
  -- ReLU / ReLU6 / Swish / h-swish backward
  let effGrad := if r.hasRelu then s!"%cbg_rel{p}"
                 else if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else if r.hasHSwish then s!"%cbg_hswish{p}"
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
    -- Swish backward: see emitConvBnBackward comment.
    s := s ++ s!"    %cbg_sig{p} = stablehlo.logistic {r.preActSSA} : {outTy}\n"
    s := s ++ s!"    %cbg_one{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_1ms{p} = stablehlo.subtract %cbg_one{p}, %cbg_sig{p} : {outTy}\n"
    s := s ++ s!"    %cbg_xt{p} = stablehlo.multiply {r.preActSSA}, %cbg_1ms{p} : {outTy}\n"
    s := s ++ s!"    %cbg_1px{p} = stablehlo.add %cbg_one{p}, %cbg_xt{p} : {outTy}\n"
    s := s ++ s!"    %cbg_dsw{p} = stablehlo.multiply %cbg_sig{p}, %cbg_1px{p} : {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_dsw{p} : {outTy}\n"
  else if r.hasHSwish then
    -- h-swish backward: piecewise. See emitConvBnBackward.
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_hsn3{p} = stablehlo.constant dense<-3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs3{p} = stablehlo.constant dense<3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs2{p} = stablehlo.constant dense<2.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs6{p} = stablehlo.constant dense<6.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs1{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs0{p} = stablehlo.constant dense<0.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hslt{p} = stablehlo.compare LT, {r.preActSSA}, %cbg_hsn3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hsgt{p} = stablehlo.compare GT, {r.preActSSA}, %cbg_hs3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hs2x{p} = stablehlo.multiply {r.preActSSA}, %cbg_hs2{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hs2xp3{p} = stablehlo.add %cbg_hs2x{p}, %cbg_hs3{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsmid{p} = stablehlo.divide %cbg_hs2xp3{p}, %cbg_hs6{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsw1{p} = stablehlo.select %cbg_hslt{p}, %cbg_hs0{p}, %cbg_hsmid{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    %cbg_hsgrad{p} = stablehlo.select %cbg_hsgt{p}, %cbg_hs1{p}, %cbg_hsw1{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_hsgrad{p} : {outTy}\n"
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

/-- Emit SGD+momentum update for one param.
    v_new = mu * v + grad
    w_new = w - lr * v_new
    + optional decoupled weight decay: w_new = w_new - wd*lr*w -/
private def emitMomentumUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) : String × String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  -- v_new = mu * velocity + grad
  s := s ++ s!"    %mu_{tag} = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %vs_{tag} = stablehlo.multiply %mu_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %vn_{tag} = stablehlo.add %vs_{tag}, {gradSSA} : {ty}\n"
  -- w_new = w - lr * v_new
  s := s ++ s!"    %lr_{tag} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %up_{tag} = stablehlo.multiply %lr_{tag}, %vn_{tag} : {ty}\n"
  s := s ++ s!"    %sub_{tag} = stablehlo.subtract {paramSSA}, %up_{tag} : {ty}\n"
  if applyWeightDecay then
    -- Decoupled weight decay: w = w - lr*v_new - wd*lr*w
    s := s ++ s!"    %wd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wdlr_{tag} = stablehlo.multiply %wd_{tag}, %lr_{tag} : {ty}\n"
    s := s ++ s!"    %wdp_{tag} = stablehlo.multiply %wdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %new_{tag} = stablehlo.subtract %sub_{tag}, %wdp_{tag} : {ty}\n"
  -- mNew = velocity (stored in m_ slot), vPassthrough = original v_ unchanged
  return (s, if applyWeightDecay then s!"%new_{tag}" else s!"%sub_{tag}", s!"%vn_{tag}", vSSA)

/-- Emit Adam update for a convBn layer (W, gamma, beta). -/
private def emitConvBnAdam (p ic oc kSize : Nat) (applyWeightDecay : Bool := true) : String × Array String × Array String := Id.run do
  let wShape := [oc, ic, kSize, kSize]
  let bShape := [oc]
  let mut s := ""
  let (s1, wNew, mwNew, vwNew) := emitAdamUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"W{p}" (applyWeightDecay := applyWeightDecay)
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

/-- Emit SGD+momentum update for a convBn layer (W, gamma, beta). -/
private def emitConvBnMomentum (p ic oc kSize : Nat) (applyWeightDecay : Bool := false) : String × Array String × Array String := Id.run do
  let wShape := [oc, ic, kSize, kSize]
  let bShape := [oc]
  let mut s := ""
  let (s1, wNew, mwNew, vwNew) := emitMomentumUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"W{p}" (applyWeightDecay := applyWeightDecay)
  s := s ++ s1
  let (s2, gNew, mgNew, vgNew) := emitMomentumUpdate s!"%g{p}" s!"%d_g{p}" s!"%m_g{p}" s!"%v_g{p}" bShape s!"g{p}"
  s := s ++ s2
  let (s3, btNew, mbtNew, vbtNew) := emitMomentumUpdate s!"%bt{p}" s!"%d_bt{p}" s!"%m_bt{p}" s!"%v_bt{p}" bShape s!"bt{p}"
  s := s ++ s3
  let retNames := #[wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
  let retTypes := #[tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape]
  return (s, retNames, retTypes)

/-- Emit the full train step (forward + loss + backward + SGD). -/
private def emitTrainStepBody (spec : NetSpec) (batchSize : Nat) (_moduleName : String)
    (labelSmoothing : Float := 0.1) (weightDecay : Float := 0.0001) (useAdam : Bool := true)
    : String := Id.run do
  let B := batchSize
  let nClasses := spec.numClasses
  let mut code : String := ""
  code := code ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  code := code ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  if useAdam then
    code := code ++ "    // Adam constants\n"
    code := code ++ "    %beta1 = stablehlo.constant dense<0.9> : tensor<f32>\n"
    code := code ++ "    %beta2 = stablehlo.constant dense<0.999> : tensor<f32>\n"
    code := code ++ "    %one_minus_b1 = stablehlo.constant dense<0.1> : tensor<f32>\n"
    code := code ++ "    %one_minus_b2 = stablehlo.constant dense<0.001> : tensor<f32>\n"
    code := code ++ "    %adam_eps = stablehlo.constant dense<1.0e-08> : tensor<f32>\n"
    code := code ++ "    // Bias correction: bc1 = 1 - β1^t, bc2 = 1 - β2^t\n"
    code := code ++ "    %one_scalar = stablehlo.constant dense<1.0> : tensor<f32>\n"
    code := code ++ "    %b1t = stablehlo.power %beta1, %t : tensor<f32>\n"
    code := code ++ "    %bc1 = stablehlo.subtract %one_scalar, %b1t : tensor<f32>\n"
    code := code ++ "    %b2t = stablehlo.power %beta2, %t : tensor<f32>\n"
    code := code ++ "    %bc2 = stablehlo.subtract %one_scalar, %b2t : tensor<f32>\n"
  else
    code := code ++ "    // SGD+momentum constants\n"
    code := code ++ "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n"
  if weightDecay > 0.0 then
    code := code ++ s!"    %wdecay = stablehlo.constant dense<{weightDecay}> : tensor<f32>\n"
  code := code ++ "\n"

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
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
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
        -- SE block (optional): insert between depthwise and project
        if useSE then
          match curShape with
          | [b, _, h, w] =>
            let tag := s!"_t{pidx}_{bi}"
            let seIn := curSSA
            let xTy := tensorTy curShape
            let pRed := pidx
            let pExp := pidx + 1
            -- 1. GAP (B, mid, H, W) → (B, mid)
            code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
            -- 2. Reduce dense
            code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
            code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
            -- 3. Swish on reduce output
            code := code ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
            -- 4. Expand dense
            code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
            code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
            -- 5. Sigmoid + broadcast multiply
            code := code ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
            code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
            code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
            curSSA := s!"%se_out{tag}"
            -- Record SE block
            records := records.push {
              layer := .globalAvgPool  -- abused marker; isSE discriminates
              pidx := none, pos
              inputSSA := seIn
              preActSSA := ""
              outputSSA := curSSA
              inShape := curShape
              outShape := curShape
              hasRelu := false
              isSE := true
              sePidxRed := pRed
              sePidxExp := pExp
              seMid := seMid
              seMidFull := mid
              seInputSSA := seIn
              seGapSSA := s!"%se_g{tag}"
              seRbSSA := s!"%se_rb{tag}"
              seRswSSA := s!"%se_rsw{tag}"
              seEbSSA := s!"%se_eb{tag}"
              seSigSSA := s!"%se_sig{tag}"
              seOutSSA := curSSA
            }
            pidx := pidx + 2
          | _ => pure ()
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
          -- Number of layer records in this block: expand?(1) + dw(1) + se?(1) + proj(1)
          let nLayersInBlock := (if expand != 1 then 1 else 0) + 1 + (if useSE then 1 else 0) + 1
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

    | .fusedMbConv ic oc expand kSize firstStride nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let useSkip := stride == 1 && blockIc == oc
        -- 1. Fused expand: k×k regular convBn, Swish
        let (s1, rec1) := emitConvBnTrainSwish pidx pos curSSA curShape blockIc mid kSize stride
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- 2. SE block (optional): insert between expand and project
        if useSE then
          match curShape with
          | [b, _, h, w] =>
            let tag := s!"_tf{pidx}_{bi}"
            let seIn := curSSA
            let xTy := tensorTy curShape
            let pRed := pidx
            let pExp := pidx + 1
            -- 1. GAP (B, mid, H, W) → (B, mid)
            code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
            -- 2. Reduce dense
            code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
            code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
            -- 3. Swish on reduce output
            code := code ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
            -- 4. Expand dense
            code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
            code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
            -- 5. Sigmoid + broadcast multiply
            code := code ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
            code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
            code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
            curSSA := s!"%se_out{tag}"
            records := records.push {
              layer := .globalAvgPool
              pidx := none, pos
              inputSSA := seIn
              preActSSA := ""
              outputSSA := curSSA
              inShape := curShape
              outShape := curShape
              hasRelu := false
              isSE := true
              sePidxRed := pRed
              sePidxExp := pExp
              seMid := seMid
              seMidFull := mid
              seInputSSA := seIn
              seGapSSA := s!"%se_g{tag}"
              seRbSSA := s!"%se_rb{tag}"
              seRswSSA := s!"%se_rsw{tag}"
              seEbSSA := s!"%se_eb{tag}"
              seSigSSA := s!"%se_sig{tag}"
              seOutSSA := curSSA
            }
            pidx := pidx + 2
          | _ => pure ()
        -- 3. Project: 1×1, NO activation (only if expand != 1)
        if expand != 1 then
          let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
          code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
          records := records.push rec3; pidx := pidx + 1
        -- 4. Skip connection: stride==1 AND blockIc==oc, NO post-add activation
        if useSkip then
          let addId := s!"fmb{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          -- Number of layer records in this block: fused(1) + se?(1) + proj?(1)
          let nLayersInBlock := 1 + (if useSE then 1 else 0) + (if expand != 1 then 1 else 0)
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

    | .mbConvV3 ic oc expandCh kSize stride useSE useHSwish =>
      -- Single block variant of mbConv:
      -- expand (if expandCh != ic) → depthwise → optional SE → project → optional skip
      let blockIn := curSSA
      let blockInShape := curShape
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let useSkip := stride == 1 && ic == oc
      -- Expand: 1×1, h-swish (if useHSwish) or ReLU6
      if expandCh != ic then
        if useHSwish then
          let (s1, rec1) := emitConvBnTrainHSwish pidx pos curSSA curShape ic mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1
        else
          let (s1, rec1) := emitConvBnTrainRelu6 pidx pos curSSA curShape ic mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1
        pidx := pidx + 1
      -- Depthwise: k×k, h-swish if useHSwish else ReLU6
      let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid kSize stride
                          (useHSwish := useHSwish)
      code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
      records := records.push rec2; pidx := pidx + 1
      -- SE block (optional)
      if useSE then
        match curShape with
        | [b, _, h, w] =>
          let tag := s!"_t3{pidx}"
          let seIn := curSSA
          let xTy := tensorTy curShape
          let pRed := pidx
          let pExp := pidx + 1
          code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
          code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
          -- Plain ReLU on reduce (MobileNetV3 SE variant)
          code := code ++ s!"    %se_rzero{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rsw{tag} = stablehlo.maximum %se_rb{tag}, %se_rzero{tag} : {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
          -- h-sigmoid on expand: ReLU6(x + 3) / 6 (MobileNetV3)
          code := code ++ s!"    %se_sig3{tag} = stablehlo.constant dense<3.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sig6{tag} = stablehlo.constant dense<6.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigxp3{tag} = stablehlo.add %se_eb{tag}, %se_sig3{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigclip{tag} = stablehlo.minimum %se_sigxp3{tag}, %se_sig6{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigr6{tag} = stablehlo.maximum %se_sigclip{tag}, %se_sigz{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sig{tag} = stablehlo.divide %se_sigr6{tag}, %se_sig6{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
          curSSA := s!"%se_out{tag}"
          records := records.push {
            layer := .globalAvgPool
            pidx := none, pos
            inputSSA := seIn
            preActSSA := ""
            outputSSA := curSSA
            inShape := curShape
            outShape := curShape
            hasRelu := false
            isSE := true
            sePidxRed := pRed
            sePidxExp := pExp
            seMid := seMid
            seMidFull := mid
            seInputSSA := seIn
            seGapSSA := s!"%se_g{tag}"
            seRbSSA := s!"%se_rb{tag}"
            seRswSSA := s!"%se_rsw{tag}"
            seEbSSA := s!"%se_eb{tag}"
            seSigSSA := s!"%se_sig{tag}"
            seOutSSA := curSSA
            seVariant := true
          }
          pidx := pidx + 2
        | _ => pure ()
      -- Project: 1×1, NO activation
      let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
      code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
      records := records.push rec3; pidx := pidx + 1
      -- Skip connection: stride==1 AND ic==oc, NO post-add activation
      if useSkip then
        let addId := s!"mb3{pidx}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
        curSSA := s!"%rb_add{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Number of layer records in this block: expand?(1) + dw(1) + se?(1) + proj(1)
        let nLayersInBlock := (if expandCh != ic then 1 else 0) + 1 + (if useSE then 1 else 0) + 1
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

    | .uib ic oc expand stride preDWk postDWk =>
      -- UIB (MobileNet V4) block: optional preDW → expand 1×1 → optional postDW → project 1×1
      -- Plain ReLU throughout. Skip if stride==1 && ic==oc.
      let blockIn := curSSA
      let blockInShape := curShape
      let mid := ic * expand
      let useSkip := stride == 1 && ic == oc
      -- 1. Optional preDW: consumes stride, with plain ReLU
      let mut effectiveStride := stride
      if preDWk > 0 then
        let (s1, rec1) := emitDepthwiseConvBnTrain pidx pos curSSA curShape ic preDWk stride
                             (useRelu := true)
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        effectiveStride := 1
      -- 2. Expand 1×1 convBn + plain ReLU
      let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape ic mid 1 1 true
      code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
      records := records.push rec2; pidx := pidx + 1
      -- 3. Optional postDW with plain ReLU
      if postDWk > 0 then
        let (s3, rec3) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid postDWk effectiveStride
                             (useRelu := true)
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
      -- 4. Project 1×1 convBn, NO activation
      let (s4, rec4) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
      code := code ++ s4; curSSA := rec4.outputSSA; curShape := rec4.outShape
      records := records.push rec4; pidx := pidx + 1
      -- 5. Skip connection: stride==1 AND ic==oc, NO post-add activation
      if useSkip then
        let addId := s!"uib{pidx}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
        curSSA := s!"%rb_add{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Number of layer records in this block: preDW?(1) + expand(1) + postDW?(1) + proj(1)
        let nLayersInBlock := (if preDWk > 0 then 1 else 0) + 1 + (if postDWk > 0 then 1 else 0) + 1
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

    | .patchEmbed ic dim pSize nP =>
      let pWIdx := pidx
      let pBIdx := pidx  -- shared pidx (W + b of conv)
      let pClsIdx := pidx + 1
      let pPosIdx := pidx + 2
      let pWSSA := s!"%W{pWIdx}"
      let pBSSA := s!"%b{pBIdx}"
      let pClsSSA := s!"%W{pClsIdx}"
      let pPosSSA := s!"%W{pPosIdx}"
      let (snip, newSSA, newShape) := emitPatchEmbedForward s!"{pos}" curSSA curShape ic dim pSize nP pWSSA pBSSA pClsSSA pPosSSA
      code := code ++ snip
      let mut peRec : FwdRec := default
      peRec := { peRec with layer := l, pidx := none, pos, inputSSA := inSSA, outputSSA := newSSA, inShape := curShape, outShape := newShape }
      peRec := { peRec with isPatchEmbed := true, pePWPidx := pWIdx, pePBPidx := pBIdx, peClsPidx := pClsIdx, pePosPidx := pPosIdx }
      peRec := { peRec with pePSize := pSize, pePIc := ic, pePDim := dim, pePNp := nP }
      records := records.push peRec
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 3

    | .transformerEncoder dim heads mlpDim nBlocks =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockShape := curShape
        let basePidx := pidx
        let tag := s!"{pos}_{bi}"
        let ty := tensorTy blockShape
        -- LN1
        let g1 := s!"%W{pidx}"; let b1 := s!"%b{pidx}"
        let (ln1Code, ln1Out, ln1Norm, ln1Istd, _ln1Mean) := emitLayerNormForward s!"{tag}_ln1" blockIn blockShape g1 b1
        code := code ++ ln1Code
        pidx := pidx + 1
        -- MHSA
        let wq := s!"%W{pidx}"; let bq := s!"%b{pidx}"; pidx := pidx + 1
        let wk := s!"%W{pidx}"; let bk := s!"%b{pidx}"; pidx := pidx + 1
        let wv := s!"%W{pidx}"; let bv := s!"%b{pidx}"; pidx := pidx + 1
        let wo := s!"%W{pidx}"; let bo := s!"%b{pidx}"; pidx := pidx + 1
        let (mhCode, mhOut, mhQ, mhK, mhV, mhSm, mhPp) := emitMHSAForward s!"{tag}_mh" ln1Out blockShape heads wq bq wk bk wv bv wo bo
        code := code ++ mhCode
        -- Residual
        code := code ++ s!"    %tb_r1{tag} = stablehlo.add {blockIn}, {mhOut} : {ty}\n"
        let r1 := s!"%tb_r1{tag}"
        -- LN2
        let g2 := s!"%W{pidx}"; let b2 := s!"%b{pidx}"; pidx := pidx + 1
        let (ln2Code, ln2Out, ln2Norm, ln2Istd, _ln2Mean) := emitLayerNormForward s!"{tag}_ln2" r1 blockShape g2 b2
        code := code ++ ln2Code
        -- fc1
        let wf1 := s!"%W{pidx}"; let bf1 := s!"%b{pidx}"; pidx := pidx + 1
        let (fc1Code, fc1Out, fc1Shape) := emitDense3D s!"{tag}_fc1" ln2Out blockShape wf1 bf1 mlpDim
        code := code ++ fc1Code
        -- GELU
        let (geCode, geOut, geT) := emitGeluForward s!"{tag}_ge" fc1Out fc1Shape
        code := code ++ geCode
        -- fc2
        let wf2 := s!"%W{pidx}"; let bf2 := s!"%b{pidx}"; pidx := pidx + 1
        let (fc2Code, fc2Out, _) := emitDense3D s!"{tag}_fc2" geOut fc1Shape wf2 bf2 dim
        code := code ++ fc2Code
        -- Residual
        code := code ++ s!"    %tb_r2{tag} = stablehlo.add {r1}, {fc2Out} : {ty}\n"
        let blockOut := s!"%tb_r2{tag}"
        -- Record block
        let mut tbRec : FwdRec := default
        tbRec := { tbRec with layer := l, pidx := none, pos, inputSSA := blockIn, outputSSA := blockOut, inShape := blockShape, outShape := blockShape }
        tbRec := { tbRec with isTransformerBlock := true, tbBasePidx := basePidx, tbHeads := heads, tbMlpDim := mlpDim, tbDim := dim }
        tbRec := { tbRec with tbLn1XSSA := blockIn, tbLn1OutSSA := ln1Out, tbLn1NormSSA := ln1Norm, tbLn1IstdSSA := ln1Istd }
        tbRec := { tbRec with tbMhsaOutSSA := mhOut, tbR1SSA := r1 }
        tbRec := { tbRec with tbLn2OutSSA := ln2Out, tbLn2NormSSA := ln2Norm, tbLn2IstdSSA := ln2Istd }
        tbRec := { tbRec with tbMhQSSA := mhQ, tbMhKSSA := mhK, tbMhVSSA := mhV, tbMhSmSSA := mhSm, tbMhPpSSA := mhPp }
        tbRec := { tbRec with tbFc1OutSSA := fc1Out, tbGeluTSSA := geT, tbGeluOutSSA := geOut }
        records := records.push tbRec
        curSSA := blockOut
      -- Final LN
      let gF := s!"%W{pidx}"; let bF := s!"%b{pidx}"
      let finalLnPidxV := pidx
      let (lnfCode, lnfOut, lnfNorm, lnfIstd, _) := emitLayerNormForward s!"{pos}_fln" curSSA curShape gF bF
      code := code ++ lnfCode
      pidx := pidx + 1
      let mut flnRec : FwdRec := default
      flnRec := { flnRec with layer := l, pidx := none, pos, inputSSA := curSSA, outputSSA := lnfOut, inShape := curShape, outShape := curShape }
      flnRec := { flnRec with isFinalLn := true, finalLnPidx := finalLnPidxV, finalLnNormSSA := lnfNorm, finalLnIstdSSA := lnfIstd, finalLnInSSA := curSSA, finalLnOutSSA := lnfOut }
      records := records.push flnRec
      curSSA := lnfOut
      -- CLS slice
      match curShape with
      | [b, _, _] =>
        let clsShape := [b, 1, dim]
        let outShape := [b, dim]
        code := code ++ s!"    %te_cls{pos} = \"stablehlo.slice\"({curSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({tensorTy curShape}) -> {tensorTy clsShape}\n"
        code := code ++ s!"    %te_out{pos} = stablehlo.reshape %te_cls{pos} : ({tensorTy clsShape}) -> {tensorTy outShape}\n"
        let clsOut := s!"%te_out{pos}"
        let mut csRec : FwdRec := default
        csRec := { csRec with layer := l, pidx := none, pos, inputSSA := curSSA, outputSSA := clsOut, inShape := curShape, outShape := outShape }
        csRec := { csRec with isClsSlice := true, clsInShape := curShape, clsInSSA := curSSA }
        records := records.push csRec
        curSSA := clsOut
        curShape := outShape
      | _ => pure ()

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
  let smoothOn := if labelSmoothing > 0.0 then 1.0 - labelSmoothing + labelSmoothing / nClasses.toFloat else 1.0
  let smoothOff := if labelSmoothing > 0.0 then labelSmoothing / nClasses.toFloat else 0.0
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
      -- Check if this is an SE block record
      if r.isSE then
        match r.inShape with
        | [b, _mid, h, w] =>
          let mid := r.seMidFull
          let seMid := r.seMid
          let pRed := r.sePidxRed
          let pExp := r.sePidxExp
          let xTy := tensorTy r.inShape
          let tag := s!"_b{pRed}"
          -- Input gradient: dL/d_y where y = x * sig_bc
          -- 1. dx_direct = grad * sig_bc  (same shape as x)
          code := code ++ s!"    %seb_sigr{tag} = stablehlo.reshape {r.seSigSSA} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %seb_sigb{tag} = stablehlo.broadcast_in_dim %seb_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %seb_dx_dir{tag} = stablehlo.multiply {gradSSA}, %seb_sigb{tag} : {xTy}\n"
          -- 2. d_sig = reduce(grad * x, dims=[2,3]) → (B, mid)
          code := code ++ s!"    %seb_gx{tag} = stablehlo.multiply {gradSSA}, {r.seInputSSA} : {xTy}\n"
          code := code ++ s!"    %seb_dsig{tag} = stablehlo.reduce(%seb_gx{tag} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
          code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
          -- 3. d_se_eb: gate backward.
          if r.seVariant then
            -- h-sigmoid backward: d/dx h_sigmoid(x) = 1/6 if -3 < x < 3 else 0
            let i1Ty2 := (tensorTy [b, mid]).replace "xf32>" "xi1>"
            code := code ++ s!"    %seb_hsgn3{tag} = stablehlo.constant dense<-3.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsg3{tag} = stablehlo.constant dense<3.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsgs{tag} = stablehlo.constant dense<0.16666667> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsgz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsglt{tag} = stablehlo.compare LT, {r.seEbSSA}, %seb_hsgn3{tag} : ({tensorTy [b, mid]}, {tensorTy [b, mid]}) -> {i1Ty2}\n"
            code := code ++ s!"    %seb_hsggt{tag} = stablehlo.compare GT, {r.seEbSSA}, %seb_hsg3{tag} : ({tensorTy [b, mid]}, {tensorTy [b, mid]}) -> {i1Ty2}\n"
            code := code ++ s!"    %seb_hsgw1{tag} = stablehlo.select %seb_hsglt{tag}, %seb_hsgz{tag}, %seb_hsgs{tag} : {i1Ty2}, {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsggrad{tag} = stablehlo.select %seb_hsggt{tag}, %seb_hsgz{tag}, %seb_hsgw1{tag} : {i1Ty2}, {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_deb{tag} = stablehlo.multiply %seb_dsig{tag}, %seb_hsggrad{tag} : {tensorTy [b, mid]}\n"
          else
            -- Sigmoid backward: d_eb = d_sig * sig * (1 - sig)
            code := code ++ s!"    %seb_one{tag} = stablehlo.constant dense<1.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_1ms{tag} = stablehlo.subtract %seb_one{tag}, {r.seSigSSA} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_ssp{tag} = stablehlo.multiply {r.seSigSSA}, %seb_1ms{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_deb{tag} = stablehlo.multiply %seb_dsig{tag}, %seb_ssp{tag} : {tensorTy [b, mid]}\n"
          -- 4. d_b_se_exp = reduce(d_se_eb, dim=0) → (mid,)
          code := code ++ s!"    %d_b{pExp} = stablehlo.reduce(%seb_deb{tag} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({tensorTy [b, mid]}, tensor<f32>) -> {tensorTy [mid]}\n"
          -- 5. d_W_se_exp: dense backward. forward: eb[b,m] = sum_s rsw[b,s] * We[m,s]
          --    d_We[m, s] = sum_b d_eb[b, m] * rsw[b, s]
          code := code ++ s!"    %seb_dWe_flat{tag} = stablehlo.dot_general %seb_deb{tag}, {r.seRswSSA},\n"
          code := code ++ s!"              contracting_dims = [0] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [b, seMid]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %d_W{pExp} = stablehlo.reshape %seb_dWe_flat{tag} : ({tensorTy [mid, seMid]}) -> {tensorTy [mid, seMid, 1, 1]}\n"
          -- 6. d_rsw: forward eb = rsw @ We^T; d_rsw[b, s] = sum_m d_eb[b, m] * We[m, s]
          code := code ++ s!"    %seb_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %seb_drsw{tag} = stablehlo.dot_general %seb_deb{tag}, %seb_We{tag},\n"
          code := code ++ s!"              contracting_dims = [1] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, seMid]}\n"
          -- 7. d_rb: reduce activation backward.
          if r.seVariant then
            -- ReLU backward: d_rb = d_rsw if rb > 0 else 0
            let i1TyR := (tensorTy [b, seMid]).replace "xf32>" "xi1>"
            code := code ++ s!"    %seb_rz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_rmask{tag} = stablehlo.compare GT, {r.seRbSSA}, %seb_rz{tag} : ({tensorTy [b, seMid]}, {tensorTy [b, seMid]}) -> {i1TyR}\n"
            code := code ++ s!"    %seb_drb{tag} = stablehlo.select %seb_rmask{tag}, %seb_drsw{tag}, %seb_rz{tag} : {i1TyR}, {tensorTy [b, seMid]}\n"
          else
            -- Swish backward: d_rb = d_rsw * (sig_r + rb * sig_r * (1 - sig_r))
            code := code ++ s!"    %seb_rsig{tag} = stablehlo.logistic {r.seRbSSA} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_one_s{tag} = stablehlo.constant dense<1.0> : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_1mrs{tag} = stablehlo.subtract %seb_one_s{tag}, %seb_rsig{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_rpart{tag} = stablehlo.multiply {r.seRbSSA}, %seb_1mrs{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_1pr{tag} = stablehlo.add %seb_one_s{tag}, %seb_rpart{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_dsw{tag} = stablehlo.multiply %seb_rsig{tag}, %seb_1pr{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_drb{tag} = stablehlo.multiply %seb_drsw{tag}, %seb_dsw{tag} : {tensorTy [b, seMid]}\n"
          -- 8. d_b_se_red = reduce(d_rb, dim=0)
          code := code ++ s!"    %d_b{pRed} = stablehlo.reduce(%seb_drb{tag} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({tensorTy [b, seMid]}, tensor<f32>) -> {tensorTy [seMid]}\n"
          -- 9. d_W_se_red: forward rm[b, s] = sum_m g[b, m] * Wr[s, m]
          --    d_Wr[s, m] = sum_b d_rb[b, s] * g[b, m]
          code := code ++ s!"    %seb_dWr_flat{tag} = stablehlo.dot_general %seb_drb{tag}, {r.seGapSSA},\n"
          code := code ++ s!"              contracting_dims = [0] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [b, mid]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %d_W{pRed} = stablehlo.reshape %seb_dWr_flat{tag} : ({tensorTy [seMid, mid]}) -> {tensorTy [seMid, mid, 1, 1]}\n"
          -- 10. d_gap: d_g[b, m] = sum_s d_rb[b, s] * Wr[s, m]
          code := code ++ s!"    %seb_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %seb_dg{tag} = stablehlo.dot_general %seb_drb{tag}, %seb_Wr{tag},\n"
          code := code ++ s!"              contracting_dims = [1] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, mid]}\n"
          -- 11. GAP backward: dx_gap = broadcast(d_g) / (H*W)
          code := code ++ s!"    %seb_dgr{tag} = stablehlo.reshape %seb_dg{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %seb_dgb{tag} = stablehlo.broadcast_in_dim %seb_dgr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %seb_gN{tag} = stablehlo.constant dense<{h * w}.0> : {xTy}\n"
          code := code ++ s!"    %seb_dx_gap{tag} = stablehlo.divide %seb_dgb{tag}, %seb_gN{tag} : {xTy}\n"
          -- 12. Total dx = dx_direct + dx_gap
          code := code ++ s!"    %seb_dx{tag} = stablehlo.add %seb_dx_dir{tag}, %seb_dx_gap{tag} : {xTy}\n"
          gradSSA := s!"%seb_dx{tag}"
          gradShape := r.inShape
        | _ => pure ()
      else if r.addSkipGrad.startsWith "proj:" || r.addSkipGrad == "identity" then
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

    | .patchEmbed _ic _dim _p _nP =>
      if r.isPatchEmbed then
        -- Patch embedding backward
        match r.inShape with
        | [b, ic, h, w] =>
          let dim := r.pePDim
          let pSize := r.pePSize
          let nP := r.pePNp
          let nT := nP + 1
          let pW := r.pePWPidx
          let pB := r.pePBPidx
          let pCls := r.peClsPidx
          let pPos := r.pePosPidx
          let outTy := tensorTy r.outShape
          let tag := s!"peb{r.pos}"
          code := code ++ s!"    %d_W{pPos} = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [nT, dim]}\n"
          let clsShape := [b, 1, dim]
          code := code ++ s!"    %{tag}_cls_sl = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({outTy}) -> {tensorTy clsShape}\n"
          code := code ++ s!"    %d_W{pCls} = stablehlo.reduce(%{tag}_cls_sl init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({tensorTy clsShape}, tensor<f32>) -> {tensorTy [dim]}\n"
          let patchShape := [b, nP, dim]
          code := code ++ s!"    %{tag}_pat_sl = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 1, 0>, limit_indices = array<i64: {b}, {nT}, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({outTy}) -> {tensorTy patchShape}\n"
          code := code ++ s!"    %{tag}_pat_tr = stablehlo.transpose %{tag}_pat_sl, dims = [0, 2, 1] : ({tensorTy patchShape}) -> {tensorTy [b, dim, nP]}\n"
          let oH := h / pSize
          let oW := w / pSize
          let convOut := [b, dim, oH, oW]
          code := code ++ s!"    %{tag}_grad = stablehlo.reshape %{tag}_pat_tr : ({tensorTy [b, dim, nP]}) -> {tensorTy convOut}\n"
          code := code ++ s!"    %d_b{pB} = stablehlo.reduce(%{tag}_grad init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({tensorTy convOut}, tensor<f32>) -> {tensorTy [dim]}\n"
          code := code ++ s!"    %{tag}_bt_in = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [ic, b, h, w]}\n"
          code := code ++ s!"    %{tag}_bt_g = stablehlo.transpose %{tag}_grad, dims = [1, 0, 2, 3] : ({tensorTy convOut}) -> {tensorTy [dim, b, oH, oW]}\n"
          code := code ++ s!"    %{tag}_dWr = \"stablehlo.convolution\"(%{tag}_bt_in, %{tag}_bt_g) " ++ "{\n"
          code := code ++ convAttrBlockFull 0 0 0 0 1 1 1 1 pSize pSize
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [ic, b, h, w]}, {tensorTy [dim, b, oH, oW]}) -> {tensorTy [ic, dim, pSize, pSize]}\n"
          code := code ++ s!"    %d_W{pW} = stablehlo.transpose %{tag}_dWr, dims = [1, 0, 2, 3] : ({tensorTy [ic, dim, pSize, pSize]}) -> {tensorTy [dim, ic, pSize, pSize]}\n"
          -- Input layer: no dx needed
          gradSSA := "// no_grad_needed"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()
    | .transformerEncoder _dim _heads _mlpDim _nBlocks =>
      -- For records emitted during forward of a transformerEncoder layer:
      -- CLS slice, final LN, or transformer block (each discriminated by flags).
      if r.isClsSlice then
        match r.clsInShape with
        | [b, n, d] =>
          let preTy := tensorTy r.clsInShape
          let postTy := tensorTy r.outShape
          let slShape := [b, 1, d]
          code := code ++ s!"    %clsb_r{r.pos} = stablehlo.reshape {gradSSA} : ({postTy}) -> {tensorTy slShape}\n"
          let rest := n - 1
          let restShape := [b, rest, d]
          code := code ++ s!"    %clsb_z{r.pos} = stablehlo.constant dense<0.0> : {tensorTy restShape}\n"
          code := code ++ s!"    %clsb_cc{r.pos} = stablehlo.concatenate %clsb_r{r.pos}, %clsb_z{r.pos}, dim = 1 : ({tensorTy slShape}, {tensorTy restShape}) -> {preTy}\n"
          gradSSA := s!"%clsb_cc{r.pos}"
          gradShape := r.clsInShape
        | _ => pure ()
      else if r.isFinalLn then
        match r.inShape with
        | [b, n, d] =>
          let p := r.finalLnPidx
          let ty := tensorTy r.inShape
          let bnTy := tensorTy [b, n]
          let dTy := tensorTy [d]
          let dF := d.toFloat
          code := code ++ s!"    %flnb_gn{p} = stablehlo.multiply {gradSSA}, {r.finalLnNormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{p} = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{p} = stablehlo.reduce(%flnb_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %flnb_g_bc{p} = stablehlo.broadcast_in_dim %W{p}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_dn{p} = stablehlo.multiply {gradSSA}, %flnb_g_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_sdn{p} = stablehlo.reduce(%flnb_dn{p} init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %flnb_dnn{p} = stablehlo.multiply %flnb_dn{p}, {r.finalLnNormSSA} : {ty}\n"
          code := code ++ s!"    %flnb_sdnn{p} = stablehlo.reduce(%flnb_dnn{p} init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %flnb_sdn_bc{p} = stablehlo.broadcast_in_dim %flnb_sdn{p}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_sdnn_bc{p} = stablehlo.broadcast_in_dim %flnb_sdnn{p}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_Nc{p} = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %flnb_t1{p} = stablehlo.multiply %flnb_Nc{p}, %flnb_dn{p} : {ty}\n"
          code := code ++ s!"    %flnb_t2{p} = stablehlo.subtract %flnb_t1{p}, %flnb_sdn_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_t3{p} = stablehlo.multiply {r.finalLnNormSSA}, %flnb_sdnn_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_t4{p} = stablehlo.subtract %flnb_t2{p}, %flnb_t3{p} : {ty}\n"
          code := code ++ s!"    %flnb_istd_bc{p} = stablehlo.broadcast_in_dim {r.finalLnIstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_invN{p} = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %flnb_scale{p} = stablehlo.multiply %flnb_istd_bc{p}, %flnb_invN{p} : {ty}\n"
          code := code ++ s!"    %flnb_dx{p} = stablehlo.multiply %flnb_scale{p}, %flnb_t4{p} : {ty}\n"
          gradSSA := s!"%flnb_dx{p}"
          gradShape := r.inShape
        | _ => pure ()
      else if r.isTransformerBlock then
        match r.inShape with
        | [b, n, d] =>
          let basePidx := r.tbBasePidx
          let pLn1   := basePidx
          let pWq    := basePidx + 1
          let pWk    := basePidx + 2
          let pWv    := basePidx + 3
          let pWo    := basePidx + 4
          let pLn2   := basePidx + 5
          let pFc1   := basePidx + 6
          let pFc2   := basePidx + 7
          let heads := r.tbHeads
          let mlpDim := r.tbMlpDim
          let dh := d / heads
          let ty := tensorTy r.inShape
          let mlpTy := tensorTy [b, n, mlpDim]
          let bnTy := tensorTy [b, n]
          let dTy := tensorTy [d]
          let mlpDTy := tensorTy [mlpDim]
          let dF := d.toFloat
          let tag := s!"tb{basePidx}"
          let dy := gradSSA
          -- fc2 backward
          code := code ++ s!"    %{tag}_dwfc2 = stablehlo.dot_general {r.tbGeluOutSSA}, {dy},\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({mlpTy}, {ty}) -> {tensorTy [mlpDim, d]}\n"
          code := code ++ s!"    %d_W{pFc2} = stablehlo.reshape %{tag}_dwfc2 : ({tensorTy [mlpDim, d]}) -> {tensorTy [mlpDim, d]}\n"
          code := code ++ s!"    %d_b{pFc2} = stablehlo.reduce({dy} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dge = stablehlo.dot_general {dy}, %W{pFc2},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [mlpDim, d]}) -> {mlpTy}\n"
          -- GELU backward
          code := code ++ s!"    %{tag}_t2 = stablehlo.multiply {r.tbGeluTSSA}, {r.tbGeluTSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_one = stablehlo.constant dense<1.0> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_1mt2 = stablehlo.subtract %{tag}_one, %{tag}_t2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_c134 = stablehlo.constant dense<0.134145> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_xsq = stablehlo.multiply {r.tbFc1OutSSA}, {r.tbFc1OutSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_cx2 = stablehlo.multiply %{tag}_c134, %{tag}_xsq : {mlpTy}\n"
          code := code ++ s!"    %{tag}_idu = stablehlo.add %{tag}_one, %{tag}_cx2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_csq = stablehlo.constant dense<0.7978845608028654> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_du = stablehlo.multiply %{tag}_idu, %{tag}_csq : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term2a = stablehlo.multiply %{tag}_1mt2, %{tag}_du : {mlpTy}\n"
          code := code ++ s!"    %{tag}_c05 = stablehlo.constant dense<0.5> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_hx = stablehlo.multiply %{tag}_c05, {r.tbFc1OutSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term2 = stablehlo.multiply %{tag}_hx, %{tag}_term2a : {mlpTy}\n"
          code := code ++ s!"    %{tag}_1pt = stablehlo.add %{tag}_one, {r.tbGeluTSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term1 = stablehlo.multiply %{tag}_c05, %{tag}_1pt : {mlpTy}\n"
          code := code ++ s!"    %{tag}_dgdx = stablehlo.add %{tag}_term1, %{tag}_term2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_dfc1 = stablehlo.multiply %{tag}_dge, %{tag}_dgdx : {mlpTy}\n"
          -- fc1 backward
          code := code ++ s!"    %{tag}_dwfc1 = stablehlo.dot_general {r.tbLn2OutSSA}, %{tag}_dfc1,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {mlpTy}) -> {tensorTy [d, mlpDim]}\n"
          code := code ++ s!"    %d_W{pFc1} = stablehlo.reshape %{tag}_dwfc1 : ({tensorTy [d, mlpDim]}) -> {tensorTy [d, mlpDim]}\n"
          code := code ++ s!"    %d_b{pFc1} = stablehlo.reduce(%{tag}_dfc1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({mlpTy}, tensor<f32>) -> {mlpDTy}\n"
          code := code ++ s!"    %{tag}_dln2 = stablehlo.dot_general %{tag}_dfc1, %W{pFc1},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({mlpTy}, {tensorTy [d, mlpDim]}) -> {ty}\n"
          -- LN2 backward
          code := code ++ s!"    %{tag}_ln2_gn = stablehlo.multiply %{tag}_dln2, {r.tbLn2NormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{pLn2} = stablehlo.reduce(%{tag}_dln2 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{pLn2} = stablehlo.reduce(%{tag}_ln2_gn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_ln2gbc = stablehlo.broadcast_in_dim %W{pLn2}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_dn = stablehlo.multiply %{tag}_dln2, %{tag}_ln2gbc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdn = stablehlo.reduce(%{tag}_ln2_dn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln2_dnn = stablehlo.multiply %{tag}_ln2_dn, {r.tbLn2NormSSA} : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdnn = stablehlo.reduce(%{tag}_ln2_dnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln2_sdn_bc = stablehlo.broadcast_in_dim %{tag}_ln2_sdn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdnn_bc = stablehlo.broadcast_in_dim %{tag}_ln2_sdnn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_Nc = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t1 = stablehlo.multiply %{tag}_ln2_Nc, %{tag}_ln2_dn : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t2 = stablehlo.subtract %{tag}_ln2_t1, %{tag}_ln2_sdn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t3 = stablehlo.multiply {r.tbLn2NormSSA}, %{tag}_ln2_sdnn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t4 = stablehlo.subtract %{tag}_ln2_t2, %{tag}_ln2_t3 : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_istdbc = stablehlo.broadcast_in_dim {r.tbLn2IstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_invN = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_scale = stablehlo.multiply %{tag}_ln2_istdbc, %{tag}_ln2_invN : {ty}\n"
          code := code ++ s!"    %{tag}_dln2_in = stablehlo.multiply %{tag}_ln2_scale, %{tag}_ln2_t4 : {ty}\n"
          -- Residual 2 accumulate
          code := code ++ s!"    %{tag}_dr1 = stablehlo.add {dy}, %{tag}_dln2_in : {ty}\n"
          -- MHSA backward
          code := code ++ s!"    %{tag}_dwo = stablehlo.dot_general {r.tbMhPpSSA}, %{tag}_dr1,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWo} = stablehlo.reshape %{tag}_dwo : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWo} = stablehlo.reduce(%{tag}_dr1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dpp = stablehlo.dot_general %{tag}_dr1, %W{pWo},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          let heTy := tensorTy [b, n, heads, dh]
          let hTy := tensorTy [b, heads, n, dh]
          let sTy := tensorTy [b, heads, n, n]
          code := code ++ s!"    %{tag}_dppr = stablehlo.reshape %{tag}_dpp : ({ty}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dattn = stablehlo.transpose %{tag}_dppr, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
          code := code ++ s!"    %{tag}_dsm = stablehlo.dot_general %{tag}_dattn, {r.tbMhVSSA},\n"
          code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              contracting_dims = [3] x [3],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({hTy}, {hTy}) -> {sTy}\n"
          code := code ++ s!"    %{tag}_dv = stablehlo.dot_general {r.tbMhSmSSA}, %{tag}_dattn,\n"
          code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              contracting_dims = [2] x [2],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
          code := code ++ s!"    %{tag}_smdsm = stablehlo.multiply {r.tbMhSmSSA}, %{tag}_dsm : {sTy}\n"
          let bhnTy := tensorTy [b, heads, n]
          code := code ++ s!"    %{tag}_sumds = stablehlo.reduce(%{tag}_smdsm init: %zf) applies stablehlo.add across dimensions = [3]\n"
          code := code ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
          code := code ++ s!"    %{tag}_sumds_bc = stablehlo.broadcast_in_dim %{tag}_sumds, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
          code := code ++ s!"    %{tag}_dsms = stablehlo.subtract %{tag}_dsm, %{tag}_sumds_bc : {sTy}\n"
          code := code ++ s!"    %{tag}_dscaled = stablehlo.multiply {r.tbMhSmSSA}, %{tag}_dsms : {sTy}\n"
          let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
          code := code ++ s!"    %{tag}_scale = stablehlo.constant dense<{invSqrtDh}> : {sTy}\n"
          code := code ++ s!"    %{tag}_dscores = stablehlo.multiply %{tag}_dscaled, %{tag}_scale : {sTy}\n"
          code := code ++ s!"    %{tag}_dq = stablehlo.dot_general %{tag}_dscores, {r.tbMhKSSA},\n"
          code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              contracting_dims = [3] x [2],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
          code := code ++ s!"    %{tag}_dk = stablehlo.dot_general %{tag}_dscores, {r.tbMhQSSA},\n"
          code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              contracting_dims = [2] x [2],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
          code := code ++ s!"    %{tag}_dqt = stablehlo.transpose %{tag}_dq, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dqr = stablehlo.reshape %{tag}_dqt : ({heTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dkt = stablehlo.transpose %{tag}_dk, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dkr = stablehlo.reshape %{tag}_dkt : ({heTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dvt = stablehlo.transpose %{tag}_dv, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dvr = stablehlo.reshape %{tag}_dvt : ({heTy}) -> {ty}\n"
          -- QKV projection backwards
          code := code ++ s!"    %{tag}_dwq = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dqr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWq} = stablehlo.reshape %{tag}_dwq : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWq} = stablehlo.reduce(%{tag}_dqr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxq = stablehlo.dot_general %{tag}_dqr, %W{pWq},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dwk = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dkr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWk} = stablehlo.reshape %{tag}_dwk : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWk} = stablehlo.reduce(%{tag}_dkr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxk = stablehlo.dot_general %{tag}_dkr, %W{pWk},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dwv = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dvr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWv} = stablehlo.reshape %{tag}_dwv : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWv} = stablehlo.reduce(%{tag}_dvr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxv = stablehlo.dot_general %{tag}_dvr, %W{pWv},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dln1a = stablehlo.add %{tag}_dxq, %{tag}_dxk : {ty}\n"
          code := code ++ s!"    %{tag}_dln1 = stablehlo.add %{tag}_dln1a, %{tag}_dxv : {ty}\n"
          -- LN1 backward
          code := code ++ s!"    %{tag}_ln1_gn = stablehlo.multiply %{tag}_dln1, {r.tbLn1NormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{pLn1} = stablehlo.reduce(%{tag}_dln1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{pLn1} = stablehlo.reduce(%{tag}_ln1_gn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_ln1gbc = stablehlo.broadcast_in_dim %W{pLn1}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_dn = stablehlo.multiply %{tag}_dln1, %{tag}_ln1gbc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdn = stablehlo.reduce(%{tag}_ln1_dn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln1_dnn = stablehlo.multiply %{tag}_ln1_dn, {r.tbLn1NormSSA} : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdnn = stablehlo.reduce(%{tag}_ln1_dnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln1_sdn_bc = stablehlo.broadcast_in_dim %{tag}_ln1_sdn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdnn_bc = stablehlo.broadcast_in_dim %{tag}_ln1_sdnn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_Nc = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t1 = stablehlo.multiply %{tag}_ln1_Nc, %{tag}_ln1_dn : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t2 = stablehlo.subtract %{tag}_ln1_t1, %{tag}_ln1_sdn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t3 = stablehlo.multiply {r.tbLn1NormSSA}, %{tag}_ln1_sdnn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t4 = stablehlo.subtract %{tag}_ln1_t2, %{tag}_ln1_t3 : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_istdbc = stablehlo.broadcast_in_dim {r.tbLn1IstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_invN = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_scale = stablehlo.multiply %{tag}_ln1_istdbc, %{tag}_ln1_invN : {ty}\n"
          code := code ++ s!"    %{tag}_dln1_in = stablehlo.multiply %{tag}_ln1_scale, %{tag}_ln1_t4 : {ty}\n"
          code := code ++ s!"    %{tag}_dblockin = stablehlo.add %{tag}_dr1, %{tag}_dln1_in : {ty}\n"
          gradSSA := s!"%{tag}_dblockin"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | _ => pure ()

  -- ═══════════════ OPTIMIZER UPDATES ═══════════════
  code := code ++ (if useAdam then "\n    // ================ ADAM UPDATES ================\n"
                   else "\n    // ================ SGD+MOMENTUM UPDATES ================\n")
  let wdActive := weightDecay > 0.0
  let mut paramRetNames : Array String := #[]
  let mut paramRetTypes : Array String := #[]
  let mut mRetNames : Array String := #[]
  let mut mRetTypes : Array String := #[]
  let mut vRetNames : Array String := #[]
  let mut vRetTypes : Array String := #[]
  let mut processedPidx : Array Nat := #[]
  -- Helper: choose Adam or momentum update for a single parameter
  let emitUpdate := fun (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String) (applyWd : Bool) =>
    if useAdam then emitAdamUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd)
    else emitMomentumUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd)
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
          let (s1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"cW{p}" wdActive
          let (s2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"cb{p}" false
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | .dense fanIn fanOut _ =>
          let wShape := [fanIn, fanOut]; let bShape := [fanOut]
          let (s1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"dW{p}" wdActive
          let (s2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"db{p}" false
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
          let (optCode, optNames, optTypes) :=
            if useAdam then emitConvBnAdam p effIc oc kSize (applyWeightDecay := wdActive)
            else emitConvBnMomentum p effIc oc kSize (applyWeightDecay := wdActive)
          code := code ++ optCode
          -- returns [wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
          paramRetNames := paramRetNames ++ optNames[:3]
          paramRetTypes := paramRetTypes ++ optTypes[:3]
          mRetNames := mRetNames ++ optNames[3:6]
          mRetTypes := mRetTypes ++ optTypes[3:6]
          vRetNames := vRetNames ++ optNames[6:]
          vRetTypes := vRetTypes ++ optTypes[6:]
        | _ => pure ()
    | none =>
      -- SE records (marker layer = globalAvgPool, isSE := true) carry 2 conv2d-style param
      -- pidxes (reduce, expand). Emit optimizer updates for them inline.
      if r.isSE then
        let mid := r.seMidFull
        let seMid := r.seMid
        let pRed := r.sePidxRed
        let pExp := r.sePidxExp
        -- Reduce: W shape [seMid, mid, 1, 1], b shape [seMid]
        let wShapeR := [seMid, mid, 1, 1]; let bShapeR := [seMid]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pRed}" s!"%d_W{pRed}" s!"%m_W{pRed}" s!"%v_W{pRed}" wShapeR s!"seRW{pRed}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pRed}" s!"%d_b{pRed}" s!"%m_b{pRed}" s!"%v_b{pRed}" bShapeR s!"seRb{pRed}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        -- Expand: W shape [mid, seMid, 1, 1], b shape [mid]
        let wShapeE := [mid, seMid, 1, 1]; let bShapeE := [mid]
        let (sa3, wN2, mwN2, vwN2) := emitUpdate s!"%W{pExp}" s!"%d_W{pExp}" s!"%m_W{pExp}" s!"%v_W{pExp}" wShapeE s!"seEW{pExp}" wdActive
        let (sa4, bN2, mbN2, vbN2) := emitUpdate s!"%b{pExp}" s!"%d_b{pExp}" s!"%m_b{pExp}" s!"%v_b{pExp}" bShapeE s!"seEb{pExp}" false
        code := code ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push wN2 |>.push bN2
        paramRetTypes := paramRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
        mRetNames := mRetNames.push mwN2 |>.push mbN2
        mRetTypes := mRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
        vRetNames := vRetNames.push vwN2 |>.push vbN2
        vRetTypes := vRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
      -- ViT patch embedding: 3 param tensors
      if r.isPatchEmbed then
        let dim := r.pePDim
        let ic := r.pePIc
        let pSize := r.pePSize
        let nP := r.pePNp
        let pW := r.pePWPidx
        let pB := r.pePBPidx
        let pCls := r.peClsPidx
        let pPos := r.pePosPidx
        let wShape := [dim, ic, pSize, pSize]
        let bShape := [dim]
        let clsShape := [dim]
        let posShape := [nP + 1, dim]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pW}" s!"%d_W{pW}" s!"%m_W{pW}" s!"%v_W{pW}" wShape s!"peW{pW}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pB}" s!"%d_b{pB}" s!"%m_b{pB}" s!"%v_b{pB}" bShape s!"peb{pB}" false
        let (sa3, clsN, mclsN, vclsN) := emitUpdate s!"%W{pCls}" s!"%d_W{pCls}" s!"%m_W{pCls}" s!"%v_W{pCls}" clsShape s!"peCls{pCls}" false
        let (sa4, posN, mposN, vposN) := emitUpdate s!"%W{pPos}" s!"%d_W{pPos}" s!"%m_W{pPos}" s!"%v_W{pPos}" posShape s!"pePos{pPos}" false
        code := code ++ sa1 ++ sa2 ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push wN |>.push bN |>.push clsN |>.push posN
        paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
        mRetNames := mRetNames.push mwN |>.push mbN |>.push mclsN |>.push mposN
        mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
        vRetNames := vRetNames.push vwN |>.push vbN |>.push vclsN |>.push vposN
        vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
      -- ViT transformer block: 8 param pairs (each W + b)
      if r.isTransformerBlock then
        let dim := r.tbDim
        let mlpDim := r.tbMlpDim
        let basePidx := r.tbBasePidx
        -- Shapes for each param pair:
        -- LN1: (dim,), (dim,); Wq,Wk,Wv,Wo: (dim, dim), (dim,); LN2: (dim,), (dim,); Wfc1: (dim, mlpDim), (mlpDim,); Wfc2: (mlpDim, dim), (dim,)
        let paramShapes : Array (List Nat × List Nat) := #[
          ([dim], [dim]),            -- LN1
          ([dim, dim], [dim]),       -- Wq
          ([dim, dim], [dim]),       -- Wk
          ([dim, dim], [dim]),       -- Wv
          ([dim, dim], [dim]),       -- Wo
          ([dim], [dim]),            -- LN2
          ([dim, mlpDim], [mlpDim]), -- Wfc1
          ([mlpDim, dim], [dim])     -- Wfc2
        ]
        let decayW : Array Bool := #[false, true, true, true, true, false, true, true]
        for i in [:8] do
          let pp := basePidx + i
          let (wShape, bShape) := paramShapes[i]!
          let decay := decayW[i]! && wdActive
          let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pp}" s!"%d_W{pp}" s!"%m_W{pp}" s!"%v_W{pp}" wShape s!"tb_w{pp}" decay
          let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pp}" s!"%d_b{pp}" s!"%m_b{pp}" s!"%v_b{pp}" bShape s!"tb_b{pp}" false
          code := code ++ sa1 ++ sa2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
      -- ViT final LN (after all transformer blocks)
      if r.isFinalLn then
        let dim := r.inShape[2]!
        let p := r.finalLnPidx
        let shape := [dim]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" shape s!"fln_w{p}" false
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" shape s!"fln_b{p}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
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
    | .mbConv ic oc expand kSize stride nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
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
        if useSE then
          let wRedTy := tensorTy [seMid, mid, 1, 1]
          let bRedTy := tensorTy [seMid]
          params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
          paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
          mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
          vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
          pidx := pidx + 1
          let wExpTy := tensorTy [mid, seMid, 1, 1]
          let bExpTy := tensorTy [mid]
          params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
          paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
          mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
          vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
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
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        -- Fused expand: regular kxk convBn
        let wTy := tensorTy [mid, blockIc, kSize, kSize]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        if useSE then
          let wRedTy := tensorTy [seMid, mid, 1, 1]
          let bRedTy := tensorTy [seMid]
          params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
          paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
          mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
          vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
          pidx := pidx + 1
          let wExpTy := tensorTy [mid, seMid, 1, 1]
          let bExpTy := tensorTy [mid]
          params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
          paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
          mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
          vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
          pidx := pidx + 1
        if expand != 1 then
          let pjTy := tensorTy [oc, mid, 1, 1]
          params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
          paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
          mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
          vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConvV3 ic oc expandCh kSize stride useSE _useHSwish =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        let wTy := tensorTy [mid, ic, 1, 1]
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
      if useSE then
        let wRedTy := tensorTy [seMid, mid, 1, 1]
        let bRedTy := tensorTy [seMid]
        params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
        paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
        mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
        vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
        pidx := pidx + 1
        let wExpTy := tensorTy [mid, seMid, 1, 1]
        let bExpTy := tensorTy [mid]
        params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
        paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
        mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
        vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
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
    | .uib ic oc expand stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      -- preDW (if preDWk > 0): W[ic, 1, k, k], gamma/beta [ic]
      if preDWk > 0 then
        let wTy := tensorTy [ic, 1, preDWk, preDWk]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyI}, %bt{pidx}: {gTyI},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyI |>.push gTyI
        mRetTypes := mRetTypes.push wTy |>.push gTyI |>.push gTyI
        vRetTypes := vRetTypes.push wTy |>.push gTyI |>.push gTyI
        pidx := pidx + 1
      -- Expand 1×1: W[mid, ic, 1, 1]
      let exTy := tensorTy [mid, ic, 1, 1]
      params := params ++ s!"      %W{pidx}: {exTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
      paramRetTypes := paramRetTypes.push exTy |>.push gTyM |>.push gTyM
      mRetTypes := mRetTypes.push exTy |>.push gTyM |>.push gTyM
      vRetTypes := vRetTypes.push exTy |>.push gTyM |>.push gTyM
      pidx := pidx + 1
      -- postDW (if postDWk > 0): W[mid, 1, k, k]
      if postDWk > 0 then
        let wTy := tensorTy [mid, 1, postDWk, postDWk]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
      -- Project 1×1: W[oc, mid, 1, 1]
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
    | .patchEmbed ic dim pSize nP =>
      let wTy := tensorTy [dim, ic, pSize, pSize]; let bTy := tensorTy [dim]
      let clsTy := tensorTy [dim]; let posTy := tensorTy [nP + 1, dim]
      params := params ++ s!"      %W{pidx}: {wTy}, %b{pidx}: {bTy},\n"
      paramRetTypes := paramRetTypes.push wTy |>.push bTy
      mRetTypes := mRetTypes.push wTy |>.push bTy
      vRetTypes := vRetTypes.push wTy |>.push bTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {clsTy},\n"
      paramRetTypes := paramRetTypes.push clsTy
      mRetTypes := mRetTypes.push clsTy
      vRetTypes := vRetTypes.push clsTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {posTy},\n"
      paramRetTypes := paramRetTypes.push posTy
      mRetTypes := mRetTypes.push posTy
      vRetTypes := vRetTypes.push posTy
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        -- LN1
        params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push dTy |>.push dTy
        mRetTypes := mRetTypes.push dTy |>.push dTy
        vRetTypes := vRetTypes.push dTy |>.push dTy
        pidx := pidx + 1
        -- Wq, bq
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wk, bk
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wv, bv
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wo, bo
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- LN2
        params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push dTy |>.push dTy
        mRetTypes := mRetTypes.push dTy |>.push dTy
        vRetTypes := vRetTypes.push dTy |>.push dTy
        pidx := pidx + 1
        -- Wfc1, bfc1
        params := params ++ s!"      %W{pidx}: {fc1Ty}, %b{pidx}: {mlpBTy},\n"
        paramRetTypes := paramRetTypes.push fc1Ty |>.push mlpBTy
        mRetTypes := mRetTypes.push fc1Ty |>.push mlpBTy
        vRetTypes := vRetTypes.push fc1Ty |>.push mlpBTy
        pidx := pidx + 1
        -- Wfc2, bfc2
        params := params ++ s!"      %W{pidx}: {fc2Ty}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push fc2Ty |>.push dTy
        mRetTypes := mRetTypes.push fc2Ty |>.push dTy
        vRetTypes := vRetTypes.push fc2Ty |>.push dTy
        pidx := pidx + 1
      -- Final LN
      params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
      paramRetTypes := paramRetTypes.push dTy |>.push dTy
      mRetTypes := mRetTypes.push dTy |>.push dTy
      vRetTypes := vRetTypes.push dTy |>.push dTy
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, dim]
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
    | .mbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        if useSE then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
          mpidx := mpidx + 1
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
    | .fusedMbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        if useSE then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
          mpidx := mpidx + 1
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
          mpidx := mpidx + 1
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
          mpidx := mpidx + 1
    | .mbConvV3 ic oc expandCh kSize _ useSE _ =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, ic, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
      mpidx := mpidx + 1
      if useSE then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
      mpidx := mpidx + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if preDWk > 0 then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %m_g{mpidx}: {gTyI}, %m_bt{mpidx}: {gTyI},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, ic, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
      mpidx := mpidx + 1
      if postDWk > 0 then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
      mpidx := mpidx + 1
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [dim, ic, pSize, pSize]}, %m_b{mpidx}: {tensorTy [dim]},\n"
      mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [dim]},\n"
      mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [nP + 1, dim]},\n"
      mpidx := mpidx + 1
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {fc1Ty}, %m_b{mpidx}: {mlpBTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {fc2Ty}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
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
    | .mbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        if useSE then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
          vpidx2 := vpidx2 + 1
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
    | .fusedMbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        if useSE then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
          vpidx2 := vpidx2 + 1
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
          vpidx2 := vpidx2 + 1
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
          vpidx2 := vpidx2 + 1
    | .mbConvV3 ic oc expandCh kSize _ useSE _ =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, ic, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
      vpidx2 := vpidx2 + 1
      if useSE then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
      vpidx2 := vpidx2 + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if preDWk > 0 then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [ic, 1, preDWk, preDWk]}, %v_g{vpidx2}: {gTyI}, %v_bt{vpidx2}: {gTyI},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, ic, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
      vpidx2 := vpidx2 + 1
      if postDWk > 0 then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, postDWk, postDWk]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
      vpidx2 := vpidx2 + 1
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [dim, ic, pSize, pSize]}, %v_b{vpidx2}: {tensorTy [dim]},\n"
      vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [dim]},\n"
      vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [nP + 1, dim]},\n"
      vpidx2 := vpidx2 + 1
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {fc1Ty}, %v_b{vpidx2}: {mlpBTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {fc2Ty}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
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
def generateTrainStep (spec : NetSpec) (batchSize : Nat) (moduleName : String := "jit_train_step")
    (labelSmoothing : Float := 0.1) (weightDecay : Float := 0.0001) (useAdam : Bool := true)
    : String :=
  s!"// {spec.name} train_step — Generated by Lean 4 → MLIR (StableHLO + VJPs)\n" ++
  s!"// Batch size: {batchSize}, optimizer: {if useAdam then "Adam" else "SGD+momentum"}\n" ++
  s!"// label_smoothing: {labelSmoothing}, weight_decay: {weightDecay}\n\n" ++
  s!"module @{moduleName} " ++ "{\n" ++
  emitTrainStepSig spec batchSize ++ " {\n" ++
  emitTrainStepBody spec batchSize moduleName labelSmoothing weightDecay useAdam ++
  "  }\n" ++
  "}\n"

end MlirCodegen
