// Minimal repro: IREE/HIP `failed to distribute` on the channel-axis
// reduction inside a full LayerNorm-over-channels of an NCHW tensor.
// (The pattern arises naturally in ConvNeXt-style architectures.)
//
// Single-reduce-only programs compile fine; the failure needs the
// surrounding LN op chain (mean -> broadcast -> diff -> var -> rsqrt
// -> normalize -> affine), which IREE fuses into a dispatch that
// then can't be distributed for the GPU backend.
//
// Compile with:
//
//   iree-compile repro_ln_nchw.mlir \
//       --iree-hal-target-backends=rocm \
//       --iree-rocm-target=gfx1100 \
//       -o /tmp/x.vmfb
//
// Result: `'func.func' op failed to distribute`, dispatch name
//   `ln_dispatch_0_reduction_32x3136x96_f32`
//
// The same MLIR compiles cleanly for `--iree-hal-target-backends=llvm-cpu`.

module @repro {
  func.func @ln_channel_axis(%x: tensor<32x96x56x56xf32>,
                              %gamma: tensor<96xf32>,
                              %beta: tensor<96xf32>)
      -> tensor<32x96x56x56xf32> {
    %zero  = stablehlo.constant dense<0.0> : tensor<f32>
    %nc    = stablehlo.constant dense<96.0> : tensor<32x56x56xf32>
    %eps   = stablehlo.constant dense<1.0e-5> : tensor<32x56x56xf32>

    // mean over the channel axis.
    %sum = stablehlo.reduce(%x init: %zero)
        applies stablehlo.add across dimensions = [1]
        : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x56x56xf32>
    %mean = stablehlo.divide %sum, %nc : tensor<32x56x56xf32>
    %meanb = stablehlo.broadcast_in_dim %mean, dims = [0, 2, 3]
        : (tensor<32x56x56xf32>) -> tensor<32x96x56x56xf32>
    %diff  = stablehlo.subtract %x, %meanb : tensor<32x96x56x56xf32>

    // var over the channel axis.
    %sq    = stablehlo.multiply %diff, %diff : tensor<32x96x56x56xf32>
    %vsum = stablehlo.reduce(%sq init: %zero)
        applies stablehlo.add across dimensions = [1]
        : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x56x56xf32>
    %var   = stablehlo.divide %vsum, %nc : tensor<32x56x56xf32>
    %ve    = stablehlo.add %var, %eps : tensor<32x56x56xf32>
    %istd  = stablehlo.rsqrt %ve : tensor<32x56x56xf32>
    %istdb = stablehlo.broadcast_in_dim %istd, dims = [0, 2, 3]
        : (tensor<32x56x56xf32>) -> tensor<32x96x56x56xf32>

    // Normalize + γ, β.
    %norm  = stablehlo.multiply %diff, %istdb : tensor<32x96x56x56xf32>
    %gb    = stablehlo.broadcast_in_dim %gamma, dims = [1]
        : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %gn    = stablehlo.multiply %norm, %gb : tensor<32x96x56x56xf32>
    %bb    = stablehlo.broadcast_in_dim %beta, dims = [1]
        : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %out   = stablehlo.add %gn, %bb : tensor<32x96x56x56xf32>
    return %out : tensor<32x96x56x56xf32>
  }
}
