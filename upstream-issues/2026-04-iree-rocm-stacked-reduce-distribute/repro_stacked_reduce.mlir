// Minimal repro: IREE/HIP `failed to distribute` for the BatchNorm
// three-term backward (4× `reduce(dimensions=[0, 2, 3])` followed by
// broadcast-back-and-rebuild ops).
//
// Single `reduce[0,2,3]` over an NCHW tensor compiles fine on HIP
// (this exact pattern is the heart of the existing convBn lowering).
// The failure shows up only when the four reductions natural to BN
// backward (β-grad, γ-grad, sum(dn), sum(dn·norm)) are present *and*
// the result is fed into the standard 3-term LayerNorm-style dx
// formula `(istd / N) · (N·dn − sum_dn − norm·sum_dnn)`.
//
// IREE appears to fuse the reductions + the rebuild chain into one
// dispatch whose tile shape (`<batch>×<small>×<spatial>`) the Distribute
// pass can't handle.
//
// Compile with:
//
//   iree-compile repro_stacked_reduce.mlir \
//       --iree-hal-target-backends=rocm \
//       --iree-rocm-target=gfx1100 \
//       -o /tmp/x.vmfb
//
// Result: `'func.func' op failed to distribute` on a dispatch named
// `<...>_dispatch_<n>_reduction_<batch>x<small>x<spatial>_f32`.
//
// The same MLIR compiles cleanly for `--iree-hal-target-backends=llvm-cpu`.

module @repro {
  // The "BN three-term backward" boiled down to a single function: take
  // dy, norm, istd_bc, γ → produce dx of the same shape as dy. Mirrors
  // exactly what we emit per BN layer in the train-step body.
  func.func @bn_backward_dx(
      %dy:      tensor<2x32x16x16xf32>,
      %norm:    tensor<2x32x16x16xf32>,
      %istd_bc: tensor<2x32x16x16xf32>,
      %gamma:   tensor<32xf32>) -> tensor<2x32x16x16xf32> {
    %zero  = stablehlo.constant dense<0.0> : tensor<f32>
    %nF    = stablehlo.constant dense<512.0> : tensor<2x32x16x16xf32>  // N = b·h·w = 2·16·16
    %invN  = stablehlo.constant dense<1.953125e-3> : tensor<2x32x16x16xf32> // 1/N

    // 1) d_β = reduce(dy, [0, 2, 3])
    %dbeta = stablehlo.reduce(%dy init: %zero)
        applies stablehlo.add across dimensions = [0, 2, 3]
        : (tensor<2x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>

    // 2) d_γ = reduce(dy · norm, [0, 2, 3])
    %dyn   = stablehlo.multiply %dy, %norm : tensor<2x32x16x16xf32>
    %dgamma = stablehlo.reduce(%dyn init: %zero)
        applies stablehlo.add across dimensions = [0, 2, 3]
        : (tensor<2x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>

    // 3) dn = dy · γ_bc; sum_dn = reduce(dn, [0, 2, 3])
    %gb    = stablehlo.broadcast_in_dim %gamma, dims = [1]
        : (tensor<32xf32>) -> tensor<2x32x16x16xf32>
    %dn    = stablehlo.multiply %dy, %gb : tensor<2x32x16x16xf32>
    %sdn   = stablehlo.reduce(%dn init: %zero)
        applies stablehlo.add across dimensions = [0, 2, 3]
        : (tensor<2x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>

    // 4) sum_dnn = reduce(dn · norm, [0, 2, 3])
    %dnn   = stablehlo.multiply %dn, %norm : tensor<2x32x16x16xf32>
    %sdnn  = stablehlo.reduce(%dnn init: %zero)
        applies stablehlo.add across dimensions = [0, 2, 3]
        : (tensor<2x32x16x16xf32>, tensor<f32>) -> tensor<32xf32>

    // Three-term rebuild → dx.
    %sdnb  = stablehlo.broadcast_in_dim %sdn,  dims = [1]
        : (tensor<32xf32>) -> tensor<2x32x16x16xf32>
    %sdnnb = stablehlo.broadcast_in_dim %sdnn, dims = [1]
        : (tensor<32xf32>) -> tensor<2x32x16x16xf32>
    %t1    = stablehlo.multiply %nF, %dn : tensor<2x32x16x16xf32>
    %t2    = stablehlo.subtract %t1, %sdnb : tensor<2x32x16x16xf32>
    %t3    = stablehlo.multiply %norm, %sdnnb : tensor<2x32x16x16xf32>
    %t4    = stablehlo.subtract %t2, %t3 : tensor<2x32x16x16xf32>
    %scale = stablehlo.multiply %istd_bc, %invN : tensor<2x32x16x16xf32>
    %dx    = stablehlo.multiply %scale, %t4 : tensor<2x32x16x16xf32>

    // Sums also returned so they aren't dead-code-eliminated.
    %u1 = stablehlo.broadcast_in_dim %dbeta, dims = [1] : (tensor<32xf32>) -> tensor<2x32x16x16xf32>
    %u2 = stablehlo.broadcast_in_dim %dgamma, dims = [1] : (tensor<32xf32>) -> tensor<2x32x16x16xf32>
    %dx2 = stablehlo.add %dx, %u1 : tensor<2x32x16x16xf32>
    %dx3 = stablehlo.add %dx2, %u2 : tensor<2x32x16x16xf32>
    return %dx3 : tensor<2x32x16x16xf32>
  }
}
