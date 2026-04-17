import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.CNN

/-!
# Depthwise Convolution

The structural simplification at the heart of MobileNet, EfficientNet,
and every "mobile" CNN since ~2017. Standard conv2d does cross-channel
mixing (every output channel sees every input channel) plus spatial
filtering. Depthwise conv **drops the cross-channel mixing**: each
input channel gets its own 2D filter and produces its own output channel.

Math-wise it's "regular conv with a constraint." Practical-wise it's
~10× cheaper because you avoid the `O(ic · oc)` cross-channel sum.

Architecturally, depthwise is always paired with a 1×1 "pointwise" conv
that does the cross-channel mixing separately. Together they form the
**depthwise-separable convolution** (Xception, MobileNet) — the same
expressive power as a regular conv, factored into two cheaper steps.

## What this file proves

The depthwise conv is **structurally a special case of regular conv**:
- Regular conv kernel: `(oc, ic, kH, kW)` — full mixing.
- Depthwise kernel:    `(c, 1, kH, kW)` — diagonal in the channel pair.

So we don't re-derive the VJPs from scratch. We state them as the
"channel-restricted" versions of `conv2d_input_grad` /
`conv2d_weight_grad` from `CNN.lean`. The transpose trick still works,
the reversed-kernel trick still works — they just operate per-channel.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Types
-- ════════════════════════════════════════════════════════════════

/-- A depthwise kernel: `(c, kH, kW)` — one filter per channel, no
    `in_channels` axis. (Equivalently, a `(c, 1, kH, kW)` kernel where
    the singleton dim has been squeezed out.)

    In the MLIR backend this is represented as a regular `(c, 1, kH, kW)`
    kernel with `feature_group_count = c`, telling StableHLO to apply
    each kernel only to its own input channel. -/
abbrev DepthwiseKernel (c kH kW : Nat) := Fin c → Fin kH → Fin kW → ℝ

-- ════════════════════════════════════════════════════════════════
-- § Forward
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise conv2d forward** (SAME padding, stride 1).

    `y[c, h, w] = (Σ_{kh, kw} x[c, h+kh−p, w+kw−p] · W[c, kh, kw]) + b[c]`

    Compare to regular conv2d (`CNN.lean` `conv2d`):
      regular: `y[o,h,w] = Σ_{c, kh, kw} x[c,...] · W[o,c,kh,kw] + b[o]`
      depthwise: same minus the `Σ_c` (no cross-channel mixing).

    Output has the same number of channels as the input (`c`, not `oc`).

    MLIR (`MlirCodegen.lean` `emitDepthwiseConvBn`):
      uses `feature_group_count = c` to tell StableHLO that each kernel
      applies only within its own channel group. -/
axiom depthwiseConv2d {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) : Tensor3 c h w

-- ════════════════════════════════════════════════════════════════
-- § Backward — three pieces, each per-channel
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise conv input-VJP** — reversed kernel, applied per channel.

    The backward function (accessed as
    `(depthwise_has_vjp3 W b).backward`, or via the named
    `depthwiseConv2d_input_grad` abbrev below) implements:

      `dx[c, h, w] = Σ_{kh, kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`

    Compare to `conv2d_has_vjp3`:
    - Regular conv: `Σ_{o, kh, kw}` — summed over all output channels.
    - Depthwise:    `Σ_{kh, kw}`     — only spatial sum, channel `c`
                                       reads only from kernel `c` and
                                       gradient channel `c`.

    The kernel-reversal trick is the same; you just don't transpose
    `(oc, ic) → (ic, oc)` because there's no cross-channel structure. -/
axiom depthwise_has_vjp3 {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w)

/-- Named accessor for the depthwise input backward — aligns with MLIR
    codegen (per-channel `stablehlo.convolution` in the backward pass). -/
noncomputable abbrev depthwiseConv2d_input_grad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : Tensor3 c h w :=
  (depthwise_has_vjp3 W b).backward x dy

/-! ### Depthwise weight gradient (Phase 7 — now axiomatized)

Per-channel transpose trick:

    `dW[c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[c, h, w]`

Compare to the regular conv weight gradient:
- Regular conv: produces `(oc, ic, kH, kW)` — every (oc, ic) pair.
- Depthwise:    produces `(c, kH, kW)` — only the diagonal `(c, c)`
                pairs survive (the rest are zero by construction).

The transpose trick works the same way: view `x` and `dy` with
channel and batch axes swapped, do a standard conv, the spatial
dims of `dy` become the kernel dims. The only difference is that
`feature_group_count` is set so the conv stays per-channel.

MLIR (the depthwise variant of the transpose trick is in
`emitDepthwiseConvBnBackward` around line 1855):

    "For depthwise: dW[c,1,kH,kW] = sum_b input[b,c,:,:] conv grad[b,c,:,:]"

**Framework.** Unlike the regular-conv weight gradient (which needs
`Kernel4.flatten` because the kernel is 4D), the depthwise kernel
`DepthwiseKernel c kH kW` is 3D — same shape as `Tensor3 c kH kW`, and
in fact definitionally equal. So we can reuse the existing `HasVJP3`
framework directly, parameterized over `W` instead of `x`. -/

/-- **Depthwise weight-VJP** — bundled axiom using `HasVJP3` directly.

    `DepthwiseKernel c kH kW` is definitionally `Tensor3 c kH kW`
    (both `Fin c → Fin kH → Fin kW → ℝ`), so `HasVJP3` applies without
    the Kernel4 flattening needed for the regular conv weight gradient.
    The `.backward` is the per-channel transpose-trick formula documented
    above, gradient-checked numerically in
    `check_axioms.py:test_depthwise_weight_grad`. -/
axiom depthwise_weight_grad_has_vjp3 {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) :
    HasVJP3 (fun W : DepthwiseKernel c kH kW => depthwiseConv2d W b x)

/-- Named accessor for the depthwise weight backward. -/
noncomputable abbrev depthwiseConv2d_weight_grad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : DepthwiseKernel c kH kW :=
  (depthwise_weight_grad_has_vjp3 b x).backward W dy

/-- **Bias gradient** — sum the cotangent over the spatial dims, per channel.

    `db[c] = Σ_{h, w} dy[c, h, w]`

    Identical to regular conv's bias gradient — the bias is per-channel
    in both cases, and it adds the same value to every spatial cell
    of its channel. The reduction is the same. -/
noncomputable def depthwiseConv2d_bias_grad {c h w : Nat}
    (dy : Tensor3 c h w) : Vec c :=
  fun cc => ∑ y : Fin h, ∑ x : Fin w, dy cc y x

-- ════════════════════════════════════════════════════════════════
-- § The relationship to regular conv
-- ════════════════════════════════════════════════════════════════

/-! ## Depthwise = constrained regular conv

Conceptually, depthwise conv is regular conv with a sparsity pattern
on the kernel: `W_regular[o, c, kh, kw]` is zero unless `o = c`. Equivalently,
`W_regular` is **block-diagonal** in the `(o, c)` channel pair.

Two consequences for the VJPs:

1. **Forward**: the cross-channel sum `Σ_c` collapses to a single term
   (the diagonal one), giving the per-channel formula above.

2. **Backward**: every formula involving `Σ_o` or `Σ_c` over channel
   indices collapses similarly. The transpose trick still produces a
   `(ic, oc, kH, kW)` tensor in principle, but only the diagonal slice
   is nonzero, and the implementation just stores the diagonal.

So you don't have to derive depthwise VJPs from scratch — you derive
them by **specializing** the regular conv VJPs to the sparsity pattern.
This is a great example of how a constraint on the forward propagates
mechanically to a constraint on the backward.

## Cost

The forward op cost goes from `O(B · oc · ic · H · W · kH · kW)` for
regular conv to `O(B · c · H · W · kH · kW)` for depthwise — saves a
factor of `oc` (typically 32–512). The backward cost reduces by the
same factor. This is why mobile architectures pair depthwise with a
cheap 1×1 pointwise conv for cross-channel mixing — together they have
the same expressive power as a regular conv at a fraction of the FLOPs.

## Where it's used

- **MobileNet v1/v2/v3** (`MainMobilenet.lean`, `MainMobilenetV2.lean`,
  `MainMobilenetV3.lean`) — depthwise everywhere.
- **EfficientNet** (`MainEfficientNet.lean`) — depthwise inside MBConv blocks.
- **MBConv** (`MainEfficientNet.lean`, `MainEfficientNetV2.lean`) — the
  block that pairs an expand 1×1 → depthwise k×k → project 1×1, with
  optional Squeeze-and-Excitation. See `SE.lean` for the SE part.
-/

/-! ## Summary of axioms in this file

- `depthwiseConv2d` — forward (black-box).
- `depthwise_has_vjp3` — input-path VJP (function + correctness bundled).
- `depthwise_weight_grad_has_vjp3` — Phase 7: the weight-path VJP,
  bundled as `HasVJP3` directly (no flattening needed; see framework
  note above). Gradient-checked numerically.

Derived (not axioms):
- `depthwiseConv2d_input_grad`, `depthwiseConv2d_weight_grad` — named
  accessors, `.backward` of the corresponding VJP.
- `depthwiseConv2d_bias_grad` — concrete sum-over-spatial formula
  (not formally tied to a `pdiv` since `depthwiseConv2d` is opaque
  wrt `b`; cross-checked numerically instead). -/

end Proofs
