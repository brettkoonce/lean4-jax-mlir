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
noncomputable def depthwiseConv2d {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) : Tensor3 c h w :=
  fun ch hi wi =>
    b ch + ∑ kh : Fin kH, ∑ kw : Fin kW,
      W ch kh kw *
        (let pH := (kH - 1) / 2
         let pW := (kW - 1) / 2
         let hh := kh.val + hi.val
         let ww := kw.val + wi.val
         if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
           x ch ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
         else 0)

-- ════════════════════════════════════════════════════════════════
-- § Backward — three pieces, each per-channel
-- ════════════════════════════════════════════════════════════════

/-- **Closed-form input gradient for depthwise conv2d** — direct formula,
    written as a sum over output positions `(ho, wo)` with reconstructed
    kernel offsets `kh_nat = hi + pH − ho`, `kw_nat = wi + pW − wo`. The
    body is nonzero only when the reconstructed `(kh_nat, kw_nat)` lies
    in `[0, kH) × [0, kW)`. No `Σ co` like regular conv2d — input channel
    `ci` reads only from kernel-channel `ci` and gradient-channel `ci`,
    because depthwise has no cross-channel mixing.

    Equivalent (under the `(ho, wo) ↔ (kh, kw)` partial bijection) to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{kh, kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`. -/
noncomputable def depthwiseConv2d_input_grad_formula {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) : Tensor3 c h w :=
  fun ci hi wi =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy ci ho wo
      else 0

/-- **Depthwise conv input-VJP** — proved from foundation rules.

    The function `v ↦ flatten (depthwiseConv2d W b (unflatten v))` is
    affine in `v`: a constant `b ohw_o(idx_out)` plus a double sum over
    `(kh, kw)` of `W ohw_o kh kw * (if pad-cond then v(reindex) else 0)`.
    Mirrors `conv2d_has_vjp3` but with one fewer sum level (no Σ c) and
    the channel for the `v`-read is the same as `ohw_o` (forced by
    structure: input-channel = output-channel in depthwise).

    The closing collapse first folds `Σ co → co=ci` (since for `co ≠ ci`,
    the indicator `idx_in = finProdFinEquiv (..., co, ...)` is false by
    channel-mismatch on the first projection), then proceeds per-(ho, wo)
    with a 2-conjunct `h_indicator` (just `kh+ho = hi+pH` and
    `kw+wo = wi+pW`; no `c = ci` since `c` isn't summed).

    The backward function (accessed as `(depthwise_has_vjp3 W b).backward`,
    or via the `depthwiseConv2d_input_grad` abbrev below) implements
    `depthwiseConv2d_input_grad_formula W dy ci hi wi`. Equivalent to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{kh, kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`. -/
noncomputable def depthwise_has_vjp3 {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w) where
  backward := fun _x dy => depthwiseConv2d_input_grad_formula W dy
  correct := by
    intro x dy ci hi wi
    set idx_in : Fin (c * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in
    -- Step 1: per-(idx_in, idx_out) pdiv formula. UN-collapsed in (kh, kw)
    -- to avoid a partial bijection between Fin h and Fin kH; the closing
    -- collapse reindexes naturally.
    have h_pdiv : ∀ idx_out : Fin (c * h * w),
        pdiv (fun v' : Vec (c * h * w) =>
                Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ kh : Fin kH, ∑ kw : Fin kW,
          W ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1) kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1,
                    ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                 (1 : ℝ) else 0)
             else 0) := by
      intro idx_out
      set ohw_wi : Fin w := (finProdFinEquiv.symm idx_out).2 with hohw_wi
      set ohw_hi : Fin h :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2 with hohw_hi
      set ohw_o : Fin c :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 with hohw_o
      -- Decompose `f = (constant b) + (sum over kh kw of W * if-pad-cond)`.
      rw [show (fun v' : Vec (c * h * w) =>
                Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v'))) =
            (fun v' k =>
              (fun (_ : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'')
                         ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten depthwiseConv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
          (Tensor3.flatten x) :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
            ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 (Tensor3.unflatten v'')
                   ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                   ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) (Tensor3.flatten x) := by
        rw [differentiableAt_pi]
        intro k'
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1,
             ⟨kh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                (Tensor3.flatten x) idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute pdiv over the kh-sum.
      rw [show (fun (v'' : Vec (c * h * w)) (k' : Fin (c * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'')
                         ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ kh : Fin kH,
              (fun (khh : Fin kH) (v''' : Vec (c * h * w)) (k'' : Fin (c * h * w)) =>
                ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''')
                         ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kh v'' k') from rfl]
      have h_kh_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (c * h * w)) (k'' : Fin (c * h * w)) =>
              ∑ kw : Fin kW,
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   (Tensor3.unflatten v''')
                     ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                     ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten x) := by
        intro khh _
        rw [differentiableAt_pi]
        intro k''
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1,
             ⟨khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k'').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_finset_sum _ _ _ h_kh_diff]
      congr 1; ext khh
      -- Distribute pdiv over the kw-sum.
      rw [show (fun (v''' : Vec (c * h * w)) (k'' : Fin (c * h * w)) =>
                ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''')
                         ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v''' k'' => ∑ kw : Fin kW,
              (fun (kww : Fin kW) (v'''' : Vec (c * h * w)) (k''' : Fin (c * h * w)) =>
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     (Tensor3.unflatten v'''')
                       ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                       ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) kw v''' k'') from rfl]
      have h_kw_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
          DifferentiableAt ℝ
            (fun (v'''' : Vec (c * h * w)) (k''' : Fin (c * h * w)) =>
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                 let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   (Tensor3.unflatten v'''')
                     ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                     ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten x) := by
        intro kww _
        rw [differentiableAt_pi]
        intro k'''
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1,
             ⟨khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kww.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_finset_sum _ _ _ h_kw_diff]
      congr 1; ext kww
      -- Per-(khh, kww) summand: factor as (W constant) * (dite in v).
      rw [show (fun (v'''' : Vec (c * h * w)) (k''' : Fin (c * h * w)) =>
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     (Tensor3.unflatten v'''')
                       ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                       ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) =
            (fun (v'''' : Vec (c * h * w)) (k''' : Fin (c * h * w)) =>
              (fun k'''' : Fin (c * h * w) =>
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) khh kww) k''' *
              (if hpad : (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
                            (finProdFinEquiv.symm k''').1).2.val ∧
                          khh.val + (finProdFinEquiv.symm
                            (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2 < h ∧
                          (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k''').2.val ∧
                          kww.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2 < w then
                v'''' (finProdFinEquiv (finProdFinEquiv
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1,
                   ⟨khh.val + (finProdFinEquiv.symm
                          (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
                  ⟨kww.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2,
                    hpad.2.2.2⟩))
              else 0)) from by
        funext v'''' k'''
        unfold Tensor3.unflatten
        rfl]
      rw [pdiv_const_mul_pi_pad_eval
        (fun k''' : Fin (c * h * w) =>
          W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww)
        (fun k''' => (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
            (finProdFinEquiv.symm k''').1).2.val ∧
          khh.val + (finProdFinEquiv.symm
            (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2 < h ∧
          (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k''').2.val ∧
          kww.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2 < w)
        (fun k''' hpad => finProdFinEquiv (finProdFinEquiv
          ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1,
           ⟨khh.val + (finProdFinEquiv.symm
            (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
          ⟨kww.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2, hpad.2.2.2⟩))]
      -- Show result matches the desired form (with ohw_o, ohw_hi, ohw_wi abbreviations).
      show W ohw_o khh kww * _ = W ohw_o khh kww * _
      congr 1
      by_cases hpad : (kH - 1) / 2 ≤ khh.val + ohw_hi.val ∧
                     khh.val + ohw_hi.val - (kH - 1) / 2 < h ∧
                     (kW - 1) / 2 ≤ kww.val + ohw_wi.val ∧
                     kww.val + ohw_wi.val - (kW - 1) / 2 < w
      · rw [dif_pos hpad, dif_pos hpad]
        by_cases heq : finProdFinEquiv (finProdFinEquiv
            (ohw_o, ⟨khh.val + ohw_hi.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kww.val + ohw_wi.val - (kW - 1) / 2, hpad.2.2.2⟩) = idx_in
        · rw [if_pos heq, if_pos heq.symm]
        · rw [if_neg heq, if_neg (fun h => heq h.symm)]
      · rw [dif_neg hpad, dif_neg hpad]
    -- Step 2: substitute h_pdiv and collapse.
    show depthwiseConv2d_input_grad_formula W dy ci hi wi =
      ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (depthwiseConv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold depthwiseConv2d_input_grad_formula pdiv3
    -- Outer Σ co collapse on RHS at co = ci. For co ≠ ci, the pdiv inner
    -- sum is 0 because the indicator `idx_in = flat(co, ...)` is false
    -- (channel mismatch on the first projection: idx_in's channel is ci).
    rw [Finset.sum_eq_single ci
        (fun co _ hco_ne => by
          apply Finset.sum_eq_zero; intro ho _
          apply Finset.sum_eq_zero; intro wo _
          rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (co, ho), wo))]
          simp only [Equiv.symm_apply_apply]
          suffices h_inner : (∑ kh : Fin kH, ∑ kw : Fin kW,
              W co kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val + ho.val
                 let ww := kw.val + wo.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   (if idx_in = finProdFinEquiv (finProdFinEquiv
                       (co, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                     (1 : ℝ) else 0)
                 else 0)) = 0 from by rw [h_inner]; ring
          apply Finset.sum_eq_zero; intro kh _
          apply Finset.sum_eq_zero; intro kw _
          suffices h_ind : ((let pH := (kH - 1) / 2
                             let pW := (kW - 1) / 2
                             let hh := kh.val + ho.val
                             let ww := kw.val + wo.val
                             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                               (if idx_in = finProdFinEquiv (finProdFinEquiv
                                   (co, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                                 (1 : ℝ) else 0)
                             else 0) : ℝ) = 0 from by rw [h_ind]; ring
          by_cases hpad : (kH - 1) / 2 ≤ kh.val + ho.val ∧
                          kh.val + ho.val - (kH - 1) / 2 < h ∧
                          (kW - 1) / 2 ≤ kw.val + wo.val ∧
                          kw.val + wo.val - (kW - 1) / 2 < w
          · rw [dif_pos hpad, if_neg ?_]
            intro h_eq
            rw [hidx_in] at h_eq
            have h_inj := finProdFinEquiv.injective h_eq
            have h_inj_pair := Prod.mk.inj h_inj
            have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
            have h_inj_inner_pair := Prod.mk.inj h_inj_inner
            exact hco_ne h_inj_inner_pair.1.symm
          · rw [dif_neg hpad])
        (fun hni => absurd (Finset.mem_univ ci) hni)]
    -- Now: LHS = ∑ ho wo, formula_inner; RHS = ∑ ho wo, pdiv3 at co=ci * dy ci ho wo.
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (ci, ho), wo))]
    simp only [Equiv.symm_apply_apply]
    -- Pull `dy ci ho wo` out of the formula's if-true branch.
    rw [show (let pH := (kH - 1) / 2
              let pW := (kW - 1) / 2
              let kh_nat := hi.val + pH - ho.val
              let kw_nat := wi.val + pW - wo.val
              if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                  wo.val ≤ wi.val + pW ∧ kw_nat < kW then
                W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy ci ho wo
              else 0) =
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let kh_nat := hi.val + pH - ho.val
             let kw_nat := wi.val + pW - wo.val
             if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
               W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩
             else 0) * dy ci ho wo from by
      by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                     hi.val + (kH - 1) / 2 - ho.val < kH ∧
                     wo.val ≤ wi.val + (kW - 1) / 2 ∧
                     wi.val + (kW - 1) / 2 - wo.val < kW
      · simp only [dif_pos hb]
      · simp only [dif_neg hb, zero_mul]]
    congr 1
    -- Convert the dependent-if indicator to a non-dependent 2-conjunct form.
    have h_indicator : ∀ (kh : Fin kH) (kw : Fin kW),
        ((let pH := (kH - 1) / 2
          let pW := (kW - 1) / 2
          let hh := kh.val + ho.val
          let ww := kw.val + wo.val
          if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
            (if idx_in = finProdFinEquiv (finProdFinEquiv
                (ci, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then (1 : ℝ) else 0)
          else 0) : ℝ) =
        (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
            kw.val + wo.val = wi.val + (kW - 1) / 2 then (1 : ℝ) else 0) := by
      intro kh kw
      by_cases hpad : (kH - 1) / 2 ≤ kh.val + ho.val ∧
                      kh.val + ho.val - (kH - 1) / 2 < h ∧
                      (kW - 1) / 2 ≤ kw.val + wo.val ∧
                      kw.val + wo.val - (kW - 1) / 2 < w
      · rw [dif_pos hpad]
        by_cases h_match : kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                           kw.val + wo.val = wi.val + (kW - 1) / 2
        · have h_idx_in_eq : idx_in = finProdFinEquiv (finProdFinEquiv
              (ci, ⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩) := by
            rw [hidx_in]
            have h_hi : (⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩ : Fin h) = hi := by
              apply Fin.ext
              show kh.val + ho.val - (kH - 1) / 2 = hi.val
              omega
            have h_wi : (⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩ : Fin w) = wi := by
              apply Fin.ext
              show kw.val + wo.val - (kW - 1) / 2 = wi.val
              omega
            rw [← h_hi, ← h_wi]
          rw [if_pos h_idx_in_eq, if_pos h_match]
        · rw [if_neg h_match]
          rw [if_neg]
          intro h_eq
          apply h_match
          rw [hidx_in] at h_eq
          have h_inj := finProdFinEquiv.injective h_eq
          have h_inj_pair := Prod.mk.inj h_inj
          have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
          have h_inj_inner_pair := Prod.mk.inj h_inj_inner
          refine ⟨?_, ?_⟩
          · have h_hi : hi.val = kh.val + ho.val - (kH - 1) / 2 :=
              Fin.ext_iff.mp h_inj_inner_pair.2
            omega
          · have h_wi : wi.val = kw.val + wo.val - (kW - 1) / 2 :=
              Fin.ext_iff.mp h_inj_pair.2
            omega
      · rw [dif_neg hpad]
        rw [if_neg]
        intro ⟨hkh_eq, hkw_eq⟩
        apply hpad
        refine ⟨?_, ?_, ?_, ?_⟩
        · rw [hkh_eq]; exact Nat.le_add_left _ _
        · rw [hkh_eq, Nat.add_sub_cancel]; exact hi.isLt
        · rw [hkw_eq]; exact Nat.le_add_left _ _
        · rw [hkw_eq, Nat.add_sub_cancel]; exact wi.isLt
    simp_rw [h_indicator]
    -- Goal: (if hb : back_cond then W ci ⟨kh*⟩ ⟨kw*⟩ else 0)
    --     = ∑ kh kw, W ci kh kw * (if (kh+ho = hi+pH ∧ kw+wo = wi+pW) then 1 else 0)
    by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                   hi.val + (kH - 1) / 2 - ho.val < kH ∧
                   wo.val ≤ wi.val + (kW - 1) / 2 ∧
                   wi.val + (kW - 1) / 2 - wo.val < kW
    · rw [dif_pos hb]
      symm
      rw [Finset.sum_eq_single ⟨hi.val + (kH - 1) / 2 - ho.val, hb.2.1⟩ ?_ ?_]
      rw [Finset.sum_eq_single ⟨wi.val + (kW - 1) / 2 - wo.val, hb.2.2.2⟩ ?_ ?_]
      · rw [if_pos]
        · ring
        refine ⟨?_, ?_⟩
        · show hi.val + (kH - 1) / 2 - ho.val + ho.val = hi.val + (kH - 1) / 2
          omega
        · show wi.val + (kW - 1) / 2 - wo.val + wo.val = wi.val + (kW - 1) / 2
          omega
      · intro kw _ hkw_ne
        rw [if_neg ?_]; · ring
        intro ⟨_, hkw_eq⟩
        apply hkw_ne
        apply Fin.ext
        show kw.val = wi.val + (kW - 1) / 2 - wo.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro kh _ hkh_ne
        apply Finset.sum_eq_zero; intro kw _
        rw [if_neg ?_]; · ring
        intro ⟨hkh_eq, _⟩
        apply hkh_ne
        apply Fin.ext
        show kh.val = hi.val + (kH - 1) / 2 - ho.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
    · rw [dif_neg hb]
      symm
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [if_neg ?_]; · ring
      intro ⟨hkh_eq, hkw_eq⟩
      apply hb
      refine ⟨?_, ?_, ?_, ?_⟩
      · have := kh.isLt; omega
      · have := kh.isLt; omega
      · have := kw.isLt; omega
      · have := kw.isLt; omega

/-- Named accessor for the depthwise input backward — aligns with MLIR
    codegen (per-channel `stablehlo.convolution` in the backward pass). -/
noncomputable abbrev depthwiseConv2d_input_grad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : Tensor3 c h w :=
  (depthwise_has_vjp3 W b).backward x dy

/-! ### Depthwise weight gradient (Phase 7 — proved from foundation rules)

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

/-- **Depthwise weight-VJP** — proved from foundation rules.

    `DepthwiseKernel c kH kW` is definitionally `Tensor3 c kH kW`, so
    `HasVJP3` applies directly. The function `W ↦ depthwiseConv2d W b x`
    is affine in W: at output (co, ho, wo) it's
    `b co + Σ_{kh, kw} W co kh kw * x_pad_term(co, kh, kw, ho, wo)`.
    Same recipe as `conv2d_weight_grad_has_vjp` but with two inner
    dims (kh, kw) instead of three (c, kh, kw) — depthwise has no
    cross-channel sum, so the "channel match" condition `co = ci` is
    a single equality rather than a packed comparison. -/
noncomputable def depthwise_weight_grad_has_vjp3 {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) :
    HasVJP3 (fun W : DepthwiseKernel c kH kW => depthwiseConv2d W b x) where
  backward := fun _W dy => fun ci hi_k wi_k =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      (let pH := (kH - 1) / 2
       let pW := (kW - 1) / 2
       let hh := hi_k.val + ho.val
       let ww := wi_k.val + wo.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0)
      * dy ci ho wo
  correct := by
    intro W dy ci hi_k wi_k
    -- Per-(co, ho, wo) pdiv3 formula.
    have h_pdiv3 : ∀ co : Fin c, ∀ ho : Fin h, ∀ wo : Fin w,
        pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x)
          W ci hi_k wi_k co ho wo =
        (if co = ci then
          (let pH := (kH - 1) / 2
           let pW := (kW - 1) / 2
           let hh := hi_k.val + ho.val
           let ww := wi_k.val + wo.val
           if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
             x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
           else 0)
        else 0) := by
      intro co ho wo
      unfold pdiv3
      -- Decompose flattened function as (constant b) + (sum over kh kw of v-reindex * x_pad).
      rw [show (fun v' : Vec (c * kH * kW) =>
                Tensor3.flatten (depthwiseConv2d
                  (Tensor3.unflatten v' : DepthwiseKernel c kH kW) b x)) =
            (fun v' k =>
              (fun (_ : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Tensor3.unflatten v'' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten depthwiseConv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
          (Tensor3.flatten W) :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
            ∑ kh : Fin kH, ∑ kw : Fin kW,
              (Tensor3.unflatten v'' : DepthwiseKernel c kH kW)
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                   ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) (Tensor3.flatten W) := by
        unfold Tensor3.unflatten; fun_prop
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                (Tensor3.flatten W)
                (finProdFinEquiv (finProdFinEquiv (ci, hi_k), wi_k))
                (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute over the double sum.
      rw [show (fun (v'' : Vec (c * kH * kW)) (k' : Fin (c * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Tensor3.unflatten v'' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ kh : Fin kH,
              (fun (khh : Fin kH) (v''' : Vec (c * kH * kW))
                  (k'' : Fin (c * h * w)) =>
                ∑ kw : Fin kW,
                  (Tensor3.unflatten v''' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kh v'' k') from rfl]
      have h_kh_summand_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (c * kH * kW)) (k'' : Fin (c * h * w)) =>
              ∑ kw : Fin kW,
                (Tensor3.unflatten v''' : DepthwiseKernel c kH kW)
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                     ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten W) := by
        intro khh _; unfold Tensor3.unflatten; fun_prop
      rw [pdiv_finset_sum _ _ _ h_kh_summand_diff]
      have h_inner_kh : ∀ khh : Fin kH,
          pdiv (fun (v''' : Vec (c * kH * kW))
                    (k'' : Fin (c * h * w)) =>
                ∑ kw : Fin kW,
                  (Tensor3.unflatten v''' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0))
                (Tensor3.flatten W)
                (finProdFinEquiv (finProdFinEquiv (ci, hi_k), wi_k))
                (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) =
          ∑ kw : Fin kW,
            (if (finProdFinEquiv (finProdFinEquiv (ci, hi_k), wi_k) :
                  Fin (c * kH * kW)) =
                finProdFinEquiv (finProdFinEquiv (co, khh), kw)
              then (1 : ℝ) else 0) *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := khh.val + ho.val
             let ww := kw.val + wo.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x co ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0) := by
        intro khh
        rw [show (fun (v''' : Vec (c * kH * kW))
                      (k'' : Fin (c * h * w)) =>
                  ∑ kw : Fin kW,
                    (Tensor3.unflatten v''' : DepthwiseKernel c kH kW)
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                      khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                           ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v''' k'' => ∑ kw : Fin kW,
                (fun (kww : Fin kW) (v'''' : Vec (c * kH * kW))
                    (k''' : Fin (c * h * w)) =>
                  (Tensor3.unflatten v'''' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kw v''' k'') from rfl]
        have h_kw_summand_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
            DifferentiableAt ℝ
              (fun (v'''' : Vec (c * kH * kW)) (k''' : Fin (c * h * w)) =>
                (Tensor3.unflatten v'''' : DepthwiseKernel c kH kW)
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                 let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                     ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten W) := by
          intro kww _; unfold Tensor3.unflatten; fun_prop
        rw [pdiv_finset_sum _ _ _ h_kw_summand_diff]
        congr 1; ext kww
        -- Per-summand: factor as (reindex v) * (constant in v).
        rw [show (fun (v'''' : Vec (c * kH * kW))
                      (k''' : Fin (c * h * w)) =>
                  (Tensor3.unflatten v'''' : DepthwiseKernel c kH kW)
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
              (fun v'''' k''' =>
                (fun (v''''' : Vec (c * kH * kW))
                    (k'''' : Fin (c * h * w)) =>
                  v''''' (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1, khh),
                      kww))) v'''' k''' *
                (fun (_ : Vec (c * kH * kW))
                    (k'''' : Fin (c * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                       ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) v'''' k''') from by
          funext v'''' k'''
          show (Tensor3.unflatten v'''' : DepthwiseKernel c kH kW)
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) khh kww * _ = _
          unfold Tensor3.unflatten
          rfl]
        have h_reindex_diff : DifferentiableAt ℝ
            (fun (v''''' : Vec (c * kH * kW)) (k'''' : Fin (c * h * w)) =>
              v''''' (finProdFinEquiv (finProdFinEquiv
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1, khh),
                  kww))) (Tensor3.flatten W) :=
          (reindexCLM (fun k'''' : Fin (c * h * w) =>
            finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1, khh),
                kww))).differentiableAt
        have h_xpad_const_diff : DifferentiableAt ℝ
            (fun (_ : Vec (c * kH * kW)) (k'''' : Fin (c * h * w)) =>
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := khh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
               let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                   ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) (Tensor3.flatten W) :=
          differentiableAt_const _
        rw [pdiv_mul _ _ _ h_reindex_diff h_xpad_const_diff]
        rw [show (fun (v''''' : Vec (c * kH * kW))
                      (k'''' : Fin (c * h * w)) =>
                  v''''' (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1, khh),
                      kww))) =
              (fun y k'''' =>
                y ((fun k''''' : Fin (c * h * w) =>
                  finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, khh),
                      kww)) k'''')) from rfl]
        rw [pdiv_reindex (fun k''''' : Fin (c * h * w) =>
            finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, khh),
                kww))]
        rw [show pdiv (fun (_ : Vec (c * kH * kW))
                          (k'''' : Fin (c * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                       ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0))
                (Tensor3.flatten W)
                (finProdFinEquiv (finProdFinEquiv (ci, hi_k), wi_k))
                (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = 0
            from pdiv_const _ _ _ _]
        -- Goal after pdiv_reindex/pdiv_const: (if (idx_W) = (idx_at(co, khh, kww)) then 1 else 0) * x_pad + x at idx_W * 0
        -- where idx_at(co, khh, kww) = finProdFinEquiv (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1, khh), kww)
        -- And finProdFinEquiv.symm (finProdFinEquiv (finProdFinEquiv (co, ho), wo)).symm... gives co.
        simp only [Equiv.symm_apply_apply]
        ring
      simp_rw [h_inner_kh]
      -- Now: ∑ kh kw, (if (idx_W of (ci, hi_k, wi_k)) = (idx_W of (co, kh, kw)) then 1 else 0) * x_pad
      -- = if co = ci then x_pad(ci, hi_k, wi_k, ho, wo) else 0.
      -- The condition reduces (by injectivity) to (ci, hi_k, wi_k) = (co, kh, kw).
      have h_indicator : ∀ kh : Fin kH, ∀ kw : Fin kW,
          ((finProdFinEquiv (finProdFinEquiv (ci, hi_k), wi_k) : Fin (c * kH * kW)) =
            finProdFinEquiv (finProdFinEquiv (co, kh), kw)) ↔
          (ci = co ∧ hi_k = kh ∧ wi_k = kw) := by
        intro kh kw
        constructor
        · intro h
          have hpair := finProdFinEquiv.injective h
          have hpair2 := finProdFinEquiv.injective (Prod.mk.inj hpair).1
          refine ⟨?_, ?_, ?_⟩
          · exact (Prod.mk.inj hpair2).1
          · exact (Prod.mk.inj hpair2).2
          · exact (Prod.mk.inj hpair).2
        · rintro ⟨hci, hhi, hwi⟩
          rw [hci, hhi, hwi]
      simp_rw [h_indicator]
      -- Triple → double sum collapse via Finset.sum_eq_single twice (kh = hi_k, kw = wi_k).
      rw [Finset.sum_eq_single hi_k
            (fun kh _ hkh_ne =>
              Finset.sum_eq_zero (fun kw _ => by
                rw [if_neg (fun ⟨_, hhi, _⟩ => hkh_ne hhi.symm), zero_mul]))
            (fun hni => absurd (Finset.mem_univ hi_k) hni)]
      rw [Finset.sum_eq_single wi_k
            (fun kw _ hkw_ne => by
              rw [if_neg (fun ⟨_, _, hwi⟩ => hkw_ne hwi.symm), zero_mul])
            (fun hni => absurd (Finset.mem_univ wi_k) hni)]
      -- Goal: (if (ci = co ∧ hi_k = hi_k ∧ wi_k = wi_k) then 1 else 0) * x_pad
      --       = if co = ci then x_pad(ci) else 0
      by_cases h_c : co = ci
      · rw [if_pos ⟨h_c.symm, rfl, rfl⟩, one_mul, if_pos h_c]
        -- After if_pos, both sides have the form `let pH := ...` containing `x ci ...` (LHS)
        -- and `x ci ...` (RHS). They should be definitionally equal since co = ci.
        rw [h_c]
      · rw [if_neg (fun ⟨h, _⟩ => h_c h.symm), zero_mul, if_neg h_c]
    -- Step 2: collapse the triple sum using h_pdiv3.
    show (∑ ho : Fin h, ∑ wo : Fin w,
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := hi_k.val + ho.val
             let ww := wi_k.val + wo.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
            * dy ci ho wo) =
          ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
            pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x)
              W ci hi_k wi_k co ho wo * dy co ho wo
    simp_rw [h_pdiv3]
    -- Now: LHS = ∑ co ho wo, (if co = ci then x_pad else 0) * dy co ho wo. Collapse co.
    rw [Finset.sum_eq_single ci
          (fun co _ hco_ne =>
            Finset.sum_eq_zero (fun ho _ =>
              Finset.sum_eq_zero (fun wo _ => by
                rw [if_neg hco_ne, zero_mul])))
          (fun hni => absurd (Finset.mem_univ ci) hni)]
    -- Now: LHS = ∑ ho wo, (if ci = ci then x_pad(ci) else 0) * dy ci ho wo. The if is true.
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    rw [if_pos rfl]

/-- Named accessor for the depthwise weight backward. -/
noncomputable abbrev depthwiseConv2d_weight_grad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : DepthwiseKernel c kH kW :=
  (depthwise_weight_grad_has_vjp3 b x).backward W dy

/-- **Depthwise bias-VJP** — proved from foundation rules. Same shape
    as `conv2d_bias_grad_has_vjp`, just simpler: depthwise has no
    Σ over input channels (input channel = output channel). The
    function `b ↦ flatten(depthwiseConv2d W b x)` decomposes as
    `(channel-reindex from b) + (W,x term constant in b)`, exactly
    like conv2d's case. -/
noncomputable def depthwise_bias_grad_has_vjp {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) :
    HasVJP (fun b : Vec c => Tensor3.flatten (depthwiseConv2d W b x)) where
  backward := fun _b dy => fun cc =>
    ∑ hi : Fin h, ∑ wi : Fin w,
      dy (finProdFinEquiv (finProdFinEquiv (cc, hi), wi))
  correct := by
    intro b dy cc
    have h_pdiv : ∀ idx : Fin (c * h * w),
        pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc idx =
        (if cc = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1
          then (1:ℝ) else 0) := by
      intro idx
      rw [show (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) =
            (fun b' k =>
              (fun y : Vec c => fun k' : Fin (c * h * w) =>
                y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b' k +
              (fun (_ : Vec c) (k' : Fin (c * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                         ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) b' k) from by
        funext b' k
        unfold Tensor3.flatten depthwiseConv2d
        rfl]
      have h_reindex_diff : DifferentiableAt ℝ
          (fun y : Vec c => fun k' : Fin (c * h * w) =>
            y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b :=
        (reindexCLM (fun k' : Fin (c * h * w) =>
          (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)).differentiableAt
      have h_const_diff : DifferentiableAt ℝ
          (fun (_ : Vec c) (k' : Fin (c * h * w)) =>
            ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                     ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) b :=
        differentiableAt_const _
      rw [pdiv_add _ _ _ h_reindex_diff h_const_diff]
      rw [show (fun y : Vec c => fun k' : Fin (c * h * w) =>
                  y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) =
            (fun y => fun k' => y ((fun k'' : Fin (c * h * w) =>
                (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) k')) from rfl]
      rw [pdiv_reindex (fun k'' : Fin (c * h * w) =>
            (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)]
      rw [show pdiv (fun (_ : Vec c) (k' : Fin (c * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)
                           ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0))
                  b cc idx = 0
          from pdiv_const _ _ _ _]
      ring
    simp_rw [h_pdiv]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx : Fin (c * h * w) =>
          (if cc = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1 then (1:ℝ) else 0)
          * dy idx)
        (fun pair : Fin (c * h) × Fin w =>
          (if cc = (finProdFinEquiv.symm pair.1).1 then (1:ℝ) else 0)
          * dy (finProdFinEquiv pair))
        (fun idx => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun pair_h : Fin (c * h) =>
          ∑ wi : Fin w,
            (if cc = (finProdFinEquiv.symm pair_h).1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (pair_h, wi)))
        (fun ch_pair : Fin c × Fin h =>
          ∑ wi : Fin w,
            (if cc = ch_pair.1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (finProdFinEquiv ch_pair, wi)))
        (fun pair_h => by
          have h_inv : finProdFinEquiv (finProdFinEquiv.symm pair_h
                : Fin c × Fin h) = pair_h :=
            Equiv.apply_symm_apply _ _
          simp_rw [h_inv])]
    rw [Fintype.sum_prod_type]
    have h_pull : ∀ cc' : Fin c,
        (∑ hi : Fin h, ∑ wi : Fin w,
          (if cc = cc' then (1:ℝ) else 0)
          * dy (finProdFinEquiv (finProdFinEquiv (cc', hi), wi))) =
        (if cc = cc' then (1:ℝ) else 0) *
          ∑ hi : Fin h, ∑ wi : Fin w,
            dy (finProdFinEquiv (finProdFinEquiv (cc', hi), wi)) := by
      intro cc'
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro hi _
      rw [Finset.mul_sum]
    simp_rw [h_pull, ite_mul, one_mul, zero_mul]
    rw [Finset.sum_ite_eq Finset.univ cc (fun cc' =>
        ∑ hi : Fin h, ∑ wi : Fin w,
          dy (finProdFinEquiv (finProdFinEquiv (cc', hi), wi)))]
    simp

/-- Named accessor for the depthwise bias backward via the VJP framework. -/
noncomputable def depthwiseConv2d_bias_grad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : Vec c :=
  (depthwise_bias_grad_has_vjp W x).backward b (Tensor3.flatten dy)

/-- **Depthwise bias gradient — closed-form formula** (documented, numerically
    verified, expected to equal `depthwiseConv2d_bias_grad` up to fp precision).

    `db[c] = Σ_{h, w} dy[c, h, w]`

    Identical to regular conv's bias gradient — the bias is per-channel
    in both cases, and it adds the same value to every spatial cell
    of its channel. The reduction is the same. -/
noncomputable def depthwiseConv2d_bias_grad_formula {c h w : Nat}
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

**None.** The forward `depthwiseConv2d` is a concrete definition (not a
black-box), and all three VJPs are theorems proved from the foundation
rules in `Tensor.lean`:

- `depthwise_has_vjp3` — input-path VJP. Phase 2 (Apr 2026): proved from
  `pdiv_add` / `pdiv_const` / `pdiv_finset_sum` / `pdiv_const_mul_pi_pad_eval`.
  Mirrors `conv2d_has_vjp3` with one fewer sum level (no Σ c) and a
  prepended Σ co collapse.
- `depthwise_weight_grad_has_vjp3` — weight-path VJP, bundled as
  `HasVJP3` directly (no flattening needed; see framework note above).
  Gradient-checked numerically.
- `depthwise_bias_grad_has_vjp` — bias-path VJP, bundled `HasVJP` on
  the flattened output. Same pattern as conv2d's bias VJP.

Pure-Mathlib closure verified via `#print axioms` (only `propext`,
`Classical.choice`, `Quot.sound`).

Derived helpers (not axioms):
- `depthwiseConv2d_input_grad`, `depthwiseConv2d_weight_grad`,
  `depthwiseConv2d_bias_grad` — named accessors, `.backward` of the
  corresponding VJP.
- `depthwiseConv2d_input_grad_formula` — the concrete sum-over-output-
  positions closed-form, used as the backward of `depthwise_has_vjp3`.
- `depthwiseConv2d_bias_grad_formula` — the concrete sum-over-spatial
  closed-form (numerically verified to equal the bias-VJP's backward). -/

/-- **Public correctness theorem for `depthwise_has_vjp3`**: the
proved input-VJP's backward equals the `pdiv3`-contracted Jacobian. -/
theorem depthwise_has_vjp3_correct {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    (depthwise_has_vjp3 (h := h) (w := w) W b).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  (depthwise_has_vjp3 (h := h) (w := w) W b).correct x dy ci hi wi

end Proofs
