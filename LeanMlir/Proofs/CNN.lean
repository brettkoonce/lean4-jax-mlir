import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP

/-!
# CNN VJP Proofs

VJP correctness for the convolutional and pooling layers used in
`mlir_poc/hand_cnn_train_step.mlir`. The architecture there is:

    x(1,28,28) → Conv(1→32) → ReLU → Conv(32→32) → ReLU → MaxPool
              → Flatten → Dense(6272→512) → ReLU → Dense(512→512)
              → ReLU → Dense(512→10) → logits

The dense and ReLU layers are inherited from `MLP.lean`. This file
adds the new operations: conv2d, max-pool, and flatten.

## The big idea: conv backward IS conv

The most pedagogically valuable result is that the VJP of a convolution
is itself expressible as **convolutions** — with appropriate kernel
reversal and axis transposition. This is why conv layers train
efficiently: there's no special backward operator. The same primitive
runs in both directions.

Two specific tricks appear in the MLIR:

1. **Input-gradient via reversed kernel**: `dx = conv(dy, reverse(Wᵀ))`
2. **Weight-gradient via the transpose trick**: `dW = conv(xᵀ, dyᵀ)`,
   where the spatial dims of the gradient become the "kernel".

We state the VJP formulas as axioms (the proofs are standard matrix
calculus on cross-correlations), and the commentary explains why each
formula has the form it does.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Tensor types for CNN
-- ════════════════════════════════════════════════════════════════

-- Tensor3 is imported from Tensor.lean

/-- A conv kernel: out_channels × in_channels × kH × kW.
    This is the OIHW layout used by StableHLO and IREE. -/
abbrev Kernel4 (oc ic kh kw : Nat) :=
  Fin oc → Fin ic → Fin kh → Fin kw → ℝ

namespace Kernel4

/-! `Kernel4 oc ic kH kW` and `Vec (oc * ic * kH * kW)` are in bijection
    by row-major flattening — mirrors `Mat.flatten` / `Tensor3.flatten`.
    We need this so that the weight-gradient VJP can be stated as a plain
    `HasVJP` (Vec → Vec) on the flattened kernel, reusing the existing
    framework instead of introducing a parallel 4D machinery.

    Nat multiplication associates left, so `oc * ic * kH * kW` parses as
    `((oc * ic) * kH) * kW` — three nested `finProdFinEquiv` calls. -/

/-- Row-major flatten: `Kernel4 oc ic kH kW → Vec (oc * ic * kH * kW)`. -/
noncomputable def flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Vec (oc * ic * kH * kW) :=
  fun k =>
    let ockH_kW := finProdFinEquiv.symm k         -- : Fin (oc*ic*kH) × Fin kW
    let ocic_kH := finProdFinEquiv.symm ockH_kW.1 -- : Fin (oc*ic) × Fin kH
    let oc_ic   := finProdFinEquiv.symm ocic_kH.1 -- : Fin oc × Fin ic
    W oc_ic.1 oc_ic.2 ocic_kH.2 ockH_kW.2

/-- Row-major unflatten: inverse of `flatten`. -/
noncomputable def unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : Kernel4 oc ic kH kW :=
  fun o c kh kw =>
    v (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw))

theorem unflatten_flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : unflatten (flatten W) = W := by
  funext o c kh kw
  unfold unflatten flatten
  simp [Equiv.symm_apply_apply]

theorem flatten_unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : flatten (unflatten v) = v := by
  funext k
  -- After `change`, Lean's Prod struct-eta already collapses the innermost
  -- pair `(c.1, c.2)` to `c = fPF.symm (..).1`, so we start by collapsing
  -- that innermost `fPF (fPF.symm ...)` directly. Two more round-trips follow,
  -- each needing an explicit Prod-eta `show` + another `Equiv.apply_symm_apply`.
  change v (finProdFinEquiv
    (finProdFinEquiv
      (finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1),
       (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2),
     (finProdFinEquiv.symm k).2)) = v k
  rw [Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1,
            (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2) =
           finProdFinEquiv.symm (finProdFinEquiv.symm k).1 from rfl,
      Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2) =
           finProdFinEquiv.symm k from rfl,
      Equiv.apply_symm_apply]

end Kernel4

-- ════════════════════════════════════════════════════════════════
-- § Conv2d
-- ════════════════════════════════════════════════════════════════

/-- **Conv2d forward** (SAME padding, stride 1).

    `y[o, h, w] = (Σ_{c, kh, kw} x[c, h+kh−p, w+kw−p] · W[o, c, kh, kw]) + b[o]`

    where `p = (kH−1)/2` is the padding offset and out-of-bounds reads
    return 0 (zero padding). The output spatial size equals the input.

    Note: this is technically *cross-correlation*, not convolution in the
    strict signal-processing sense. ML literature uses "convolution" loosely;
    the difference (kernel flipping) only matters when comparing against
    classical signal-processing references.

    MLIR (`hand_cnn_train_step.mlir`):
      %cv0 = "stablehlo.convolution"(%x, %W0) {
        padding = dense<[[1, 1], [1, 1]]>, ...
      }
      %h0pre = stablehlo.add %cv0, broadcast(%b0) -/
noncomputable def conv2d {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) : Tensor3 oc h w :=
  fun o hi wi =>
    b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      W o c kh kw *
        (let pH := (kH - 1) / 2
         let pW := (kW - 1) / 2
         let hh := kh.val + hi.val
         let ww := kw.val + wi.val
         if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
           x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
         else 0)

/-- **Differentiability of an `if hpad : P then v(σ hpad) else 0` term.**
    The dependent-`if` would otherwise stymie `fun_prop`. By proof
    irrelevance, the chosen branch is a CLM (eval-at-σ if `P`, the
    constant `0` otherwise) — both differentiable in `v`. -/
lemma differentiableAt_pad_eval {n : Nat} (P : Prop) [Decidable P]
    (σ : P → Fin n) (v : Vec n) :
    DifferentiableAt ℝ (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) v := by
  by_cases hP : P
  · rw [show (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) =
            (fun y => y (σ hP)) from by funext y; rw [dif_pos hP]]
    fun_prop
  · rw [show (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) =
            (fun _ => (0 : ℝ)) from by funext y; rw [dif_neg hP]]
    exact differentiableAt_const _

/-- **Pdiv of a per-output dependent if-eval-or-zero family.**
    Given a per-output dependent if `fun v k' ↦ if h : P k' then v (σ k' h) else 0`,
    its pdiv at `(idx_in, idx_out)` is the indicator
    `if P holds at idx_out ∧ σ matches idx_in then 1 else 0`. The proof uses
    `fderiv_apply` to extract the `idx_out`-th component, then `by_cases` on
    `P idx_out` to discharge the dependent-if. -/
lemma pdiv_pi_pad_eval {n m : Nat}
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            if h : P k' then v' (σ k' h) else (0 : ℝ))
          v idx_in idx_out =
    if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0 := by
  unfold pdiv
  have h_diff_pi : DifferentiableAt ℝ (fun (v' : Vec n) (k' : Fin m) =>
      if h : P k' then v' (σ k' h) else (0 : ℝ)) v := by
    rw [differentiableAt_pi]
    intro k'
    exact differentiableAt_pad_eval (P k') (σ k') v
  rw [show fderiv ℝ (fun (v' : Vec n) (k' : Fin m) =>
              if h : P k' then v' (σ k' h) else (0 : ℝ)) v (basisVec idx_in) idx_out
        = fderiv ℝ (fun v' : Vec n =>
            (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out)
            v (basisVec idx_in) from by
    rw [fderiv_apply h_diff_pi idx_out]; rfl]
  by_cases hpad : P idx_out
  · rw [show (fun v' : Vec n =>
          (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out) =
        (fun v' : Vec n => v' (σ idx_out hpad)) from by
      funext v'
      show (if h : P idx_out then v' (σ idx_out h) else (0 : ℝ)) = v' (σ idx_out hpad)
      rw [dif_pos hpad]]
    rw [show (fun v' : Vec n => v' (σ idx_out hpad)) =
          ((ContinuousLinearMap.proj (σ idx_out hpad) : Vec n →L[ℝ] ℝ) : Vec n → ℝ)
        from rfl]
    rw [ContinuousLinearMap.fderiv]
    show (ContinuousLinearMap.proj (σ idx_out hpad) : Vec n →L[ℝ] ℝ) (basisVec idx_in) = _
    rw [ContinuousLinearMap.proj_apply, basisVec_apply, dif_pos hpad]
  · rw [show (fun v' : Vec n =>
          (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out) =
        (fun _ => (0 : ℝ)) from by
      funext v'
      show (if h : P idx_out then v' (σ idx_out h) else (0 : ℝ)) = 0
      rw [dif_neg hpad]]
    rw [(hasFDerivAt_const (0 : ℝ) v).fderiv, dif_neg hpad]
    rfl

/-- **Pdiv of `c_const * pad-eval` family.** Combines `pdiv_mul`,
    `pdiv_const`, and `pdiv_pi_pad_eval` for the conv2d per-summand
    pattern: a `k'`-varying constant times the dependent if-eval-or-zero. -/
lemma pdiv_const_mul_pi_pad_eval {n m : Nat}
    (c_const : Fin m → ℝ)
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            c_const k' *
            (if h : P k' then v' (σ k' h) else (0 : ℝ)))
          v idx_in idx_out =
    c_const idx_out *
    (if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0) := by
  have h_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (k' : Fin m) => c_const k') v := differentiableAt_const _
  have h_pad_diff : DifferentiableAt ℝ
      (fun (v' : Vec n) (k' : Fin m) =>
        if h : P k' then v' (σ k' h) else (0 : ℝ)) v := by
    rw [differentiableAt_pi]
    intro k'
    exact differentiableAt_pad_eval (P k') (σ k') v
  rw [pdiv_mul _ _ _ h_const_diff h_pad_diff]
  rw [show pdiv (fun (_ : Vec n) (k' : Fin m) => c_const k') v idx_in idx_out = 0
      from pdiv_const _ _ _ _]
  rw [pdiv_pi_pad_eval]
  ring

/-- **Closed-form input gradient for conv2d** — direct formula, written as
    a sum over output positions `(co, ho, wo)` with reconstructed kernel
    offsets `kh_nat = hi + pH − ho`, `kw_nat = wi + pW − wo`. The body is
    nonzero only when the reconstructed `(kh_nat, kw_nat)` lies in
    `[0, kH) × [0, kW)` — i.e., when the input position `(hi, wi)` is
    actually reachable from output `(ho, wo)` via some valid kernel offset.
    Equivalent (under the `(ho, wo) ↔ (kh, kw)` partial bijection
    `ho = hi+pH-kh`) to the MLIR-aligned "reversed-kernel" formula
    `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] · dy[o, h+kh−p, w+kw−p]`. -/
noncomputable def conv2d_input_grad_formula {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  fun ci hi wi =>
    ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧ wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy co ho wo
      else 0

/-- **Conv2d input-VJP** — proved from foundation rules.

    The function `v ↦ flatten (conv2d W b (unflatten v))` is affine in
    `v`: a constant `b o(idx_out)` plus a triple sum over `(c, kh, kw)`
    of `W o(idx_out) c kh kw * (if pad-cond then v(reindex) else 0)`.
    Each summand factors as `(constant W) * (if-pad-conditional in v)`,
    so `pdiv_add` + `pdiv_const` + `pdiv_finset_sum` (×3) + `pdiv_mul` +
    a `by_cases` on the pad condition (CLM-projection on the pad-true
    branch, constant zero on the pad-false branch) collapse the
    per-`(idx_in, idx_out)` pdiv. Reindex `Fin (oc*h*w) ↔ Fin oc × Fin h × Fin w`
    on the sum-over-`idx_out`, then a triple `Finset.sum_eq_single` over
    `(c, kh, kw)` (matching `idx_in`'s decoded `(ci, hi, wi)`) gives the
    closed-form input gradient `conv2d_input_grad_formula`.

    The backward function (accessed as `(conv2d_has_vjp3 W b).backward`,
    or via the `conv2d_input_grad` abbrev below) implements
    `conv2d_input_grad_formula W dy ci hi wi` — a direct sum over
    `(co, kh, kw)` of `W co ci kh kw * dy co ho_nat wo_nat` for valid
    `(ho_nat, wo_nat)`. Equivalent (under `kh ↔ kH−1−kh`) to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] · dy[o, h+kh−p, w+kw−p]`.

    MLIR emits the reversed-kernel form directly:
      %W1_t   = stablehlo.transpose %W1, dims = [1, 0, 2, 3]   -- swap oc↔ic
      %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3]         -- flip spatial
      %d_h0   = "stablehlo.convolution"(%d_h1pre, %W1_rev) ...
-/
noncomputable def conv2d_has_vjp3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP3 (conv2d W b : Tensor3 ic h w → Tensor3 oc h w) where
  backward := fun _x dy => conv2d_input_grad_formula W dy
  correct := by
    intro x dy ci hi wi
    -- Set abbreviation for idx_in (the input position we're computing grad at).
    set idx_in : Fin (ic * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in
    -- Step 1: per-`(idx_in, idx_out)` pdiv formula. We keep the UN-collapsed
    -- form `∑ c kh kw, W * indicator` because the closing reindex collapses
    -- it into the natural `(co, kh, kw)` loop without needing a partial
    -- bijection between `Fin h` and `Fin kH`.
    have h_pdiv : ∀ idx_out : Fin (oc * h * w),
        pdiv (fun v' : Vec (ic * h * w) =>
                Tensor3.flatten (conv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1) c kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   (c, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                 (1 : ℝ) else 0)
             else 0) := by
      intro idx_out
      -- Set abbreviations for idx_out components (idx_out unpacks to (ohw_o, ohw_hi, ohw_wi)).
      set ohw_wi : Fin w := (finProdFinEquiv.symm idx_out).2 with hohw_wi
      set ohw_hi : Fin h :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2 with hohw_hi
      set ohw_o : Fin oc :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 with hohw_o
      -- Decompose `f = (constant b) + (sum over c kh kw of W * if-pad-cond)`.
      rw [show (fun v' : Vec (ic * h * w) =>
                Tensor3.flatten (conv2d W b (Tensor3.unflatten v'))) =
            (fun v' k =>
              (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) (Tensor3.flatten x) :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) (Tensor3.flatten x) := by
        rw [differentiableAt_pi]
        intro k'
        apply DifferentiableAt.fun_sum; intro c _
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (c, ⟨kh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                (Tensor3.flatten x) idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute pdiv over the c-sum.
      rw [show (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ c : Fin ic,
              (fun (cc : Fin ic) (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) c v'' k') from rfl]
      have h_c_diff : ∀ cc ∈ (Finset.univ : Finset (Fin ic)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
              ∑ kh : Fin kH, ∑ kw : Fin kW,
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten x) := by
        intro cc _
        rw [differentiableAt_pi]
        intro k''
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨kh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k'').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_finset_sum _ _ _ h_c_diff]
      -- Now distribute over kh and kw inside each c-summand.
      have h_inner_c : ∀ cc : Fin ic,
          pdiv (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) (Tensor3.flatten x) idx_in idx_out =
          ∑ kh : Fin kH, ∑ kw : Fin kW,
            W ohw_o cc kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val + ohw_hi.val
             let ww := kw.val + ohw_wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   (cc, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then (1 : ℝ) else 0)
             else 0) := by
        intro cc
        -- Distribute over kh.
        rw [show (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v''' k'' => ∑ kh : Fin kH,
                (fun (khh : Fin kH) (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) kh v''' k'') from rfl]
        have h_kh_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
            DifferentiableAt ℝ
              (fun (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) (Tensor3.flatten x) := by
          intro khh _
          rw [differentiableAt_pi]
          intro k'''
          apply DifferentiableAt.fun_sum; intro kw _
          apply DifferentiableAt.mul (differentiableAt_const _)
          unfold Tensor3.unflatten
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + (finProdFinEquiv.symm
                (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kw.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kh_diff]
        congr 1; ext khh
        -- Distribute over kw.
        rw [show (fun (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v'''' k''' => ∑ kw : Fin kW,
                (fun (kww : Fin kW) (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kw v'''' k''') from rfl]
        have h_kw_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
            DifferentiableAt ℝ
              (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) (Tensor3.flatten x) := by
          intro kww _
          rw [differentiableAt_pi]
          intro k''''
          apply DifferentiableAt.mul (differentiableAt_const _)
          unfold Tensor3.unflatten
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + (finProdFinEquiv.symm
                (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kw_diff]
        congr 1; ext kww
        -- Per-(cc, khh, kww) summand: factor as (W constant) * (dite in v).
        -- After unfolding `Tensor3.unflatten`, the inner becomes
        -- `if hpad : ... then v(σ) else 0`, fitting `pdiv_const_mul_pi_pad_eval`.
        rw [show (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
              (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                (fun k''''' : Fin (oc * h * w) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1) cc khh kww) k'''' *
                (if hpad : (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
                              (finProdFinEquiv.symm k'''').1).2.val ∧
                            khh.val + (finProdFinEquiv.symm
                              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2 < h ∧
                            (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k'''').2.val ∧
                            kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2 < w then
                  v''''' (finProdFinEquiv (finProdFinEquiv
                    (cc, ⟨khh.val + (finProdFinEquiv.symm
                            (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
                    ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2,
                      hpad.2.2.2⟩))
                else 0)) from by
          funext v''''' k''''
          unfold Tensor3.unflatten
          rfl]
        rw [pdiv_const_mul_pi_pad_eval
          (fun k''''' : Fin (oc * h * w) =>
            W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1) cc khh kww)
          (fun k'''' => (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val ∧
            khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2 < h ∧
            (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k'''').2.val ∧
            kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2 < w)
          (fun k'''' hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2, hpad.2.2.2⟩))]
        -- Show the result matches the desired form (with ohw_o, ohw_hi, ohw_wi abbreviations).
        show W ohw_o cc khh kww * _ = W ohw_o cc khh kww * _
        congr 1
        -- Goal: dite-form on the LHS = dite-form on the RHS (after symmetrizing the
        -- equality direction: σ idx_out h = idx_in vs idx_in = ...).
        by_cases hpad : (kH - 1) / 2 ≤ khh.val + ohw_hi.val ∧
                       khh.val + ohw_hi.val - (kH - 1) / 2 < h ∧
                       (kW - 1) / 2 ≤ kww.val + ohw_wi.val ∧
                       kww.val + ohw_wi.val - (kW - 1) / 2 < w
        · rw [dif_pos hpad, dif_pos hpad]
          by_cases heq : finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + ohw_hi.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kww.val + ohw_wi.val - (kW - 1) / 2, hpad.2.2.2⟩) = idx_in
          · rw [if_pos heq, if_pos heq.symm]
          · rw [if_neg heq, if_neg (fun h => heq h.symm)]
        · rw [dif_neg hpad, dif_neg hpad]
      simp_rw [h_inner_c]
    -- Step 2: substitute h_pdiv into the RHS sum and collapse.
    show conv2d_input_grad_formula W dy ci hi wi =
      ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (conv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold conv2d_input_grad_formula pdiv3
    apply Finset.sum_congr rfl; intro co _
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    -- For each (co, ho, wo), substitute h_pdiv at idx_out := flat(co, ho, wo).
    rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (co, ho), wo))]
    -- Simplify the decoding (Equiv.symm_apply_apply ⊢ ohw_o = co, ohw_hi = ho, ohw_wi = wo).
    simp only [Equiv.symm_apply_apply]
    -- Pull `dy co ho wo` out of the formula's if-true branch.
    rw [show (let pH := (kH - 1) / 2
              let pW := (kW - 1) / 2
              let kh_nat := hi.val + pH - ho.val
              let kw_nat := wi.val + pW - wo.val
              if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                  wo.val ≤ wi.val + pW ∧ kw_nat < kW then
                W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy co ho wo
              else 0) =
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let kh_nat := hi.val + pH - ho.val
             let kw_nat := wi.val + pW - wo.val
             if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
               W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩
             else 0) * dy co ho wo from by
      by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                     hi.val + (kH - 1) / 2 - ho.val < kH ∧
                     wo.val ≤ wi.val + (kW - 1) / 2 ∧
                     wi.val + (kW - 1) / 2 - wo.val < kW
      · simp only [dif_pos hb]
      · simp only [dif_neg hb, zero_mul]]
    -- Goal: (if hb : back_cond then W co ci ⟨kh*⟩ ⟨kw*⟩ else 0) * dy co ho wo
    --     = (∑ c kh kw, W co c kh kw * indicator) * dy co ho wo
    congr 1
    -- Convert the dependent-if indicator to a non-dependent conjunction-form.
    have h_indicator : ∀ (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
        ((let pH := (kH - 1) / 2
          let pW := (kW - 1) / 2
          let hh := kh.val + ho.val
          let ww := kw.val + wo.val
          if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
            (if idx_in = finProdFinEquiv (finProdFinEquiv
                (c, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then (1 : ℝ) else 0)
          else 0) : ℝ) =
        (if c = ci ∧ kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                     kw.val + wo.val = wi.val + (kW - 1) / 2 then (1 : ℝ) else 0) := by
      intro c kh kw
      by_cases hpad : (kH - 1) / 2 ≤ kh.val + ho.val ∧
                      kh.val + ho.val - (kH - 1) / 2 < h ∧
                      (kW - 1) / 2 ≤ kw.val + wo.val ∧
                      kw.val + wo.val - (kW - 1) / 2 < w
      · rw [dif_pos hpad]
        by_cases h_match : c = ci ∧ kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                                    kw.val + wo.val = wi.val + (kW - 1) / 2
        · -- Build the explicit Fin equality for the indicator's RHS.
          have h_idx_in_eq : idx_in = finProdFinEquiv (finProdFinEquiv
              (c, ⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩) := by
            rw [hidx_in]
            have h_c : c = ci := h_match.1
            have h_hi : (⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩ : Fin h) = hi := by
              apply Fin.ext
              show kh.val + ho.val - (kH - 1) / 2 = hi.val
              omega
            have h_wi : (⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩ : Fin w) = wi := by
              apply Fin.ext
              show kw.val + wo.val - (kW - 1) / 2 = wi.val
              omega
            rw [← h_c, ← h_hi, ← h_wi]
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
          refine ⟨h_inj_inner_pair.1.symm, ?_, ?_⟩
          · have h_hi : hi.val = kh.val + ho.val - (kH - 1) / 2 :=
              Fin.ext_iff.mp h_inj_inner_pair.2
            omega
          · have h_wi : wi.val = kw.val + wo.val - (kW - 1) / 2 :=
              Fin.ext_iff.mp h_inj_pair.2
            omega
      · rw [dif_neg hpad]
        rw [if_neg]
        intro ⟨_, hkh_eq, hkw_eq⟩
        apply hpad
        refine ⟨?_, ?_, ?_, ?_⟩
        · rw [hkh_eq]; exact Nat.le_add_left _ _
        · rw [hkh_eq, Nat.add_sub_cancel]; exact hi.isLt
        · rw [hkw_eq]; exact Nat.le_add_left _ _
        · rw [hkw_eq, Nat.add_sub_cancel]; exact wi.isLt
    simp_rw [h_indicator]
    -- Goal: (if hb : back_cond then W co ci ⟨kh*⟩ ⟨kw*⟩ else 0)
    --     = ∑ c kh kw, W co c kh kw * (if c = ci ∧ ... then 1 else 0)
    by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                   hi.val + (kH - 1) / 2 - ho.val < kH ∧
                   wo.val ≤ wi.val + (kW - 1) / 2 ∧
                   wi.val + (kW - 1) / 2 - wo.val < kW
    · rw [dif_pos hb]
      -- Σ c collapses on c = ci, then Σ kh on kh = ⟨hi+pH-ho, hb.2.1⟩, then Σ kw similarly.
      symm
      rw [Finset.sum_eq_single ci ?_ ?_]
      rw [Finset.sum_eq_single ⟨hi.val + (kH - 1) / 2 - ho.val, hb.2.1⟩ ?_ ?_]
      rw [Finset.sum_eq_single ⟨wi.val + (kW - 1) / 2 - wo.val, hb.2.2.2⟩ ?_ ?_]
      · -- Main: W co ci ⟨kh*⟩ ⟨kw*⟩ * (if ci=ci ∧ ... then 1 else 0) = W co ci ⟨kh*⟩ ⟨kw*⟩
        rw [if_pos]
        · ring
        refine ⟨rfl, ?_, ?_⟩
        · show hi.val + (kH - 1) / 2 - ho.val + ho.val = hi.val + (kH - 1) / 2
          omega
        · show wi.val + (kW - 1) / 2 - wo.val + wo.val = wi.val + (kW - 1) / 2
          omega
      · intro kw _ hkw_ne
        rw [if_neg ?_]; · ring
        intro ⟨_, _, hkw_eq⟩
        apply hkw_ne
        apply Fin.ext
        show kw.val = wi.val + (kW - 1) / 2 - wo.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro kh _ hkh_ne
        apply Finset.sum_eq_zero; intro kw _
        rw [if_neg ?_]; · ring
        intro ⟨_, hkh_eq, _⟩
        apply hkh_ne
        apply Fin.ext
        show kh.val = hi.val + (kH - 1) / 2 - ho.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro c _ hc_ne
        apply Finset.sum_eq_zero; intro kh _
        apply Finset.sum_eq_zero; intro kw _
        rw [if_neg (fun ⟨hcc, _, _⟩ => hc_ne hcc)]; ring
      · intro hni; exact absurd (Finset.mem_univ ci) hni
    · rw [dif_neg hb]
      -- Show the inner sum is 0: for !back_cond, no (c, kh, kw) satisfies the indicator.
      symm
      apply Finset.sum_eq_zero; intro c _
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [if_neg ?_]; · ring
      intro ⟨_, hkh_eq, hkw_eq⟩
      apply hb
      refine ⟨?_, ?_, ?_, ?_⟩
      · have := kh.isLt; omega
      · have := kh.isLt; omega
      · have := kw.isLt; omega
      · have := kw.isLt; omega

/-- Named accessor for the conv2d input backward — aligns with MLIR
    codegen (`stablehlo.convolution` in the backward pass). -/
noncomputable abbrev conv2d_input_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  (conv2d_has_vjp3 W b).backward x dy

/-! ### Weight gradient (now proved from foundation via `unfold + fun_prop`)

The conv weight gradient implements the **transpose trick**:

    `dW[o, c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[o, h, w]`

Here's the slick observation: this *is* a convolution, with the input
and gradient playing the roles of "input" and "kernel" respectively.

- View the input `x : (ic, H, W)` as `(ic, 1, H, W)` (treat channels as batch).
- View the gradient `dy : (oc, H, W)` as `(oc, 1, H, W)` (same trick).
- Now do a standard convolution: input shape `(ic, 1, H, W)`, kernel
  shape `(oc, 1, H, W)`. The "spatial" dims of the kernel are H×W (the
  whole image), so the output is the small `(ic, oc, kH, kW)` weight
  gradient — produced by sliding the gradient as a giant kernel.
- Transpose the output `(ic, oc, kH, kW) → (oc, ic, kH, kW)` to match
  the kernel layout.

This avoids needing a separate "convolution-with-funny-dimension-numbers"
op; we use the same forward conv operator, just with shapes reinterpreted.
Critical for backends like IREE that don't accept non-standard
`dimension_numbers` (see `iree-org/iree#21955`).

MLIR (Conv 1 backward — exactly this trick):
    %x_t      = stablehlo.transpose %x, dims = [1, 0, 2, 3]      -- (1,128,28,28)
    %dh0p_t   = stablehlo.transpose %d_h0pre, dims = [1, 0, 2, 3] -- (32,128,28,28)
    %d_W0_raw = "stablehlo.convolution"(%x_t, %dh0p_t) ...        -- (1,32,3,3)
    %d_W0     = stablehlo.transpose %d_W0_raw, dims = [1, 0, 2, 3] -- (32,1,3,3)

**Framework.** `HasVJP3` covered only input→output VJPs. For the
weight gradient we reuse the plain `HasVJP` on `Vec` by flattening
both the kernel (`Kernel4.flatten : Kernel4 → Vec (oc*ic*kH*kW)`) and
the output (`Tensor3.flatten : Tensor3 → Vec (oc*h*w)`). The axiom
asserts existence of a correct backward for the flattened function;
the user-facing `conv2d_weight_grad` wrapper does the flatten / unflatten
housekeeping so callers see the natural `Kernel4` type.

Numerical validation: `check_jacobians.py:test_conv2d_weight_grad`
gradient-checks the transpose-trick formula against finite differences. -/

/-- **Conv2d weight-VJP** — proved from foundation rules.

    The function `v ↦ flatten (conv2d (unflatten v) b x)` is affine in
    `v`: a constant `b o(idx_out)` plus a triple sum over `(c, kh, kw)`
    of `(unflatten v) o(idx_out) c kh kw * x_pad_term`. Each summand
    factors as `(reindex of v) * (x-only constant)`, so `pdiv_add` +
    `pdiv_const` + `pdiv_finset_sum` (×3) + `pdiv_mul` + `pdiv_reindex`
    collapse the per-(idx_in, idx_out) pdiv. Triple-sum collapse via
    `Finset.sum_eq_single` gives the transpose-trick backward
    `dW[o', c', kh', kw'] = Σ_{hi, wi} x_pad_term(...) · dy(flat(o', hi, wi))`. -/
noncomputable def conv2d_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)) where
  backward := fun _v dy => fun idx_in =>
    let kw' := (finProdFinEquiv.symm idx_in).2
    let kh' := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let o' := (finProdFinEquiv.symm
              (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1
    let c' := (finProdFinEquiv.symm
              (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2
    ∑ hi : Fin h, ∑ wi : Fin w,
      (let pH := (kH - 1) / 2
       let pW := (kW - 1) / 2
       let hh := kh'.val + hi.val
       let ww := kw'.val + wi.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0)
      * dy (finProdFinEquiv (finProdFinEquiv (o', hi), wi))
  correct := by
    intro v dy idx_in
    -- Set abbreviations for the unpacked idx_in components (kernel position).
    set kw' : Fin kW := (finProdFinEquiv.symm idx_in).2 with hkw'
    set kh' : Fin kH :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2 with hkh'
    set o' : Fin oc := (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1 with ho'
    set c' : Fin ic := (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2 with hc'
    -- Identity: idx_in unpacks to (o', c', kh', kw') and re-packs back.
    have h_idx_in_eq :
        finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o', c'), kh'), kw') = idx_in := by
      rw [ho', hc', hkh', hkw']
      rw [show ((finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1,
                (finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2) =
               finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1 from rfl,
          Equiv.apply_symm_apply]
      rw [show ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1,
                (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2) =
               finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1 from rfl,
          Equiv.apply_symm_apply]
      rw [show ((finProdFinEquiv.symm idx_in).1, (finProdFinEquiv.symm idx_in).2) =
               finProdFinEquiv.symm idx_in from rfl,
          Equiv.apply_symm_apply]
    -- Step 1: per-idx_out pdiv formula.
    have h_pdiv : ∀ idx_out : Fin (oc * h * w),
        pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v idx_in idx_out =
        (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 = o' then
          (let pH := (kH - 1) / 2
           let pW := (kW - 1) / 2
           let hh := kh'.val +
             (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
           let ww := kw'.val + (finProdFinEquiv.symm idx_out).2.val
           if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
             x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
           else 0)
         else 0) := by
      intro idx_out
      -- Set abbreviations for idx_out components, prevents auto-resugaring.
      set ohw_wi : Fin w := (finProdFinEquiv.symm idx_out).2 with hohw_wi
      set ohw_hi : Fin h :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2 with hohw_hi
      set ohw_o : Fin oc :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 with hohw_o
      -- Decompose f = (constant b) + (sum over c kh kw of (reindex_v * x_pad_const)).
      rw [show (fun v' : Vec (oc * ic * kH * kW) =>
                Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) =
            (fun v' k =>
              (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v'')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              (Kernel4.unflatten v'')
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) v := by
        unfold Kernel4.unflatten; fun_prop
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                v idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute over the triple sum.
      rw [show (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v'')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ c : Fin ic,
              (fun (cc : Fin ic) (v''' : Vec (oc * ic * kH * kW))
                  (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) c v'' k') from rfl]
      have h_c_summand_diff : ∀ cc ∈ (Finset.univ : Finset (Fin ic)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (oc * ic * kH * kW)) (k'' : Fin (oc * h * w)) =>
              ∑ kh : Fin kH, ∑ kw : Fin kW,
                (Kernel4.unflatten v''')
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) v := by
        intro cc _; unfold Kernel4.unflatten; fun_prop
      rw [pdiv_finset_sum _ _ _ h_c_summand_diff]
      have h_inner_c : ∀ cc : Fin ic,
          pdiv (fun (v''' : Vec (oc * ic * kH * kW))
                    (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v idx_in idx_out =
          ∑ kh : Fin kH, ∑ kw : Fin kW,
            (if idx_in = finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              (ohw_o, cc), kh), kw)
              then (1 : ℝ) else 0) *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val + ohw_hi.val
             let ww := kw.val + ohw_wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0) := by
        intro cc
        rw [show (fun (v''' : Vec (oc * ic * kH * kW))
                      (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    (Kernel4.unflatten v''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                      cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v''' k'' => ∑ kh : Fin kH,
                (fun (khh : Fin kH) (v'''' : Vec (oc * ic * kH * kW))
                    (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    (Kernel4.unflatten v'''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                      cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) kh v''' k'') from rfl]
        have h_kh_summand_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
            DifferentiableAt ℝ
              (fun (v'''' : Vec (oc * ic * kH * kW)) (k''' : Fin (oc * h * w)) =>
                ∑ kw : Fin kW,
                  (Kernel4.unflatten v'''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                    cc khh kw *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                   let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) v := by
          intro khh _; unfold Kernel4.unflatten; fun_prop
        rw [pdiv_finset_sum _ _ _ h_kh_summand_diff]
        congr 1; ext khh
        rw [show (fun (v'''' : Vec (oc * ic * kH * kW))
                      (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    (Kernel4.unflatten v'''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                      cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v'''' k''' => ∑ kw : Fin kW,
                (fun (kww : Fin kW) (v''''' : Vec (oc * ic * kH * kW))
                    (k'''' : Fin (oc * h * w)) =>
                  (Kernel4.unflatten v''''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                    cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kw v'''' k''') from rfl]
        have h_kw_summand_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
            DifferentiableAt ℝ
              (fun (v''''' : Vec (oc * ic * kH * kW)) (k'''' : Fin (oc * h * w)) =>
                (Kernel4.unflatten v''''')
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                  cc khh kww *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                 let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) v := by
          intro kww _; unfold Kernel4.unflatten; fun_prop
        rw [pdiv_finset_sum _ _ _ h_kw_summand_diff]
        congr 1; ext kww
        -- Per-summand: factor as (reindex v) * (constant in v).
        rw [show (fun (v''''' : Vec (oc * ic * kH * kW))
                      (k'''' : Fin (oc * h * w)) =>
                  (Kernel4.unflatten v''''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                    cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
              (fun v''''' k'''' =>
                (fun (v'''''' : Vec (oc * ic * kH * kW))
                    (k''''' : Fin (oc * h * w)) =>
                  v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                      khh), kww))) v''''' k'''' *
                (fun (_ : Vec (oc * ic * kH * kW))
                    (k''''' : Fin (oc * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) v''''' k'''') from by
          funext v''''' k''''
          unfold Kernel4.unflatten
          rfl]
        have h_reindex_diff : DifferentiableAt ℝ
            (fun (v'''''' : Vec (oc * ic * kH * kW))
                 (k''''' : Fin (oc * h * w)) =>
              v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                  khh), kww))) v :=
          (reindexCLM (fun k''''' : Fin (oc * h * w) =>
            finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                khh), kww))).differentiableAt
        have h_xpad_const_diff : DifferentiableAt ℝ
            (fun (_ : Vec (oc * ic * kH * kW))
                 (k''''' : Fin (oc * h * w)) =>
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := khh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
               let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) v :=
          differentiableAt_const _
        rw [pdiv_mul _ _ _ h_reindex_diff h_xpad_const_diff]
        rw [show (fun (v'''''' : Vec (oc * ic * kH * kW))
                      (k''''' : Fin (oc * h * w)) =>
                  v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                      khh), kww))) =
              (fun y k''''' =>
                y ((fun k'''''' : Fin (oc * h * w) =>
                  finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''''').1).1, cc),
                      khh), kww)) k''''')) from rfl]
        rw [pdiv_reindex (fun k'''''' : Fin (oc * h * w) =>
            finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''''').1).1, cc),
                khh), kww))]
        rw [show pdiv (fun (_ : Vec (oc * ic * kH * kW))
                          (k''''' : Fin (oc * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0))
                v idx_in idx_out = 0
            from pdiv_const _ _ _ _]
        ring
      simp_rw [h_inner_c]
      -- Now: ∑ c kh kw, (if idx_in = ker_idx(ohw_o, c, kh, kw) then 1 else 0) * xpad
      --      = if ohw_o = o' then xpad(c', kh', kw') else 0.
      -- Rewrite indicator: by injectivity, idx_in = ker_idx(ohw_o, c, kh, kw)
      -- iff (ohw_o, c, kh, kw) = (o', c', kh', kw').
      have h_indicator : ∀ c : Fin ic, ∀ kh : Fin kH, ∀ kw : Fin kW,
          (idx_in = finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              (ohw_o, c), kh), kw)) ↔
          (ohw_o = o' ∧ c = c' ∧ kh = kh' ∧ kw = kw') := by
        intro c kh kw
        constructor
        · intro h
          rw [← h_idx_in_eq] at h
          have hpair := finProdFinEquiv.injective h
          have hpair2 := finProdFinEquiv.injective (Prod.mk.inj hpair).1
          have hpair3 := finProdFinEquiv.injective (Prod.mk.inj hpair2).1
          refine ⟨?_, ?_, ?_, ?_⟩
          · exact (Prod.mk.inj hpair3).1.symm
          · exact (Prod.mk.inj hpair3).2.symm
          · exact (Prod.mk.inj hpair2).2.symm
          · exact (Prod.mk.inj hpair).2.symm
        · rintro ⟨ho_eq, hc_eq, hkh_eq, hkw_eq⟩
          rw [← h_idx_in_eq, ← ho_eq, ← hc_eq, ← hkh_eq, ← hkw_eq]
      simp_rw [h_indicator]
      -- Triple sum collapse via Finset.sum_eq_single (3 levels deep).
      -- For c ≠ c': inner sum is 0 since the conjunction (... ∧ c = c' ∧ ...) is false.
      rw [Finset.sum_eq_single c'
            (fun c _ hc_ne =>
              Finset.sum_eq_zero (fun kh _ =>
                Finset.sum_eq_zero (fun kw _ => by
                  rw [if_neg (fun ⟨_, hc, _, _⟩ => hc_ne hc), zero_mul])))
            (fun hni => absurd (Finset.mem_univ c') hni)]
      rw [Finset.sum_eq_single kh'
            (fun kh _ hkh_ne =>
              Finset.sum_eq_zero (fun kw _ => by
                rw [if_neg (fun ⟨_, _, hkh, _⟩ => hkh_ne hkh), zero_mul]))
            (fun hni => absurd (Finset.mem_univ kh') hni)]
      rw [Finset.sum_eq_single kw'
            (fun kw _ hkw_ne => by
              rw [if_neg (fun ⟨_, _, _, hkw⟩ => hkw_ne hkw), zero_mul])
            (fun hni => absurd (Finset.mem_univ kw') hni)]
      -- Final: (if (ohw_o = o' ∧ c' = c' ∧ kh' = kh' ∧ kw' = kw') then 1 else 0) * xpad(c', kh', kw')
      --        = if ohw_o = o' then xpad(c', kh', kw') else 0
      by_cases h_o : ohw_o = o'
      · rw [if_pos ⟨h_o, rfl, rfl, rfl⟩, one_mul, if_pos h_o]
      · rw [if_neg (fun ⟨h, _⟩ => h_o h), zero_mul, if_neg h_o]
    -- Step 2: substitute h_pdiv into the backward sum and collapse.
    show (∑ hi : Fin h, ∑ wi : Fin w,
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val + hi.val
             let ww := kw'.val + wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
            * dy (finProdFinEquiv (finProdFinEquiv (o', hi), wi))) =
          ∑ idx_out : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
              v idx_in idx_out * dy idx_out
    simp_rw [h_pdiv]
    -- Now goal: ∑ hi wi, xpad * dy(flat(o', hi, wi)) = ∑ idx_out, (if o(idx_out) = o' then xpad' else 0) * dy(idx_out)
    -- Convert idx_out → ((o, hi), wi) → (o, hi, wi) via two Fintype.sum_equiv's.
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx_out : Fin (oc * h * w) =>
          (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 = o' then
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw'.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
           else 0) * dy idx_out)
        (fun pair : Fin (oc * h) × Fin w =>
          (if (finProdFinEquiv.symm pair.1).1 = o' then
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val + (finProdFinEquiv.symm pair.1).2.val
             let ww := kw'.val + pair.2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
           else 0) * dy (finProdFinEquiv pair))
        (fun idx_out => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun pair_h : Fin (oc * h) =>
          ∑ wi : Fin w,
            (if (finProdFinEquiv.symm pair_h).1 = o' then
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh'.val + (finProdFinEquiv.symm pair_h).2.val
               let ww := kw'.val + wi.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)
             else 0) * dy (finProdFinEquiv (pair_h, wi)))
        (fun ohi : Fin oc × Fin h =>
          ∑ wi : Fin w,
            (if ohi.1 = o' then
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh'.val + ohi.2.val
               let ww := kw'.val + wi.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)
             else 0) * dy (finProdFinEquiv (finProdFinEquiv ohi, wi)))
        (fun pair_h => by
          have h_inv : finProdFinEquiv (finProdFinEquiv.symm pair_h
              : Fin oc × Fin h) = pair_h :=
            Equiv.apply_symm_apply _ _
          simp_rw [h_inv])]
    rw [Fintype.sum_prod_type]
    -- Goal: ∑ hi, ∑ wi, xpad * dy(flat(o', hi, wi)) = ∑ o, ∑ hi, ∑ wi, (if o = o' then xpad else 0) * dy(...)
    -- Pull (if o = o' then ... else 0) and collapse o-sum via Finset.sum_eq_single.
    rw [Finset.sum_eq_single o'
          (fun o _ ho_ne =>
            Finset.sum_eq_zero (fun hi _ =>
              Finset.sum_eq_zero (fun wi _ => by
                rw [if_neg ho_ne, zero_mul])))
          (fun hni => absurd (Finset.mem_univ o') hni)]
    -- Now the outer (if o' = o' then ... else 0) collapses to the body.
    apply Finset.sum_congr rfl
    intro hi _
    apply Finset.sum_congr rfl
    intro wi _
    rw [if_pos rfl]

/-- Named accessor for the conv2d weight backward — aligns with MLIR
    codegen (the "transpose trick" `stablehlo.convolution` in the backward
    pass). Unwraps the flattening so callers see `Kernel4 → Kernel4`. -/
noncomputable def conv2d_weight_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Kernel4 oc ic kH kW :=
  Kernel4.unflatten
    ((conv2d_weight_grad_has_vjp b x).backward
      (Kernel4.flatten W) (Tensor3.flatten dy))

/-- **Conv2d bias-VJP** — proved from foundation rules. Now that `conv2d`
    is a real def, the function `b ↦ flatten (conv2d W b x)` decomposes
    as `(channel-reindex from b) + (W,x-only term constant in b)`. Apply
    `pdiv_add` + `pdiv_reindex` + `pdiv_const`, then collapse the
    Kronecker over the `(c, hi, wi)` decomposition of `Fin (oc*h*w)`.
    The backward is `db[o] = Σ_{hi, wi} dy[o, hi, wi]` (matches
    `conv2d_bias_grad_formula` below). -/
noncomputable def conv2d_bias_grad_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) :
    HasVJP (fun b : Vec oc => Tensor3.flatten (conv2d W b x)) where
  backward := fun _b dy => fun o =>
    ∑ hi : Fin h, ∑ wi : Fin w,
      dy (finProdFinEquiv (finProdFinEquiv (o, hi), wi))
  correct := by
    intro b dy o
    -- Step 1: pdiv decomposition. f decomposes as `b ↦ b(chan idx)` + `(W,x term)`.
    have h_pdiv : ∀ idx : Fin (oc * h * w),
        pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o idx =
        (if o = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1
          then (1:ℝ) else 0) := by
      intro idx
      rw [show (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) =
            (fun b' k =>
              (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
                y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b' k +
              (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) b' k) from by
        funext b' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_reindex_diff : DifferentiableAt ℝ
          (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
            y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b :=
        (reindexCLM (fun k' : Fin (oc * h * w) =>
          (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)).differentiableAt
      have h_const_diff : DifferentiableAt ℝ
          (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) b :=
        differentiableAt_const _
      rw [pdiv_add _ _ _ h_reindex_diff h_const_diff]
      rw [show (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
                  y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) =
            (fun y => fun k' => y ((fun k'' : Fin (oc * h * w) =>
                (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) k')) from rfl]
      rw [pdiv_reindex (fun k'' : Fin (oc * h * w) =>
            (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)]
      rw [show pdiv (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
                  ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0))
                  b o idx = 0
          from pdiv_const _ _ _ _]
      ring
    -- Step 2: substitute and collapse the Kronecker via two stages of finProdFinEquiv.
    simp_rw [h_pdiv]
    -- Σ idx, [if o = chan(idx) then 1 else 0] * dy idx
    -- Convert idx ∈ Fin (oc*h*w) to ((c, hi), wi) via finProdFinEquiv.symm twice.
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx : Fin (oc * h * w) =>
          (if o = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1 then (1:ℝ) else 0)
          * dy idx)
        (fun pair : Fin (oc * h) × Fin w =>
          (if o = (finProdFinEquiv.symm pair.1).1 then (1:ℝ) else 0)
          * dy (finProdFinEquiv pair))
        (fun idx => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun pair_h : Fin (oc * h) =>
          ∑ wi : Fin w,
            (if o = (finProdFinEquiv.symm pair_h).1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (pair_h, wi)))
        (fun ch_pair : Fin oc × Fin h =>
          ∑ wi : Fin w,
            (if o = ch_pair.1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (finProdFinEquiv ch_pair, wi)))
        (fun pair_h => by
          have h_inv : finProdFinEquiv (finProdFinEquiv.symm pair_h
                : Fin oc × Fin h) = pair_h :=
            Equiv.apply_symm_apply _ _
          simp_rw [h_inv])]
    rw [Fintype.sum_prod_type]
    -- Goal is now ∑ c, ∑ hi, ∑ wi, (if o = c then 1 else 0) * dy (flat (c, hi, wi))
    have h_pull : ∀ c : Fin oc,
        (∑ hi : Fin h, ∑ wi : Fin w,
          (if o = c then (1:ℝ) else 0)
          * dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi))) =
        (if o = c then (1:ℝ) else 0) *
          ∑ hi : Fin h, ∑ wi : Fin w,
            dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi)) := by
      intro c
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro hi _
      rw [Finset.mul_sum]
    simp_rw [h_pull, ite_mul, one_mul, zero_mul]
    rw [Finset.sum_ite_eq Finset.univ o (fun c =>
        ∑ hi : Fin h, ∑ wi : Fin w,
          dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi)))]
    simp

/-- Named accessor for the conv2d bias backward via the VJP framework. -/
noncomputable def conv2d_bias_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Vec oc :=
  (conv2d_bias_grad_has_vjp W x).backward b (Tensor3.flatten dy)

/-- **Conv2d bias gradient — closed-form formula** (documented, numerically
    verified, expected to equal `conv2d_bias_grad` up to fp precision).

    `db[o] = Σ_{h, w} dy[o, h, w]`

    Each output cell adds the same `b[o]`, so its gradient accumulates
    the contributions from every spatial position. MLIR emits this as
    a `stablehlo.reduce` across the spatial (and batch) dims. -/
noncomputable def conv2d_bias_grad_formula {oc h w : Nat}
    (dy : Tensor3 oc h w) : Vec oc :=
  fun o => ∑ y : Fin h, ∑ x : Fin w, dy o y x

-- ════════════════════════════════════════════════════════════════
-- § MaxPool
-- ════════════════════════════════════════════════════════════════

/-- **MaxPool 2×2 stride 2 forward** — concrete definition.

    Each output cell is the maximum of a 2×2 window of input cells:
    `y[c, h, w] = max{ x[c, 2h+a, 2w+b] : a, b ∈ {0,1} }`. No longer
    an axiom — replaced with the explicit four-way max.

    MLIR:
      %pool = "stablehlo.reduce_window"(%h1, %neginf) ({
        ^bb0(%a, %b): stablehlo.return (stablehlo.maximum %a, %b)
      }) {window_dimensions = [1, 1, 2, 2], window_strides = [1, 1, 2, 2]} -/
noncomputable def maxPool2 {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) : Tensor3 c h w :=
  fun ch hi wi =>
    let i0 : Fin (2*h) := ⟨2*hi.val,     by have := hi.isLt; omega⟩
    let i1 : Fin (2*h) := ⟨2*hi.val + 1, by have := hi.isLt; omega⟩
    let j0 : Fin (2*w) := ⟨2*wi.val,     by have := wi.isLt; omega⟩
    let j1 : Fin (2*w) := ⟨2*wi.val + 1, by have := wi.isLt; omega⟩
    max (max (x ch i0 j0) (x ch i1 j0)) (max (x ch i0 j1) (x ch i1 j1))

/-- **MaxPool2 input-VJP** — gradient routes only to the argmax positions.

    The backward function implements:

      `dx[c, 2h+a, 2w+b] = dy[c, h, w] · 𝟙[(a,b) is the argmax of the window]`

    Conceptually, max-pool is a piecewise selection: each output is one
    specific input. So the Jacobian is a sparse 0/1 matrix and the VJP
    just routes the gradient to the chosen input.

    MLIR uses `stablehlo.select_and_scatter` to implement this directly:
      %d_h1 = "stablehlo.select_and_scatter"(%h1, %d_pool, %zf) ({
        -- selector: pick the GE element
        ^bb0(%a, %b): stablehlo.return (stablehlo.compare GE, %a, %b)
      }, {
        -- scatter: accumulate by addition (no overlap with stride = window)
        ^bb0(%a, %b): stablehlo.return (stablehlo.add %a, %b)
      }) {window_dimensions = [1,1,2,2], window_strides = [1,1,2,2]}

    **Canonical (junk-at-tie) witness.** `HasVJP3.correct` is
    satisfied by the canonical pdiv3-derived backward via `rfl`. At
    argmax-tie boundaries `maxPool2` is not differentiable, so `pdiv3`
    agrees with `fderiv`'s junk default of `0` and the canonical
    witness is also `0` there. The codegen emits the
    `select_and_scatter` formula above instead — see
    `LeanMlir/Proofs/README.md` for the trust-boundary discussion. -/
noncomputable def maxPool2_has_vjp3 {c h w : Nat} :
    HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w) where
  backward x dy ci hi wi :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo
  correct _ _ _ _ _ := rfl

/-- Named accessor for the maxPool2 input backward — aligns with MLIR
    `stablehlo.select_and_scatter` in codegen. -/
noncomputable abbrev maxPool2_input_grad {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w) : Tensor3 c (2*h) (2*w) :=
  maxPool2_has_vjp3.backward x dy

-- ════════════════════════════════════════════════════════════════
-- § Flatten
-- ════════════════════════════════════════════════════════════════

/-! ## Reshape (flatten / unflatten)

Flatten is a permutation of indices, so its VJP is just the inverse
permutation. No gradient computation needed.

MLIR:
    %flat = stablehlo.reshape %pool
            : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>

The flatten / unflatten bijection is **already defined** in
`Tensor.lean` as `Tensor3.flatten` / `Tensor3.unflatten` (used by the
`pdiv3` derivation in Phase 5). We reuse those here rather than
duplicating — see `Tensor3.flatten_unflatten` / `unflatten_flatten`
for the mutual-inverse proofs. -/

-- ════════════════════════════════════════════════════════════════
-- § The full CNN backward pass
-- ════════════════════════════════════════════════════════════════

/-- **Walking through the CNN backward pass**.

    Unlike the MLP, where the chain rule (`vjp_comp`) gave us the whole
    backward pass in one go, here the layer types vary (Tensor3 ↔ Vec
    via flatten) so a uniform `HasVJP`-style composition would need a
    type family. For pedagogical clarity, we instead trace the backward
    pass step-by-step, matching `hand_cnn_train_step.mlir`.

    Forward:
        x ────conv W₀── h₀pre ──relu── h₀ ──conv W₁── h₁pre ──relu── h₁
          ──maxPool── pool ──flatten── d₀in ──dense W₂── d₀pre ──relu── d₀
          ──dense W₃── d₁pre ──relu── d₁ ──dense W₄── logits

    Backward (each step labeled with which lemma justifies it):

        d_logits  = softmax_ce_grad logits label                  [softmaxCE_grad]
        d_W4      = outer d₁ d_logits                             [dense_weight_grad]
        d_b4      = d_logits                                      [dense_bias_grad]
        d_d₁      = mulVec W₄ d_logits                            [dense_has_vjp]
        d_d₁pre   = relu_back d₁pre d_d₁                          [relu_has_vjp]
        d_W3      = outer d₀ d_d₁pre                              [dense_weight_grad]
        d_b3      = d_d₁pre                                       [dense_bias_grad]
        d_d₀      = mulVec W₃ d_d₁pre                             [dense_has_vjp]
        d_d₀pre   = relu_back d₀pre d_d₀                          [relu_has_vjp]
        d_W2      = outer d₀in d_d₀pre                            [dense_weight_grad]
        d_b2      = d_d₀pre                                       [dense_bias_grad]
        d_d₀in    = mulVec W₂ d_d₀pre                             [dense_has_vjp]
        d_pool    = unflatten d_d₀in                              [flatten VJP = unflatten]
        d_h₁      = maxPool2_input_grad h₁ d_pool                 [maxPool2_input_grad]
        d_h₁pre   = relu_back h₁pre d_h₁                          [relu_has_vjp, lifted to T3]
        d_W1      = conv2d_weight_grad W₁ b₁ h₀ d_h₁pre           [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b1      = conv2d_bias_grad W₁ b₁ h₀ d_h₁pre             [conv2d_bias_grad_has_vjp]
        d_h₀      = conv2d_input_grad W₁ b₁ h₀ d_h₁pre            [conv2d_has_vjp3]     ← reversed kernel
        d_h₀pre   = relu_back h₀pre d_h₀                          [relu_has_vjp, lifted to T3]
        d_W0      = conv2d_weight_grad W₀ b₀ x d_h₀pre            [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b0      = conv2d_bias_grad W₀ b₀ x d_h₀pre              [conv2d_bias_grad_has_vjp]

    Each line of the backward pass corresponds to a single line in
    `hand_cnn_train_step.mlir` (lines 134–272). The backward pass is just
    the forward layers walked in reverse, replacing each forward operation
    with its VJP. The MLIR is the literal compiled-down version of this
    derivation.

    The novelty over the MLP is in the conv layers, where the VJP turns
    out to be — itself — a convolution, just with reversed/transposed
    kernels (`conv2d_input_grad`) or swapped axes (`conv2d_weight_grad`'s
    transpose trick). Once you accept those two tricks, the entire CNN
    backprop fits in a page.
-/
example : True := trivial  -- anchor for the docstring above

/-! ## Summary of axioms in this file

- `conv2d`, `maxPool2` — forward operations (black-box forward).
- `maxPool2_has_vjp3` — input-path VJP for maxPool2 (argmax-routing
  subgradient convention).

Derived (not axioms):
- `conv2d_has_vjp3` — Phase 4: input-path VJP, proved from foundation
  rules using the per-coord pdiv chain plus a custom `pdiv_pi_pad_eval`
  helper for the dependent-`if hpad : pad then v(σ hpad) else 0`
  pattern. Backward function is `conv2d_input_grad_formula` (sum over
  `(co, ho, wo)` with reconstructed kernel offsets `kh = hi+pH-ho`,
  `kw = wi+pW-wo`).
- `conv2d_weight_grad_has_vjp` — Phase 7: the weight-path VJP, bundled
  as a plain `HasVJP` on the Kernel4-flattened function. Numerically
  gradient-checked against the transpose-trick formula in
  `check_jacobians.py:test_conv2d_weight_grad`.
- `conv2d_bias_grad_has_vjp` — Phase 9: the bias-path VJP, same bundled
  `HasVJP` pattern. The closed-form "sum output cotangent over spatial
  dims per channel" is expressed as `conv2d_bias_grad_formula`; the
  named `conv2d_bias_grad` extracts the backward via the VJP.
- `conv2d_input_grad`, `maxPool2_input_grad`, `conv2d_weight_grad`,
  `conv2d_bias_grad` — named accessors, defined as `.backward` (plus
  flatten / unflatten housekeeping for the weight / bias variants) of
  the corresponding VJP.
- `conv2d_input_grad_formula`, `conv2d_bias_grad_formula` — the
  concrete closed-form formulas (numerically verified to equal the
  VJP's backward).
- 3D reshape (`Tensor3.flatten` / `Tensor3.unflatten`) imported from
  `Tensor.lean`; 4D reshape (`Kernel4.flatten` / `Kernel4.unflatten`)
  defined here, both proved bijections. -/

/-- **Public correctness theorem for `maxPool2_has_vjp3`**: the
canonical-witness backward equals the `pdiv3`-contracted Jacobian
by definition. The codegen substitutes the standard argmax-routing
convention at non-smooth tiebreaks (see `LeanMlir/Proofs/README.md`'s
Codegen Trust Boundary). -/
theorem maxPool2_has_vjp3_correct {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp3 (c := c) (h := h) (w := w)).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  maxPool2_has_vjp3.correct x dy ci hi wi

end Proofs
