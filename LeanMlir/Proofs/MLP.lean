import LeanMlir.Proofs.Tensor
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# MLP VJP Proofs

Formal VJP correctness for the layers of a 3-layer MLP.
All definitions over `ℝ`, proofs use Mathlib's `Finset.sum`.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Dense Layer:  y = xW + b
-- ════════════════════════════════════════════════════════════════

noncomputable def dense {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) : Vec n :=
  fun j => (∑ i : Fin m, x i * W i j) + b j

/-- **Dense Jacobian** — `∂(W·x + b)_j/∂x_i = W_{ij}`. Now a theorem,
    derived from the foundation axioms (`pdiv_add`, `pdiv_const`,
    `pdiv_finset_sum`, `pdiv_mul`, `pdiv_reindex`). The proof factors
    `dense W b` into `(∑ i', x i' * W i' j) + b j`, distributes pdiv
    over the outer sum and finset sum, applies the product rule per
    summand, and collapses the Kronecker δ. -/
theorem pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j := by
  unfold dense
  -- Step 1: rewrite as `(sum x' * W) + (constant b)` to apply pdiv_add.
  rw [show (fun x' : Vec m => fun j' : Fin n =>
              (∑ i' : Fin m, x' i' * W i' j') + b j') =
        (fun x' j' =>
          (fun y : Vec m => fun j'' : Fin n => ∑ i' : Fin m, y i' * W i' j'') x' j' +
          (fun _ : Vec m => b) x' j') from rfl]
  -- Differentiable evidence for the sum-of-bilinear-summands and the constant.
  have h_summand_diff : ∀ i' ∈ (Finset.univ : Finset (Fin m)),
      DifferentiableAt ℝ
        (fun (x' : Vec m) (j'' : Fin n) => x' i' * W i' j'') x := by
    intro i' _
    have h_y : DifferentiableAt ℝ (fun (y : Vec m) (_ : Fin n) => y i') x :=
      (reindexCLM (fun _ : Fin n => i')).differentiableAt
    have h_W : DifferentiableAt ℝ (fun (_ : Vec m) (j'' : Fin n) => W i' j'') x :=
      differentiableAt_const _
    exact h_y.mul h_W
  have h_sum_diff : DifferentiableAt ℝ
      (fun (y : Vec m) (j'' : Fin n) => ∑ i' : Fin m, y i' * W i' j'') x := by
    have : (fun (y : Vec m) (j'' : Fin n) => ∑ i' : Fin m, y i' * W i' j'') =
           (fun y : Vec m => ∑ i' : Fin m,
             fun j'' : Fin n => y i' * W i' j'') := by
      funext y j''; rw [Finset.sum_apply]
    rw [this]
    exact DifferentiableAt.fun_sum (fun i' _ => h_summand_diff i' (Finset.mem_univ i'))
  have h_const_diff : DifferentiableAt ℝ (fun _ : Vec m => b) x :=
    differentiableAt_const _
  rw [pdiv_add _ _ _ h_sum_diff h_const_diff, pdiv_const, add_zero]
  -- Step 2: distribute pdiv over the finset sum.
  rw [pdiv_finset_sum (Finset.univ : Finset (Fin m))
      (fun i' x' j'' => x' i' * W i' j'') x h_summand_diff i j]
  -- Step 3: each summand is `(fun x' _ => x' i') * (fun _ _ => W i' j)`. Apply pdiv_mul.
  have hterm : ∀ i' : Fin m,
      pdiv (fun x' : Vec m => fun j' : Fin n => x' i' * W i' j') x i j =
      if i = i' then W i' j else 0 := by
    intro i'
    rw [show (fun x' : Vec m => fun j' : Fin n => x' i' * W i' j') =
          (fun x' j' =>
            (fun y : Vec m => fun j'' : Fin n => y i') x' j' *
            (fun _ : Vec m => fun j'' : Fin n => W i' j'') x' j') from rfl]
    have h_y_diff : DifferentiableAt ℝ
        (fun (y : Vec m) (_ : Fin n) => y i') x :=
      (reindexCLM (fun _ : Fin n => i')).differentiableAt
    have h_W_diff : DifferentiableAt ℝ
        (fun (_ : Vec m) (_ : Fin n) => W i' j) x :=
      differentiableAt_const _
    rw [pdiv_mul _ _ _ h_y_diff (differentiableAt_const _)]
    -- The reindex factor: `fun y j'' => y i'` = reindex via `fun _ : Fin n => i'`.
    rw [show (fun y : Vec m => fun j'' : Fin n => y i') =
          (fun y => fun j'' => y ((fun _ : Fin n => i') j'')) from rfl]
    rw [pdiv_reindex (fun _ : Fin n => i')]
    -- The const factor: pdiv = 0.
    rw [show pdiv (fun _ : Vec m => fun j'' : Fin n => W i' j'') x i j = 0
        from pdiv_const _ _ _ _]
    -- Goal: (if i = i' then 1 else 0) * W i' j + x i' * 0 = if i = i' then W i' j else 0
    by_cases h : i = i'
    · rw [if_pos h, if_pos h]; ring
    · rw [if_neg h, if_neg h]; ring
  simp_rw [hterm]
  -- Step 4: collapse the Kronecker sum.
  rw [Finset.sum_ite_eq Finset.univ i (fun i' => W i' j)]
  simp

/-- **Jacobian of dense wrt W** — `∂dense(W, b, x)_j/∂W_{i, j'} = x_i·δ(j, j')`.
    Now a theorem, derived from foundation axioms (`pdiv_add`,
    `pdiv_const`, `pdiv_finset_sum`, `pdiv_mul`, `pdiv_reindex`) over
    the flatten bijection. Symmetric counterpart to `pdiv_dense`. -/
theorem pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 := by
  -- Step 1: unfold dense + unflatten to an explicit Vec (m*n) → Vec n form.
  rw [show (fun v : Vec (m * n) => dense (Mat.unflatten v) b x) =
        (fun v : Vec (m * n) => fun jo : Fin n =>
          (∑ i' : Fin m, x i' * v (finProdFinEquiv (i', jo))) + b jo) from by
      funext v jo; unfold dense Mat.unflatten; rfl]
  -- Step 2: split into (sum) + (constant bias) and apply pdiv_add + pdiv_const.
  rw [show (fun v : Vec (m * n) => fun jo : Fin n =>
              (∑ i' : Fin m, x i' * v (finProdFinEquiv (i', jo))) + b jo) =
        (fun v jo =>
          (fun w : Vec (m * n) => fun jo' : Fin n =>
              ∑ i' : Fin m, x i' * w (finProdFinEquiv (i', jo'))) v jo +
          (fun _ : Vec (m * n) => b) v jo) from rfl]
  -- Differentiable evidence for the sum-of-bilinear-summands and the constant.
  have h_summand_diff : ∀ i' ∈ (Finset.univ : Finset (Fin m)),
      DifferentiableAt ℝ
        (fun (v : Vec (m * n)) (jo : Fin n) =>
          x i' * v (finProdFinEquiv (i', jo))) (Mat.flatten W) := by
    intro i' _
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec (m * n)) (_ : Fin n) => x i') (Mat.flatten W) :=
      differentiableAt_const _
    have h_reindex : DifferentiableAt ℝ
        (fun (w : Vec (m * n)) (jo' : Fin n) => w (finProdFinEquiv (i', jo'))) (Mat.flatten W) :=
      (reindexCLM (fun jo' : Fin n => finProdFinEquiv (i', jo'))).differentiableAt
    exact h_const.mul h_reindex
  have h_sum_diff : DifferentiableAt ℝ
      (fun (w : Vec (m * n)) (jo' : Fin n) =>
        ∑ i' : Fin m, x i' * w (finProdFinEquiv (i', jo'))) (Mat.flatten W) := by
    have h_eq : (fun (w : Vec (m * n)) (jo' : Fin n) =>
                  ∑ i' : Fin m, x i' * w (finProdFinEquiv (i', jo'))) =
                (fun w : Vec (m * n) => ∑ i' : Fin m,
                  fun jo' : Fin n => x i' * w (finProdFinEquiv (i', jo'))) := by
      funext w jo'; rw [Finset.sum_apply]
    rw [h_eq]
    exact DifferentiableAt.fun_sum (fun i' _ => h_summand_diff i' (Finset.mem_univ i'))
  have h_const_diff : DifferentiableAt ℝ (fun _ : Vec (m * n) => b) (Mat.flatten W) :=
    differentiableAt_const _
  rw [pdiv_add _ _ _ h_sum_diff h_const_diff, pdiv_const, add_zero]
  -- Step 3: distribute pdiv over the finset sum (over Fin m).
  rw [pdiv_finset_sum (Finset.univ : Finset (Fin m))
      (fun i' v jo => x i' * v (finProdFinEquiv (i', jo)))
      (Mat.flatten W) h_summand_diff (finProdFinEquiv (i, j')) j]
  -- Step 4: each summand is (const x_i') × (reindex v at (i', jo)). Apply pdiv_mul.
  have hterm : ∀ i' : Fin m,
      pdiv (fun v : Vec (m * n) => fun jo : Fin n =>
              x i' * v (finProdFinEquiv (i', jo)))
           (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if i = i' ∧ j' = j then x i else 0 := by
    intro i'
    rw [show (fun v : Vec (m * n) => fun jo : Fin n =>
                x i' * v (finProdFinEquiv (i', jo))) =
          (fun v jo =>
            (fun (_ : Vec (m * n)) (_ : Fin n) => x i') v jo *
            (fun (w : Vec (m * n)) (jo' : Fin n) =>
                w (finProdFinEquiv (i', jo'))) v jo) from rfl]
    have h_const_inner : DifferentiableAt ℝ
        (fun (_ : Vec (m * n)) (_ : Fin n) => x i') (Mat.flatten W) :=
      differentiableAt_const _
    have h_reindex_inner : DifferentiableAt ℝ
        (fun (w : Vec (m * n)) (jo' : Fin n) =>
          w (finProdFinEquiv (i', jo'))) (Mat.flatten W) :=
      (reindexCLM (fun jo' : Fin n => finProdFinEquiv (i', jo'))).differentiableAt
    rw [pdiv_mul _ _ _ h_const_inner h_reindex_inner]
    -- Const factor pdiv = 0.
    rw [show pdiv (fun (_ : Vec (m * n)) (_ : Fin n) => x i') (Mat.flatten W)
              (finProdFinEquiv (i, j')) j = 0
        from pdiv_const _ _ _ _]
    -- Reindex factor via pdiv_reindex with σ = `fun jo => finProdFinEquiv (i', jo)`.
    rw [show (fun (w : Vec (m * n)) (jo' : Fin n) =>
                w (finProdFinEquiv (i', jo'))) =
          (fun w => fun jo' =>
            w ((fun jo'' : Fin n => finProdFinEquiv (i', jo'')) jo')) from rfl]
    rw [pdiv_reindex (fun jo'' : Fin n => finProdFinEquiv (i', jo''))]
    -- Goal: 0 * x i' + x i' * (if (fPF (i, j')) = fPF (i', j) then 1 else 0)
    --       = if i = i' ∧ j' = j then x i else 0
    by_cases h : i = i' ∧ j' = j
    · obtain ⟨hii', hj'j⟩ := h
      subst hii'; subst hj'j
      simp
    · have hne : finProdFinEquiv (i, j') ≠ finProdFinEquiv (i', j) := by
        intro heq
        apply h
        have := finProdFinEquiv.injective heq
        exact ⟨(Prod.mk.inj this).1, (Prod.mk.inj this).2⟩
      rw [if_neg hne, if_neg h]
      ring
  simp_rw [hterm]
  -- Step 5: collapse the sum over i'. ∑ i', if i = i' ∧ j' = j then x i else 0.
  by_cases hj'j : j' = j
  · subst hj'j
    simp only [and_true]
    rw [Finset.sum_ite_eq Finset.univ i (fun _ => x i)]
    simp
  · rw [if_neg (fun h => hj'j h.symm)]
    simp_rw [show ∀ i' : Fin m, (i = i' ∧ j' = j) ↔ False from
      fun i' => ⟨fun h => hj'j h.2, False.elim⟩]
    simp

/-- Dense VJP — proved. -/
noncomputable def dense_has_vjp {m n : Nat} (W : Mat m n) (b : Vec n) :
    HasVJP (dense W b) where
  backward := fun _x dy => Mat.mulVec W dy
  correct := by
    intro x dy i
    simp only [Mat.mulVec]
    congr 1; ext j; rw [pdiv_dense]

/-- **Dense weight gradient is the outer product** — theorem (Phase 7).

    `Mat.outer x dy` is the cotangent-contracted Jacobian of `dense(W, b, x)`
    with respect to `W`, at every index. This promotes the previous vacuous
    `rfl` about `Mat.outer` into a real theorem connecting the outer product
    to the actual weight gradient of `dense`.

    `(Mat.outer x dy) i j = ∑ k, pdiv (…) (Mat.flatten W) (fPF (i, j)) k · dy k` -/
theorem dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k := by
  simp_rw [pdiv_dense_W]
  -- Σ k, (if k = j then x i else 0) * dy k  collapses to x i * dy j
  rw [Finset.sum_eq_single j
      (fun k _ hne => by rw [if_neg hne]; ring)
      (fun h => absurd (Finset.mem_univ j) h)]
  simp [Mat.outer]

/-- **Dense bias gradient is identity** — theorem (Phase 7).

    `∂ dense(W, b, x)_j / ∂ b_{j'} = δ(j, j')`, so the bias backward is
    just `dy` itself. Derived from `pdiv_add` + `pdiv_const` + `pdiv_id`
    — no new axiom. -/
theorem pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 := by
  -- Rewrite `fun b' => dense W b' x` as `(constant in b') + (identity on b')`.
  have hDec : (fun b' : Vec n => dense W b' x) =
              (fun b' k => (fun (_ : Vec n) (k' : Fin n) =>
                              ∑ i' : Fin m, x i' * W i' k') b' k +
                           (fun (y : Vec n) => y) b' k) := by
    funext b' k; rfl
  have h_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (k' : Fin n) => ∑ i' : Fin m, x i' * W i' k') b :=
    differentiableAt_const _
  have h_id_diff : DifferentiableAt ℝ (fun y : Vec n => y) b :=
    differentiableAt_id
  rw [hDec, pdiv_add _ _ _ h_const_diff h_id_diff, pdiv_const, pdiv_id]
  ring

theorem dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j := by
  simp_rw [pdiv_dense_b W b x]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [if_neg (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ i) h)]
  simp

/-- **Dense weight backward** — named accessor.
    `dW = x ⊗ dy` (outer product). -/
noncomputable def dense_weight_grad {m n : Nat}
    (x : Vec m) (dy : Vec n) : Mat m n :=
  Mat.outer x dy

/-- **Dense bias backward** — named accessor. `db = dy`. -/
def dense_bias_grad {n : Nat} (dy : Vec n) : Vec n := dy

-- ════════════════════════════════════════════════════════════════
-- § ReLU:  y = max(x, 0)
-- ════════════════════════════════════════════════════════════════

noncomputable def relu (n : Nat) (x : Vec n) : Vec n :=
  fun i => if x i > 0 then x i else 0

/-- **ReLU partial derivative — proved via local-diagonal-CLM transport.**

    At a smooth point (`∀ k, x k ≠ 0`), `relu n` agrees with a fixed
    diagonal indicator CLM on `Metric.ball x r` for `r := min |x k|` —
    every coordinate keeps its sign in that ball. `EventuallyEq`
    transports `fderiv` from the CLM (which is itself, since CLMs are
    their own `fderiv`) to ReLU, and direct evaluation at `basisVec i`
    reads off the entry. -/
theorem pdiv_relu (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0)
    (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0 := by
  -- Local diagonal indicator CLM at x.
  let phi : Vec n →L[ℝ] Vec n :=
    ContinuousLinearMap.pi fun k =>
      if x k > 0 then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)
  -- Coordinate evaluation: phi y k = if x k > 0 then y k else 0.
  have h_phi_eval : ∀ y : Vec n, ∀ k : Fin n,
      phi y k = if x k > 0 then y k else 0 := by
    intro y k
    show (ContinuousLinearMap.pi (fun k' =>
            if x k' > 0 then ContinuousLinearMap.proj k'
                        else (0 : Vec n →L[ℝ] ℝ))) y k = _
    rw [ContinuousLinearMap.pi_apply]
    by_cases hxk : x k > 0
    · rw [if_pos hxk, if_pos hxk]; rfl
    · rw [if_neg hxk, if_neg hxk]; rfl
  -- n = 0: Fin 0 is empty, vacuously true.
  rcases Nat.eq_zero_or_pos n with hn0 | hn_pos
  · subst hn0; exact i.elim0
  -- Pick r := min |x k|.
  haveI : Nonempty (Fin n) := ⟨⟨0, hn_pos⟩⟩
  let r : ℝ := Finset.univ.inf' Finset.univ_nonempty (fun k : Fin n => |x k|)
  have hr_pos : 0 < r := by
    refine (Finset.lt_inf'_iff _).mpr ?_
    intro k _; exact abs_pos.mpr (h_smooth k)
  have hr_le : ∀ k : Fin n, r ≤ |x k| := fun k =>
    Finset.inf'_le _ (Finset.mem_univ k)
  -- Local agreement: relu n y = phi y on Metric.ball x r.
  have h_local : Set.EqOn (relu n) (⇑phi) (Metric.ball x r) := by
    intro y hy
    have hy_norm : ‖y - x‖ < r := by
      rw [Metric.mem_ball, dist_eq_norm] at hy; exact hy
    funext k
    have h_close : |y k - x k| < |x k| := by
      have h1 : |y k - x k| ≤ ‖y - x‖ := by
        have h2 : ‖(y - x) k‖ ≤ ‖y - x‖ := norm_le_pi_norm (y - x) k
        rw [Real.norm_eq_abs] at h2
        exact h2
      linarith [hr_le k]
    show (relu n y) k = phi y k
    rw [h_phi_eval]
    show (if y k > 0 then y k else 0) = if x k > 0 then y k else 0
    rcases lt_or_gt_of_ne (h_smooth k) with hxk_neg | hxk_pos
    · -- x k < 0 ⇒ y k < 0 (within radius |x k|).
      have hyk_neg : y k < 0 := by
        have h_abs : |y k - x k| < -x k := by rwa [abs_of_neg hxk_neg] at h_close
        have h_lt : y k - x k < -x k := (abs_lt.mp h_abs).2
        linarith
      rw [if_neg (not_lt.mpr hyk_neg.le), if_neg (not_lt.mpr hxk_neg.le)]
    · -- x k > 0 ⇒ y k > 0.
      have hyk_pos : 0 < y k := by
        have h_abs : |y k - x k| < x k := by rwa [abs_of_pos hxk_pos] at h_close
        have h_lt : -(x k) < y k - x k := (abs_lt.mp h_abs).1
        linarith
      rw [if_pos hyk_pos, if_pos hxk_pos]
  -- Promote to EventuallyEq at x.
  have h_evt : (relu n) =ᶠ[nhds x] (⇑phi : Vec n → Vec n) :=
    h_local.eventuallyEq_of_mem (Metric.ball_mem_nhds x hr_pos)
  -- Transport: HasFDerivAt (relu n) phi x.
  have h_fderiv : HasFDerivAt (relu n) phi x :=
    (phi.hasFDerivAt).congr_of_eventuallyEq h_evt
  -- Compute pdiv directly.
  unfold pdiv
  rw [h_fderiv.fderiv, h_phi_eval, basisVec_apply]
  -- Goal: (if x j > 0 then (if j = i then 1 else 0) else 0)
  --     = if i = j then (if x i > 0 then 1 else 0) else 0
  by_cases hij : i = j
  · subst hij; rw [if_pos rfl, if_pos rfl]
  · rw [if_neg (fun h : j = i => hij h.symm), if_neg hij]
    by_cases hxj : x j > 0
    · rw [if_pos hxj]
    · rw [if_neg hxj]

/-- **ReLU bundled VJP — canonical (junk-at-kink) witness.**

    `HasVJP.correct` is satisfied by the canonical pdiv-derived backward:
    at smooth points it is the diagonal indicator (per `pdiv_relu`); at
    points where some coordinate is zero, `pdiv (relu n) x` agrees with
    `fderiv`'s junk default of `0`, so the canonical backward is `0`
    there too — and `correct` holds by `rfl`.

    The codegen (`MlirCodegen.lean`) emits the standard subgradient
    formula `if x > 0 then dy else 0` instead, which agrees with the
    canonical witness at smooth points and differs at the kinks (the
    convention `relu'(0) := 0` used by every ML framework). The
    Lean-vs-codegen gap at the kinks is the codegen trust boundary —
    see `LeanMlir/Proofs/README.md`. -/
noncomputable def relu_has_vjp (n : Nat) : HasVJP (relu n) where
  backward x dy i := ∑ j : Fin n, pdiv (relu n) x i j * dy j
  correct _ _ _  := rfl

-- ════════════════════════════════════════════════════════════════
-- § Softmax Cross-Entropy Loss
-- ════════════════════════════════════════════════════════════════

noncomputable def softmax (c : Nat) (z : Vec c) : Vec c :=
  let e : Vec c := fun j => Real.exp (z j)
  let total := ∑ k : Fin c, e k
  fun j => e j / total

noncomputable def oneHot (c : Nat) (label : Fin c) : Vec c :=
  fun j => if j = label then 1 else 0

noncomputable def crossEntropy (c : Nat) (logits : Vec c) (label : Fin c) : ℝ :=
  -(Real.log (softmax c logits label))

-- `softmaxCE_grad` is proved in `Attention.lean` (after `pdiv_softmax` is
-- available). Its statement and proof live there; this file keeps only
-- `softmax`, `oneHot`, and `crossEntropy` definitions used downstream.

-- ════════════════════════════════════════════════════════════════
-- § MLP Composition
-- ════════════════════════════════════════════════════════════════

noncomputable def mlpForward {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    Vec d₀ → Vec d₃ :=
  dense W₂ b₂ ∘ relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀

/-- **MLP composition VJP — canonical witness.**

    The MLP forward composes `dense W b` (everywhere `Differentiable`)
    with `relu` (non-`Differentiable` at the kinks). `vjp_comp` would
    require `Differentiable ℝ (relu n)`, which doesn't hold globally,
    so the chain-rule route is blocked. The canonical pdiv-derived
    backward inhabits `HasVJP.correct` directly via `rfl` — the
    codegen substitutes the subgradient formula at the kinks (see
    `LeanMlir/Proofs/README.md` for the trust-boundary discussion). -/
noncomputable def mlp_has_vjp {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) where
  backward x dy i :=
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j
  correct _ _ _  := rfl

/-! ## Public correctness theorems for the canonical-witness defs

Each `_has_vjp` def above bundles a backward function with a `.correct`
field; these `_correct` theorems expose that field as a top-level
proposition so consumers (downstream code, `tests/comparator/`,
doc-gen4) can refer to the contract directly without reaching into
record internals. -/

/-- **Public correctness theorem for `relu_has_vjp`**: the canonical
witness's backward equals the `pdiv`-contracted Jacobian by definition. -/
theorem relu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  (relu_has_vjp n).correct x dy i

/-- **Public correctness theorem for `mlp_has_vjp`**: same pattern as
`relu_has_vjp_correct`, lifted to the three-layer MLP forward. -/
theorem mlp_has_vjp_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).correct x dy i

end Proofs
