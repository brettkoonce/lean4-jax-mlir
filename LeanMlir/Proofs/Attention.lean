import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN          -- needed for Kernel4 in patchEmbed
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Attention — the Capstone

The fanciest architectural primitive in modern vision and language
models, formalized in one file. If you're reading the book straight
through, this is the chapter where everything you've learned clicks
together and you realize **there's nothing left to learn**.

## The cast of characters

Scaled dot-product attention:

    out = softmax((Q * K^T) / sqrt(d)) * V

where `Q = X Wq`, `K = X Wk`, `V = X Wv` — three dense projections of
the same input `X`. Every piece is something we already have:

| Piece                 | Chapter            | VJP move                 |
|-----------------------|--------------------|--------------------------|
| `Q = X Wq`            | `MLP.lean`         | dense backward           |
| `K = X Wk`            | `MLP.lean`         | dense backward           |
| `V = X Wv`            | `MLP.lean`         | dense backward           |
| `Q * K^T`             | (matmul = dense)   | chain rule               |
| `/ sqrt(d)`           | (scalar)           | chain rule + scale       |
| **`softmax(...)`**    | **this file**      | **closed-form collapse** |
| `... * V`             | (matmul = dense)   | chain rule               |
| three-way fan-in at X | `Residual.lean`    | `biPath_has_vjp`         |

So the **only genuinely new ingredient in attention** is the standalone
softmax VJP (previously we only had it bundled inside CE loss). Once
that's in hand, everything else is composition via tools we built in
earlier chapters.

## Structure of this file

1. **Standalone softmax VJP** — the last closed-form trick.
2. **Scaled dot-product attention** — SDPA as a composition.
3. **Multi-head wrapper** — reshape/transpose boilerplate, no new math.
4. **Transformer block** — LN -> MHSA -> + -> LN -> MLP -> +, pure composition.
5. **Final commentary** — why the taxonomy is complete.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 0. Differentiable helpers for the matrix-VJP building blocks
--
-- After the foundation flip, every `vjpMat_comp` and `biPathMat_has_vjp`
-- call requires `Differentiable` evidence for the flattened versions of
-- the composed matrix functions. The four helpers below cover the linear
-- building blocks (matmul-by-const-left/right, scalar-scale, transpose);
-- non-linear ingredients (rowSoftmax, layerNorm, gelu) get dedicated
-- Diff theorems further down where they're introduced.
-- ════════════════════════════════════════════════════════════════

lemma matmul_right_const_flat_diff {m p q : Nat} (D : Mat p q) :
    Differentiable ℝ (fun v : Vec (m * p) =>
      Mat.flatten (Mat.mul (Mat.unflatten v) D)) := by
  unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop

lemma matmul_left_const_flat_diff {m p q : Nat} (C : Mat m p) :
    Differentiable ℝ (fun v : Vec (p * q) =>
      Mat.flatten (Mat.mul C (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop

lemma scalarScale_flat_diff {m n : Nat} (s : ℝ) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (fun r c => s * (Mat.unflatten v) r c)) := by
  unfold Mat.unflatten Mat.flatten; fun_prop

lemma transpose_flat_diff {m n : Nat} :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (Mat.transpose (Mat.unflatten v) : Mat n m)) := by
  unfold Mat.unflatten Mat.flatten Mat.transpose; fun_prop

/-- Differentiability of the flattened per-token dense map.
    `fun X => fun n => dense W b (X n)` is linear in `X`, so the
    flattened version is `Differentiable` everywhere. -/
lemma dense_per_token_flat_diff {N inD outD : Nat}
    (W : Mat inD outD) (b : Vec outD) :
    Differentiable ℝ (fun v : Vec (N * inD) =>
      Mat.flatten ((fun X : Mat N inD => fun n => dense W b (X n))
                   (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten dense; fun_prop

/-- Differentiability of the flattened per-token GELU map.
    `geluScalar = 0.5 · x · (1 + tanh(√(2/π)(x + 0.044715·x³)))`. With
    `Real.differentiable_tanh` available to `fun_prop`, the proof
    discharges automatically. -/
theorem gelu_per_token_flat_diff (N D : Nat) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => gelu D (X n))
                   (Mat.unflatten v))) := by
  unfold Mat.unflatten Mat.flatten gelu geluScalar; fun_prop

/-- Differentiability of `dense W b` as a function of the input. -/
lemma dense_diff {m n : Nat} (W : Mat m n) (b : Vec n) :
    Differentiable ℝ (dense W b) := by
  unfold dense; fun_prop

/-- Differentiability of `softmax c` — same recipe as `rowSoftmax_flat_diff`,
    but on the unflattened vector. -/
lemma softmax_diff (c : Nat) : Differentiable ℝ (softmax c) := by
  match c with
  | 0 =>
    -- Codomain Vec 0 — trivially differentiable.
    rw [show (softmax 0 : Vec 0 → Vec 0) = (fun _ : Vec 0 => fun (k : Fin 0) => (0 : ℝ)) by
      funext _ k; exact k.elim0]
    intro v
    exact (differentiable_const _).differentiableAt
  | c + 1 =>
    rw [differentiable_pi]
    intro k
    -- The k-th coord is `exp(z k) * (Σ j, exp(z j))⁻¹`.
    have h_fn : (fun z : Vec (c + 1) => softmax (c + 1) z k) =
                (fun z : Vec (c + 1) =>
                  Real.exp (z k) * (∑ j : Fin (c + 1), Real.exp (z j))⁻¹) := by
      funext z
      show (let e := fun j => Real.exp (z j); let total := ∑ k', e k'; e k / total) = _
      rw [div_eq_mul_inv]
    rw [h_fn]
    have h_num : Differentiable ℝ (fun z : Vec (c + 1) => Real.exp (z k)) := by fun_prop
    have h_denom : Differentiable ℝ
        (fun z : Vec (c + 1) => ∑ j : Fin (c + 1), Real.exp (z j)) := by fun_prop
    have h_ne : ∀ z : Vec (c + 1),
        (∑ j : Fin (c + 1), Real.exp (z j)) ≠ 0 := fun z =>
      (Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty).ne'
    have h_inv : Differentiable ℝ
        (fun z : Vec (c + 1) => (∑ j : Fin (c + 1), Real.exp (z j))⁻¹) :=
      fun z => (h_denom z).inv (h_ne z)
    exact h_num.mul h_inv

/-- Differentiability of `layerNormForward D ε γ β`.
    `layerNormForward = bnForward = bnAffine ∘ bnNormalize` (definitionally),
    where the chain is differentiable when `ε > 0`. -/
lemma layerNorm_diff (D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (layerNormForward D ε γ β) := by
  show Differentiable ℝ (bnForward D ε γ β)
  rw [bnForward_eq_compose]
  -- bnForward = bnAffine ∘ bnNormalize
  apply Differentiable.comp
  · -- bnAffine is differentiable
    unfold bnAffine; fun_prop
  · -- bnNormalize is differentiable (centered * istdBroadcast, both diff)
    rw [show bnNormalize D ε =
          (fun y : Vec D => fun k : Fin D =>
            bnCentered D y k * bnIstdBroadcast D ε y k) from by
      funext y; exact bnXhat_eq_product D ε y]
    have h_centered : Differentiable ℝ (bnCentered D) := by
      unfold bnCentered bnMean; fun_prop
    exact h_centered.mul (bnIstdBroadcast_diff D ε hε)

/-- Differentiability of the flattened per-token LayerNorm map.
    Now a theorem (was an axiom): each output coord projects through a
    row-projection CLM into `layerNorm_diff` at that row. -/
theorem layerNorm_per_token_flat_diff (N D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormForward D ε γ β (X n))
                   (Mat.unflatten v))) := by
  rw [differentiable_pi]
  intro idx
  set p := finProdFinEquiv.symm idx
  -- The idx-coord equals layerNormForward applied to the p.1-th row, then
  -- coord p.2: layerNormForward (row_proj_p.1 v) p.2.
  have h_eq : (fun v : Vec (N * D) =>
        Mat.flatten ((fun X : Mat N D => fun n => layerNormForward D ε γ β (X n))
                     (Mat.unflatten v)) idx) =
      (fun w : Vec D => layerNormForward D ε γ β w p.2) ∘
      (fun v : Vec (N * D) => fun j' : Fin D => v (finProdFinEquiv (p.1, j'))) := by
    funext v
    show Mat.flatten _ idx = _
    show layerNormForward D ε γ β (Mat.unflatten v p.1) p.2 = _
    rfl
  rw [h_eq]
  -- (layerNormForward _ _ _ _ · p.2) is the p.2-coord projection of layerNormForward;
  -- diff via layerNorm_diff + differentiableAt_pi.mp.
  have h_outer : Differentiable ℝ (fun w : Vec D => layerNormForward D ε γ β w p.2) :=
    fun w => differentiableAt_pi.mp ((layerNorm_diff D ε γ β hε) w) p.2
  have h_proj : Differentiable ℝ
      (fun v : Vec (N * D) => fun j' : Fin D => v (finProdFinEquiv (p.1, j'))) :=
    (reindexCLM (fun j' : Fin D => finProdFinEquiv (p.1, j'))).differentiable
  exact h_outer.comp h_proj

/-- Differentiability of the flattened identity matrix map.
    `Mat.flatten ∘ id ∘ Mat.unflatten = id` on `Vec (a*b)`. -/
lemma identity_mat_flat_diff (a b : Nat) :
    Differentiable ℝ (fun v : Vec (a * b) =>
      Mat.flatten ((fun X : Mat a b => X) (Mat.unflatten v))) := by
  show Differentiable ℝ (fun v : Vec (a * b) => Mat.flatten (Mat.unflatten v))
  have h_eq : (fun v : Vec (a * b) => Mat.flatten (Mat.unflatten v)) = id := by
    funext v; exact Mat.flatten_unflatten v
  rw [h_eq]; exact differentiable_id

-- ════════════════════════════════════════════════════════════════
-- § 1. Standalone Softmax VJP
-- ════════════════════════════════════════════════════════════════

/-! ## The softmax Jacobian

For `p = softmax(z)` with `p_j = exp(z_j) / sum_k exp(z_k)`, the quotient
rule gives:

    dp_j/dz_i = p_j * (delta_{ij} - p_i)

This is the famous "diag minus outer product" form:

    J = diag(p) - p * p^T

Dense (every output depends on every input), but **rank-1 correction
to a diagonal** — which means the VJP has a closed-form collapse, just
like BatchNorm did.
-/

/-- **Partial derivative of softmax** (quotient rule on the exponentials).

    `d(softmax(z))_j/dz_i = softmax(z)_j * (delta_{ij} - softmax(z)_i)`

    Proved (was an axiom). The j-th coord of `softmax c z` is
    `Real.exp (z j) / S` with `S := Σ_k Real.exp (z k) > 0`, so the j-th
    output coord function `z' ↦ exp(z' j) * (Σ_k exp(z' k))⁻¹` has
    `HasFDerivAt` derivative built from `HasFDerivAt.exp`,
    `HasFDerivAt.fun_sum`, `(hasDerivAt_inv ·).comp_hasFDerivAt`, and
    `HasFDerivAt.mul`. Evaluating that CLM at `basisVec i` and
    collapsing `Σ_k exp(z k) · δ_{ki} = exp(z i)` gives the formula. -/
theorem pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) := by
  cases c with
  | zero => exact j.elim0
  | succ c' =>
  unfold pdiv
  -- Convert (fderiv ℝ softmax z) (basisVec i) j → fderiv of the j-th coord function.
  have h_swap : fderiv ℝ (softmax (c' + 1)) z (basisVec i) j =
                fderiv ℝ (fun z' : Vec (c' + 1) => softmax (c' + 1) z' j) z (basisVec i) := by
    rw [fderiv_apply (softmax_diff (c' + 1) z) j]
    rfl
  rw [h_swap]
  rw [show (fun z' : Vec (c' + 1) => softmax (c' + 1) z' j) =
         (fun z' => Real.exp (z' j) * (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹) from by
    funext z'
    show Real.exp (z' j) / (∑ k : Fin (c' + 1), Real.exp (z' k)) = _
    rw [div_eq_mul_inv]]
  set S : ℝ := ∑ k : Fin (c' + 1), Real.exp (z k) with hS_def
  have hS_pos : (0 : ℝ) < S :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty
  have hS_ne : S ≠ 0 := hS_pos.ne'
  -- HasFDerivAt building blocks
  have h_proj : ∀ k : Fin (c' + 1),
      HasFDerivAt (fun z' : Vec (c' + 1) => z' k)
                  (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ) z :=
    fun k => (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ).hasFDerivAt
  have h_exp : ∀ k : Fin (c' + 1),
      HasFDerivAt (fun z' : Vec (c' + 1) => Real.exp (z' k))
                  (Real.exp (z k) • (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ)) z :=
    fun k => (h_proj k).exp
  have h_sum : HasFDerivAt
      (fun z' : Vec (c' + 1) => ∑ k : Fin (c' + 1), Real.exp (z' k))
      (∑ k : Fin (c' + 1), Real.exp (z k) •
          (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ)) z :=
    HasFDerivAt.fun_sum (fun k _ => h_exp k)
  have h_inv : HasFDerivAt
      (fun z' : Vec (c' + 1) => (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹)
      ((-(S ^ 2)⁻¹) • (∑ k : Fin (c' + 1), Real.exp (z k) •
          (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ))) z :=
    (hasDerivAt_inv hS_ne).comp_hasFDerivAt z h_sum
  have h_mul : HasFDerivAt
      (fun z' : Vec (c' + 1) =>
          Real.exp (z' j) * (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹)
      (Real.exp (z j) • ((-(S ^ 2)⁻¹) • (∑ k : Fin (c' + 1), Real.exp (z k) •
            (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ))) +
       S⁻¹ • (Real.exp (z j) • (ContinuousLinearMap.proj j : Vec (c' + 1) →L[ℝ] ℝ))) z :=
    (h_exp j).mul h_inv
  rw [h_mul.fderiv]
  -- Evaluate the resulting CLM at basisVec i and simplify.
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply, smul_eq_mul,
             ContinuousLinearMap.sum_apply, ContinuousLinearMap.proj_apply, basisVec_apply]
  -- Collapse the Kronecker sum: Σ_k exp(z k) * (if k = i then 1 else 0) = exp(z i).
  rw [show (∑ k : Fin (c' + 1), Real.exp (z k) * (if k = i then (1 : ℝ) else 0)) =
        Real.exp (z i) from by
      rw [Finset.sum_eq_single i]
      · rw [if_pos rfl, mul_one]
      · intros b _ hb; rw [if_neg hb, mul_zero]
      · intro h; exact absurd (Finset.mem_univ i) h]
  -- Unfold softmax on the RHS and convert `if j = i` to `if i = j`.
  show Real.exp (z j) * (-(S ^ 2)⁻¹ * Real.exp (z i)) +
       S⁻¹ * (Real.exp (z j) * (if j = i then (1 : ℝ) else 0)) =
       (Real.exp (z j) / S) * ((if i = j then (1 : ℝ) else 0) - Real.exp (z i) / S)
  have h_if : (if j = i then (1 : ℝ) else 0) = (if i = j then (1 : ℝ) else 0) := by
    by_cases h : i = j
    · rw [if_pos h, if_pos h.symm]
    · rw [if_neg h, if_neg (fun heq => h heq.symm)]
  rw [h_if]
  field_simp
  ring

/-- **Softmax VJP — the closed-form collapse.**

    `back(z, dy)_i = p_i * (dy_i - <p, dy>)`

    where `p = softmax(z)` and `<p, dy> = sum_j p_j * dy_j` is one scalar.

    **Read this carefully.** The naive VJP would be:
      dz_i = sum_j J_{ji} * dy_j = sum_j (p_j * (delta_{ij} - p_i)) * dy_j

    That's O(c) per entry, O(c^2) total. But expanding:
      dz_i = p_i * dy_i - p_i * sum_j p_j * dy_j
           = p_i * (dy_i - <p, dy>)

    The rank-1 correction lets you **precompute one scalar** (`<p, dy>`)
    and apply it to every entry. **Total work: O(c).** Same optimization
    pattern as BN (one reduction + a broadcast) and max-pool (one
    comparison + a select).

    **Interpretation.** Softmax outputs a probability distribution. Its
    backward subtracts the "weighted average of the incoming gradient
    under that distribution" from each entry, then scales by the
    entry's probability. Entries with low probability get small
    gradients (because the softmax flattened them in the forward);
    entries with high probability get gradients proportional to how
    much they deviate from the weighted-average cotangent.

    This is the one place where "softmax means softly select one thing"
    maps directly to "softmax backward selectively amplifies the
    gradient for the winning class." -/
noncomputable def softmax_has_vjp (c : Nat) : HasVJP (softmax c) where
  backward := fun z dy =>
    let p : Vec c := softmax c z
    let s : ℝ := ∑ j : Fin c, p j * dy j  -- <p, dy>
    fun i => p i * (dy i - s)
  correct := by
    intro z dy i
    -- Goal: p_i * (dy_i - <p, dy>) = sum_j pdiv(softmax) z i j * dy_j
    -- RHS by pdiv_softmax: sum_j (p_j * (delta_{ij} - p_i)) * dy_j
    --                    = p_i * dy_i - p_i * sum_j p_j * dy_j
    --                    = p_i * (dy_i - <p, dy>)
    simp only [pdiv_softmax]
    set p := softmax c z
    -- Reduce to: ∑ j, p j * (δ_ij - p i) * dy j = p i * dy i - p i * ∑ j, p j * dy j
    suffices h : ∑ j : Fin c, p j * ((if i = j then (1:ℝ) else 0) - p i) * dy j
        = p i * dy i - p i * ∑ j : Fin c, p j * dy j by
      rw [h]; ring
    -- Distribute and split the sum
    simp_rw [mul_sub, sub_mul]
    rw [Finset.sum_sub_distrib]
    congr 1
    -- First sum: Kronecker delta collapses to p i * dy i
    · simp [mul_ite, ite_mul, mul_one, mul_zero, zero_mul]
    -- Second sum: factor p i out
    · rw [Finset.mul_sum]; congr 1; ext j; ring

/-- **Softmax cross-entropy scalar gradient** — proved (was an axiom in
    MLP.lean; relocated here to use `pdiv_softmax`).

    `∂(-log softmax(z)[label])/∂z_j = softmax(z)_j - onehot(label)_j`

    Stated using `pdiv` on a `Vec 1`-valued wrapper (cross-entropy is
    naturally scalar, but `pdiv` is defined for `Vec → Vec`; we just
    take the only output index). Proof: `fderiv_apply` extracts the
    only coord, then `HasFDerivAt.log` (with `softmax z label > 0`)
    composed with `softmax_diff` gives the derivative of the inner
    `Real.log`. Negating and evaluating at `basisVec j` reduces via
    `pdiv_softmax` to the expected formula. -/
theorem softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j := by
  cases c with
  | zero => exact label.elim0
  | succ c' =>
  have h_softmax_pos : ∀ z : Vec (c' + 1), 0 < softmax (c' + 1) z label := fun z =>
    div_pos (Real.exp_pos _)
      (Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty)
  have hp_ne : softmax (c' + 1) logits label ≠ 0 := (h_softmax_pos logits).ne'
  -- Differentiability infrastructure.
  have h_softmax_label_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => softmax (c' + 1) z label) :=
    fun z => differentiableAt_pi.mp ((softmax_diff (c' + 1)) z) label
  have h_log_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => Real.log (softmax (c' + 1) z label)) :=
    fun z => (h_softmax_label_diff z).log (h_softmax_pos z).ne'
  have h_ce_pi_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => fun _ : Fin 1 => crossEntropy (c' + 1) z label) := by
    rw [differentiable_pi]
    intro _
    show Differentiable ℝ (fun z => -(Real.log (softmax (c' + 1) z label)))
    exact h_log_diff.neg
  unfold pdiv
  -- Step 1: extract the single (0-th) coord of the Vec 1-valued function.
  rw [show fderiv ℝ (fun z : Vec (c' + 1) => fun _ : Fin 1 => crossEntropy (c' + 1) z label)
                  logits (basisVec j) 0
        = fderiv ℝ (fun z : Vec (c' + 1) => crossEntropy (c' + 1) z label)
                  logits (basisVec j) from by
    rw [fderiv_apply (h_ce_pi_diff logits) 0]; rfl]
  -- Step 2: HasFDerivAt chain for crossEntropy = -log ∘ softmax_label.
  have h_softmax_at : HasFDerivAt (fun z : Vec (c' + 1) => softmax (c' + 1) z label)
      (fderiv ℝ (fun z => softmax (c' + 1) z label) logits) logits :=
    (h_softmax_label_diff logits).hasFDerivAt
  have h_log_at : HasFDerivAt
      (fun z : Vec (c' + 1) => Real.log (softmax (c' + 1) z label))
      ((softmax (c' + 1) logits label)⁻¹ •
        fderiv ℝ (fun z => softmax (c' + 1) z label) logits) logits :=
    h_softmax_at.log hp_ne
  have h_ce_at : HasFDerivAt
      (fun z : Vec (c' + 1) => crossEntropy (c' + 1) z label)
      (-((softmax (c' + 1) logits label)⁻¹ •
          fderiv ℝ (fun z => softmax (c' + 1) z label) logits)) logits := by
    show HasFDerivAt (fun z => -(Real.log (softmax (c' + 1) z label))) _ logits
    exact h_log_at.neg
  rw [h_ce_at.fderiv]
  -- Step 3: simplify CLM application at basisVec j.
  simp only [ContinuousLinearMap.neg_apply, ContinuousLinearMap.smul_apply, smul_eq_mul]
  -- Step 4: rewrite fderiv of `softmax z label` (in z) as pdiv softmax, then apply pdiv_softmax.
  rw [show fderiv ℝ (fun z : Vec (c' + 1) => softmax (c' + 1) z label) logits (basisVec j)
        = pdiv (softmax (c' + 1)) logits j label from by
    show _ = fderiv ℝ (softmax (c' + 1)) logits (basisVec j) label
    rw [fderiv_apply ((softmax_diff (c' + 1)) logits) label]; rfl]
  rw [pdiv_softmax]
  -- Step 5: oneHot unfolds to `if j = label then 1 else 0`; algebra cancels p[label].
  show -((softmax (c' + 1) logits label)⁻¹ *
        (softmax (c' + 1) logits label *
          ((if j = label then (1 : ℝ) else 0) - softmax (c' + 1) logits j))) =
       softmax (c' + 1) logits j - (if j = label then (1 : ℝ) else 0)
  field_simp
  ring

-- ════════════════════════════════════════════════════════════════
-- § 2. Scaled Dot-Product Attention
-- ════════════════════════════════════════════════════════════════

/-! ## Attention as a composition

For a single sequence of `n` tokens, each with feature dim `d`, let
`X : Mat n d` be the input. Attention produces `out : Mat n d` via:

    Q = X * Wq        -- (n x d), dense projection
    K = X * Wk        -- (n x d)
    V = X * Wv        -- (n x d)
    scores = Q * K^T   -- (n x n)
    scaled = scores / sqrt(d)
    weights = softmax_row(scaled)   -- softmax applied per row
    out = weights * V               -- (n x d)

Because the input `X` is a matrix, we need matrix-level types. We work
with `Mat n d` throughout this section (already defined in `Tensor.lean`).

**Row-wise softmax** is just "apply the 1D softmax to each row
independently." Its VJP is just "apply the 1D softmax VJP to each row
independently." No new derivation; the fan-out structure is trivially
parallel.
-/

/-- Row-wise softmax of a matrix. -/
noncomputable def rowSoftmax {m n : Nat} (A : Mat m n) : Mat m n :=
  fun i => softmax n (A i)

/-- **Smoothness of `rowSoftmax`** — proved from Mathlib calculus
    (VJP.md follow-up B).

    `rowSoftmax M r c = exp(M r c) / Σⱼ exp(M r j)`. The denominator is
    everywhere positive (sum of `Real.exp_pos` terms over a nonempty
    index set when `n ≥ 1`), so the function is C^∞ via `Real.exp`,
    `Finset.sum`, and `div` with positivity. The `n = 0` case is
    trivial because `Vec (m * 0) = Vec 0` is 0-dimensional. -/
theorem rowSoftmax_flat_diff (m n : Nat) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m n)) := by
  match n with
  | 0 =>
    -- m * 0 = 0; codomain Vec 0 — trivially differentiable.
    intro v
    -- Reduce the goal via funext to the pointwise version, then handle Fin 0 = ∅.
    have : (fun v : Vec (m * 0) => Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m 0)) =
           (fun _ : Vec (m * 0) => fun (k : Fin (m * 0)) => (0 : ℝ)) := by
      funext v k
      exact (k.elim0 : False).elim
    rw [this]
    exact (differentiable_const _).differentiableAt
  | n + 1 =>
    rw [differentiable_pi]
    intro k
    set p := finProdFinEquiv.symm k
    -- Rewrite the k-th coordinate as `Real.exp (linear) * (Σ Real.exp (linear))⁻¹`
    -- so that we can chain `Differentiable.mul` with `Differentiable.inv`
    -- (the multivariate `Differentiable.div` requires a scalar domain).
    have h_fn :
        (fun v : Vec ((m * (n + 1))) =>
          Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m (n + 1)) k) =
        (fun v : Vec ((m * (n + 1))) =>
          Real.exp (v (finProdFinEquiv (p.1, p.2))) *
          (∑ j : Fin (n + 1), Real.exp (v (finProdFinEquiv (p.1, j))))⁻¹) := by
      funext v
      show rowSoftmax (Mat.unflatten v) p.1 p.2 = _
      unfold rowSoftmax softmax Mat.unflatten
      rw [div_eq_mul_inv]
    rw [h_fn]
    have h_num : Differentiable ℝ
        (fun v : Vec (m * (n + 1)) => Real.exp (v (finProdFinEquiv (p.1, p.2)))) := by
      fun_prop
    have h_denom : Differentiable ℝ
        (fun v : Vec (m * (n + 1)) =>
          ∑ j : Fin (n + 1), Real.exp (v (finProdFinEquiv (p.1, j)))) := by
      fun_prop
    have h_ne : ∀ v : Vec (m * (n + 1)),
        (∑ j : Fin (n + 1), Real.exp (v (finProdFinEquiv (p.1, j)))) ≠ 0 := by
      intro v
      exact (Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty).ne'
    -- inv of the denominator is differentiable everywhere (denom ≠ 0).
    have h_inv : Differentiable ℝ
        (fun v : Vec (m * (n + 1)) =>
          (∑ j : Fin (n + 1), Real.exp (v (finProdFinEquiv (p.1, j))))⁻¹) :=
      fun v => (h_denom v).inv (h_ne v)
    exact h_num.mul h_inv

/-- **Row-wise softmax VJP** — proved, no sorry.

    Rows are independent, so the Jacobian is block-diagonal with the
    standalone softmax Jacobian in each block. The backward just
    applies `softmax_has_vjp` per row. -/
noncomputable def rowSoftmax_has_vjp_mat {m n : Nat} :
    HasVJPMat (fun A : Mat m n => fun r => softmax n (A r)) where
  backward := fun A dY => fun r c => (softmax_has_vjp n).backward (A r) (dY r) c
  correct := by
    intro A dY i j
    -- Replace pdivMat of the row-independent fn with its row/vector form.
    simp_rw [pdivMat_rowIndep _ (softmax_diff n)]
    -- Goal: (softmax_has_vjp n).backward (A i) (dY i) j =
    --       Σ k, Σ l, (if i = k then pdiv (softmax n) (A i) j l else 0) * dY k l
    -- Push the *dY through the if-else, then pull the if-else out of the inner sum.
    have h : ∀ k : Fin m,
        (∑ l : Fin n, (if i = k then pdiv (softmax n) (A i) j l else 0) * dY k l) =
        if i = k then ∑ l : Fin n, pdiv (softmax n) (A i) j l * dY k l else 0 := by
      intro k
      by_cases hik : i = k
      · simp [hik]
      · simp [hik]
    simp_rw [h]
    -- Now: Σ k, if i = k then Σ l, ... * dY k l else 0.  Collapse at k = i.
    rw [Finset.sum_ite_eq Finset.univ i
        (fun k => ∑ l : Fin n, pdiv (softmax n) (A i) j l * dY k l)]
    simp only [Finset.mem_univ, if_true]
    -- Goal: (softmax_has_vjp n).backward (A i) (dY i) j =
    --       Σ l, pdiv (softmax n) (A i) j l * dY i l
    exact (softmax_has_vjp n).correct (A i) (dY i) j

/-- Alias so `rowSoftmax_has_vjp_mat` types against the actual `rowSoftmax`
    definition (definitionally equal, but lets Lean unify on the name). -/
noncomputable def rowSoftmax_has_vjp_mat' (m n : Nat) :
    HasVJPMat (@rowSoftmax m n) :=
  rowSoftmax_has_vjp_mat

/-- **Scaled dot-product attention**, for a single sequence and a
    single head. `Q K V : Mat n d`.

    `sdpa Q K V = softmax_row(Q * K^T / sqrt(d)) * V`

    MLIR (`emitMHSAForward`, lines 754-781):
      %mh_sc   = dot_general %mh_q, %mh_k, contracting_dims = [3] x [3]
      %mh_ss   = multiply %mh_sc, broadcast(1/sqrt(d))
      %mh_sm   = softmax(%mh_ss) -- via reduce max, shift, exp, reduce sum, divide
      %mh_av   = dot_general %mh_sm, %mh_v, contracting_dims = [3] x [2]
-/
noncomputable def sdpa (n d : Nat) (Q K V : Mat n d) : Mat n d :=
  let scores : Mat n n := Mat.mul Q (Mat.transpose K)
  let scale : ℝ := 1 / Real.sqrt (↑d)
  let scaled : Mat n n := fun i j => scale * scores i j
  let weights : Mat n n := rowSoftmax scaled
  Mat.mul weights V

/-! ## The backward pass through SDPA (by hand, then compositionally)

Working backward from `d_out : Mat n d`, four steps:

**Step 1.** Through the final matmul `out = weights * V`. By the dense
layer VJP generalized to matrices (same derivation as `dense_has_vjp`,
just with a batch dimension):

    d_V       = weights^T * d_out     -- (n x d)
    d_weights = d_out * V^T           -- (n x n)

**Step 2.** Through the per-row softmax. Each row is independent, so
we apply `softmax_has_vjp` row-by-row:

    d_scaled_i = weights_i * (d_weights_i - <weights_i, d_weights_i> * 1)

**Step 3.** Through the scalar scale `scaled = scores / sqrt(d)`. Just
divide the incoming gradient by `sqrt(d)`:

    d_scores = d_scaled / sqrt(d)

**Step 4.** Through `scores = Q * K^T`. Same matrix-matmul VJP as
step 1, but now Q and K both flow back:

    d_Q = d_scores * K                       -- (n x d)
    d_K = d_scores^T * Q                     -- (n x d)

**Step 5.** Three parallel dense backwards from Q, K, V back to X.
Each uses `dense_has_vjp`:

    d_X_via_Q = d_Q * Wq^T
    d_X_via_K = d_K * Wk^T
    d_X_via_V = d_V * Wv^T

**Step 6.** Fan-in at X — the three paths **add**:

    d_X = d_X_via_Q + d_X_via_K + d_X_via_V

This is `biPath_has_vjp` from `Residual.lean`, applied twice (to
combine three paths). The three-way fan-in **is** the attention
backward pass at the input. Q, K, V are parallel branches reading
from `X`, so their gradients accumulate at `X`.

And the parameter gradients (for W_q, W_k, W_v, W_o) are collected
at each dense layer along the way — exactly as with any other dense
layer in the book.

**There is no novel structural move in attention.** It's three dense
layers, two matmuls, one row-softmax, one scale, and a three-way
fan-in. Every piece has been proved. The composition is mechanical.
-/

/-! ### The backward, concretely

Earlier drafts of this section ended with a single `axiom sdpa_has_vjp`
whose type was just `(... functions) × (... functions) × (... functions)`.
That was **vacuous as a correctness claim** — a triple of zero functions
satisfies it. The current state is:

1. **Concrete definitions** of `sdpa_back_Q`, `sdpa_back_K`, `sdpa_back_V`
   transcribed from the step-by-step derivation above.
2. **Honest correctness theorems** stated in terms of `pdivMat` (the
   matrix-level partial derivative primitive from `Tensor.lean`),
   proved compositionally via the four-step `vjpMat_comp` chain
   described in §Q-correctness below.

The concrete formulas are also numerically gradient-checked in
`check_axioms.py` (`test_sdpa_back_Q/K/V`) for cross-validation.
-/

/-- `1 / sqrt(d)`, the SDPA scale factor. -/
noncomputable def sdpa_scale (d : Nat) : ℝ := 1 / Real.sqrt (↑d)

/-- Softmax-weights under the SDPA scale, reused by all three backwards. -/
noncomputable def sdpa_weights (n d : Nat) (Q K : Mat n d) : Mat n n :=
  let scores : Mat n n := Mat.mul Q (Mat.transpose K)
  let scaled : Mat n n := fun i j => sdpa_scale d * scores i j
  rowSoftmax scaled

/-- Gradient flowing into `weights` from the final matmul `out = weights · V`. -/
noncomputable def sdpa_dWeights {n d : Nat} (V dOut : Mat n d) : Mat n n :=
  Mat.mul dOut (Mat.transpose V)

/-- Per-row softmax VJP: `p_i * (dw_i - <p_i, dw_i>)`. -/
noncomputable def sdpa_dScaled (n d : Nat) (Q K V dOut : Mat n d) : Mat n n :=
  let p : Mat n n := sdpa_weights n d Q K
  let dw : Mat n n := sdpa_dWeights V dOut
  fun i j =>
    let s : ℝ := ∑ k : Fin n, p i k * dw i k
    p i j * (dw i j - s)

/-- Gradient w.r.t. the pre-softmax scores, after undoing the `/ sqrt(d)` scale. -/
noncomputable def sdpa_dScores (n d : Nat) (Q K V dOut : Mat n d) : Mat n n :=
  fun i j => sdpa_scale d * sdpa_dScaled n d Q K V dOut i j

/-- **Backward w.r.t. Q**: `dQ = dScores · K`. -/
noncomputable def sdpa_back_Q (n d : Nat) (Q K V dOut : Mat n d) : Mat n d :=
  Mat.mul (sdpa_dScores n d Q K V dOut) K

/-- **Backward w.r.t. K**: `dK = dScores^T · Q`. -/
noncomputable def sdpa_back_K (n d : Nat) (Q K V dOut : Mat n d) : Mat n d :=
  Mat.mul (Mat.transpose (sdpa_dScores n d Q K V dOut)) Q

/-- **Backward w.r.t. V**: `dV = weights^T · dOut`. (V does not appear on the
    RHS: `V`'s gradient flows only through the final matmul, not through
    `weights`.) -/
noncomputable def sdpa_back_V (n d : Nat) (Q K _V dOut : Mat n d) : Mat n d :=
  Mat.mul (Mat.transpose (sdpa_weights n d Q K)) dOut

/-! ## Q and K correctness via compositional SDPA forward chain

For Q (with K, V fixed), `sdpa n d · K V` is the composition:

    Q ↦ Q · K^T   ↦   scale * _   ↦   rowSoftmax _   ↦   _ · V

Four steps, four already-proved `HasVJPMat` building blocks:

1. `matmul_right_const_has_vjp (Mat.transpose K)` — ∂(Q · K^T)/∂Q
2. `scalarScale_has_vjp (sdpa_scale d)` — ∂(scale · scores)/∂scores
3. `rowSoftmax_has_vjp_mat` — ∂(rowSoftmax scaled)/∂scaled
4. `matmul_right_const_has_vjp V` — ∂(weights · V)/∂weights

Chain them with `vjpMat_comp` thrice → a `HasVJPMat` for the full
Q-path. Then show the chain's backward function equals `sdpa_back_Q`
pointwise (trivial — the chain's backward literally computes the same
nested formula) and invoke its `.correct` to discharge the goal. -/

/-- Explicit 4-composition forward for SDPA, varying Q with K, V fixed. -/
noncomputable def sdpa_Q_chain (n d : Nat) (K V : Mat n d) : Mat n d → Mat n d :=
  (fun w : Mat n n => Mat.mul w V) ∘
  (@rowSoftmax n n) ∘
  (fun s : Mat n n => fun r c => sdpa_scale d * s r c) ∘
  (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))

theorem sdpa_Q_chain_eq (n d : Nat) (Q K V : Mat n d) :
    sdpa_Q_chain n d K V Q = sdpa n d Q K V := by
  unfold sdpa_Q_chain sdpa sdpa_scale
  rfl

/-- `HasVJPMat` for the chain — built by nesting `vjpMat_comp` thrice. -/
noncomputable def sdpa_Q_chain_has_vjp (n d : Nat) (K V : Mat n d) :
    HasVJPMat (sdpa_Q_chain n d K V) :=
  -- Innermost (matmul Q' Kt → scalar scale):
  let inner_has_vjp :=
    vjpMat_comp _ (fun s : Mat n n => fun r c => sdpa_scale d * s r c)
      (matmul_right_const_flat_diff (Mat.transpose K))
      (scalarScale_flat_diff (sdpa_scale d))
      (matmul_right_const_has_vjp (Mat.transpose K))
      (scalarScale_has_vjp (sdpa_scale d))
  -- Diff of the innermost composition (scalar_scale ∘ matmul_right_const) — linear in v.
  have inner_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun s : Mat n n => fun r c => sdpa_scale d * s r c)
          ((fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K)) (Mat.unflatten v)))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul; fun_prop
  -- Middle chain (… → rowSoftmax):
  let middle_has_vjp :=
    vjpMat_comp _ (@rowSoftmax n n)
      inner_diff (rowSoftmax_flat_diff n n)
      inner_has_vjp (rowSoftmax_has_vjp_mat' n n)
  -- Diff of the middle composition (rowSoftmax ∘ scaled-matmul) via composition.
  have middle_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) (((fun s : Mat n n => fun r c => sdpa_scale d * s r c) ∘
          (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))) (Mat.unflatten v)))) := by
    have h_eq : (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) (((fun s : Mat n n => fun r c => sdpa_scale d * s r c) ∘
          (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))) (Mat.unflatten v)))) =
        (fun u : Vec (n * n) => Mat.flatten ((@rowSoftmax n n) (Mat.unflatten u))) ∘
        (fun v : Vec (n * d) =>
          Mat.flatten (((fun s : Mat n n => fun r c => sdpa_scale d * s r c) ∘
            (fun Q' : Mat n d => Mat.mul Q' (Mat.transpose K))) (Mat.unflatten v))) := by
      funext v; simp [Mat.unflatten_flatten]
    rw [h_eq]
    exact (rowSoftmax_flat_diff n n).comp inner_diff
  -- Outermost (… → matmul w V):
  vjpMat_comp _ (fun w : Mat n n => Mat.mul w V)
    middle_diff (matmul_right_const_flat_diff V)
    middle_has_vjp
    (matmul_right_const_has_vjp V)

/-- **Correctness of `sdpa_back_Q`** — proved, no sorry.

    Two moves: (1) replace `fun Q' => sdpa n d Q' K V` by the chain via
    `sdpa_Q_chain_eq`; (2) apply the chain's `.correct` and verify that
    the chain's backward reduces to `sdpa_back_Q` (pure unfolding). -/
theorem sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l := by
  have hfwd : (fun Q' : Mat n d => sdpa n d Q' K V) = sdpa_Q_chain n d K V := by
    funext Q'; exact (sdpa_Q_chain_eq n d Q' K V).symm
  rw [hfwd]
  rw [← (sdpa_Q_chain_has_vjp n d K V).correct Q dOut i j]
  -- Goal: sdpa_back_Q ... = (sdpa_Q_chain_has_vjp ...).backward Q dOut i j
  unfold sdpa_back_Q sdpa_dScores sdpa_dScaled sdpa_dWeights sdpa_weights
    sdpa_Q_chain_has_vjp
  rfl

/-! ## K case

K enters through a transpose before the first matmul. One extra step in
the chain: K ↦ K^T, then follow the Q chain (but with the matmul being
"left factor constant" this time because Q is fixed and K^T is on the
right). -/

noncomputable def sdpa_K_chain (n d : Nat) (Q V : Mat n d) : Mat n d → Mat n d :=
  (fun w : Mat n n => Mat.mul w V) ∘
  (@rowSoftmax n n) ∘
  (fun s : Mat n n => fun r c => sdpa_scale d * s r c) ∘
  (fun Kt' : Mat d n => Mat.mul Q Kt') ∘
  (fun K' : Mat n d => Mat.transpose K')

theorem sdpa_K_chain_eq (n d : Nat) (Q K V : Mat n d) :
    sdpa_K_chain n d Q V K = sdpa n d Q K V := by
  unfold sdpa_K_chain sdpa sdpa_scale
  rfl

noncomputable def sdpa_K_chain_has_vjp (n d : Nat) (Q V : Mat n d) :
    HasVJPMat (sdpa_K_chain n d Q V) :=
  -- Innermost (transpose → matmul Q · Kt):
  let l1_has_vjp :=
    vjpMat_comp _ (fun Kt' : Mat d n => Mat.mul Q Kt')
      transpose_flat_diff
      (matmul_left_const_flat_diff Q)
      (@transpose_has_vjp n d)
      (matmul_left_const_has_vjp Q)
  have l1_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun Kt' : Mat d n => Mat.mul Q Kt')
          (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul Mat.transpose; fun_prop
  -- Add scalar scale:
  let l2_has_vjp :=
    vjpMat_comp _ (fun s : Mat n n => fun r c => sdpa_scale d * s r c)
      l1_diff
      (scalarScale_flat_diff (sdpa_scale d))
      l1_has_vjp
      (scalarScale_has_vjp (sdpa_scale d))
  have l2_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((fun s : Mat n n => fun r c => sdpa_scale d * s r c)
          ((fun Kt' : Mat d n => Mat.mul Q Kt')
            (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n)))) := by
    unfold Mat.unflatten Mat.flatten Mat.mul Mat.transpose; fun_prop
  -- Add rowSoftmax:
  let l3_has_vjp :=
    vjpMat_comp _ (@rowSoftmax n n)
      l2_diff (rowSoftmax_flat_diff n n)
      l2_has_vjp (rowSoftmax_has_vjp_mat' n n)
  have l3_diff : Differentiable ℝ
      (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) ((fun s : Mat n n => fun r c => sdpa_scale d * s r c)
          ((fun Kt' : Mat d n => Mat.mul Q Kt')
            (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n))))) := by
    have h_eq : (fun v : Vec (n * d) =>
        Mat.flatten ((@rowSoftmax n n) ((fun s : Mat n n => fun r c => sdpa_scale d * s r c)
          ((fun Kt' : Mat d n => Mat.mul Q Kt')
            (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n))))) =
        (fun u : Vec (n * n) => Mat.flatten ((@rowSoftmax n n) (Mat.unflatten u))) ∘
        (fun v : Vec (n * d) =>
          Mat.flatten ((fun s : Mat n n => fun r c => sdpa_scale d * s r c)
            ((fun Kt' : Mat d n => Mat.mul Q Kt')
              (Mat.transpose (Mat.unflatten v : Mat n d) : Mat d n)))) := by
      funext v; simp [Mat.unflatten_flatten]
    rw [h_eq]; exact (rowSoftmax_flat_diff n n).comp l2_diff
  -- Outermost (… → matmul w V):
  vjpMat_comp _ (fun w : Mat n n => Mat.mul w V)
    l3_diff (matmul_right_const_flat_diff V)
    l3_has_vjp
    (matmul_right_const_has_vjp V)

/-- **Correctness of `sdpa_back_K`** — proved, no sorry.

    Same shape as Q, but the chain goes through a leading transpose
    step. The resulting backward computes `∑ k, Q k j * dScores k i`
    whereas `sdpa_back_K` is `Mat.mul (Mat.transpose dScores) Q`, which
    expands to `∑ k, dScores k i * Q k j`. Equal by `mul_comm` at the
    summand level. -/
theorem sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l := by
  have hfwd : (fun K' : Mat n d => sdpa n d Q K' V) = sdpa_K_chain n d Q V := by
    funext K'; exact (sdpa_K_chain_eq n d Q K' V).symm
  rw [hfwd]
  rw [← (sdpa_K_chain_has_vjp n d Q V).correct K dOut i j]
  unfold sdpa_back_K sdpa_dScores sdpa_dScaled sdpa_dWeights sdpa_weights
    sdpa_K_chain_has_vjp vjpMat_comp
    matmul_right_const_has_vjp matmul_left_const_has_vjp transpose_has_vjp
    scalarScale_has_vjp rowSoftmax_has_vjp_mat' rowSoftmax_has_vjp_mat
    softmax_has_vjp rowSoftmax
  -- Both sides now in sum-of-products form; differ only by mul_comm at the summand.
  simp only [Mat.mul, Mat.transpose, Function.comp]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- The final matmul in SDPA: for fixed Q, K, the function `V' ↦ sdpa Q K V'`
    is `V' ↦ W · V'` where `W = sdpa_weights Q K`. Pure rewrite; definitional. -/
theorem sdpa_eq_mul_weights (n d : Nat) (Q K V : Mat n d) :
    sdpa n d Q K V = Mat.mul (sdpa_weights n d Q K) V := by
  unfold sdpa sdpa_weights sdpa_scale
  rfl

/-- **Correctness of `sdpa_back_V`** — proved, no sorry.

    The V-path is the simplest case: `V'` only enters through the final
    matmul `out = weights · V'`. So `fun V' => sdpa n d Q K V'` is just
    `fun V' => Mat.mul W V'` where W is fixed (= `sdpa_weights n d Q K`),
    and the VJP comes directly from `matmul_left_const_has_vjp`. -/
theorem sdpa_back_V_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_V n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l := by
  -- Replace `fun V' => sdpa n d Q K V'` by `fun V' => Mat.mul W V'`.
  have hfwd : (fun V' : Mat n d => sdpa n d Q K V') =
              (fun V' : Mat n d => Mat.mul (sdpa_weights n d Q K) V') := by
    funext V'; exact sdpa_eq_mul_weights n d Q K V'
  rw [hfwd]
  -- Apply the matmul VJP correctness backward (i.e., rewrite the RHS
  -- into the VJP's backward) and then match `sdpa_back_V`.
  rw [← (matmul_left_const_has_vjp (sdpa_weights n d Q K)).correct V dOut i j]
  -- Goal: sdpa_back_V n d Q K V dOut i j = Σ k, W k i * dOut k j
  unfold sdpa_back_V Mat.mul Mat.transpose
  rfl

/-- **Bundled SDPA ternary VJP.** Packages `sdpa_back_{Q, K, V}_correct`
    into a single `HasVJPMat3` instance. The backward triple
    `(sdpa_back_Q, sdpa_back_K, sdpa_back_V)` gives per-input
    gradients; correctness is the three existing per-input theorems
    in one structure. -/
noncomputable def sdpa_has_vjp_mat3 (n d : Nat) :
    HasVJPMat3 (sdpa n d) where
  backward := fun Q K V dY =>
    (sdpa_back_Q n d Q K V dY,
     sdpa_back_K n d Q K V dY,
     sdpa_back_V n d Q K V dY)
  correct_1 := sdpa_back_Q_correct n d
  correct_2 := sdpa_back_K_correct n d
  correct_3 := sdpa_back_V_correct n d

-- ════════════════════════════════════════════════════════════════
-- § 3. Multi-Head wrapping (Phase 3 — proved via column-stacking)
-- ════════════════════════════════════════════════════════════════

/-! ## Multi-head: parallelism over a partition

Multi-head attention is:

  1. Project `X : Mat N D` three ways: `Q = X·Wq + bq`, `K = X·Wk + bk`, `V = X·Wv + bv`.
  2. Reshape each projection `(N, D) → (N, heads, d_head)` by slicing the feature axis.
  3. Run SDPA independently on each of the `heads` slices.
  4. Concatenate the head outputs back to `(N, D)`.
  5. Apply the output projection `Y = concat · Wo + bo`.

In the MLIR (`emitMHSAForward`):
    reshape (B, N, D) -> (B, N, H, D_h)
    transpose -> (B, H, N, D_h)
    [SDPA per head, using batching_dims = [0, 1]]
    transpose -> (B, N, H, D_h)
    reshape -> (B, N, D)
    dense projection (the "output projection" `Wo`)

**Earlier** this section just narrated "no new VJP math" and moved on.
The current state proves `mhsa_has_vjp_mat` end-to-end: we *define*
`mhsa_layer` concretely in Lean (Q/K/V projections → per-head slice
→ sdpa-per-head → concat → Wo projection), then prove its `HasVJPMat`
via the new `pdivMat_colIndep` + `colSlabwise_has_vjp_mat` framework
(Phase 3, Apr 2026), which lifts the per-head SDPA backward over the
head axis. Formula remains numerically gradient-checked in
`check_axioms.py` for cross-validation. -/

/-- Multi-head SDPA on a single sequence: `Mat N (heads·d_head) → Mat N (heads·d_head)`.

    Concretely defined (not opaque):
    1. Q, K, V projections (each a per-token dense with its own Wq/Wk/Wv).
    2. For each head `h : Fin heads`, extract the `(N, d_head)` slice of Q/K/V
       by indexing `finProdFinEquiv (h, k)` in the combined axis.
    3. Run `sdpa` on each slice.
    4. Concatenate the head outputs back along the feature axis.
    5. Output projection Wo · concat + bo (per-token dense).

    The bundled VJP theorem below packages the correctness of this
    whole thing — composing dense Jacobians, the per-head SDPA
    jacobians (we already proved `sdpa_back_{Q,K,V}_correct`), and
    the reshape/unreshape `pdiv_reindex` facts, with the per-head
    independence handled by Phase 3's column-stacking framework
    (`pdivMat_colIndep` + `colSlabwise_has_vjp_mat`). -/
noncomputable def mhsa_layer (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) : Mat N (heads * d_head) :=
  let D := heads * d_head
  -- Q / K / V projections
  let Q : Mat N D := fun n j => (∑ k : Fin D, X n k * Wq k j) + bq j
  let K : Mat N D := fun n j => (∑ k : Fin D, X n k * Wk k j) + bk j
  let V : Mat N D := fun n j => (∑ k : Fin D, X n k * Wv k j) + bv j
  -- Per-head SDPA. `finProdFinEquiv (h, j) : Fin (heads * d_head)` picks out
  -- column `j` of head `h`. Extract, apply sdpa, note the result per head.
  let perHead : Fin heads → Mat N d_head := fun h =>
    let Qh : Mat N d_head := fun n j => Q n (finProdFinEquiv (h, j))
    let Kh : Mat N d_head := fun n j => K n (finProdFinEquiv (h, j))
    let Vh : Mat N d_head := fun n j => V n (finProdFinEquiv (h, j))
    sdpa N d_head Qh Kh Vh
  -- Concatenate heads: output[n, fPF(h, j)] = perHead h n j.
  let concat : Mat N D := fun n hj =>
    let hj' := finProdFinEquiv.symm hj
    perHead hj'.1 n hj'.2
  -- Output projection
  fun n j => (∑ k : Fin D, concat n k * Wo k j) + bo j

/-! ## Phase 3: Column-stacked SDPA — the bridge from `HasVJPMat3` to multi-head.

    The two `mhsa_*` axioms below were the project floor for two reasons:
    (1) joint differentiability of `(Q, K, V) ↦ sdpa Q K V`, which doesn't
    follow from the existing per-input `_flat_diff` lemmas; (2) the per-head
    "vmap" structure, which `colSlabwise_has_vjp_mat` (Phase 1) handles for
    *unary* per-slab functions but SDPA is naturally ternary.

    The fix: column-stack `(Q | K | V)` into a single `Mat n (3 * d_head)`
    "qkv slab", define `mhsa_g : Mat n (3 * d_head) → Mat n d_head` as the
    unary view of SDPA on this slab, and lift via the existing framework.

    Both `mhsa_g_flat_diff` and `mhsa_g_has_vjp_mat` are then mechanical
    composition of existing pieces: the joint `_flat_diff` factors through
    `rowSoftmax_flat_diff` after stage-by-stage chaining, and the VJP comes
    from `sdpa_has_vjp_mat3` plus a "column-third projection" argument that
    matches the `(c : Fin 3)` index of the slab to the Q/K/V partial. -/

/-- Column-stacked SDPA: takes a slab `Mat n (3 * d_head)` whose columns
    encode `(c : Fin 3, j : Fin d_head)` via `finProdFinEquiv`, with `c = 0`
    being the Q-third, `c = 1` the K-third, `c = 2` the V-third. Returns
    `sdpa` applied to those three thirds. -/
noncomputable def mhsa_g (n d : Nat) (slab : Mat n (3 * d)) : Mat n d :=
  sdpa n d
    (fun r j => slab r (finProdFinEquiv ((0 : Fin 3), j)))
    (fun r j => slab r (finProdFinEquiv ((1 : Fin 3), j)))
    (fun r j => slab r (finProdFinEquiv ((2 : Fin 3), j)))

/-- Pre-softmax matrix in `mhsa_g`: `scale · Q · K^T` as a function of slab.
    Each entry is a polynomial in the slab's coords (linear projections
    times each other), so `fun_prop` discharges flat-diff after unfolding. -/
noncomputable def mhsa_pre_weights (n d : Nat) (slab : Mat n (3 * d)) : Mat n n :=
  fun r c => sdpa_scale d *
    Mat.mul
      (fun r' j => slab r' (finProdFinEquiv ((0 : Fin 3), j)))
      (Mat.transpose (fun r' j => slab r' (finProdFinEquiv ((1 : Fin 3), j))))
      r c

theorem mhsa_pre_weights_flat_diff (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_pre_weights n d) (Mat.unflatten v))) := by
  unfold mhsa_pre_weights Mat.flatten Mat.unflatten Mat.mul Mat.transpose
  fun_prop

/-- Post-softmax weights in `mhsa_g`: `rowSoftmax(scale · Q · K^T)`. -/
noncomputable def mhsa_weights (n d : Nat) (slab : Mat n (3 * d)) : Mat n n :=
  rowSoftmax (mhsa_pre_weights n d slab)

theorem mhsa_weights_flat_diff (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_weights n d) (Mat.unflatten v))) := by
  have h_eq : (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_weights n d) (Mat.unflatten v))) =
      (fun u : Vec (n * n) => Mat.flatten (rowSoftmax (Mat.unflatten u))) ∘
      (fun v : Vec (n * (3 * d)) =>
        Mat.flatten ((mhsa_pre_weights n d) (Mat.unflatten v))) := by
    funext v
    show Mat.flatten (rowSoftmax (mhsa_pre_weights n d (Mat.unflatten v))) =
         Mat.flatten (rowSoftmax (Mat.unflatten
           (Mat.flatten (mhsa_pre_weights n d (Mat.unflatten v)))))
    rw [Mat.unflatten_flatten]
  rw [h_eq]
  exact (rowSoftmax_flat_diff n n).comp (mhsa_pre_weights_flat_diff n d)

/-- **Joint flat-diff of column-stacked SDPA.**

    The blocker for Phase 3 (per `mhsa.md`): joint diff in `(Q, K, V)`
    doesn't follow from the existing per-input `_flat_diff` lemmas. Here
    we prove it by treating the qkv-slab as the variable, factoring SDPA
    as `Mat.mul ∘ rowSoftmax ∘ scaled-matmul`, and chaining: pre-softmax
    is fun_prop-able (polynomial in slab coords), rowSoftmax composes via
    `rowSoftmax_flat_diff`, final matmul-with-V splits per output coord
    into a sum of products of two diff scalars. -/
theorem mhsa_g_flat_diff (n d : Nat) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_g n d) (Mat.unflatten v))) := by
  rw [differentiable_pi]
  intro idx
  set p := finProdFinEquiv.symm idx with hp_def
  -- The idx-th coord = (mhsa_g (unflatten v))[p.1, p.2]
  --                  = Σ s, weights[p.1, s] * V[s, p.2]
  --                  = Σ s, Mat.flatten weights (fPF(p.1, s)) * v (fPF(s, fPF3(2, p.2)))
  have h_eq : (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_g n d) (Mat.unflatten v)) idx) =
      (fun v : Vec (n * (3 * d)) =>
        ∑ s : Fin n,
          Mat.flatten ((mhsa_weights n d) (Mat.unflatten v)) (finProdFinEquiv (p.1, s)) *
          v (finProdFinEquiv (s, finProdFinEquiv ((2 : Fin 3), p.2)))) := by
    funext v
    show (mhsa_g n d (Mat.unflatten v)) p.1 p.2 = _
    unfold mhsa_g sdpa
    show Mat.mul (rowSoftmax (fun i j => sdpa_scale d *
      Mat.mul
        (fun r j' => Mat.unflatten v r (finProdFinEquiv ((0 : Fin 3), j')))
        (Mat.transpose (fun r j' => Mat.unflatten v r (finProdFinEquiv ((1 : Fin 3), j'))))
        i j))
      (fun r j => Mat.unflatten v r (finProdFinEquiv ((2 : Fin 3), j))) p.1 p.2 = _
    unfold Mat.mul
    -- LHS: Σ s, weights[p.1, s] * V[s, p.2]
    -- RHS: Σ s, Mat.flatten (mhsa_weights ...) (fPF(p.1, s)) * v (fPF(s, fPF3(2, p.2)))
    apply Finset.sum_congr rfl
    intro s _
    -- The second factors `Mat.unflatten v s (fPF(2, p.2))` and `v (fPF(s, fPF(2, p.2)))`
    -- are def-equal by `Mat.unflatten`, so `congr 1` auto-closes that side; only the
    -- weights side remains.
    congr 1
    -- Goal: rowSoftmax (...) p.1 s = Mat.flatten (mhsa_weights …) (fPF(p.1, s))
    show rowSoftmax _ p.1 s = Mat.flatten ((mhsa_weights n d) (Mat.unflatten v)) _
    unfold Mat.flatten mhsa_weights
    simp only [Equiv.symm_apply_apply]
    show rowSoftmax _ p.1 s = rowSoftmax (mhsa_pre_weights n d (Mat.unflatten v)) p.1 s
    unfold mhsa_pre_weights Mat.mul
    rfl
  rw [h_eq]
  -- Each summand is product of two differentiable scalar functions.
  apply Differentiable.fun_sum
  intro s _
  have h_w : Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_weights n d) (Mat.unflatten v)) (finProdFinEquiv (p.1, s))) :=
    fun v => differentiableAt_pi.mp ((mhsa_weights_flat_diff n d) v) _
  have h_v : Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      v (finProdFinEquiv (s, finProdFinEquiv ((2 : Fin 3), p.2)))) := by
    fun_prop
  exact h_w.mul h_v

/-! ### Column-stacked SDPA VJP

    `HasVJPMat (mhsa_g n d)`: the backward column-stacks
    `(sdpa_back_Q, sdpa_back_K, sdpa_back_V)` according to the c-third
    of the slab column index. Correctness reduces to `sdpa_has_vjp_mat3`
    after observing that perturbing the c-th third of the slab only
    perturbs the c-th input of SDPA. -/

/-- Column projection `slab ↦ slab^[c]` for a fixed `c : Fin 3`.
    Linear, so its flat form is a `reindexCLM`. -/
noncomputable def mhsa_proj_c {n d : Nat} (c : Fin 3) (slab : Mat n (3 * d)) : Mat n d :=
  fun r j => slab r (finProdFinEquiv (c, j))

theorem mhsa_proj_c_flat_diff (n d : Nat) (c : Fin 3) :
    Differentiable ℝ (fun v : Vec (n * (3 * d)) =>
      Mat.flatten ((mhsa_proj_c c) (Mat.unflatten v) : Mat n d)) := by
  unfold mhsa_proj_c Mat.flatten Mat.unflatten
  fun_prop

/-- The flat form of the column projection `mhsa_proj_c c` is exactly the
    `reindexCLM` `σ_c idx = fPF((decode idx).1, fPF(c, (decode idx).2))`. -/
noncomputable def mhsa_proj_c_CLM (n d : Nat) (c : Fin 3) :
    Vec (n * (3 * d)) →L[ℝ] Vec (n * d) :=
  reindexCLM (fun idx : Fin (n * d) =>
    finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                     finProdFinEquiv (c, (finProdFinEquiv.symm idx).2)))

theorem mhsa_proj_c_eq_CLM (n d : Nat) (c : Fin 3) (v : Vec (n * (3 * d))) :
    Mat.flatten ((mhsa_proj_c c) (Mat.unflatten v) : Mat n d) = mhsa_proj_c_CLM n d c v := by
  funext idx
  show Mat.flatten (fun r j => Mat.unflatten v r (finProdFinEquiv (c, j))) idx =
       v (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                           finProdFinEquiv (c, (finProdFinEquiv.symm idx).2)))
  unfold Mat.flatten Mat.unflatten
  rfl

/-- "Lift to slab third c": embeds `Vec (n * d)` into `Vec (n * (3 * d))` by
    placing `u` in the c-th column third and zero elsewhere. Linear, hence
    a CLM. The dual of `mhsa_proj_c_CLM`. Constructed from per-coord CLMs
    via `ContinuousLinearMap.pi`: each output coord is either a projection
    (if the index is in the c-third) or zero. -/
noncomputable def mhsa_lift_c_CLM (n d : Nat) (c : Fin 3) :
    Vec (n * d) →L[ℝ] Vec (n * (3 * d)) :=
  ContinuousLinearMap.pi (fun idx : Fin (n * (3 * d)) =>
    if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
    then ContinuousLinearMap.proj
      (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
    else 0)

theorem mhsa_lift_c_CLM_apply (n d : Nat) (c : Fin 3) (u : Vec (n * d))
    (idx : Fin (n * (3 * d))) :
    mhsa_lift_c_CLM n d c u idx =
      (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
       then u (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                                (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
       else 0) := by
  show (ContinuousLinearMap.pi _) u idx = _
  rw [ContinuousLinearMap.pi_apply]
  by_cases hc : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
  · rw [if_pos hc, if_pos hc]
    rfl
  · rw [if_neg hc, if_neg hc]
    rfl

/-- "Embed Q' into slab at the c-th third, keep other thirds at slab's values."
    Affine function: `mhsa_lift_c_CLM c · u + (slab with c-th third zeroed)`. -/
noncomputable def mhsa_embed_c (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) : Vec (n * (3 * d)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = c then u (finProdFinEquiv (p.1, q.2)) else Mat.flatten slab idx

theorem mhsa_embed_c_eq (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) :
    mhsa_embed_c n d c slab u = mhsa_lift_c_CLM n d c u +
      (fun idx =>
        if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
        then 0 else Mat.flatten slab idx) := by
  funext idx
  show (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
        then u (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
        else Mat.flatten slab idx) =
       mhsa_lift_c_CLM n d c u idx +
       (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
        then 0 else Mat.flatten slab idx)
  rw [mhsa_lift_c_CLM_apply]
  by_cases hcond : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
  · rw [if_pos hcond, if_pos hcond, if_pos hcond, add_zero]
  · rw [if_neg hcond, if_neg hcond, if_neg hcond, zero_add]

theorem mhsa_embed_c_hasFDerivAt (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u₀ : Vec (n * d)) :
    HasFDerivAt (mhsa_embed_c n d c slab) (mhsa_lift_c_CLM n d c) u₀ := by
  rw [show (mhsa_embed_c n d c slab : Vec (n * d) → Vec (n * (3 * d))) =
        (fun u => mhsa_lift_c_CLM n d c u +
          (fun idx =>
            let p := finProdFinEquiv.symm idx
            let q := finProdFinEquiv.symm p.2
            if q.1 = c then 0 else Mat.flatten slab idx))
      from funext (mhsa_embed_c_eq n d c slab)]
  exact (mhsa_lift_c_CLM n d c).hasFDerivAt.add_const _

/-- The composition `mhsa_g ∘ mhsa_embed_c c slab` equals "SDPA with the c-th
    argument variable, the other two fixed at `slab`'s projections". This is
    the freezing identity. -/
theorem mhsa_g_comp_embed (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d))
    (u : Vec (n * d)) :
    Mat.flatten ((mhsa_g n d) (Mat.unflatten (mhsa_embed_c n d c slab u))) =
      (if c = (0 : Fin 3) then
         Mat.flatten (sdpa n d (Mat.unflatten u)
                        (mhsa_proj_c (1 : Fin 3) slab) (mhsa_proj_c (2 : Fin 3) slab))
       else if c = (1 : Fin 3) then
         Mat.flatten (sdpa n d (mhsa_proj_c (0 : Fin 3) slab)
                        (Mat.unflatten u) (mhsa_proj_c (2 : Fin 3) slab))
       else
         Mat.flatten (sdpa n d (mhsa_proj_c (0 : Fin 3) slab)
                        (mhsa_proj_c (1 : Fin 3) slab) (Mat.unflatten u))) := by
  -- Three cases on c. For each, show that the c-th projection of
  -- (mhsa_embed_c c slab u) is `Mat.unflatten u` and the other two are `mhsa_proj_c · slab`.
  have h_proj_match : ∀ (c' : Fin 3),
      mhsa_proj_c c' (Mat.unflatten (mhsa_embed_c n d c slab u) : Mat n (3 * d)) =
      (if c' = c then (Mat.unflatten u : Mat n d) else mhsa_proj_c c' slab) := by
    intro c'
    funext r j
    show Mat.unflatten (mhsa_embed_c n d c slab u) r (finProdFinEquiv (c', j)) = _
    unfold Mat.unflatten mhsa_embed_c
    -- Unfold and decode the index.
    show (if (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv (r, finProdFinEquiv (c', j)))).2).1 = c
          then u (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv (r, finProdFinEquiv (c', j)))).1,
                                   (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv (r, finProdFinEquiv (c', j)))).2).2))
          else Mat.flatten slab (finProdFinEquiv (r, finProdFinEquiv (c', j)))) = _
    simp only [Equiv.symm_apply_apply]
    by_cases hc' : c' = c
    · subst hc'
      rw [if_pos rfl]
      simp [if_pos rfl]
    · rw [if_neg hc']
      rw [if_neg hc']
      unfold Mat.flatten mhsa_proj_c
      simp only [Equiv.symm_apply_apply]
  unfold mhsa_g
  by_cases hc0 : c = (0 : Fin 3)
  · subst hc0
    rw [if_pos rfl]
    have h0 := h_proj_match (0 : Fin 3)
    have h1 := h_proj_match (1 : Fin 3)
    have h2 := h_proj_match (2 : Fin 3)
    simp [if_pos rfl] at h0
    simp [show (1 : Fin 3) ≠ (0 : Fin 3) from by decide] at h1
    simp [show (2 : Fin 3) ≠ (0 : Fin 3) from by decide] at h2
    show Mat.flatten (sdpa n d
      (fun r j => (Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u) : Mat n (3 * d)) r (finProdFinEquiv ((0 : Fin 3), j)))
      (fun r j => Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u) r (finProdFinEquiv ((1 : Fin 3), j)))
      (fun r j => Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u) r (finProdFinEquiv ((2 : Fin 3), j)))) = _
    show Mat.flatten (sdpa n d (mhsa_proj_c (0 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u)))
      (mhsa_proj_c (1 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u)))
      (mhsa_proj_c (2 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (0 : Fin 3) slab u)))) = _
    rw [h0, h1, h2]
  · rw [if_neg hc0]
    by_cases hc1 : c = (1 : Fin 3)
    · subst hc1
      rw [if_pos rfl]
      have h0 := h_proj_match (0 : Fin 3)
      have h1 := h_proj_match (1 : Fin 3)
      have h2 := h_proj_match (2 : Fin 3)
      simp [show (0 : Fin 3) ≠ (1 : Fin 3) from by decide] at h0
      simp [if_pos rfl] at h1
      simp [show (2 : Fin 3) ≠ (1 : Fin 3) from by decide] at h2
      show Mat.flatten (sdpa n d (mhsa_proj_c (0 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (1 : Fin 3) slab u)))
        (mhsa_proj_c (1 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (1 : Fin 3) slab u)))
        (mhsa_proj_c (2 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (1 : Fin 3) slab u)))) = _
      rw [h0, h1, h2]
    · rw [if_neg hc1]
      have hc2 : c = (2 : Fin 3) := by
        fin_cases c
        · exact absurd rfl hc0
        · exact absurd rfl hc1
        · rfl
      subst hc2
      have h0 := h_proj_match (0 : Fin 3)
      have h1 := h_proj_match (1 : Fin 3)
      have h2 := h_proj_match (2 : Fin 3)
      simp [show (0 : Fin 3) ≠ (2 : Fin 3) from by decide] at h0
      simp [show (1 : Fin 3) ≠ (2 : Fin 3) from by decide] at h1
      simp [if_pos rfl] at h2
      show Mat.flatten (sdpa n d (mhsa_proj_c (0 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (2 : Fin 3) slab u)))
        (mhsa_proj_c (1 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (2 : Fin 3) slab u)))
        (mhsa_proj_c (2 : Fin 3) (Mat.unflatten (mhsa_embed_c n d (2 : Fin 3) slab u)))) = _
      rw [h0, h1, h2]

theorem mhsa_embed_c_at_proj (n d : Nat) (c : Fin 3) (slab : Mat n (3 * d)) :
    mhsa_embed_c n d c slab (Mat.flatten (mhsa_proj_c c slab)) = Mat.flatten slab := by
  funext idx
  show (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
        then Mat.flatten (mhsa_proj_c c slab)
              (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                                (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2))
        else Mat.flatten slab idx) = _
  by_cases hcond : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
  · rw [if_pos hcond]
    show Mat.flatten (mhsa_proj_c c slab)
          (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                            (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2)) =
         Mat.flatten slab idx
    unfold Mat.flatten mhsa_proj_c
    simp only [Equiv.symm_apply_apply]
    -- slab[(decode idx).1, fPF(c, (decode (decode idx).2).2)] = slab[(decode idx).1, (decode idx).2]
    -- Need: fPF(c, (decode (decode idx).2).2) = (decode idx).2
    -- (decode idx).2 : Fin (3 * d) decodes as (q.1, q.2). hcond: q.1 = c. Hence fPF(c, q.2) = fPF(q.1, q.2) = (decode idx).2.
    show slab (finProdFinEquiv.symm idx).1
              (finProdFinEquiv (c, (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2)) =
         slab (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
    congr 1
    rw [show c = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 from hcond.symm]
    exact (Equiv.apply_symm_apply _ _)
  · rw [if_neg hcond]

/-- **Helper for `pdivMat_mhsa_g_split` (per-c chain rule).**
    For each `c : Fin 3`, the chain rule gives:
    `fderiv flat_g flat_slab ∘L mhsa_lift_c_CLM = fderiv flat_freeze_c flat_proj_c_slab`.
    Used in `pdivMat_mhsa_g_split` after the basis-vector lift identity. -/
theorem pdivMat_mhsa_g_split_chain (n d : Nat) (slab : Mat n (3 * d)) (c : Fin 3)
    (freeze_fn : Mat n d → Mat n d)
    (h_g_freeze_eq : ∀ u : Vec (n * d),
      Mat.flatten ((mhsa_g n d) (Mat.unflatten (mhsa_embed_c n d c slab u))) =
      Mat.flatten (freeze_fn (Mat.unflatten u))) :
    (fderiv ℝ (fun v : Vec (n * (3 * d)) => Mat.flatten ((mhsa_g n d) (Mat.unflatten v)))
              (Mat.flatten slab)).comp (mhsa_lift_c_CLM n d c) =
    fderiv ℝ (fun u : Vec (n * d) => Mat.flatten (freeze_fn (Mat.unflatten u)))
              (Mat.flatten (mhsa_proj_c c slab)) := by
  set flat_g : Vec (n * (3 * d)) → Vec (n * d) := fun v =>
    Mat.flatten ((mhsa_g n d) (Mat.unflatten v))
  set flat_freeze : Vec (n * d) → Vec (n * d) := fun u =>
    Mat.flatten (freeze_fn (Mat.unflatten u))
  set u₀ : Vec (n * d) := Mat.flatten (mhsa_proj_c c slab)
  -- flat_g ∘ embed = flat_freeze (pointwise, by hypothesis).
  have h_comp : flat_g ∘ (mhsa_embed_c n d c slab) = flat_freeze := by
    funext u
    exact h_g_freeze_eq u
  -- HasFDerivAt for embed at u₀.
  have h_embed_at : mhsa_embed_c n d c slab u₀ = Mat.flatten slab := mhsa_embed_c_at_proj n d c slab
  have h_embed_diff : HasFDerivAt (mhsa_embed_c n d c slab) (mhsa_lift_c_CLM n d c) u₀ :=
    mhsa_embed_c_hasFDerivAt n d c slab u₀
  -- HasFDerivAt for flat_g at (embed u₀) = Mat.flatten slab.
  have h_g_at : HasFDerivAt flat_g (fderiv ℝ flat_g (Mat.flatten slab)) (mhsa_embed_c n d c slab u₀) := by
    rw [h_embed_at]
    exact ((mhsa_g_flat_diff n d) (Mat.flatten slab)).hasFDerivAt
  -- Composition gives HasFDerivAt for flat_g ∘ embed = flat_freeze.
  have h_chain : HasFDerivAt (flat_g ∘ mhsa_embed_c n d c slab)
      ((fderiv ℝ flat_g (Mat.flatten slab)).comp (mhsa_lift_c_CLM n d c)) u₀ :=
    h_g_at.comp u₀ h_embed_diff
  rw [h_comp] at h_chain
  -- Conclude: fderiv freeze u₀ = the comp.
  exact h_chain.fderiv.symm

/-- **`pdivMat` of `mhsa_g` splits per-c into the corresponding `pdivMat` of
    SDPA against its c-th argument.** The freezing lemma: changes in the
    c-th column third of the slab only perturb the c-th input of SDPA.
    Proved via the chain rule `mhsa_g ∘ mhsa_embed_c = freeze_c`. -/
theorem pdivMat_mhsa_g_split (n d : Nat) (slab : Mat n (3 * d))
    (i : Fin n) (c : Fin 3) (j : Fin d) (k : Fin n) (l : Fin d) :
    pdivMat (mhsa_g n d) slab i (finProdFinEquiv (c, j)) k l =
    (if c = (0 : Fin 3) then
       pdivMat (fun Q' : Mat n d => sdpa n d Q' (mhsa_proj_c (1 : Fin 3) slab)
                                      (mhsa_proj_c (2 : Fin 3) slab))
               (mhsa_proj_c (0 : Fin 3) slab) i j k l
     else if c = (1 : Fin 3) then
       pdivMat (fun K' : Mat n d => sdpa n d (mhsa_proj_c (0 : Fin 3) slab) K'
                                      (mhsa_proj_c (2 : Fin 3) slab))
               (mhsa_proj_c (1 : Fin 3) slab) i j k l
     else
       pdivMat (fun V' : Mat n d => sdpa n d (mhsa_proj_c (0 : Fin 3) slab)
                                      (mhsa_proj_c (1 : Fin 3) slab) V')
               (mhsa_proj_c (2 : Fin 3) slab) i j k l) := by
  -- Compute mhsa_lift_c_CLM (basisVec (fPF(i, j))) = basisVec (fPF(i, fPF(c, j))).
  have h_lift_basis : mhsa_lift_c_CLM n d c (basisVec (finProdFinEquiv (i, j))) =
      basisVec (finProdFinEquiv (i, finProdFinEquiv (c, j))) := by
    funext idx
    rw [mhsa_lift_c_CLM_apply, basisVec_apply, basisVec_apply]
    -- Both basis vectors collapse to 1 at exactly one index.
    -- LHS = 1 ↔ (decode (decode idx).2).1 = c ∧ fPF((decode idx).1, (decode (decode idx).2).2) = fPF(i, j)
    -- RHS = 1 ↔ idx = fPF(i, fPF(c, j))
    -- These are equivalent by injectivity of fPF.
    by_cases hidx : idx = finProdFinEquiv (i, finProdFinEquiv (c, j))
    · subst hidx
      simp [Equiv.symm_apply_apply]
    · rw [if_neg hidx]
      -- Show LHS = 0.
      by_cases hcond : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).1 = c
      · rw [if_pos hcond, if_neg]
        intro heq
        apply hidx
        have h_inj := finProdFinEquiv.injective heq
        have h_p1 : (finProdFinEquiv.symm idx).1 = i := (Prod.mk.inj h_inj).1
        have h_p2 : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2).2 = j := (Prod.mk.inj h_inj).2
        -- Reconstruct idx = fPF(i, fPF(c, j)) from h_p1, hcond, h_p2.
        have h_inner_eq : (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2) = (c, j) := by
          apply Prod.ext
          · exact hcond
          · exact h_p2
        have h_p2_full : (finProdFinEquiv.symm idx).2 = finProdFinEquiv (c, j) := by
          rw [show (finProdFinEquiv.symm idx).2 =
                finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm idx).2)
              from (Equiv.apply_symm_apply _ _).symm]
          rw [h_inner_eq]
        have h_full : (finProdFinEquiv.symm idx) = (i, finProdFinEquiv (c, j)) :=
          Prod.ext h_p1 h_p2_full
        have key : finProdFinEquiv (finProdFinEquiv.symm idx) = idx :=
          Equiv.apply_symm_apply _ _
        rw [← key, h_full]
      · rw [if_neg hcond]
  -- The pdiv on the LHS of the goal:
  unfold pdivMat pdiv
  -- Key step: rewrite basisVec (fPF(i, fPF(c, j))) = mhsa_lift_c_CLM (basisVec (fPF(i, j))).
  rw [show basisVec (finProdFinEquiv (i, finProdFinEquiv (c, j))) =
          mhsa_lift_c_CLM n d c (basisVec (finProdFinEquiv (i, j)))
      from h_lift_basis.symm]
  -- Now: fderiv flat_g flat_slab (mhsa_lift_c_CLM (basis_e_(i,j))) (fPF(k, l))
  --    = ((fderiv flat_g flat_slab).comp (mhsa_lift_c_CLM)) (basis_e_(i,j)) (fPF(k, l))
  rw [show fderiv ℝ (fun v : Vec (n * (3 * d)) =>
              Mat.flatten ((mhsa_g n d) (Mat.unflatten v))) (Mat.flatten slab)
            (mhsa_lift_c_CLM n d c (basisVec (finProdFinEquiv (i, j))))
          = ((fderiv ℝ (fun v : Vec (n * (3 * d)) =>
                        Mat.flatten ((mhsa_g n d) (Mat.unflatten v))) (Mat.flatten slab)).comp
              (mhsa_lift_c_CLM n d c)) (basisVec (finProdFinEquiv (i, j)))
      from rfl]
  -- Case on c.
  by_cases hc0 : c = (0 : Fin 3)
  · subst hc0
    rw [if_pos rfl]
    rw [pdivMat_mhsa_g_split_chain n d slab (0 : Fin 3)
          (fun Q' => sdpa n d Q' (mhsa_proj_c (1 : Fin 3) slab) (mhsa_proj_c (2 : Fin 3) slab))
          (fun u => by
            have h := mhsa_g_comp_embed n d (0 : Fin 3) slab u
            rw [if_pos rfl] at h
            exact h)]
  · rw [if_neg hc0]
    by_cases hc1 : c = (1 : Fin 3)
    · subst hc1
      rw [if_pos rfl]
      rw [pdivMat_mhsa_g_split_chain n d slab (1 : Fin 3)
            (fun K' => sdpa n d (mhsa_proj_c (0 : Fin 3) slab) K' (mhsa_proj_c (2 : Fin 3) slab))
            (fun u => by
              have h := mhsa_g_comp_embed n d (1 : Fin 3) slab u
              rw [if_neg (by decide : (1 : Fin 3) ≠ (0 : Fin 3)), if_pos rfl] at h
              exact h)]
    · rw [if_neg hc1]
      have hc2 : c = (2 : Fin 3) := by
        fin_cases c
        · exact absurd rfl hc0
        · exact absurd rfl hc1
        · rfl
      subst hc2
      rw [pdivMat_mhsa_g_split_chain n d slab (2 : Fin 3)
            (fun V' => sdpa n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab) V')
            (fun u => by
              have h := mhsa_g_comp_embed n d (2 : Fin 3) slab u
              rw [if_neg (by decide : (2 : Fin 3) ≠ (0 : Fin 3)),
                  if_neg (by decide : (2 : Fin 3) ≠ (1 : Fin 3))] at h
              exact h)]

/-- **HasVJPMat for column-stacked SDPA.** Backward column-stacks the three
    `sdpa_back_*` outputs by their `c : Fin 3` slot. Correctness comes from
    `pdivMat_mhsa_g_split` (case-splits on c into the corresponding
    one-input SDPA pdivMat) and `sdpa_has_vjp_mat3.correct_*`. -/
noncomputable def mhsa_g_has_vjp_mat (n d : Nat) :
    HasVJPMat (mhsa_g n d) where
  backward := fun slab dY r kj =>
    let p := finProdFinEquiv.symm kj
    if p.1 = (0 : Fin 3) then
      sdpa_back_Q n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                      (mhsa_proj_c (2 : Fin 3) slab) dY r p.2
    else if p.1 = (1 : Fin 3) then
      sdpa_back_K n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                      (mhsa_proj_c (2 : Fin 3) slab) dY r p.2
    else
      sdpa_back_V n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                      (mhsa_proj_c (2 : Fin 3) slab) dY r p.2
  correct := by
    intro slab dY i kj
    -- Decompose kj as (c, j) via finProdFinEquiv.symm.
    set p := finProdFinEquiv.symm kj with hp_def
    have hkj : kj = finProdFinEquiv (p.1, p.2) := (Equiv.apply_symm_apply _ _).symm
    -- Rewrite RHS pdivMat with the (c, j) form.
    show (if p.1 = (0 : Fin 3) then
            sdpa_back_Q n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                            (mhsa_proj_c (2 : Fin 3) slab) dY i p.2
          else if p.1 = (1 : Fin 3) then
            sdpa_back_K n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                            (mhsa_proj_c (2 : Fin 3) slab) dY i p.2
          else
            sdpa_back_V n d (mhsa_proj_c (0 : Fin 3) slab) (mhsa_proj_c (1 : Fin 3) slab)
                            (mhsa_proj_c (2 : Fin 3) slab) dY i p.2)
       = ∑ k' : Fin n, ∑ l' : Fin d, pdivMat (mhsa_g n d) slab i kj k' l' * dY k' l'
    rw [hkj]
    simp_rw [pdivMat_mhsa_g_split]
    -- Now goal: ... = ∑ k' l', (if p.1 = 0 then ... else if p.1 = 1 then ... else ...) * dY[k', l']
    -- Pull the if outside the sum, then apply sdpa_back_*_correct.
    by_cases hc0 : p.1 = (0 : Fin 3)
    · rw [if_pos hc0]
      simp_rw [if_pos hc0]
      exact sdpa_back_Q_correct n d (mhsa_proj_c (0 : Fin 3) slab)
                                 (mhsa_proj_c (1 : Fin 3) slab)
                                 (mhsa_proj_c (2 : Fin 3) slab) dY i p.2
    · rw [if_neg hc0]
      simp_rw [if_neg hc0]
      by_cases hc1 : p.1 = (1 : Fin 3)
      · rw [if_pos hc1]
        simp_rw [if_pos hc1]
        exact sdpa_back_K_correct n d (mhsa_proj_c (0 : Fin 3) slab)
                                   (mhsa_proj_c (1 : Fin 3) slab)
                                   (mhsa_proj_c (2 : Fin 3) slab) dY i p.2
      · rw [if_neg hc1]
        simp_rw [if_neg hc1]
        exact sdpa_back_V_correct n d (mhsa_proj_c (0 : Fin 3) slab)
                                   (mhsa_proj_c (1 : Fin 3) slab)
                                   (mhsa_proj_c (2 : Fin 3) slab) dY i p.2

/-- Flat-diff for `colSlabApply g`: each output coord is `(g (slab h ·)) [n, j_out]`,
    factoring through the slab-projection CLM (linear) and `g` (flat-diff). -/
theorem colSlabApply_flat_diff {n heads d_in d_out : Nat}
    (g : Mat n d_in → Mat n d_out)
    (hg_diff : Differentiable ℝ
                 (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v)))) :
    Differentiable ℝ (fun v : Vec (n * (heads * d_in)) =>
      Mat.flatten (colSlabApply g (Mat.unflatten v) : Mat n (heads * d_out))) := by
  rw [differentiable_pi]
  intro idx
  set p := finProdFinEquiv.symm idx with hp_def
  set q := finProdFinEquiv.symm p.2 with hq_def
  -- Output coord (idx) decomposes via (n', (h, j_out)).
  -- Output: (colSlabApply g (unflatten v))[n', fPF(h, j_out)]
  --       = g (slab h (unflatten v))[n', j_out]
  -- Slab h is a CLM in v.
  set slabProj : Vec (n * (heads * d_in)) →L[ℝ] Vec (n * d_in) :=
    reindexCLM (fun idx' : Fin (n * d_in) =>
      finProdFinEquiv ((finProdFinEquiv.symm idx').1,
                       finProdFinEquiv (q.1, (finProdFinEquiv.symm idx').2)))
  have h_eq : (fun v : Vec (n * (heads * d_in)) =>
      Mat.flatten (colSlabApply g (Mat.unflatten v) : Mat n (heads * d_out)) idx) =
      (fun w : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten w)) (finProdFinEquiv (p.1, q.2))) ∘
      (fun v : Vec (n * (heads * d_in)) => slabProj v) := by
    funext v
    show (colSlabApply g (Mat.unflatten v) : Mat n (heads * d_out))
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = _
    show (colSlabApply g (Mat.unflatten v) : Mat n (heads * d_out)) p.1 p.2 = _
    -- Unfold colSlabApply: (n', kj) → g(slab kj.1 (unflatten v))[n', kj.2]
    show g (fun r' j_in => (Mat.unflatten v : Mat n (heads * d_in)) r'
                            (finProdFinEquiv ((finProdFinEquiv.symm p.2).1, j_in))) p.1 (finProdFinEquiv.symm p.2).2 = _
    -- The RHS unfolds (Function.comp etc.):
    show _ = Mat.flatten (g (Mat.unflatten (slabProj v))) (finProdFinEquiv (p.1, q.2))
    -- Need the slab projection to match: slab q.1 (unflatten v) = unflatten (slabProj v).
    have h_slab_eq :
        (fun r' j_in => (Mat.unflatten v : Mat n (heads * d_in)) r'
            (finProdFinEquiv ((finProdFinEquiv.symm p.2).1, j_in))) =
        (Mat.unflatten (slabProj v) : Mat n d_in) := by
      funext r' j_in
      show (Mat.unflatten v : Mat n (heads * d_in)) r'
              (finProdFinEquiv (q.1, j_in)) = _
      show v (finProdFinEquiv (r', finProdFinEquiv (q.1, j_in))) = slabProj v (finProdFinEquiv (r', j_in))
      show _ = v (finProdFinEquiv ((finProdFinEquiv.symm
                  (finProdFinEquiv (r', j_in))).1,
                  finProdFinEquiv (q.1, (finProdFinEquiv.symm
                    (finProdFinEquiv (r', j_in))).2)))
      rw [Equiv.symm_apply_apply]
    rw [h_slab_eq]
    show g (Mat.unflatten (slabProj v) : Mat n d_in) p.1 q.2 = _
    unfold Mat.flatten
    show _ = g (Mat.unflatten (slabProj v) : Mat n d_in)
              (finProdFinEquiv.symm (finProdFinEquiv (p.1, q.2))).1
              (finProdFinEquiv.symm (finProdFinEquiv (p.1, q.2))).2
    rw [Equiv.symm_apply_apply]
  rw [h_eq]
  -- The composition: (per-coord projection of g) ∘ slabProj.
  have h_outer : Differentiable ℝ (fun w : Vec (n * d_in) =>
      Mat.flatten (g (Mat.unflatten w)) (finProdFinEquiv (p.1, q.2))) :=
    fun w => differentiableAt_pi.mp (hg_diff w) _
  exact h_outer.comp slabProj.differentiable

-- ════════════════════════════════════════════════════════════════
-- § 3.5 Multi-head composition: replace the two axioms with theorems.
-- ════════════════════════════════════════════════════════════════

/-- Combined Q/K/V weight matrix: stack `Wq | Wk | Wv` with the per-head
    interleave layout. Output column `(h, c, j) ↦ (Wq | Wk | Wv)[k, fPF(h, j)]`
    based on `c : Fin 3`. Used to express the three Q/K/V projections as a
    single per-token dense, enabling clean composition with `colSlabApply mhsa_g`. -/
noncomputable def mhsa_qkv_W (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head)) :
    Mat (heads * d_head) (heads * (3 * d_head)) :=
  fun k idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = (0 : Fin 3) then Wq k (finProdFinEquiv (p.1, q.2))
    else if q.1 = (1 : Fin 3) then Wk k (finProdFinEquiv (p.1, q.2))
    else Wv k (finProdFinEquiv (p.1, q.2))

noncomputable def mhsa_qkv_b (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head)) :
    Vec (heads * (3 * d_head)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.2
    if q.1 = (0 : Fin 3) then bq (finProdFinEquiv (p.1, q.2))
    else if q.1 = (1 : Fin 3) then bk (finProdFinEquiv (p.1, q.2))
    else bv (finProdFinEquiv (p.1, q.2))

@[simp] theorem mhsa_qkv_W_eq0 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_W heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((0 : Fin 3), j))) = Wq k (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_W
  simp [Equiv.symm_apply_apply]

@[simp] theorem mhsa_qkv_W_eq1 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_W heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((1 : Fin 3), j))) = Wk k (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_W
  simp [Equiv.symm_apply_apply, show (1 : Fin 3) ≠ (0 : Fin 3) from by decide]

@[simp] theorem mhsa_qkv_W_eq2 (heads d_head : Nat)
    (Wq Wk Wv : Mat (heads * d_head) (heads * d_head))
    (k : Fin (heads * d_head)) (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_W heads d_head Wq Wk Wv k
      (finProdFinEquiv (h, finProdFinEquiv ((2 : Fin 3), j))) = Wv k (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_W
  simp [Equiv.symm_apply_apply,
        show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]

@[simp] theorem mhsa_qkv_b_eq0 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_b heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((0 : Fin 3), j))) = bq (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_b
  simp [Equiv.symm_apply_apply]

@[simp] theorem mhsa_qkv_b_eq1 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_b heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((1 : Fin 3), j))) = bk (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_b
  simp [Equiv.symm_apply_apply, show (1 : Fin 3) ≠ (0 : Fin 3) from by decide]

@[simp] theorem mhsa_qkv_b_eq2 (heads d_head : Nat)
    (bq bk bv : Vec (heads * d_head))
    (h : Fin heads) (j : Fin d_head) :
    mhsa_qkv_b heads d_head bq bk bv
      (finProdFinEquiv (h, finProdFinEquiv ((2 : Fin 3), j))) = bv (finProdFinEquiv (h, j)) := by
  unfold mhsa_qkv_b
  simp [Equiv.symm_apply_apply,
        show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]

/-- The mhsa_layer factorization: it equals
    `output_dense ∘ colSlabApply mhsa_g ∘ qkv_stack_dense`.

    All three pieces have HasVJPMat and flat-diff:
    - `qkv_stack_dense` uses `mhsa_qkv_W`, `mhsa_qkv_b` as a single per-token dense.
    - `colSlabApply mhsa_g` lifts `mhsa_g_has_vjp_mat` per-head.
    - `output_dense` is the standard per-token dense for Wo, bo. -/
theorem mhsa_layer_eq_compose (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) :
    mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo X =
    (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n))
      (colSlabApply (mhsa_g N d_head) (heads := heads)
        ((fun X' : Mat N (heads * d_head) => fun n =>
           dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))
         X)) := by
  funext n j
  -- Both sides compute the same value at (n, j).
  -- LHS = mhsa_layer ... at (n, j) = Σ k, concat[n, k] * Wo[k, j] + bo[j]
  -- RHS = output_dense ... at (n, j) = Σ k, (colSlabApply mhsa_g (qkv_stack X)) [n, k] * Wo[k, j] + bo[j]
  -- Reduce to: concat[n, k] = (colSlabApply mhsa_g (qkv_stack X))[n, k] for all k.
  show (∑ k : Fin (heads * d_head),
          (let perHead : Fin heads → Mat N d_head := fun h n' j' =>
             sdpa N d_head
               (fun n'' j'' => (fun n''' j''' => (∑ k' : Fin (heads * d_head),
                                                  X n''' k' * Wq k' j''') + bq j''')
                              n'' (finProdFinEquiv (h, j'')))
               (fun n'' j'' => (fun n''' j''' => (∑ k' : Fin (heads * d_head),
                                                  X n''' k' * Wk k' j''') + bk j''')
                              n'' (finProdFinEquiv (h, j'')))
               (fun n'' j'' => (fun n''' j''' => (∑ k' : Fin (heads * d_head),
                                                  X n''' k' * Wv k' j''') + bv j''')
                              n'' (finProdFinEquiv (h, j'')))
               n' j'
           (fun n' hj => perHead (finProdFinEquiv.symm hj).1 n' (finProdFinEquiv.symm hj).2) n k) *
            Wo k j) + bo j = _
  show _ =
    (∑ k : Fin (heads * d_head),
       (colSlabApply (mhsa_g N d_head) (heads := heads)
          (fun n' => fun idx =>
            (∑ k' : Fin (heads * d_head),
              X n' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k' idx) +
            (mhsa_qkv_b heads d_head bq bk bv) idx)) n k * Wo k j) + bo j
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  -- For each k = fPF(h, j_out): concat[n, k] = perHead h n j_out
  -- and (colSlabApply mhsa_g qkv_stack X)[n, k] = mhsa_g (slab h qkv_stack X)[n, j_out]
  -- = sdpa(slab_0_3, slab_1_3, slab_2_3)[n, j_out]
  -- where each slab equals the corresponding Q, K, V slab of X by construction.
  congr 1
  set p_k := finProdFinEquiv.symm k with hp_k_def
  show (sdpa N d_head
          (fun n'' j'' => (∑ k' : Fin (heads * d_head),
                            X n'' k' * Wq k' (finProdFinEquiv (p_k.1, j''))) +
                          bq (finProdFinEquiv (p_k.1, j'')))
          (fun n'' j'' => (∑ k' : Fin (heads * d_head),
                            X n'' k' * Wk k' (finProdFinEquiv (p_k.1, j''))) +
                          bk (finProdFinEquiv (p_k.1, j'')))
          (fun n'' j'' => (∑ k' : Fin (heads * d_head),
                            X n'' k' * Wv k' (finProdFinEquiv (p_k.1, j''))) +
                          bv (finProdFinEquiv (p_k.1, j''))))
        n p_k.2
      = (colSlabApply (mhsa_g N d_head) (heads := heads)
          (fun n' idx =>
            (∑ k' : Fin (heads * d_head),
              X n' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k' idx) +
            (mhsa_qkv_b heads d_head bq bk bv) idx)) n k
  show _ = mhsa_g N d_head
    (fun r' j_in => (fun n' idx =>
      (∑ k' : Fin (heads * d_head),
        X n' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k' idx) +
      (mhsa_qkv_b heads d_head bq bk bv) idx) r' (finProdFinEquiv (p_k.1, j_in)))
    n p_k.2
  unfold mhsa_g
  -- Three goals: Q, K, V arg equality.
  congr 1
  · -- Q part
    funext n'' j''
    show _ = (∑ k' : Fin (heads * d_head),
               X n'' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k'
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((0 : Fin 3), j'')))) +
             (mhsa_qkv_b heads d_head bq bk bv)
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((0 : Fin 3), j'')))
    simp only [mhsa_qkv_W_eq0, mhsa_qkv_b_eq0]
  · -- K part (note: congr 1 gave 3 goals; remaining ones are K and V flat)
    funext n'' j''
    show _ = (∑ k' : Fin (heads * d_head),
               X n'' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k'
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((1 : Fin 3), j'')))) +
             (mhsa_qkv_b heads d_head bq bk bv)
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((1 : Fin 3), j'')))
    simp only [mhsa_qkv_W_eq1, mhsa_qkv_b_eq1]
  · -- V part
    funext n'' j''
    show _ = (∑ k' : Fin (heads * d_head),
               X n'' k' * (mhsa_qkv_W heads d_head Wq Wk Wv) k'
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((2 : Fin 3), j'')))) +
             (mhsa_qkv_b heads d_head bq bk bv)
                 (finProdFinEquiv (p_k.1, finProdFinEquiv ((2 : Fin 3), j'')))
    simp only [mhsa_qkv_W_eq2, mhsa_qkv_b_eq2]

/-- **Multi-head SDPA VJP (Phase 8).** Now a theorem (was an axiom),
    composed from `mhsa_g_has_vjp_mat`, `colSlabwise_has_vjp_mat`, and
    the per-token dense framework. -/
noncomputable def mhsa_has_vjp_mat (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) := by
  rw [show mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo =
        (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) ∘
        (colSlabApply (mhsa_g N d_head) (heads := heads)) ∘
        (fun X' : Mat N (heads * d_head) => fun n =>
           dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))
      from by
        funext X
        exact mhsa_layer_eq_compose N heads d_head Wq Wk Wv Wo bq bk bv bo X]
  -- VJPs and diffs for each piece (inline `dense_per_token_has_vjp_mat` since
  -- it's defined later in this file; use `rowwise_has_vjp_mat` directly).
  have h_qkv_vjp : HasVJPMat (fun X' : Mat N (heads * d_head) => fun n =>
      dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n)) :=
    rowwise_has_vjp_mat (dense_has_vjp (mhsa_qkv_W heads d_head Wq Wk Wv)
                                        (mhsa_qkv_b heads d_head bq bk bv))
                        (dense_diff (mhsa_qkv_W heads d_head Wq Wk Wv)
                                    (mhsa_qkv_b heads d_head bq bk bv))
  have h_qkv_diff := dense_per_token_flat_diff
                      (N := N) (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv)
  have h_g_diff := mhsa_g_flat_diff N d_head
  have h_body_vjp : HasVJPMat (colSlabApply (mhsa_g N d_head) (heads := heads)) :=
    colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d_head) h_g_diff
  have h_body_diff : Differentiable ℝ (fun v : Vec (N * (heads * (3 * d_head))) =>
      Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads)) (Mat.unflatten v)
                   : Mat N (heads * d_head))) :=
    colSlabApply_flat_diff (mhsa_g N d_head) h_g_diff
  have h_output_vjp : HasVJPMat (fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) :=
    rowwise_has_vjp_mat (dense_has_vjp Wo bo) (dense_diff Wo bo)
  have h_output_diff := dense_per_token_flat_diff (N := N) Wo bo
  -- Compose body ∘ qkv first.
  have h_body_qkv_vjp : HasVJPMat
      ((colSlabApply (mhsa_g N d_head) (heads := heads)) ∘
       (fun X' : Mat N (heads * d_head) => fun n =>
          dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))) :=
    vjpMat_comp _ _ h_qkv_diff h_body_diff h_qkv_vjp h_body_vjp
  have h_body_qkv_diff : Differentiable ℝ
      (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads) ∘
          (fun X' : Mat N (heads * d_head) => fun n =>
            dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n)))
          (Mat.unflatten v) : Mat N (heads * d_head))) := by
    have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads) ∘
          (fun X' : Mat N (heads * d_head) => fun n =>
            dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n)))
          (Mat.unflatten v) : Mat N (heads * d_head))) =
        (fun u : Vec (N * (heads * (3 * d_head))) =>
          Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads)) (Mat.unflatten u)
                       : Mat N (heads * d_head))) ∘
        (fun v : Vec (N * (heads * d_head)) =>
          Mat.flatten ((fun X' : Mat N (heads * d_head) => fun n =>
            dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))
            (Mat.unflatten v))) := by
      funext v
      simp [Function.comp, Mat.unflatten_flatten]
    rw [h_eq]
    exact h_body_diff.comp h_qkv_diff
  -- Final compose with output.
  exact vjpMat_comp _ _ h_body_qkv_diff h_output_diff h_body_qkv_vjp h_output_vjp

/-- **Differentiability of the flattened multi-head SDPA layer** — theorem
    (was an axiom). Composition of three `_flat_diff` lemmas. -/
theorem mhsa_layer_flat_diff (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo
                     (Mat.unflatten v))) := by
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) =>
        Mat.flatten ((fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n))
                     (Mat.unflatten u))) ∘
      (fun u : Vec (N * (heads * (3 * d_head))) =>
        Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads)) (Mat.unflatten u)
                     : Mat N (heads * d_head))) ∘
      (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten ((fun X' : Mat N (heads * d_head) => fun n =>
            dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))
            (Mat.unflatten v))) := by
    funext v
    rw [mhsa_layer_eq_compose]
    simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (dense_per_token_flat_diff (N := N) Wo bo).comp
    ((colSlabApply_flat_diff (mhsa_g N d_head) (mhsa_g_flat_diff N d_head)).comp
     (dense_per_token_flat_diff (N := N) (mhsa_qkv_W heads d_head Wq Wk Wv)
                                (mhsa_qkv_b heads d_head bq bk bv)))

-- ════════════════════════════════════════════════════════════════
-- § 4. Transformer Block (Phase 8 — composition, no hand-waving)
-- ════════════════════════════════════════════════════════════════

/-! ## Per-token liftings (theorems)

Every per-token operation in a transformer (LN, dense, GELU) lifts from
`HasVJP` on `Vec D` to `HasVJPMat` on `Mat N D` via the single helper
`rowwise_has_vjp_mat` (Tensor.lean). These are theorems — no new axioms. -/

/-- Per-token layer norm across a sequence. Applies `layerNormForward`
    to each row of the `(N, D)` input; the backward is block-diagonal. -/
noncomputable def layerNorm_per_token_has_vjp_mat (N D : Nat) (ε γ β : ℝ)
    (hε : 0 < ε) :
    HasVJPMat (fun X : Mat N D => fun n => layerNormForward D ε γ β (X n)) :=
  rowwise_has_vjp_mat (layerNorm_has_vjp D ε γ β hε) (layerNorm_diff D ε γ β hε)

/-- Per-token dense projection across a sequence.
    `Q = X · W + b`, row-by-row dense with shared weights. -/
noncomputable def dense_per_token_has_vjp_mat (N inD outD : Nat)
    (W : Mat inD outD) (b : Vec outD) :
    HasVJPMat (fun X : Mat N inD => fun n => dense W b (X n)) :=
  rowwise_has_vjp_mat (dense_has_vjp W b) (dense_diff W b)

/-- Per-token GELU across a sequence. Elementwise activation,
    so diagonal Jacobian both across rows and within a row. -/
noncomputable def gelu_per_token_has_vjp_mat (N D : Nat) :
    HasVJPMat (fun X : Mat N D => fun n => gelu D (X n)) :=
  rowwise_has_vjp_mat (gelu_has_vjp D) (gelu_diff D)

/-! ## A transformer encoder block

From `emitTransformerBlockForward` (line 796 of MlirCodegen.lean):

    block(x) = h1 + MLP(LN2(h1))       where h1 = x + MHSA(LN1(x))

Expanding:

    h1 = x + MHSA(LN1(x))       -- attention sub-layer with residual
    out = h1 + MLP(LN2(h1))     -- MLP sub-layer with residual

where `MLP(z) = dense(Wfc2, bfc2, gelu(dense(Wfc1, bfc1, z)))`.

Every piece is now a `HasVJPMat` on `Mat N D`:
- `LN1`, `LN2` — `layerNorm_per_token_has_vjp_mat` (theorem via `rowwise_has_vjp_mat`)
- `MHSA`       — `mhsa_has_vjp_mat` (bundled axiom — Phase 8)
- `MLP`        — two `dense_per_token_has_vjp_mat` + one `gelu_per_token_has_vjp_mat`, glued with `vjpMat_comp`
- `+` residuals — `biPathMat_has_vjp` (theorem, Tensor.lean) with identity

The transformer block theorem below glues these with `vjpMat_comp` and
`biPathMat_has_vjp`. Zero new axioms — every piece is a theorem. -/

/-- MLP sublayer of a transformer block: `dense ∘ GELU ∘ dense` applied per-token.

    Concretely: `MLP(z) = Wfc2 · gelu(Wfc1 · z + bfc1) + bfc2`, applied row-wise. -/
noncomputable def transformerMlp (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    Mat N D → Mat N D :=
  (fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
  (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) ∘
  (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n))

/-- Differentiability of the flattened `transformerMlp` — composition of
    `dense ∘ gelu ∘ dense` per-token. Built from the three per-token-flat
    Diff helpers via `Differentiable.comp`, with the usual `Mat.unflatten_flatten`
    rewrite to push the bijection through `∘`. -/
lemma transformerMlp_flat_diff (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten (transformerMlp N D mlpDim Wfc1 bfc1 Wfc2 bfc2
                     (Mat.unflatten v))) := by
  unfold transformerMlp Mat.unflatten Mat.flatten dense gelu geluScalar
  -- After unfolding, the three layers compose as one explicit function.
  -- `dense` and the two row indexings are linear; `geluScalar` is the
  -- only obstacle (handled by `gelu_per_token_flat_diff`, now a
  -- theorem via Real.differentiable_tanh). We factor through the
  -- gelu helper rather than reproving it inline.
  -- Strategy: show the function equals `flat_dense₂ ∘ flat_gelu ∘ flat_dense₁`
  -- through `Mat.unflatten_flatten` round-trips, then chain `Differentiable.comp`.
  have h1 : Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => dense Wfc1 bfc1 (X n))
                   (Mat.unflatten v))) :=
    dense_per_token_flat_diff Wfc1 bfc1
  have h2 : Differentiable ℝ (fun v : Vec (N * mlpDim) =>
      Mat.flatten ((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))
                   (Mat.unflatten v))) :=
    gelu_per_token_flat_diff N mlpDim
  have h3 : Differentiable ℝ (fun v : Vec (N * mlpDim) =>
      Mat.flatten ((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n))
                   (Mat.unflatten v))) :=
    dense_per_token_flat_diff Wfc2 bfc2
  -- Restate the goal using the composed form via Mat.flatten/unflatten round-trip.
  have h_eq : (fun v : Vec (N * D) =>
        Mat.flatten ((((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
                       (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))) ∘
                      (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n)))
                     (Mat.unflatten v))) =
      (fun u : Vec (N * mlpDim) => Mat.flatten
         (((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
           (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)))
            (Mat.unflatten u))) ∘
      (fun v : Vec (N * D) => Mat.flatten
         ((fun X : Mat N D => fun n => dense Wfc1 bfc1 (X n)) (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  -- After the round-trip simp, the goal is the composition of the outer two
  -- with the innermost dense₁. The outer two compose similarly:
  have h_outer_eq : (fun u : Vec (N * mlpDim) => Mat.flatten
        (((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
          (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)))
           (Mat.unflatten u))) =
      (fun u : Vec (N * mlpDim) => Mat.flatten
        ((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n))
           (Mat.unflatten u))) ∘
      (fun u : Vec (N * mlpDim) => Mat.flatten
        ((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))
           (Mat.unflatten u))) := by
    funext u; simp [Function.comp, Mat.unflatten_flatten]
  -- Rebuild the goal step by step.
  show Differentiable ℝ (fun v : Vec (N * D) =>
        Mat.flatten ((((fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n)) ∘
                       (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))) ∘
                      (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n)))
                     (Mat.unflatten v)))
  rw [h_eq]
  rw [h_outer_eq]
  exact (h3.comp h2).comp h1

/-- `HasVJPMat` for the MLP sublayer — chain of two `vjpMat_comp`
    steps over per-token liftings (`dense ∘ gelu ∘ dense`). Theorem,
    no longer axiom: every Diff hypothesis is discharged by the
    per-token-flat helpers above. -/
noncomputable def transformerMlp_has_vjp_mat (N D mlpDim : Nat)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    HasVJPMat (transformerMlp N D mlpDim Wfc1 bfc1 Wfc2 bfc2) :=
  -- Inner composition: gelu ∘ dense₁
  let inner_has_vjp :=
    vjpMat_comp _ (fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n))
      (dense_per_token_flat_diff Wfc1 bfc1)
      (gelu_per_token_flat_diff N mlpDim)
      (dense_per_token_has_vjp_mat N D mlpDim Wfc1 bfc1)
      (gelu_per_token_has_vjp_mat N mlpDim)
  -- Diff of the inner composition (gelu ∘ dense₁), via Mat.flatten/unflatten
  -- round-trip + Differentiable.comp.
  have inner_diff : Differentiable ℝ
      (fun v : Vec (N * D) =>
        Mat.flatten (((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) ∘
                      (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n)))
                     (Mat.unflatten v))) := by
    have h_eq : (fun v : Vec (N * D) =>
          Mat.flatten (((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) ∘
                        (fun X : Mat N D      => fun n => dense Wfc1 bfc1 (X n)))
                       (Mat.unflatten v))) =
        (fun u : Vec (N * mlpDim) => Mat.flatten
            ((fun Y : Mat N mlpDim => fun n => gelu mlpDim (Y n)) (Mat.unflatten u))) ∘
        (fun v : Vec (N * D) => Mat.flatten
            ((fun X : Mat N D => fun n => dense Wfc1 bfc1 (X n)) (Mat.unflatten v))) := by
      funext v; simp [Function.comp, Mat.unflatten_flatten]
    rw [h_eq]
    exact (gelu_per_token_flat_diff N mlpDim).comp (dense_per_token_flat_diff Wfc1 bfc1)
  -- Outer composition: dense₂ ∘ (gelu ∘ dense₁)
  vjpMat_comp _ (fun Y : Mat N mlpDim => fun n => dense Wfc2 bfc2 (Y n))
    inner_diff
    (dense_per_token_flat_diff Wfc2 bfc2)
    inner_has_vjp
    (dense_per_token_has_vjp_mat N mlpDim D Wfc2 bfc2)

/-- Attention sublayer: `X ↦ X + MHSA(LN1(X))`. Top-level composition;
    the `biPathMat` skip-adds identity to the MHSA∘LN1 branch. -/
noncomputable def transformerAttnSublayer (N heads d_head : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormForward (heads * d_head) ε γ1 β1 (X n)))

/-- MLP sublayer: `h ↦ h + MLP(LN2(h))`. Same biPathMat structure. -/
noncomputable def transformerMlpSublayer (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormForward (heads * d_head) ε γ2 β2 (X n)))

/-- **Transformer encoder block forward**: MLP-sublayer ∘ attention-sublayer.
    Signature matches the codegen: `Mat N (heads·d_head) → Mat N (heads·d_head)`. -/
noncomputable def transformerBlock (N heads d_head mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
  (transformerAttnSublayer N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo)

/-- Differentiability of the flattened attention sublayer's non-trivial arm
    (`mhsa ∘ LN1`). Used by both the sublayer VJP proof and any downstream
    composition that needs Diff for the sublayer's arm. -/
lemma transformerAttnSublayer_inner_flat_diff
    (N heads d_head : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormForward (heads * d_head) ε γ1 β1 (X n)))
         (Mat.unflatten v))) := by
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ1 β1 (X n)))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          ((fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ1 β1 (X n)) (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo).comp
        (layerNorm_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)

/-- Differentiability of the flattened attention sublayer.
    `biPathMat (id) (mhsa ∘ LN1)` flattens to a sum, both arms Differentiable. -/
lemma transformerAttnSublayer_flat_diff
    (N heads d_head : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerAttnSublayer N heads d_head ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) := by
  unfold transformerAttnSublayer biPathMat
  -- Goal: Differentiable of `fun v k => (id (Mat.unflatten v) + (mhsa ∘ LN) (Mat.unflatten v)) ...`.
  -- Each output coordinate is a sum of two coordinates of the two arms — both Diff.
  have h_id := identity_mat_flat_diff N (heads * d_head)
  have h_inner := transformerAttnSublayer_inner_flat_diff N heads d_head ε γ1 β1 hε
                    Wq Wk Wv Wo bq bk bv bo
  -- The biPathMat unfolds to `fun M r s => F M r s + G M r s`. Flattened:
  --   fun v k => F (Mat.unflatten v) (fPF.symm k).1 (fPF.symm k).2 +
  --              G (Mat.unflatten v) (fPF.symm k).1 (fPF.symm k).2
  -- = (Mat.flatten ∘ F ∘ Mat.unflatten) v k + (Mat.flatten ∘ G ∘ Mat.unflatten) v k
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten (fun (r : Fin N) (s : Fin (heads * d_head)) =>
          (fun X : Mat N (heads * d_head) => X) (Mat.unflatten v) r s +
          ((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ1 β1 (X n)))
            (Mat.unflatten v) r s)) =
      fun v => fun k =>
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten ((fun X : Mat N (heads * d_head) => X) (Mat.unflatten v'))) v k +
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ1 β1 (X n)))
            (Mat.unflatten v'))) v k := by
    funext v k; unfold Mat.flatten; rfl
  rw [h_eq]
  exact h_id.add h_inner

/-- Attention sublayer VJP: `biPathMat` of identity and `mhsa ∘ LN1`.
    Theorem, no longer axiom: discharges the `Differentiable` hypotheses
    using `identity_mat_flat_diff` for the skip arm and the inner Diff
    helper above for the `mhsa ∘ LN1` arm. -/
noncomputable def transformerAttnSublayer_has_vjp_mat (N heads d_head : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (transformerAttnSublayer N heads d_head ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo) :=
  -- Inner arm composition: mhsa ∘ LN1
  let inner_has_vjp :=
    vjpMat_comp _ (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo)
      (layerNorm_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)
      (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo)
      (layerNorm_per_token_has_vjp_mat N (heads * d_head) ε γ1 β1 hε)
      (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo)
  biPathMat_has_vjp _ _
    (identity_mat_flat_diff N (heads * d_head))
    (transformerAttnSublayer_inner_flat_diff N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (identityMat_has_vjp N (heads * d_head))
    inner_has_vjp

/-- Differentiability of the MLP sublayer's non-trivial arm
    (`transformerMlp ∘ LN2`). Composition of `transformerMlp_flat_diff`
    and `layerNorm_per_token_flat_diff`. -/
lemma transformerMlpSublayer_inner_flat_diff
    (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormForward (heads * d_head) ε γ2 β2 (X n)))
         (Mat.unflatten v))) := by
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ2 β2 (X n)))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          ((fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ2 β2 (X n)) (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2).comp
        (layerNorm_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)

/-- Differentiability of the flattened MLP sublayer.
    `biPathMat (id) (transformerMlp ∘ LN2)` flattens to a sum, both arms Differentiable. -/
lemma transformerMlpSublayer_flat_diff
    (N heads d_head mlpDim : Nat) (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2
                     Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
  unfold transformerMlpSublayer biPathMat
  have h_id := identity_mat_flat_diff N (heads * d_head)
  have h_inner := transformerMlpSublayer_inner_flat_diff N heads d_head mlpDim ε γ2 β2 hε
                    Wfc1 bfc1 Wfc2 bfc2
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten (fun (r : Fin N) (s : Fin (heads * d_head)) =>
          (fun X : Mat N (heads * d_head) => X) (Mat.unflatten v) r s +
          ((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ2 β2 (X n)))
            (Mat.unflatten v) r s)) =
      fun v => fun k =>
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten ((fun X : Mat N (heads * d_head) => X) (Mat.unflatten v'))) v k +
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γ2 β2 (X n)))
            (Mat.unflatten v'))) v k := by
    funext v k; unfold Mat.flatten; rfl
  rw [h_eq]
  exact h_id.add h_inner

/-- MLP sublayer VJP: `biPathMat` of identity and `transformerMlp ∘ LN2`.
    Theorem, no longer axiom: same recipe as `transformerAttnSublayer_has_vjp_mat`. -/
noncomputable def transformerMlpSublayer_has_vjp_mat (N heads d_head mlpDim : Nat)
    (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2
                 Wfc1 bfc1 Wfc2 bfc2) :=
  let inner_has_vjp :=
    vjpMat_comp _ (transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
      (layerNorm_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)
      (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
      (layerNorm_per_token_has_vjp_mat N (heads * d_head) ε γ2 β2 hε)
      (transformerMlp_has_vjp_mat N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
  biPathMat_has_vjp _ _
    (identity_mat_flat_diff N (heads * d_head))
    (transformerMlpSublayer_inner_flat_diff N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (identityMat_has_vjp N (heads * d_head))
    inner_has_vjp

/-- Differentiability of the flattened transformer block.
    `MlpSublayer ∘ AttnSublayer`; both sublayers' flat Diff are theorems above. -/
lemma transformerBlock_flat_diff (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerBlock N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2
                   (Mat.unflatten v))) := by
  unfold transformerBlock
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
            (transformerAttnSublayer N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2
             (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerAttnSublayer N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
             (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (transformerMlpSublayer_flat_diff N heads d_head mlpDim ε γ2 β2 hε
            Wfc1 bfc1 Wfc2 bfc2).comp
        (transformerAttnSublayer_flat_diff N heads d_head ε γ1 β1 hε
            Wq Wk Wv Wo bq bk bv bo)

/-- **Transformer block VJP** — composition of attn + mlp sublayers.
    Theorem, no longer axiom: a single `vjpMat_comp` of the two sublayer
    theorems with their Diff helpers. -/
noncomputable def transformerBlock_has_vjp_mat (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) :=
  vjpMat_comp _ (transformerMlpSublayer N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayer_flat_diff N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayer_flat_diff N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayer_has_vjp_mat N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayer_has_vjp_mat N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)

-- ════════════════════════════════════════════════════════════════
-- § 5. The ViT finale — k-block transformer tower
-- ════════════════════════════════════════════════════════════════

/-! ## Stacking transformer blocks

ViT-Tiny has 12 transformer blocks; ViT-Base has 12, ViT-Large has 24.
The stack is just k-fold composition of individual blocks. By
`vjpMat_comp` and induction on k, if each block has a `HasVJPMat`
then so does the stack — for any depth.

For the formal theorem we use a single shared parameter tuple across
blocks (a mild simplification; in practice every block has its own
weights). The Jacobian-composition structure doesn't change — the
theorem generalizes trivially to per-block parameters by replacing
the Nat induction with a `Fin k` parameter function, which is
mechanical once the single-shared-param case is proved. -/

/-- k-fold iterated transformer block, sharing parameters across all
    k layers. Defined by `Nat.rec` so the `HasVJPMat` proof is a
    straightforward induction on k. -/
noncomputable def transformerTower (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  Nat.rec (motive := fun _ => Mat N (heads * d_head) → Mat N (heads * d_head))
    (fun X => X)
    (fun _ acc =>
      (transformerBlock N heads d_head mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
         γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘ acc)
    k

/-- Differentiability of the flattened k-fold transformer tower.
    Induction on `k`: zero case is identity, successor case is
    `block ∘ tower(k)` composed via `Differentiable.comp`. -/
lemma transformerTower_flat_diff (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerTower k N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2
                   (Mat.unflatten v))) := by
  induction k with
  | zero =>
    -- transformerTower 0 ... = fun X => X
    show Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten ((fun X : Mat N (heads * d_head) => X) (Mat.unflatten v)))
    exact identity_mat_flat_diff N (heads * d_head)
  | succ k' ih =>
    -- transformerTower (k'+1) = block ∘ transformerTower k'  (defeq via Nat.rec)
    show Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (((transformerBlock N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
                    (transformerTower k' N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
                   (Mat.unflatten v)))
    have h_eq : (fun v : Vec (N * (heads * d_head)) =>
          Mat.flatten (((transformerBlock N heads d_head mlpDim ε γ1 β1
                           Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
                        (transformerTower k' N heads d_head mlpDim ε γ1 β1
                           Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
                       (Mat.unflatten v))) =
        (fun u : Vec (N * (heads * d_head)) => Mat.flatten
            (transformerBlock N heads d_head mlpDim ε γ1 β1
               Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten u))) ∘
        (fun v : Vec (N * (heads * d_head)) => Mat.flatten
            (transformerTower k' N heads d_head mlpDim ε γ1 β1
               Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
      funext v; simp [Function.comp, Mat.unflatten_flatten]
    rw [h_eq]
    exact (transformerBlock_flat_diff N heads d_head mlpDim ε γ1 β1 hε
              Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).comp ih

/-- **Transformer tower VJP** — k-fold composition. Theorem, no longer axiom:
    induction on `k` via `vjpMat_comp` and `transformerBlock_has_vjp_mat`. -/
noncomputable def transformerTower_has_vjp_mat (k N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerTower k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) := by
  induction k with
  | zero =>
    show HasVJPMat (fun X : Mat N (heads * d_head) => X)
    exact identityMat_has_vjp N (heads * d_head)
  | succ k' ih =>
    show HasVJPMat ((transformerBlock N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
                    (transformerTower k' N heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
    exact vjpMat_comp _ _
      (transformerTower_flat_diff k' N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
      (transformerBlock_flat_diff N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
      ih
      (transformerBlock_has_vjp_mat N heads d_head mlpDim ε γ1 β1 hε
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-! ## ViT body: tower + final LN

`vit_body` is the ViT backbone operating on a single `(N, D)` sequence,
*after* the patch embedding produced a `Mat N D` input and *before* the
classifier head slices the CLS token and runs dense+softmax CE.

    patch_embed(X : Tensor3 ic h w) : Mat N D   ← outside Mat-land
    vit_body(M : Mat N D) : Mat N D             ← the backbone (this file)
    classifier(M) = dense(W_cls, b_cls, M[0])   ← Mat → Vec, then softmax CE loss

The backbone is `finalLN ∘ transformerTower`. Both sides are `Mat N D`,
so `vjpMat_comp` glues the two VJPs.

The patch-embedding and classifier-head steps exit `Mat`-land (they
change type to/from `Tensor3` and `Vec` respectively). Both are trivial
compositions of already-proved theorems (`conv2d_has_vjp3` / `pdiv_reindex`
for patch embed, `pdiv_reindex` / `dense_has_vjp` / `softmaxCE_grad` for
the classifier) but they don't fit in the uniform `HasVJPMat` frame.
We mark them as future work; closing this would require a unified
rank-polymorphic VJP framework that's not needed for the pedagogy. -/

/-- **ViT body** — transformer tower followed by final per-token LayerNorm.

    Composition is `finalLN ∘ transformerTower`; matches the codegen's
    `emitForwardBody` ordering for a `.transformerEncoder` followed by
    the implicit final LN block. -/
noncomputable def vit_body (k N heads d_head mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)  -- final LN params
    : Mat N (heads * d_head) → Mat N (heads * d_head) :=
  (fun X : Mat N (heads * d_head) => fun n =>
      layerNormForward (heads * d_head) ε γF βF (X n)) ∘
  (transformerTower k N heads d_head mlpDim ε γ1 β1
     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-- Differentiability of the flattened ViT body.
    `finalLN ∘ transformerTower` — both have flat Diff theorems above. -/
lemma vit_body_flat_diff (k N heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (vit_body k N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                   (Mat.unflatten v))) := by
  unfold vit_body
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γF βF (X n)) ∘
            (transformerTower k N heads d_head mlpDim ε γ1 β1
               Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          ((fun X : Mat N (heads * d_head) => fun n =>
              layerNormForward (heads * d_head) ε γF βF (X n)) (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerTower k N heads d_head mlpDim ε γ1 β1
             Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (layerNorm_per_token_flat_diff N (heads * d_head) ε γF βF hε).comp
        (transformerTower_flat_diff k N heads d_head mlpDim ε γ1 β1 hε
           Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)

/-- **The ViT body VJP** — `finalLN ∘ transformerTower`. Theorem, no longer
    axiom: a single `vjpMat_comp` of the tower + final LN with their
    Diff helpers.

    Conceptually still the punchline: a depth-k ViT backbone has a
    correct VJP, composed entirely from proved building blocks. With
    Phase 3's column-stacking framework, even `mhsa_has_vjp_mat` and
    its flat-diff sibling are theorems now, so this whole chain is
    pure-Mathlib closure with no project axioms. -/
noncomputable def vit_body_has_vjp_mat (k N heads d_head mlpDim : Nat) (ε : ℝ)
    (hε : 0 < ε)
    (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ) :
    HasVJPMat (vit_body k N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF) :=
  vjpMat_comp _ (fun X : Mat N (heads * d_head) => fun n =>
                   layerNormForward (heads * d_head) ε γF βF (X n))
    (transformerTower_flat_diff k N heads d_head mlpDim ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (layerNorm_per_token_flat_diff N (heads * d_head) ε γF βF hε)
    (transformerTower_has_vjp_mat k N heads d_head mlpDim ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (layerNorm_per_token_has_vjp_mat N (heads * d_head) ε γF βF hε)

-- ════════════════════════════════════════════════════════════════
-- § 6. The end of the road
-- ════════════════════════════════════════════════════════════════

/-! ## What we've proved (and what's left)

**Proved (zero sorry's, machine-checked):**
- Dense, ReLU (`MLP.lean`)
- Softmax cross-entropy loss gradient (`MLP.lean`)
- Conv2d, MaxPool, Flatten (`CNN.lean`)
- BatchNorm closed-form backward (`BatchNorm.lean`)
- Residual / biPath fan-in (`Residual.lean`)
- Depthwise conv (`Depthwise.lean`)
- Squeeze-and-Excitation / elementwise product VJP (`SE.lean`)
- LayerNorm, GELU (`LayerNorm.lean`)
- Standalone softmax VJP (this file)
- Scaled dot-product attention backwards `sdpa_back_{Q,K,V}` —
  proved via `vjpMat_comp` composition of four matrix-level VJP
  building blocks (matmul, scalarScale, rowSoftmax, matmul). Formulas
  are also numerically gradient-checked as a belt-and-braces check.

**Three calculus rules do all the structural work** (now theorems
proved from Mathlib's `fderiv`, formerly axioms):

    pdiv_comp   (chain rule — functions compose, derivatives compose)
    pdiv_add    (linearity — derivatives of sums are sums of derivatives)
    pdiv_mul    (product rule — derivatives of elementwise products)

**Five closed-form "Jacobian-structure tricks"** handle the layers
whose Jacobians are dense but exploitable:

1. **Diagonal** (activations) — collapse the sum_j to one term.
2. **Sparse toeplitz** (conv, depthwise) — reversed/transposed kernels.
3. **Binary selection** (max-pool) — route gradients to argmax cells.
4. **Rank-1 correction to diagonal** (softmax, BN, LN, IN, GN) — one
   extra scalar reduction, everything else is pointwise.
5. **Outer product + reductions** (dense, matmul) — rank-1 update
   accumulation.

**That is the complete taxonomy.** I've thought hard about this and
cannot find a sixth trick or a fourth calculus rule anywhere in the
modern architecture zoo. Every paper, every block, every optimization
is a rearrangement of these ten things.

## What this means for the reader

If you've read this far, you have a complete decoder for the
architecture-of-the-month. Pick any paper — Swin, ConvNeXt, CLIP,
Mamba, anything — and walk through the forward pass. For each
operation, ask:

  1. Is it **composition** of known ops? -> chain rule.
  2. Is it a **sum of branches**? -> fan-in add.
  3. Is it an **elementwise / scalar product of branches**? -> fan-in mul.
  4. Is it an **activation**? -> diagonal Jacobian template.
  5. Is it a **normalization**? -> closed-form three-term formula.
  6. Is it a **convolution or linear map**? -> the structured-matmul
     machinery.
  7. Is it an **attention or softmax-based selection**? -> the closed-form
     rank-1 collapse.

If the answer is "none of the above" — which it won't be — then you've
found the first genuinely new layer of the decade, and you get to
write the next chapter of this book.

Until then, welcome to the end of the road. -/

-- ════════════════════════════════════════════════════════════════
-- § 7. The ACTUAL grand finale — full ViT as a single `HasVJP`
-- ════════════════════════════════════════════════════════════════

/-! ## Bridging ranks: from Mat-land back to Vec-land

`vit_body_has_vjp_mat` lives in `HasVJPMat` territory. The pieces at the
boundaries — patch embedding (image → tokens) and classifier head (tokens
→ logits) — change tensor rank. Rather than invent new mixed-rank VJP
frameworks, we flatten everything to `Vec` at the interfaces and compose
via plain `HasVJP`, glued by `vjp_comp`.

Two ingredients needed:

- **`hasVJPMat_to_hasVJP`** (Tensor.lean, Phase 10) — bridges any
  `HasVJPMat` to `HasVJP` on the flattened endpoints. One theorem, no
  new axioms.
- **`cls_slice_flat_has_vjp`** — gathers row 0 of a flattened
  `Mat (N+1) D`. Derivable from `pdiv_reindex`. -/

/-- CLS token extraction, stated on the flattened matrix. Row 0 of a
    `Mat (N+1) D` is a `Vec D`; on the flattened `Vec ((N+1)*D)` this is
    the gather `v ↦ fun k => v (fPF (0, k))`. -/
noncomputable def cls_slice_flat (N D : Nat) :
    Vec ((N + 1) * D) → Vec D :=
  fun v k => v (finProdFinEquiv ((0 : Fin (N + 1)), k))

/-- **CLS slice VJP** — gather-style; backward scatters `dy` to row 0.
    Derived from `pdiv_reindex`. -/
noncomputable def cls_slice_flat_has_vjp (N D : Nat) :
    HasVJP (cls_slice_flat N D) where
  backward := fun _v dy => fun idx =>
    let p := finProdFinEquiv.symm idx
    if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0
  correct := by
    intro v dy idx
    set p := finProdFinEquiv.symm idx with hp
    show (if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0) = _
    -- Goal RHS: ∑ j, pdiv cls_slice_flat v idx j * dy j
    -- pdiv_reindex gives: pdiv = if idx = fPF (0, j) then 1 else 0.
    simp_rw [show ∀ j : Fin D,
        pdiv (cls_slice_flat N D) v idx j =
          (if idx = finProdFinEquiv ((0 : Fin (N + 1)), j) then 1 else 0) from
      fun j => pdiv_reindex
                 (fun k : Fin D => finProdFinEquiv ((0 : Fin (N + 1)), k))
                 v idx j]
    -- Sum of (if idx = fPF(0, j) then 1 else 0) * dy j over j.
    -- If p.1 = 0 then idx = fPF (0, p.2), so the unique matching j is p.2.
    by_cases hrow : p.1 = (0 : Fin (N + 1))
    · rw [if_pos hrow]
      rw [Finset.sum_eq_single p.2
          (fun j _ hne => by
            rw [if_neg ?_, zero_mul]
            intro heq
            apply hne
            -- idx = fPF (0, j), and idx = fPF p = fPF (p.1, p.2) = fPF (0, p.2)
            have hfpf : finProdFinEquiv (p.1, p.2) = idx := by
              show finProdFinEquiv p = idx
              rw [hp]; exact Equiv.apply_symm_apply _ _
            have heq2 : finProdFinEquiv (p.1, p.2) = finProdFinEquiv ((0 : Fin (N + 1)), j) := by
              rw [hfpf]; exact heq
            have := finProdFinEquiv.injective heq2
            have : p.2 = j := (Prod.mk.inj this).2
            exact this.symm)
          (fun h => absurd (Finset.mem_univ p.2) h)]
      rw [if_pos]
      · ring
      · -- idx = fPF (0, p.2)
        have hfpf : finProdFinEquiv (p.1, p.2) = idx := by
          show finProdFinEquiv p = idx
          rw [hp]; exact Equiv.apply_symm_apply _ _
        rw [hrow] at hfpf
        exact hfpf.symm
    · rw [if_neg hrow]
      symm
      apply Finset.sum_eq_zero
      intro j _
      rw [if_neg ?_, zero_mul]
      intro heq
      apply hrow
      -- idx = fPF (0, j), so p.1 = 0.
      have hfpf : finProdFinEquiv (p.1, p.2) = idx := by
        show finProdFinEquiv p = idx
        rw [hp]; exact Equiv.apply_symm_apply _ _
      have heq2 : finProdFinEquiv (p.1, p.2) = finProdFinEquiv ((0 : Fin (N + 1)), j) := by
        rw [hfpf]; exact heq
      exact (Prod.mk.inj (finProdFinEquiv.injective heq2)).1

/-- **Classifier head**: flattened CLS slice + dense projection to `Vec nClasses`.

    `fun v : Vec ((N+1)*D) => dense W_cls b_cls (cls_slice_flat v)` -/
noncomputable def classifier_flat (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    Vec ((N + 1) * D) → Vec nClasses :=
  (dense Wcls bcls) ∘ (cls_slice_flat N D)

/-- Differentiability of `cls_slice_flat` — linear reindex. -/
lemma cls_slice_flat_diff (N D : Nat) :
    Differentiable ℝ (cls_slice_flat N D) := by
  unfold cls_slice_flat; fun_prop

/-- Differentiability of `dense W b` as a function of the input vector — linear. -/
lemma dense_input_diff {m n : Nat} (W : Mat m n) (b : Vec n) :
    Differentiable ℝ (dense W b) := by
  unfold dense; fun_prop

/-- **Classifier head VJP** — composition via `vjp_comp`. Theorem, no
    longer axiom: `cls_slice_flat` and `dense` are both linear, so their
    Diff hypotheses discharge by `fun_prop`. -/
noncomputable def classifier_flat_has_vjp (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    HasVJP (classifier_flat N D nClasses Wcls bcls) :=
  vjp_comp (cls_slice_flat N D) (dense Wcls bcls)
    (cls_slice_flat_diff N D)
    (dense_input_diff Wcls bcls)
    (cls_slice_flat_has_vjp N D)
    (dense_has_vjp Wcls bcls)

/-! ## Patch embedding — Phase 6 (de-opaqued, no longer axiomatic)

The patch embedding takes a flattened image `Vec (ic*H*W)` and produces
a flattened `Vec ((N+1)*D)` interpreted as `Mat (N+1) D`:

1. Conv projection with stride = patchSize: per-patch dense projection
   `W_conv : Kernel4 D ic patchSize patchSize` + bias `b_conv : Vec D`.
2. Reshape spatial `(D, H', W')` to tokens `(N, D)` — pure permutation.
3. Prepend learnable CLS token at row 0 → `(N+1, D)`.
4. Add learnable positional embedding matrix → `(N+1, D)`.

This was previously an `opaque` definition + two bundled axioms
(`patchEmbed_flat_has_vjp` / `patchEmbed_flat_diff`). Phase 6 (Apr 2026)
de-opaques: the forward is a concrete `def` and both axioms become
theorems via foundation rules.

The `N` parameter is independent of `(H, W, patchSize)` — out-of-range
patches contribute zero (via a `hpad` guard on the image read), so the
API does not require `N = (H/patchSize) * (W/patchSize)`. -/

/-- **Patch embedding forward** on flattened endpoints.

    Output at flat-index `idx_out = finProdFinEquiv (n, d)`:
    - `n = 0`: `cls_token d + pos_embed 0 d`.
    - `n > 0` (let `p := n - 1`, `h' := p / (W/patchSize)`,
      `w' := p % (W/patchSize)`):
      `b_conv d + Σ c kh kw, W_conv d c kh kw * img(c, h'*P+kh, w'*P+kw)
       + pos_embed n d`,
      where the image read is guarded by `h'*P+kh < H ∧ w'*P+kw < W`
      (returns 0 if out of range).

    This handles arbitrary `N` cleanly: for `n` whose decoded patch
    `(h', w')` falls outside the image grid, the inner sum is identically
    zero. -/
noncomputable def patchEmbed_flat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D) :=
  fun img =>
    fun idx_out =>
      let n := (finProdFinEquiv.symm idx_out).1
      let d := (finProdFinEquiv.symm idx_out).2
      pos_embed n d +
        (if n.val = 0 then
          cls_token d
         else
          b_conv d +
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv d c kh kw *
              (let W' := W / patchSize
               let p := n.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
               else 0))

/-- **Patch embedding differentiability** — proved from foundation rules.
    `patchEmbed_flat` is linear in `img` plus constants
    (`pos_embed`, `cls_token`, `b_conv`); the only non-trivial part is
    the dependent `if hpad : ... then img(σ hpad) else 0` pattern handled
    by `differentiableAt_pad_eval`. -/
lemma patchEmbed_flat_diff
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Differentiable ℝ (patchEmbed_flat ic H W patchSize N D
                       W_conv b_conv cls_token pos_embed) := by
  unfold patchEmbed_flat
  intro img
  rw [differentiableAt_pi]
  intro idx_out
  -- The body at idx_out is a constant (pos_embed) + (if n=0 then constant else
  -- b_conv + Σ c kh kw, W_conv * pad-guarded img-read). All terms are
  -- DifferentiableAt in img.
  apply DifferentiableAt.add (differentiableAt_const _)
  by_cases hn : (finProdFinEquiv.symm idx_out).1.val = 0
  · simp only [hn, if_true]
    exact differentiableAt_const _
  · simp only [hn, if_false]
    apply DifferentiableAt.add (differentiableAt_const _)
    apply DifferentiableAt.fun_sum; intro c _
    apply DifferentiableAt.fun_sum; intro kh _
    apply DifferentiableAt.fun_sum; intro kw _
    apply DifferentiableAt.mul (differentiableAt_const _)
    -- The pad-eval pattern: if hpad : (hh < H ∧ ww < W) then img(σ hpad) else 0
    exact differentiableAt_pad_eval _
      (fun hpad => finProdFinEquiv (finProdFinEquiv
        (c, ⟨((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
              patchSize + kh.val, hpad.1⟩),
        ⟨((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
          patchSize + kw.val, hpad.2⟩)) _

/-- **Closed-form input gradient for `patchEmbed_flat`** — direct formula,
    written as a sum over patches `p : Fin N` with reconstructed kernel
    offsets `(kh, kw)` matching the input position `(hh, ww)` decoded from
    `idx_in`. Equivalent (under the patch-row decomposition `h' := p/(W/P)`,
    `w' := p%(W/P)`) to the standard "deconvolution" formula for patchEmbed.

    The CLS row (n = 0) does not appear here — `idx_in` only flows through
    the conv-projection branch (n > 0), so the gradient sums over
    `p : Fin N` (corresponding to output rows `n = p+1`). -/
noncomputable def patchEmbed_input_grad_formula
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize)
    (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    ∑ p : Fin N, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if _h_match : h' * patchSize + kh.val = hh.val ∧
                    w' * patchSize + kw.val = ww.val then
        ∑ d : Fin D, W_conv d c kh kw *
          dy (finProdFinEquiv (p.succ, d))
      else 0

/-- **Patch embedding VJP — proved from foundation rules.**

    Phase 6b (Apr 2026): the forward (de-opaqued in Phase 6a) is linear in
    `img` plus constants; the only non-trivial pattern is the dependent
    `if hpad : ... then img(σ) else 0` handled by `pdiv_const_mul_pi_pad_eval`
    (already used for conv2d/depthwise input-VJPs). Closing collapse mirrors
    `conv2d_has_vjp3`, with one new wrinkle: split `Σ n : Fin (N+1)` into
    `n = 0` (CLS row, contributes 0 to img-grad) + `Σ p : Fin N` (n = p+1)
    via `Fin.sum_univ_succ`.

    Backward: `patchEmbed_input_grad_formula W_conv dy`. -/
noncomputable def patchEmbed_flat_has_vjp
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    HasVJP (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed) where
  backward := fun _img dy => patchEmbed_input_grad_formula ic H W patchSize N D W_conv dy
  correct := by
    intro img dy idx_in
    -- Set abbreviations for idx_in's decoded components.
    set c_in : Fin ic :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1 with hc_in
    set hh_in : Fin H :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2 with hhh_in
    set ww_in : Fin W := (finProdFinEquiv.symm idx_in).2 with hww_in
    have hidx_in : idx_in = finProdFinEquiv (finProdFinEquiv (c_in, hh_in), ww_in) := by
      show idx_in = finProdFinEquiv (finProdFinEquiv
        ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1,
         (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2),
        (finProdFinEquiv.symm idx_in).2)
      rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
    -- Step 1: per-(idx_in, idx_out) pdiv lemma. We decompose the forward as
    --   F = const_F + var_F
    -- where var_F absorbs the `n.val = 0` split into the pad guard:
    --   var_F img k = ∑ c kh kw, W_conv * (if hpad : ¬(n.val=0) ∧ hh<H ∧ ww<W
    --                                      then img(σ) else 0).
    -- This makes var_F linear in img with no `if n.val = 0` test, so
    -- pdiv_finset_sum + pdiv_const_mul_pi_pad_eval apply uniformly.
    have h_pdiv : ∀ idx_out : Fin ((N + 1) * D),
        pdiv (patchEmbed_flat ic H W patchSize N D
                W_conv b_conv cls_token pos_embed) img idx_in idx_out =
        if _hn0 : (finProdFinEquiv.symm idx_out).1.val = 0 then 0
        else
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
              (let W' := W / patchSize
               let p := (finProdFinEquiv.symm idx_out).1.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 (if idx_in = finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
               else 0) := by
      intro idx_out
      -- Decompose F = const_F + var_F via the absorb-trick: var_F's pad-guard
      -- includes `¬(n.val = 0)` as a conjunct, so var_F is identically 0 at
      -- output rows with n.val = 0. This bypasses the `if n.val = 0` test in
      -- the original forward and gives us a uniform Σ c kh kw form.
      rw [show (patchEmbed_flat ic H W patchSize N D
              W_conv b_conv cls_token pos_embed) =
          (fun img' k =>
            (fun (_ : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
              pos_embed (finProdFinEquiv.symm k').1 (finProdFinEquiv.symm k').2 +
              (if (finProdFinEquiv.symm k').1.val = 0 then
                 cls_token (finProdFinEquiv.symm k').2
               else b_conv (finProdFinEquiv.symm k').2)) img' k +
            (fun (img'' : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
              ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k').2 c kh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : ¬((finProdFinEquiv.symm k').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img'' (finProdFinEquiv (finProdFinEquiv
                       (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) img' k) from by
        funext img' k
        unfold patchEmbed_flat
        by_cases hn0_k : (finProdFinEquiv.symm k).1.val = 0
        · -- LHS = pos_embed n d + cls_token d
          -- RHS = (pos_embed n d + cls_token d) + Σ_zero  (each summand = 0 because pad-guard ⊥)
          simp_rw [if_pos hn0_k]
          rw [show (∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv ((finProdFinEquiv.symm k).2) c kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k).1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : ¬((finProdFinEquiv.symm k).1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img' (finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0) : ℝ) = 0 from ?_]
          · ring
          apply Finset.sum_eq_zero; intro c _
          apply Finset.sum_eq_zero; intro kh _
          apply Finset.sum_eq_zero; intro kw _
          rw [show ((let W' := W / patchSize
                     let p := (finProdFinEquiv.symm k).1.val - 1
                     let h' := p / W'
                     let w' := p % W'
                     let hh := h' * patchSize + kh.val
                     let ww := w' * patchSize + kw.val
                     if hpad : ¬((finProdFinEquiv.symm k).1.val = 0) ∧
                               hh < H ∧ ww < W then
                       img' (finProdFinEquiv (finProdFinEquiv
                         (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                     else 0) : ℝ) = 0 from ?_]
          · ring
          rw [dif_neg]
          intro h
          exact h.1 hn0_k
        · -- LHS = pos_embed + b_conv + Σ_orig (with 2-conj pad).
          -- RHS = (pos_embed + b_conv) + Σ_new (with 3-conj pad).
          -- Σ_orig = Σ_new because ¬(n.val=0) is true, so 3-conj ⇔ 2-conj dites equal.
          simp_rw [if_neg hn0_k]
          rw [show (∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv ((finProdFinEquiv.symm k).2) c kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k).1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : ¬((finProdFinEquiv.symm k).1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img' (finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0)) =
                (∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv ((finProdFinEquiv.symm k).2) c kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k).1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : hh < H ∧ ww < W then
                   img' (finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
                 else 0)) from ?_]
          · ring
          apply Finset.sum_congr rfl; intro c _
          apply Finset.sum_congr rfl; intro kh _
          apply Finset.sum_congr rfl; intro kw _
          congr 1
          by_cases hpad' : ((finProdFinEquiv.symm k).1.val - 1) / (W / patchSize) *
                              patchSize + kh.val < H ∧
                           ((finProdFinEquiv.symm k).1.val - 1) % (W / patchSize) *
                              patchSize + kw.val < W
          · rw [dif_pos ⟨hn0_k, hpad'⟩, dif_pos hpad']
          · rw [dif_neg ?_, dif_neg hpad']
            intro h
            exact hpad' h.2]
      -- Now apply pdiv_add to split into pdiv const_F (= 0) + pdiv var_F.
      have h_const_diff : DifferentiableAt ℝ
          (fun (_ : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
            pos_embed (finProdFinEquiv.symm k').1 (finProdFinEquiv.symm k').2 +
            (if (finProdFinEquiv.symm k').1.val = 0 then
               cls_token (finProdFinEquiv.symm k').2
             else b_conv (finProdFinEquiv.symm k').2)) img :=
        differentiableAt_const _
      have h_var_diff : DifferentiableAt ℝ
          (fun (img'' : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
            ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              W_conv (finProdFinEquiv.symm k').2 c kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k').1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : ¬((finProdFinEquiv.symm k').1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img'' (finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0)) img := by
        rw [differentiableAt_pi]
        intro k'
        apply DifferentiableAt.fun_sum; intro c _
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (c, ⟨((finProdFinEquiv.symm k').1.val - 1) / (W / patchSize) *
                  patchSize + kh.val, hpad.2.1⟩),
            ⟨((finProdFinEquiv.symm k').1.val - 1) % (W / patchSize) *
              patchSize + kw.val, hpad.2.2⟩)) _
      rw [pdiv_add _ _ _ h_const_diff h_var_diff]
      rw [show pdiv (fun (_ : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
            pos_embed (finProdFinEquiv.symm k').1 (finProdFinEquiv.symm k').2 +
            (if (finProdFinEquiv.symm k').1.val = 0 then
               cls_token (finProdFinEquiv.symm k').2
             else b_conv (finProdFinEquiv.symm k').2)) img idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute pdiv over Σ c.
      rw [show (fun (img'' : Vec (ic * H * W)) (k' : Fin ((N + 1) * D)) =>
            ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              W_conv (finProdFinEquiv.symm k').2 c kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k').1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : ¬((finProdFinEquiv.symm k').1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img'' (finProdFinEquiv (finProdFinEquiv
                     (c, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0)) =
          (fun img'' k' => ∑ c : Fin ic,
            (fun (cc : Fin ic) (img''' : Vec (ic * H * W))
                 (k'' : Fin ((N + 1) * D)) =>
              ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k'').2 cc kh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k'').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : ¬((finProdFinEquiv.symm k'').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) c img'' k') from rfl]
      have h_c_diff : ∀ cc ∈ (Finset.univ : Finset (Fin ic)),
          DifferentiableAt ℝ
            (fun (img''' : Vec (ic * H * W)) (k'' : Fin ((N + 1) * D)) =>
              ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k'').2 cc kh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k'').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : ¬((finProdFinEquiv.symm k'').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) img := by
        intro cc _
        rw [differentiableAt_pi]
        intro k''
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨((finProdFinEquiv.symm k'').1.val - 1) / (W / patchSize) *
                  patchSize + kh.val, hpad.2.1⟩),
            ⟨((finProdFinEquiv.symm k'').1.val - 1) % (W / patchSize) *
              patchSize + kw.val, hpad.2.2⟩)) _
      rw [pdiv_finset_sum _ _ _ h_c_diff]
      -- For each c, distribute over kh-sum.
      have h_inner_c : ∀ cc : Fin ic,
          pdiv (fun (img''' : Vec (ic * H * W)) (k'' : Fin ((N + 1) * D)) =>
            ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              W_conv (finProdFinEquiv.symm k'').2 cc kh kw *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k'').1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + kh.val
                 let ww := w' * patchSize + kw.val
                 if hpad : ¬((finProdFinEquiv.symm k'').1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img''' (finProdFinEquiv (finProdFinEquiv
                     (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0)) img idx_in idx_out =
          ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv (finProdFinEquiv.symm idx_out).2 cc kh kw *
              (let W' := W / patchSize
               let p := (finProdFinEquiv.symm idx_out).1.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : ¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                         hh < H ∧ ww < W then
                 (if idx_in = finProdFinEquiv (finProdFinEquiv
                     (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩) then (1 : ℝ) else 0)
               else 0) := by
        intro cc
        -- Distribute over kh.
        rw [show (fun (img''' : Vec (ic * H * W)) (k'' : Fin ((N + 1) * D)) =>
              ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k'').2 cc kh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k'').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + kh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : ¬((finProdFinEquiv.symm k'').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) =
            (fun img''' k'' => ∑ kh : Fin patchSize,
              (fun (khh : Fin patchSize) (img'''' : Vec (ic * H * W))
                   (k''' : Fin ((N + 1) * D)) =>
                ∑ kw : Fin patchSize,
                  W_conv (finProdFinEquiv.symm k''').2 cc khh kw *
                    (let W' := W / patchSize
                     let p := (finProdFinEquiv.symm k''').1.val - 1
                     let h' := p / W'
                     let w' := p % W'
                     let hh := h' * patchSize + khh.val
                     let ww := w' * patchSize + kw.val
                     if hpad : ¬((finProdFinEquiv.symm k''').1.val = 0) ∧
                               hh < H ∧ ww < W then
                       img'''' (finProdFinEquiv (finProdFinEquiv
                         (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                     else 0)) kh img''' k'') from rfl]
        have h_kh_diff : ∀ khh ∈ (Finset.univ : Finset (Fin patchSize)),
            DifferentiableAt ℝ
              (fun (img'''' : Vec (ic * H * W)) (k''' : Fin ((N + 1) * D)) =>
                ∑ kw : Fin patchSize,
                  W_conv (finProdFinEquiv.symm k''').2 cc khh kw *
                    (let W' := W / patchSize
                     let p := (finProdFinEquiv.symm k''').1.val - 1
                     let h' := p / W'
                     let w' := p % W'
                     let hh := h' * patchSize + khh.val
                     let ww := w' * patchSize + kw.val
                     if hpad : ¬((finProdFinEquiv.symm k''').1.val = 0) ∧
                               hh < H ∧ ww < W then
                       img'''' (finProdFinEquiv (finProdFinEquiv
                         (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                     else 0)) img := by
          intro khh _
          rw [differentiableAt_pi]
          intro k'''
          apply DifferentiableAt.fun_sum; intro kw _
          apply DifferentiableAt.mul (differentiableAt_const _)
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨((finProdFinEquiv.symm k''').1.val - 1) / (W / patchSize) *
                    patchSize + khh.val, hpad.2.1⟩),
              ⟨((finProdFinEquiv.symm k''').1.val - 1) % (W / patchSize) *
                patchSize + kw.val, hpad.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kh_diff]
        congr 1; ext khh
        -- Distribute over kw.
        rw [show (fun (img'''' : Vec (ic * H * W)) (k''' : Fin ((N + 1) * D)) =>
              ∑ kw : Fin patchSize,
                W_conv (finProdFinEquiv.symm k''').2 cc khh kw *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k''').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + khh.val
                   let ww := w' * patchSize + kw.val
                   if hpad : ¬((finProdFinEquiv.symm k''').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img'''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) =
            (fun img'''' k''' => ∑ kw : Fin patchSize,
              (fun (kww : Fin patchSize) (img''''' : Vec (ic * H * W))
                   (k'''' : Fin ((N + 1) * D)) =>
                W_conv (finProdFinEquiv.symm k'''').2 cc khh kww *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k'''').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + khh.val
                   let ww := w' * patchSize + kww.val
                   if hpad : ¬((finProdFinEquiv.symm k'''').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img''''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) kw img'''' k''') from rfl]
        have h_kw_diff : ∀ kww ∈ (Finset.univ : Finset (Fin patchSize)),
            DifferentiableAt ℝ
              (fun (img''''' : Vec (ic * H * W)) (k'''' : Fin ((N + 1) * D)) =>
                W_conv (finProdFinEquiv.symm k'''').2 cc khh kww *
                  (let W' := W / patchSize
                   let p := (finProdFinEquiv.symm k'''').1.val - 1
                   let h' := p / W'
                   let w' := p % W'
                   let hh := h' * patchSize + khh.val
                   let ww := w' * patchSize + kww.val
                   if hpad : ¬((finProdFinEquiv.symm k'''').1.val = 0) ∧
                             hh < H ∧ ww < W then
                     img''''' (finProdFinEquiv (finProdFinEquiv
                       (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                   else 0)) img := by
          intro kww _
          rw [differentiableAt_pi]
          intro k''''
          apply DifferentiableAt.mul (differentiableAt_const _)
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨((finProdFinEquiv.symm k'''').1.val - 1) / (W / patchSize) *
                    patchSize + khh.val, hpad.2.1⟩),
              ⟨((finProdFinEquiv.symm k'''').1.val - 1) % (W / patchSize) *
                patchSize + kww.val, hpad.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kw_diff]
        congr 1; ext kww
        -- Per-(cc, khh, kww) summand: factor as (W constant) * (dite in img).
        rw [show (fun (img''''' : Vec (ic * H * W)) (k'''' : Fin ((N + 1) * D)) =>
              W_conv (finProdFinEquiv.symm k'''').2 cc khh kww *
                (let W' := W / patchSize
                 let p := (finProdFinEquiv.symm k'''').1.val - 1
                 let h' := p / W'
                 let w' := p % W'
                 let hh := h' * patchSize + khh.val
                 let ww := w' * patchSize + kww.val
                 if hpad : ¬((finProdFinEquiv.symm k'''').1.val = 0) ∧
                           hh < H ∧ ww < W then
                   img''''' (finProdFinEquiv (finProdFinEquiv
                     (cc, ⟨hh, hpad.2.1⟩), ⟨ww, hpad.2.2⟩))
                 else 0)) =
            (fun (img''''' : Vec (ic * H * W)) (k'''' : Fin ((N + 1) * D)) =>
              (fun k''''' : Fin ((N + 1) * D) =>
                W_conv (finProdFinEquiv.symm k''''').2 cc khh kww) k'''' *
              (if hpad : ¬((finProdFinEquiv.symm k'''').1.val = 0) ∧
                        ((finProdFinEquiv.symm k'''').1.val - 1) / (W / patchSize) *
                          patchSize + khh.val < H ∧
                        ((finProdFinEquiv.symm k'''').1.val - 1) % (W / patchSize) *
                          patchSize + kww.val < W then
                img''''' (finProdFinEquiv (finProdFinEquiv
                  (cc, ⟨((finProdFinEquiv.symm k'''').1.val - 1) / (W / patchSize) *
                       patchSize + khh.val, hpad.2.1⟩),
                  ⟨((finProdFinEquiv.symm k'''').1.val - 1) % (W / patchSize) *
                       patchSize + kww.val, hpad.2.2⟩))
              else 0)) from rfl]
        rw [pdiv_const_mul_pi_pad_eval
          (fun k'''' : Fin ((N + 1) * D) =>
            W_conv (finProdFinEquiv.symm k'''').2 cc khh kww)
          (fun k'''' => ¬((finProdFinEquiv.symm k'''').1.val = 0) ∧
            ((finProdFinEquiv.symm k'''').1.val - 1) / (W / patchSize) *
              patchSize + khh.val < H ∧
            ((finProdFinEquiv.symm k'''').1.val - 1) % (W / patchSize) *
              patchSize + kww.val < W)
          (fun k'''' hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨((finProdFinEquiv.symm k'''').1.val - 1) / (W / patchSize) *
                  patchSize + khh.val, hpad.2.1⟩),
            ⟨((finProdFinEquiv.symm k'''').1.val - 1) % (W / patchSize) *
              patchSize + kww.val, hpad.2.2⟩))]
        -- Bridge: pdiv_const_mul_pi_pad_eval gives the form `σ = idx_in` (sigma on left);
        -- h_inner_c expects `idx_in = σ`. Symmetrize via by_cases on σ_eq.
        show W_conv (finProdFinEquiv.symm idx_out).2 cc khh kww * _ =
             W_conv (finProdFinEquiv.symm idx_out).2 cc khh kww * _
        congr 1
        by_cases hpad : ¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                        ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + khh.val < H ∧
                        ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kww.val < W
        · rw [dif_pos hpad, dif_pos hpad]
          by_cases heq : finProdFinEquiv (finProdFinEquiv
              (cc, ⟨((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                     patchSize + khh.val, hpad.2.1⟩),
              ⟨((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                     patchSize + kww.val, hpad.2.2⟩) = idx_in
          · rw [if_pos heq, if_pos heq.symm]
          · rw [if_neg heq, if_neg (fun h => heq h.symm)]
        · rw [dif_neg hpad, dif_neg hpad]
      simp_rw [h_inner_c]
      -- Now case-split on hn0 to convert the 3-conjunct dite to either 0 (n=0)
      -- or the desired 2-conjunct dite (n ≠ 0).
      by_cases hn0 : (finProdFinEquiv.symm idx_out).1.val = 0
      · rw [dif_pos hn0]
        apply Finset.sum_eq_zero; intro c _
        apply Finset.sum_eq_zero; intro kh _
        apply Finset.sum_eq_zero; intro kw _
        -- Per-summand: dite has 3-conj cond; ¬(n_out.val = 0) is False (since hn0),
        -- so dite = 0 and summand = W * 0 = 0.
        have h_neg : ¬(¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                      ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val < H ∧
                      ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val < W) :=
          fun h => h.1 hn0
        show W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
             (if hpad : ¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                       ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val < H ∧
                       ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val < W then
                (if idx_in = finProdFinEquiv (finProdFinEquiv
                    (c, ⟨((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val, hpad.2.1⟩),
                    ⟨((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val, hpad.2.2⟩) then (1 : ℝ) else 0)
              else 0) = 0
        rw [dif_neg h_neg, mul_zero]
      · rw [dif_neg hn0]
        apply Finset.sum_congr rfl; intro c _
        apply Finset.sum_congr rfl; intro kh _
        apply Finset.sum_congr rfl; intro kw _
        -- Per-summand: bridge 3-conj let-form to 2-conj let-form using ¬hn0.
        show W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
             (if hpad : ¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                       ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val < H ∧
                       ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val < W then
                (if idx_in = finProdFinEquiv (finProdFinEquiv
                    (c, ⟨((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val, hpad.2.1⟩),
                    ⟨((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val, hpad.2.2⟩) then (1 : ℝ) else 0)
              else 0) =
             W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
             (if hpad : ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val < H ∧
                       ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val < W then
                (if idx_in = finProdFinEquiv (finProdFinEquiv
                    (c, ⟨((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                          patchSize + kh.val, hpad.1⟩),
                    ⟨((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                          patchSize + kw.val, hpad.2⟩) then (1 : ℝ) else 0)
              else 0)
        congr 1
        by_cases hpad' : ((finProdFinEquiv.symm idx_out).1.val - 1) /
                            (W / patchSize) * patchSize + kh.val < H ∧
                         ((finProdFinEquiv.symm idx_out).1.val - 1) %
                            (W / patchSize) * patchSize + kw.val < W
        · rw [dif_pos ⟨hn0, hpad'⟩, dif_pos hpad']
        · have h_neg' : ¬(¬((finProdFinEquiv.symm idx_out).1.val = 0) ∧
                          ((finProdFinEquiv.symm idx_out).1.val - 1) / (W / patchSize) *
                              patchSize + kh.val < H ∧
                          ((finProdFinEquiv.symm idx_out).1.val - 1) % (W / patchSize) *
                              patchSize + kw.val < W) :=
            fun h => hpad' h.2
          rw [dif_neg h_neg', dif_neg hpad']
    -- Step 2: closing collapse.
    show patchEmbed_input_grad_formula ic H W patchSize N D W_conv dy idx_in =
         ∑ idx_out : Fin ((N + 1) * D),
           pdiv (patchEmbed_flat ic H W patchSize N D
                   W_conv b_conv cls_token pos_embed) img idx_in idx_out * dy idx_out
    unfold patchEmbed_input_grad_formula
    -- Substitute h_pdiv on RHS.
    simp_rw [h_pdiv]
    -- Reindex Σ idx_out → Σ pair via finProdFinEquiv.symm.
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx_out : Fin ((N + 1) * D) =>
          (if _hn0 : (finProdFinEquiv.symm idx_out).1.val = 0 then (0 : ℝ)
           else
             ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
               W_conv (finProdFinEquiv.symm idx_out).2 c kh kw *
                 (let W' := W / patchSize
                  let p := (finProdFinEquiv.symm idx_out).1.val - 1
                  let h' := p / W'
                  let w' := p % W'
                  let hh := h' * patchSize + kh.val
                  let ww := w' * patchSize + kw.val
                  if hpad : hh < H ∧ ww < W then
                    (if idx_in = finProdFinEquiv (finProdFinEquiv
                        (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
                  else 0)) * dy idx_out)
        (fun pair : Fin (N + 1) × Fin D =>
          (if _hn0 : pair.1.val = 0 then (0 : ℝ)
           else
             ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
               W_conv pair.2 c kh kw *
                 (let W' := W / patchSize
                  let p := pair.1.val - 1
                  let h' := p / W'
                  let w' := p % W'
                  let hh := h' * patchSize + kh.val
                  let ww := w' * patchSize + kw.val
                  if hpad : hh < H ∧ ww < W then
                    (if idx_in = finProdFinEquiv (finProdFinEquiv
                        (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
                  else 0)) * dy (finProdFinEquiv pair))
        (fun idx_out => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    -- Split Σ n via Fin.sum_univ_succ: n=0 row + ∑ p:Fin N (n=p.succ).
    rw [Fin.sum_univ_succ]
    -- The n=0 term contributes 0 (each summand has dif_pos rfl → 0, then 0 * dy = 0).
    rw [show (∑ d : Fin D,
        (if _hn0 : ((0 : Fin (N + 1)).val = 0) then (0 : ℝ)
         else
           ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
             W_conv ((0 : Fin (N + 1)), d).2 c kh kw *
               (let W' := W / patchSize
                let p := ((0 : Fin (N + 1)), d).1.val - 1
                let h' := p / W'
                let w' := p % W'
                let hh := h' * patchSize + kh.val
                let ww := w' * patchSize + kw.val
                if hpad : hh < H ∧ ww < W then
                  (if idx_in = finProdFinEquiv (finProdFinEquiv
                      (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
                else 0)) * dy (finProdFinEquiv ((0 : Fin (N + 1)), d))) = 0 from by
      apply Finset.sum_eq_zero; intro d _
      have h_zero : ((0 : Fin (N + 1)).val = 0) := Fin.val_zero (N + 1)
      rw [dif_pos h_zero, zero_mul]]
    rw [zero_add]
    -- Per-p: simplify dif_neg at p.succ + match formula.
    apply Finset.sum_congr rfl; intro p _
    have h_p_ne : (Fin.succ p).val ≠ 0 := Nat.succ_ne_zero _
    simp_rw [dif_neg h_p_ne]
    -- The `(p.succ).val - 1 = p.val` simplification.
    have h_p_succ_sub : (Fin.succ p).val - 1 = p.val := by
      show p.val + 1 - 1 = p.val
      omega
    -- Per-p, prove the indicator helper: dite-form ⇔ conjunction-form.
    have h_indicator : ∀ c : Fin ic, ∀ kh kw : Fin patchSize,
        ((let W' := W / patchSize
          let p_val := (Fin.succ p).val - 1
          let h' := p_val / W'
          let w' := p_val % W'
          let hh := h' * patchSize + kh.val
          let ww := w' * patchSize + kw.val
          if hpad : hh < H ∧ ww < W then
            (if idx_in = finProdFinEquiv (finProdFinEquiv
                (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩) then (1 : ℝ) else 0)
          else 0) : ℝ) =
        (if c = c_in ∧
              p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
              p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
          (1 : ℝ) else 0) := by
      intro c kh kw
      simp only [h_p_succ_sub]
      by_cases hpad : p.val / (W / patchSize) * patchSize + kh.val < H ∧
                      p.val % (W / patchSize) * patchSize + kw.val < W
      · rw [dif_pos hpad]
        by_cases h_match : c = c_in ∧
                           p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                           p.val % (W / patchSize) * patchSize + kw.val = ww_in.val
        · have h_idx_in_eq : idx_in = finProdFinEquiv (finProdFinEquiv
              (c, ⟨p.val / (W / patchSize) * patchSize + kh.val, hpad.1⟩),
              ⟨p.val % (W / patchSize) * patchSize + kw.val, hpad.2⟩) := by
            rw [hidx_in]
            have h_c : c_in = c := h_match.1.symm
            have h_hh : (⟨p.val / (W / patchSize) * patchSize + kh.val, hpad.1⟩ : Fin H) =
                hh_in := by
              apply Fin.ext
              show p.val / (W / patchSize) * patchSize + kh.val = hh_in.val
              exact h_match.2.1
            have h_ww : (⟨p.val % (W / patchSize) * patchSize + kw.val, hpad.2⟩ : Fin W) =
                ww_in := by
              apply Fin.ext
              show p.val % (W / patchSize) * patchSize + kw.val = ww_in.val
              exact h_match.2.2
            rw [h_c, ← h_hh, ← h_ww]
          rw [if_pos h_idx_in_eq, if_pos h_match]
        · rw [if_neg h_match, if_neg]
          intro h_eq
          apply h_match
          rw [hidx_in] at h_eq
          have h_inj := finProdFinEquiv.injective h_eq
          have h_inj_pair := Prod.mk.inj h_inj
          have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
          have h_inj_inner_pair := Prod.mk.inj h_inj_inner
          refine ⟨h_inj_inner_pair.1.symm, ?_, ?_⟩
          · exact (Fin.ext_iff.mp h_inj_inner_pair.2).symm
          · exact (Fin.ext_iff.mp h_inj_pair.2).symm
      · rw [dif_neg hpad]
        rw [if_neg]
        intro ⟨_, h_hh, h_ww⟩
        apply hpad
        refine ⟨?_, ?_⟩
        · rw [h_hh]; exact hh_in.isLt
        · rw [h_ww]; exact ww_in.isLt
    -- Use h_indicator to rewrite dite to conjunction-indicator.
    simp_rw [h_indicator]
    -- Now: LHS = ∑ kh kw, (if h_match then ∑ d, W d c_in kh kw * dy else 0)
    --      RHS = ∑ d, (∑ c kh kw, W * (if c=c_in ∧ h_match then 1 else 0)) * dy
    -- Strategy: reduce both sides to the canonical form
    --     ∑ d, ∑ kh, ∑ kw, (if h_match then W d c_in kh kw * dy else 0)
    -- Step 1: convert LHS.
    have h_lhs_canonical :
        (∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
              ∑ d : Fin D, W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
            else 0) =
        ∑ d : Fin D, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            (if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
              W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
            else 0) := by
      -- Pull if into Σ d, then swap orders to get Σ d outermost.
      rw [show (∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                      p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
                ∑ d : Fin D, W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
              else 0) =
            (∑ kh : Fin patchSize, ∑ kw : Fin patchSize, ∑ d : Fin D,
              (if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                      p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
                W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
              else 0)) from by
        apply Finset.sum_congr rfl; intro kh _
        apply Finset.sum_congr rfl; intro kw _
        by_cases h_match : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                           p.val % (W / patchSize) * patchSize + kw.val = ww_in.val
        · rw [dif_pos h_match]
          apply Finset.sum_congr rfl; intro d _
          rw [dif_pos h_match]
        · rw [dif_neg h_match]
          symm
          apply Finset.sum_eq_zero; intro d _
          rw [dif_neg h_match]]
      -- Now: ∑ kh, ∑ kw, ∑ d, body. Move Σ d outermost via two sum_comm applications.
      -- Step 1: swap inner ∑ kw and ∑ d (under ∑ kh binder) via a per-kh helper.
      rw [show (∑ kh : Fin patchSize, ∑ kw : Fin patchSize, ∑ d : Fin D,
            (if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
              W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
            else 0)) =
            (∑ kh : Fin patchSize, ∑ d : Fin D, ∑ kw : Fin patchSize,
            (if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
              W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
            else 0)) from by
        apply Finset.sum_congr rfl; intro kh _
        exact Finset.sum_comm]
      -- Step 2: swap outer ∑ kh and ∑ d.
      rw [Finset.sum_comm]
    rw [h_lhs_canonical]
    -- Step 2: convert RHS.
    have h_rhs_canonical :
        (∑ d : Fin D,
            (∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              W_conv d c kh kw *
                (if c = c_in ∧
                    p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
                  (1 : ℝ) else 0)) *
            dy (finProdFinEquiv (Fin.succ p, d))) =
        ∑ d : Fin D, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            (if _h : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
              W_conv d c_in kh kw * dy (finProdFinEquiv (Fin.succ p, d))
            else 0) := by
      apply Finset.sum_congr rfl; intro d _
      -- Per-d: (∑ c kh kw, W * indicator) * dy = ∑ kh kw, [if h_match then W d c_in * dy else 0]
      rw [Finset.sum_mul]  -- (∑ c, ...) * dy = ∑ c, ... * dy
      -- Goal: ∑ c, (∑ kh, ∑ kw, W * indicator) * dy = ∑ kh, ∑ kw, ...
      -- Hmm wait, Finset.sum_mul applied to outermost ∑ c.
      rw [Finset.sum_eq_single c_in]
      rotate_left
      · -- Other c contribute 0.
        intro c _ hc_ne
        rw [show ((∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
              W_conv d c kh kw *
                (if c = c_in ∧
                    p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                    p.val % (W / patchSize) * patchSize + kw.val = ww_in.val then
                  (1 : ℝ) else 0)) : ℝ) = 0 from by
          apply Finset.sum_eq_zero; intro kh _
          apply Finset.sum_eq_zero; intro kw _
          rw [if_neg (fun ⟨h, _⟩ => hc_ne h), mul_zero]]
        rw [zero_mul]
      · intro h; exact absurd (Finset.mem_univ c_in) h
      -- Now: (∑ kh, ∑ kw, W d c_in kh kw * (if c_in = c_in ∧ h_match then 1 else 0)) * dy
      -- = ∑ kh, ∑ kw, [if h_match then W d c_in kh kw * dy else 0]
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl; intro kh _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl; intro kw _
      -- Per-(d, kh, kw): W d c_in kh kw * (if c_in = c_in ∧ h_match then 1 else 0) * dy
      -- = if h_match then W d c_in kh kw * dy else 0
      by_cases h_match : p.val / (W / patchSize) * patchSize + kh.val = hh_in.val ∧
                         p.val % (W / patchSize) * patchSize + kw.val = ww_in.val
      · rw [if_pos ⟨rfl, h_match⟩, dif_pos h_match]; ring
      · rw [if_neg (fun ⟨_, h⟩ => h_match h), dif_neg h_match]; ring
    rw [h_rhs_canonical]

/-! ## The full ViT theorem

Compose patch embed + ViT body (via the Mat→Vec bridge) + classifier.
All three are `HasVJP`s on `Vec`, so `vjp_comp` chains them directly. -/

/-- **vit_full** — full ViT forward from flattened image pixels to logits.

    `Vec (ic*H*W) → Vec nClasses`

    Composition: `patchEmbed → (flatten ∘ vit_body ∘ unflatten) → classifier`.
    Uses `D := heads * d_head` directly (no separate `D` parameter) so the
    type-level reinterpretation at the body is a no-op. -/
noncomputable def vit_full
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifier_flat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten
      (vit_body kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
       (Mat.unflatten v))) ∘
  (patchEmbed_flat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- Differentiability of the classifier head — composition of linear ops. -/
lemma classifier_flat_diff (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses) :
    Differentiable ℝ (classifier_flat N D nClasses Wcls bcls) := by
  unfold classifier_flat
  exact (dense_input_diff Wcls bcls).comp (cls_slice_flat_diff N D)

/-- **vit_full VJP — the grand finale.** Theorem, no longer axiom: three
    `vjp_comp` steps glueing `patchEmbed_flat_has_vjp`,
    `hasVJPMat_to_hasVJP (vit_body_has_vjp_mat ...)`, and
    `classifier_flat_has_vjp`. Each `vjp_comp`'s Diff hypotheses are
    discharged by the per-stage Diff theorems above. -/
noncomputable def vit_full_has_vjp
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vit_full ic H W patchSize N mlpDim heads d_head kBlocks nClasses
              W_conv b_conv cls_token pos_embed
              ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
              γ2 β2 Wfc1 bfc1 Wfc2 bfc2
              γF βF Wcls bcls) :=
  -- Inner: patchEmbed
  let body_bridge : HasVJP (fun v : Vec ((N + 1) * (heads * d_head)) =>
        Mat.flatten (vit_body kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
                       Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                     (Mat.unflatten v))) :=
    hasVJPMat_to_hasVJP (vit_body_has_vjp_mat kBlocks (N + 1) heads d_head mlpDim
                          ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo
                          γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF)
  let body_bridge_diff := vit_body_flat_diff kBlocks (N + 1) heads d_head mlpDim
                            ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo
                            γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
  let patch_diff := patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
                      W_conv b_conv cls_token pos_embed
  let patch_has_vjp := patchEmbed_flat_has_vjp ic H W patchSize N (heads * d_head)
                        W_conv b_conv cls_token pos_embed
  -- Inner composition: body_bridge ∘ patchEmbed
  let inner_has_vjp := vjp_comp _ _ patch_diff body_bridge_diff
                        patch_has_vjp body_bridge
  have inner_diff : Differentiable ℝ
      ((fun v : Vec ((N + 1) * (heads * d_head)) =>
          Mat.flatten (vit_body kBlocks (N + 1) heads d_head mlpDim ε γ1 β1
                         Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF
                       (Mat.unflatten v))) ∘
        (patchEmbed_flat ic H W patchSize N (heads * d_head)
           W_conv b_conv cls_token pos_embed)) :=
    body_bridge_diff.comp patch_diff
  -- Outer: classifier_flat ∘ (body_bridge ∘ patchEmbed)
  vjp_comp _ _ inner_diff
    (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls)
    inner_has_vjp
    (classifier_flat_has_vjp N (heads * d_head) nClasses Wcls bcls)

end Proofs
