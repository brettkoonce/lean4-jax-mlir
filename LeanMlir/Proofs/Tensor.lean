import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.Ring
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.FDeriv.Linear

/-!
# Tensor Algebra for VJP Proofs

Vectors, matrices, and operations over `ℝ`, using Mathlib's `Finset.sum`.

Partial derivatives (`pdiv`) and their composition rules (chain rule,
linearity, product rule) are now **defined and proved** from Mathlib's
Fréchet derivative `fderiv`. The post-foundation-flip definition is

  `pdiv f x i j := fderiv ℝ f x (basisVec i) j`

and every former axiom (`pdiv_id`, `pdiv_const`, `pdiv_reindex`,
`pdiv_add`, `pdiv_comp`, `pdiv_mul`) is now a theorem proved against
Mathlib's API. The bilinear rules carry `Differentiable` hypotheses
that propagate through every downstream chapter.

The post-flip path: every claim downstream of this file is either a
definition Lean unfolds or a theorem typechecked against Mathlib —
no project axioms remain. `#print axioms vit_full_has_vjp` lists only
Lean core (`propext`, `Classical.choice`, `Quot.sound`).
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Types
-- ════════════════════════════════════════════════════════════════

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (m n : Nat) := Fin m → Fin n → ℝ

-- ════════════════════════════════════════════════════════════════
-- § Matrix Operations
-- ════════════════════════════════════════════════════════════════

namespace Mat

noncomputable def mulVec (A : Mat m n) (v : Vec n) : Vec m :=
  fun i => ∑ j : Fin n, A i j * v j

def outer (u : Vec m) (v : Vec n) : Mat m n :=
  fun i j => u i * v j

noncomputable def mul (A : Mat m n) (B : Mat n p) : Mat m p :=
  fun i k => ∑ j : Fin n, A i j * B j k

/-- Matrix transpose: swap rows and columns. -/
def transpose (A : Mat m n) : Mat n m :=
  fun j i => A i j

end Mat

-- ════════════════════════════════════════════════════════════════
-- § Differentiation (Mathlib-grounded)
--
-- `pdiv f x i j` is the (i, j) entry of the Jacobian of
-- `f : Vec m → Vec n` at `x`, recovered by applying `fderiv ℝ f x` to
-- the i-th standard basis vector and reading off the j-th coordinate.
-- All six structural rules (id, const, reindex, sum, product, chain)
-- are theorems. The bilinear rules (sum/product/chain) carry
-- `DifferentiableAt` hypotheses, the form required to be consistent
-- with `fderiv`'s junk-default at non-smooth points.
-- ════════════════════════════════════════════════════════════════

/-- Standard basis vector `eᵢ` in `Vec m`: 1 at index i, 0 elsewhere.
    Avoids `Pi.single`'s dependent-type elaboration friction in
    contexts where the codomain family isn't immediately apparent. -/
@[reducible] def basisVec {m : Nat} (i : Fin m) : Vec m :=
  fun k => if k = i then (1 : ℝ) else 0

@[simp] theorem basisVec_apply {m : Nat} (i j : Fin m) :
    basisVec i j = if j = i then (1 : ℝ) else 0 := rfl

/-- The reindex map `y ↦ (k ↦ y (σ k))` packaged as a continuous linear
    map. Used to discharge `pdiv_reindex` and to provide
    `DifferentiableAt` evidence for reindex-shaped subexpressions. -/
noncomputable def reindexCLM {a b : Nat} (σ : Fin b → Fin a) :
    Vec a →L[ℝ] Vec b :=
  { toFun := fun y k => y (σ k)
    map_add' := by intros; rfl
    map_smul' := by intros; rfl
    cont := continuous_pi (fun k => continuous_apply (σ k)) }

@[simp] theorem reindexCLM_apply {a b : Nat} (σ : Fin b → Fin a) (y : Vec a) :
    reindexCLM σ y = fun k => y (σ k) := rfl

/-- **Partial derivative.** The (i, j) entry of the Jacobian of
    `f : Vec m → Vec n` at `x`. -/
noncomputable def pdiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) : ℝ :=
  fderiv ℝ f x (basisVec i) j

/-- **Identity Jacobian** — `δᵢⱼ`. -/
theorem pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 := by
  unfold pdiv
  rw [show (fun y : Vec n => y) = id from rfl, fderiv_id]
  show basisVec i j = _
  rw [basisVec_apply]
  rcases eq_or_ne j i with h | h
  · subst h; simp
  · rw [if_neg h, if_neg (fun h' => h h'.symm)]

/-- **Constant function Jacobian** — zero. -/
theorem pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 := by
  unfold pdiv
  rw [(hasFDerivAt_const c x).fderiv]
  rfl

/-- **Reindex Jacobian** — sparse, hits 1 only at i = σ(j). Subsumes
    `pdiv_id` (set a = b, σ = id). Covers transpose, flatten,
    unflatten, slicing, any permutation. -/
theorem pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 := by
  unfold pdiv
  rw [show (fun y : Vec a => fun k : Fin b => y (σ k)) =
        (reindexCLM σ : Vec a → Vec b) from rfl]
  rw [ContinuousLinearMap.fderiv]
  show basisVec i (σ j) = _
  rw [basisVec_apply]
  rcases eq_or_ne (σ j) i with h | h
  · subst h; simp
  · rw [if_neg h, if_neg (fun h' => h h'.symm)]

/-- **Product rule** for `pdiv`. `Vec n` is a normed algebra over ℝ
    via `Pi.normedAlgebra`, so `fderiv_mul` applies directly to the
    pointwise product `f * g`. Requires both factors to be
    `DifferentiableAt x`. -/
theorem pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j := by
  unfold pdiv
  rw [show (fun y : Vec m => fun k => f y k * g y k) = (f * g) from rfl]
  rw [fderiv_mul hf hg]
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
             smul_eq_mul, Pi.add_apply, Pi.mul_apply]
  ring

/-- **Sum rule** for `pdiv`. Requires both summands to be
    `DifferentiableAt x`. -/
theorem pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j := by
  unfold pdiv
  rw [show (fun y => fun k => f y k + g y k) = (f + g) from rfl]
  rw [fderiv_add hf hg]
  rfl

/-- **Chain rule** for `pdiv`. Requires `f` differentiable at `x` and
    `g` differentiable at `f x`. -/
theorem pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k := by
  unfold pdiv
  rw [fderiv_comp x hg hf]
  show fderiv ℝ g (f x) (fderiv ℝ f x (basisVec i)) k = _
  set v : Vec n := fderiv ℝ f x (basisVec i) with hv
  have hv_decomp : v = ∑ j : Fin n, v j • (basisVec j : Vec n) := by
    funext j'
    rw [Finset.sum_apply]
    simp_rw [Pi.smul_apply, basisVec_apply, smul_eq_mul, mul_ite, mul_one, mul_zero]
    rw [Finset.sum_ite_eq Finset.univ j' (fun j => v j)]
    simp
  conv_lhs => rw [hv_decomp]
  rw [map_sum]
  rw [Finset.sum_apply]
  congr 1
  funext j
  rw [(fderiv ℝ g (f x)).map_smul]
  show v j * fderiv ℝ g (f x) (basisVec j) k = _
  rfl

/-- **Finset-sum rule** — derived from `pdiv_add` and `pdiv_const` by
    induction on the Finset. Linearity of the derivative extended to
    arbitrary finite sums. Requires each `f s` to be differentiable
    at `x`. -/
theorem pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j := by
  induction S using Finset.induction_on with
  | empty =>
    simp only [Finset.sum_empty]
    exact pdiv_const (fun _ : Fin n => (0 : ℝ)) x i j
  | @insert a T ha ih =>
    have hdiff_a : DifferentiableAt ℝ (f a) x :=
      hdiff a (Finset.mem_insert_self a T)
    have hdiff_T : ∀ s ∈ T, DifferentiableAt ℝ (f s) x := fun s hs =>
      hdiff s (Finset.mem_insert_of_mem hs)
    have hdiff_sumT :
        DifferentiableAt ℝ (fun y : Vec m => fun k : Fin n => ∑ s ∈ T, f s y k) x := by
      have heq_curry : (fun y : Vec m => fun k : Fin n => ∑ s ∈ T, f s y k)
                     = (fun y : Vec m => ∑ s ∈ T, f s y) := by
        funext y k; rw [Finset.sum_apply]
      rw [heq_curry]
      exact DifferentiableAt.fun_sum (fun s hs => hdiff_T s hs)
    have heq :
        (fun (y : Vec m) (k : Fin n) => ∑ s ∈ insert a T, f s y k) =
        (fun y k => f a y k + (fun y' k' => ∑ s ∈ T, f s y' k') y k) := by
      funext y k
      rw [Finset.sum_insert ha]
    rw [heq, pdiv_add _ _ _ hdiff_a hdiff_sumT, ih hdiff_T,
        Finset.sum_insert ha]

-- ════════════════════════════════════════════════════════════════
-- § VJP Framework
-- ════════════════════════════════════════════════════════════════

structure HasVJP {m n : Nat} (f : Vec m → Vec n) where
  backward : Vec m → Vec n → Vec m
  correct : ∀ (x : Vec m) (dy : Vec n) (i : Fin m),
    backward x dy i = ∑ j : Fin n, pdiv f x i j * dy j

/-- **Chain rule for VJPs** — proved, no sorry. Requires `f` and `g`
    to be differentiable everywhere. -/
noncomputable def vjp_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (g ∘ f) where
  backward := fun x dy => hf.backward x (hg.backward (f x) dy)
  correct := by
    intro x dy i
    rw [hf.correct]
    simp_rw [hg.correct]
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    congr 1; ext k
    rw [pdiv_comp _ _ _ (hf_diff x) (hg_diff (f x))]
    simp_rw [← mul_assoc]
    rw [← Finset.sum_mul]

/-- **Additive fan-in** — proved, no sorry. Requires `f` and `g` to be
    differentiable everywhere. -/
@[reducible] noncomputable def biPath {m n : Nat} (f g : Vec m → Vec n) : Vec m → Vec n :=
  fun x i => f x i + g x i

noncomputable def biPath_has_vjp {m n : Nat}
    (f g : Vec m → Vec n)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (biPath f g) where
  backward := fun x dy i => hf.backward x dy i + hg.backward x dy i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j; rw [pdiv_add _ _ _ (hf_diff x) (hg_diff x)]; ring

/-- **Multiplicative fan-in** — proved, no sorry. Requires `f` and `g`
    to be differentiable everywhere. -/
@[reducible] noncomputable def elemwiseProduct {n : Nat}
    (f g : Vec n → Vec n) : Vec n → Vec n :=
  fun x i => f x i * g x i

noncomputable def elemwiseProduct_has_vjp {n : Nat}
    (f g : Vec n → Vec n)
    (hf_diff : Differentiable ℝ f) (hg_diff : Differentiable ℝ g)
    (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (elemwiseProduct f g) where
  backward := fun x dy i =>
    hf.backward x (fun j => g x j * dy j) i +
    hg.backward x (fun j => f x j * dy j) i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j
    rw [pdiv_mul _ _ _ (hf_diff x) (hg_diff x)]; ring

/-- **Identity VJP** — proved, no sorry. -/
def identity_has_vjp (n : Nat) : HasVJP (fun (x : Vec n) => x) where
  backward := fun _x dy => dy
  correct := by
    intro x dy i
    simp_rw [pdiv_id]
    simp [Finset.mem_univ]

-- ════════════════════════════════════════════════════════════════
-- § Matrix ↔ Vector flattening (row-major)
-- ════════════════════════════════════════════════════════════════

/-! `Mat m n` and `Vec (m * n)` are in bijection by row-major flattening.
This bijection lets us **define** `pdivMat` in terms of `pdiv` rather
than introducing parallel axioms, and so **derive** the rank-2 chain,
sum, and identity rules as theorems. The 5 local Jacobian axioms
(matmul, scalarScale, transpose, rowIndep) remain — they're genuine
calculus facts about specific operations, not structural framework. -/

namespace Mat

/-- Row-major flatten: `Mat m n → Vec (m * n)`. Uses Mathlib's
    `finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)`. -/
noncomputable def flatten {m n : Nat} (A : Mat m n) : Vec (m * n) :=
  fun k => let p := finProdFinEquiv.symm k; A p.1 p.2

/-- Row-major unflatten: `Vec (m * n) → Mat m n`. -/
noncomputable def unflatten {m n : Nat} (v : Vec (m * n)) : Mat m n :=
  fun i j => v (finProdFinEquiv (i, j))

/-- Unflatten is a left inverse of flatten. -/
theorem unflatten_flatten {m n : Nat} (A : Mat m n) :
    unflatten (flatten A) = A := by
  funext i j
  unfold unflatten flatten
  simp [Equiv.symm_apply_apply]

/-- Flatten is a left inverse of unflatten. -/
theorem flatten_unflatten {m n : Nat} (v : Vec (m * n)) :
    flatten (unflatten v) = v := by
  funext k
  change v (finProdFinEquiv (finProdFinEquiv.symm k)) = v k
  rw [Equiv.apply_symm_apply]

end Mat

-- ════════════════════════════════════════════════════════════════
-- § Matrix-level differentiation (derived from `pdiv`)
-- ════════════════════════════════════════════════════════════════

/-- **Matrix partial derivative**, defined in terms of `pdiv` on the
    row-major flattened `Vec` form. No longer an axiom — the rank-2
    structural rules (chain/sum/id) now follow as theorems. -/
noncomputable def pdivMat {a b c d : Nat} (f : Mat a b → Mat c d) (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) : ℝ :=
  pdiv (fun v : Vec (a * b) => Mat.flatten (f (Mat.unflatten v)))
    (Mat.flatten A) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l))

/-- **Chain rule for `pdivMat`** — now a theorem, derived from `pdiv_comp`
    via the row-major flatten bijection. -/
theorem pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by
  unfold pdivMat
  have h_compose :
      (fun v : Vec (a * b) => Mat.flatten ((G ∘ F) (Mat.unflatten v))) =
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) ∘
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) := by
    funext v
    simp [Function.comp, Mat.unflatten_flatten]
  have h_mid :
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)
      = Mat.flatten (F A) := by
    simp [Mat.unflatten_flatten]
  have hG_diff' : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u)))
      ((fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)) := by
    rw [h_mid]; exact hG_diff
  rw [h_compose, pdiv_comp _ _ _ hF_diff hG_diff']
  simp_rw [h_mid]
  -- Step 3: convert the single sum over Fin (c*d) to a double sum over Fin c × Fin d.
  rw [Fintype.sum_equiv finProdFinEquiv.symm
      (fun r =>
        pdiv (fun v => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)
          (finProdFinEquiv (i, j)) r *
        pdiv (fun u => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A))
          r (finProdFinEquiv (k, l)))
      (fun pq =>
        pdiv (fun v => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)
          (finProdFinEquiv (i, j)) (finProdFinEquiv pq) *
        pdiv (fun u => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A))
          (finProdFinEquiv pq) (finProdFinEquiv (k, l)))
      (fun r => by
        show _ = _ * _
        rw [Equiv.apply_symm_apply])]
  rw [Fintype.sum_prod_type]

/-- **Sum rule for `pdivMat`** — theorem, via `pdiv_add`. Requires both
    flattened summands to be differentiable at `flatten A`. -/
theorem pdivMat_add {a b c d : Nat}
    (F G : Mat a b → Mat c d) (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))) (Mat.flatten A))
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) :
    pdivMat (fun M r s => F M r s + G M r s) A i j k l
    = pdivMat F A i j k l + pdivMat G A i j k l := by
  unfold pdivMat
  have h_flat : (fun v : Vec (a * b) =>
                  Mat.flatten ((fun M r s => F M r s + G M r s) (Mat.unflatten v))) =
                (fun v k => (fun w => Mat.flatten (F (Mat.unflatten w))) v k +
                            (fun w => Mat.flatten (G (Mat.unflatten w))) v k) := by
    funext v k
    unfold Mat.flatten
    rfl
  rw [h_flat, pdiv_add _ _ _ hF_diff hG_diff]

/-- **Identity Jacobian for `pdivMat`** — theorem, via `pdiv_id`. -/
theorem pdivMat_id {a b : Nat} (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin a) (l : Fin b) :
    pdivMat (fun M : Mat a b => M) A i j k l =
    if i = k ∧ j = l then 1 else 0 := by
  unfold pdivMat
  -- flatten ∘ id ∘ unflatten = id (on Vec (a*b))
  have h_id : (fun v : Vec (a * b) => Mat.flatten (Mat.unflatten v)) =
              (fun v : Vec (a * b) => v) := by
    funext v; exact Mat.flatten_unflatten v
  rw [h_id, pdiv_id]
  -- Now: (if finProdFinEquiv (i,j) = finProdFinEquiv (k,l) then 1 else 0)
  --    = if i = k ∧ j = l then 1 else 0
  by_cases h : i = k ∧ j = l
  · obtain ⟨hik, hjl⟩ := h
    subst hik; subst hjl
    simp
  · rw [if_neg h, if_neg]
    intro heq
    apply h
    have := finProdFinEquiv.injective heq
    exact ⟨(Prod.mk.inj this).1, (Prod.mk.inj this).2⟩

-- ════════════════════════════════════════════════════════════════
-- § Matrix VJP Framework
-- ════════════════════════════════════════════════════════════════

/-- Matrix-level VJP: given a matrix-valued function of a matrix, a
    correct backward function contracts the `pdivMat` Jacobian against
    the output cotangent. Mirrors `HasVJP` for `Vec`. -/
structure HasVJPMat {a b c d : Nat} (f : Mat a b → Mat c d) where
  backward : Mat a b → Mat c d → Mat a b
  correct : ∀ (A : Mat a b) (dY : Mat c d) (i : Fin a) (j : Fin b),
    backward A dY i j = ∑ k : Fin c, ∑ l : Fin d,
      pdivMat f A i j k l * dY k l

/-- **Chain rule for matrix VJPs** — proved, no sorry.
    Direct transcription of `vjp_comp` to rank-2 indices. -/
noncomputable def vjpMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (hF_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))))
    (hG_diff : Differentiable ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))))
    (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (G ∘ F) where
  backward := fun A dY => hF.backward A (hG.backward (F A) dY)
  correct := by
    intro A dY i j
    rw [hF.correct]
    simp_rw [hG.correct]
    have hF_diff_at : DifferentiableAt ℝ
        (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A) :=
      hF_diff (Mat.flatten A)
    have hG_diff_at : DifferentiableAt ℝ
        (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)) :=
      hG_diff (Mat.flatten (F A))
    conv_rhs =>
      arg 2; ext k; arg 2; ext l
      rw [show pdivMat (G ∘ F) A i j k l * dY k l =
          (∑ p : Fin c, ∑ q : Fin d,
            pdivMat F A i j p q * pdivMat G (F A) p q k l) * dY k l
        from by rw [← pdivMat_comp _ _ _ hF_diff_at hG_diff_at]]
    simp_rw [Finset.sum_mul, mul_assoc, Finset.mul_sum]
    -- LHS: ∑p ∑q, pdivMat F · ∑k ∑l, pdivMat G · dY
    -- RHS: ∑k ∑l ∑p ∑q, pdivMat F · pdivMat G · dY
    -- Pack (p,q) and (k,l) into products, swap, unpack.
    calc _ = ∑ pq ∈ Finset.univ ×ˢ Finset.univ,
             ∑ kl ∈ Finset.univ ×ˢ Finset.univ,
               pdivMat F A i j pq.1 pq.2 *
                 (pdivMat G (F A) pq.1 pq.2 kl.1 kl.2 * dY kl.1 kl.2) := by
             simp_rw [Finset.sum_product]
         _ = ∑ kl ∈ Finset.univ ×ˢ Finset.univ,
             ∑ pq ∈ Finset.univ ×ˢ Finset.univ,
               pdivMat F A i j pq.1 pq.2 *
                 (pdivMat G (F A) pq.1 pq.2 kl.1 kl.2 * dY kl.1 kl.2) :=
             Finset.sum_comm
         _ = _ := by simp_rw [Finset.sum_product]

/-- **Additive fan-in for matrices** — proved, no sorry. -/
@[reducible] noncomputable def biPathMat {a b c d : Nat}
    (F G : Mat a b → Mat c d) : Mat a b → Mat c d :=
  fun M r s => F M r s + G M r s

noncomputable def biPathMat_has_vjp {a b c d : Nat}
    (F G : Mat a b → Mat c d)
    (hF_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))))
    (hG_diff : Differentiable ℝ
      (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))))
    (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (biPathMat F G) where
  backward := fun A dY i j => hF.backward A dY i j + hG.backward A dY i j
  correct := by
    intro A dY i j
    rw [hF.correct, hG.correct, ← Finset.sum_add_distrib]
    congr 1; ext k
    rw [← Finset.sum_add_distrib]
    congr 1; ext l
    rw [pdivMat_add _ _ _ (hF_diff (Mat.flatten A)) (hG_diff (Mat.flatten A))]; ring

/-- **Identity VJP for matrices** — proved, no sorry. -/
noncomputable def identityMat_has_vjp (a b : Nat) :
    HasVJPMat (fun (M : Mat a b) => M) where
  backward := fun _A dY => dY
  correct := by
    intro A dY i j
    -- ∑ k ∑ l, (if i=k ∧ j=l then 1 else 0) * dY k l = dY i j
    simp_rw [pdivMat_id]
    -- Collapse the two-dimensional Kronecker sum to dY i j.
    have : ∀ (k : Fin a) (l : Fin b),
        (if i = k ∧ j = l then (1 : ℝ) else 0) * dY k l =
        (if i = k then (if j = l then dY k l else 0) else 0) := by
      intro k l
      by_cases hik : i = k <;> by_cases hjl : j = l <;> simp [hik, hjl]
    simp_rw [this]
    rw [Finset.sum_eq_single i (by intro k _ hne; simp [Ne.symm hne]) (by simp)]
    simp only [if_true]
    rw [Finset.sum_eq_single j (by intro l _ hne; simp [Ne.symm hne]) (by simp)]
    simp

/-- **Bridge: `HasVJPMat` → `HasVJP` via the `Mat.flatten` bijection.**

    Given a matrix-level VJP for `f : Mat a b → Mat c d`, produce a
    vector-level VJP for the flattened version
    `fun v : Vec (a*b) => Mat.flatten (f (Mat.unflatten v))`. The backward
    reshapes the input/output flat vectors to matrices, applies the
    matrix backward, and flattens the result.

    Lets us compose `HasVJPMat` pieces (vit_body, transformer blocks)
    with rank-crossing pieces (patch embed, classifier head) that live
    natively as `Vec → Vec` by first bridging everything to `HasVJP`. -/
noncomputable def hasVJPMat_to_hasVJP {a b c d : Nat} {f : Mat a b → Mat c d}
    (hf : HasVJPMat f) :
    HasVJP (fun v : Vec (a * b) =>
              Mat.flatten (f (Mat.unflatten v))) where
  backward := fun v dy => fun idx =>
    let ij := finProdFinEquiv.symm idx
    hf.backward (Mat.unflatten v) (Mat.unflatten dy) ij.1 ij.2
  correct := by
    intro v dy idx
    set ij := finProdFinEquiv.symm idx with hij
    show hf.backward (Mat.unflatten v) (Mat.unflatten dy) ij.1 ij.2 = _
    rw [hf.correct]
    unfold pdivMat
    simp only [Mat.flatten_unflatten]
    have hidx : finProdFinEquiv (ij.1, ij.2) = idx := by
      show finProdFinEquiv ij = idx
      rw [hij]; exact Equiv.apply_symm_apply _ _
    simp_rw [hidx]
    -- Goal: ∑ k ∑ l, pdiv F v idx (fPF (k,l)) * Mat.unflatten dy k l = ∑ j', pdiv F v idx j' * dy j'
    -- Step-by-step conversion using `calc`:
    -- Σ k Σ l, ... = Σ p : Fin c × Fin d, ... = Σ j' : Fin (c*d), ...
    set F : Vec (a * b) → Vec (c * d) :=
      fun w => Mat.flatten (f (Mat.unflatten w)) with hF
    calc (∑ k : Fin c, ∑ l : Fin d,
              pdiv F v idx (finProdFinEquiv (k, l)) *
              Mat.unflatten dy k l)
        = ∑ p : Fin c × Fin d,
              pdiv F v idx (finProdFinEquiv p) *
              Mat.unflatten dy p.1 p.2 := by
          rw [Fintype.sum_prod_type]
      _ = ∑ p : Fin c × Fin d,
              pdiv F v idx (finProdFinEquiv p) *
              dy (finProdFinEquiv p) := by
          apply Finset.sum_congr rfl
          intro p _; rfl
      _ = ∑ j' : Fin (c * d), pdiv F v idx j' * dy j' := by
          exact Fintype.sum_equiv finProdFinEquiv
            (fun p : Fin c × Fin d =>
              pdiv F v idx (finProdFinEquiv p) * dy (finProdFinEquiv p))
            (fun j' : Fin (c * d) => pdiv F v idx j' * dy j')
            (fun _ => rfl)

-- ════════════════════════════════════════════════════════════════
-- § Matrix VJP Building Blocks (matmul, row-independent functions)
-- ════════════════════════════════════════════════════════════════

/-! The three axioms here are local Jacobians for the operations that
appear in scaled dot-product attention's backward pass:

1. **`pdivMat_matmul_left_const`** — right-factor varies, left factor fixed:
   `∂(C · B')_{kl} / ∂B'_{ij} = C_{ki} · [l = j]`.
2. **`pdivMat_matmul_right_const`** — left factor varies, right factor fixed:
   `∂(A' · D)_{kl} / ∂A'_{ij} = D_{jl} · [i = k]`.
3. **`pdivMat_rowIndep`** — functions that act row-wise have block-diagonal
   Jacobians, with the per-row block equal to the vector Jacobian of the
   row function `g`.

Each is a direct transcription of an elementary calculus fact. They are
numerically gradient-checked in `check_jacobians.py`. -/

/-- **Matmul Jacobian (left-const)** — theorem, derived from
    `pdiv_finset_sum` + `pdiv_mul` + `pdiv_const` + `pdiv_reindex`. -/
theorem pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 := by
  unfold pdivMat
  -- Step 1: flatten(Mat.mul C (unflatten v)) at idx = Σ_s C_{k'(idx), s} · v(fPF(s, l'(idx)))
  have h_reduces :
      (fun v : Vec (p * q) =>
        Mat.flatten ((fun B' : Mat p q => Mat.mul C B') (Mat.unflatten v))) =
      (fun v : Vec (p * q) => fun idx : Fin (m * q) =>
        ∑ s : Fin p,
          C (finProdFinEquiv.symm idx).1 s *
          v (finProdFinEquiv (s, (finProdFinEquiv.symm idx).2))) := by
    funext v idx
    show Mat.mul C (Mat.unflatten v)
           (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = _
    unfold Mat.mul Mat.unflatten
    rfl
  rw [h_reduces]
  -- Step 2: linearity distributes pdiv over the Σ_s.
  -- Each summand is `(const_in_v) * (reindex of v)` — Differentiable.
  have h_summand_diff : ∀ s ∈ (Finset.univ : Finset (Fin p)),
      DifferentiableAt ℝ
        (fun (v : Vec (p * q)) (idx : Fin (m * q)) =>
          C (finProdFinEquiv.symm idx).1 s *
          v (finProdFinEquiv (s, (finProdFinEquiv.symm idx).2)))
        (Mat.flatten B) := by
    intro s _
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec (p * q)) (idx : Fin (m * q)) =>
          C (finProdFinEquiv.symm idx).1 s) (Mat.flatten B) :=
      differentiableAt_const _
    have h_reindex : DifferentiableAt ℝ
        (fun (w : Vec (p * q)) (idx : Fin (m * q)) =>
          w (finProdFinEquiv (s, (finProdFinEquiv.symm idx).2))) (Mat.flatten B) :=
      (reindexCLM (fun idx : Fin (m * q) =>
        finProdFinEquiv (s, (finProdFinEquiv.symm idx).2))).differentiableAt
    exact h_const.mul h_reindex
  rw [pdiv_finset_sum _ _ _ h_summand_diff]
  -- Step 3: each summand is a product (const · reindex); pdiv_mul + pdiv_const + pdiv_reindex.
  have hterm : ∀ s : Fin p,
      pdiv (fun v : Vec (p * q) => fun idx : Fin (m * q) =>
              C (finProdFinEquiv.symm idx).1 s *
              v (finProdFinEquiv (s, (finProdFinEquiv.symm idx).2)))
           (Mat.flatten B) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l)) =
      C k s * (if finProdFinEquiv (i, j) = finProdFinEquiv (s, l) then 1 else 0) := by
    intro s
    -- Factor as (const fn) · (reindex fn):
    have h_prod :
        (fun v : Vec (p * q) => fun idx : Fin (m * q) =>
          C (finProdFinEquiv.symm idx).1 s *
          v (finProdFinEquiv (s, (finProdFinEquiv.symm idx).2))) =
        (fun v idx =>
          (fun (_ : Vec (p * q)) (idx' : Fin (m * q)) =>
            C (finProdFinEquiv.symm idx').1 s) v idx *
          (fun (w : Vec (p * q)) (idx' : Fin (m * q)) =>
            w (finProdFinEquiv (s, (finProdFinEquiv.symm idx').2))) v idx) := rfl
    have h_const_diff : DifferentiableAt ℝ
        (fun (_ : Vec (p * q)) (idx' : Fin (m * q)) =>
          C (finProdFinEquiv.symm idx').1 s) (Mat.flatten B) :=
      differentiableAt_const _
    have h_reindex_diff : DifferentiableAt ℝ
        (fun (w : Vec (p * q)) (idx' : Fin (m * q)) =>
          w (finProdFinEquiv (s, (finProdFinEquiv.symm idx').2))) (Mat.flatten B) :=
      (reindexCLM (fun idx' : Fin (m * q) =>
        finProdFinEquiv (s, (finProdFinEquiv.symm idx').2))).differentiableAt
    rw [h_prod, pdiv_mul _ _ _ h_const_diff h_reindex_diff]
    rw [show pdiv (fun _ : Vec (p * q) => fun idx' : Fin (m * q) =>
              C (finProdFinEquiv.symm idx').1 s)
            (Mat.flatten B) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l)) = 0
        from pdiv_const _ _ _ _]
    rw [pdiv_reindex (fun idx' => finProdFinEquiv (s, (finProdFinEquiv.symm idx').2))]
    -- (fPF.symm (fPF (k, l))).2 = l and (fPF.symm (fPF (k, l))).1 = k
    simp only [Equiv.symm_apply_apply]
    ring
  simp_rw [hterm]
  -- Step 4: collapse the Finset sum.
  -- Only s = i contributes (when j = l); otherwise all terms are zero.
  have hkey : ∀ s : Fin p,
      C k s * (if finProdFinEquiv (i, j) = finProdFinEquiv (s, l) then (1:ℝ) else 0) =
      if s = i ∧ l = j then C k s else 0 := by
    intro s
    by_cases hs : s = i ∧ l = j
    · obtain ⟨hsi, hlj⟩ := hs
      subst hsi; subst hlj; simp
    · have hne : finProdFinEquiv (i, j) ≠ finProdFinEquiv (s, l) := by
        intro heq
        apply hs
        have := finProdFinEquiv.injective heq
        exact ⟨(Prod.mk.inj this).1.symm, (Prod.mk.inj this).2.symm⟩
      rw [if_neg hne]; simp [hs]
  simp_rw [hkey]
  -- Goal: ∑ s, (if s = i ∧ l = j then C k s else 0) = if l = j then C k i else 0
  by_cases hlj : l = j
  · rw [if_pos hlj]
    -- Each `s = i ∧ l = j` term reduces to `s = i` (given hlj).
    simp_rw [show ∀ s : Fin p, (s = i ∧ l = j) ↔ (s = i) from
      fun s => ⟨And.left, fun h => ⟨h, hlj⟩⟩]
    rw [Finset.sum_ite_eq' Finset.univ i (fun s => C k s)]
    simp
  · rw [if_neg hlj]
    -- All terms false; sum is 0.
    simp_rw [show ∀ s : Fin p, (s = i ∧ l = j) ↔ False from
      fun s => ⟨fun h => hlj h.2, False.elim⟩]
    simp

/-- **Matmul Jacobian (right-const)** — theorem, same recipe as the
    left-const case with roles swapped. -/
theorem pdivMat_matmul_right_const {m p q : Nat} (A : Mat m p) (D : Mat p q)
    (i : Fin m) (j : Fin p) (k : Fin m) (l : Fin q) :
    pdivMat (fun A' : Mat m p => Mat.mul A' D) A i j k l =
    if i = k then D j l else 0 := by
  unfold pdivMat
  have h_reduces :
      (fun v : Vec (m * p) =>
        Mat.flatten ((fun A' : Mat m p => Mat.mul A' D) (Mat.unflatten v))) =
      (fun v : Vec (m * p) => fun idx : Fin (m * q) =>
        ∑ s : Fin p,
          v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, s)) *
          D s (finProdFinEquiv.symm idx).2) := by
    funext v idx
    show Mat.mul (Mat.unflatten v) D
           (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = _
    unfold Mat.mul Mat.unflatten
    rfl
  rw [h_reduces]
  have h_summand_diff : ∀ s ∈ (Finset.univ : Finset (Fin p)),
      DifferentiableAt ℝ
        (fun (v : Vec (m * p)) (idx : Fin (m * q)) =>
          v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, s)) *
          D s (finProdFinEquiv.symm idx).2)
        (Mat.flatten A) := by
    intro s _
    have h_reindex : DifferentiableAt ℝ
        (fun (w : Vec (m * p)) (idx : Fin (m * q)) =>
          w (finProdFinEquiv ((finProdFinEquiv.symm idx).1, s))) (Mat.flatten A) :=
      (reindexCLM (fun idx : Fin (m * q) =>
        finProdFinEquiv ((finProdFinEquiv.symm idx).1, s))).differentiableAt
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec (m * p)) (idx : Fin (m * q)) =>
          D s (finProdFinEquiv.symm idx).2) (Mat.flatten A) :=
      differentiableAt_const _
    exact h_reindex.mul h_const
  rw [pdiv_finset_sum _ _ _ h_summand_diff]
  have hterm : ∀ s : Fin p,
      pdiv (fun v : Vec (m * p) => fun idx : Fin (m * q) =>
              v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, s)) *
              D s (finProdFinEquiv.symm idx).2)
           (Mat.flatten A) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l)) =
      D s l * (if finProdFinEquiv (i, j) = finProdFinEquiv (k, s) then 1 else 0) := by
    intro s
    have h_prod :
        (fun v : Vec (m * p) => fun idx : Fin (m * q) =>
          v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, s)) *
          D s (finProdFinEquiv.symm idx).2) =
        (fun v idx =>
          (fun (w : Vec (m * p)) (idx' : Fin (m * q)) =>
            w (finProdFinEquiv ((finProdFinEquiv.symm idx').1, s))) v idx *
          (fun (_ : Vec (m * p)) (idx' : Fin (m * q)) =>
            D s (finProdFinEquiv.symm idx').2) v idx) := rfl
    have h_reindex_diff : DifferentiableAt ℝ
        (fun (w : Vec (m * p)) (idx' : Fin (m * q)) =>
          w (finProdFinEquiv ((finProdFinEquiv.symm idx').1, s))) (Mat.flatten A) :=
      (reindexCLM (fun idx' : Fin (m * q) =>
        finProdFinEquiv ((finProdFinEquiv.symm idx').1, s))).differentiableAt
    have h_const_diff : DifferentiableAt ℝ
        (fun (_ : Vec (m * p)) (idx' : Fin (m * q)) =>
          D s (finProdFinEquiv.symm idx').2) (Mat.flatten A) :=
      differentiableAt_const _
    rw [h_prod, pdiv_mul _ _ _ h_reindex_diff h_const_diff]
    rw [pdiv_reindex (fun idx' => finProdFinEquiv ((finProdFinEquiv.symm idx').1, s))]
    rw [show pdiv (fun _ : Vec (m * p) => fun idx' : Fin (m * q) =>
              D s (finProdFinEquiv.symm idx').2)
            (Mat.flatten A) (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l)) = 0
        from pdiv_const _ _ _ _]
    simp only [Equiv.symm_apply_apply]
    ring
  simp_rw [hterm]
  have hkey : ∀ s : Fin p,
      D s l * (if finProdFinEquiv (i, j) = finProdFinEquiv (k, s) then (1:ℝ) else 0) =
      if s = j ∧ i = k then D s l else 0 := by
    intro s
    by_cases hs : s = j ∧ i = k
    · obtain ⟨hsj, hik⟩ := hs
      subst hsj; subst hik; simp
    · have hne : finProdFinEquiv (i, j) ≠ finProdFinEquiv (k, s) := by
        intro heq
        apply hs
        have := finProdFinEquiv.injective heq
        exact ⟨(Prod.mk.inj this).2.symm, (Prod.mk.inj this).1⟩
      rw [if_neg hne]; simp [hs]
  simp_rw [hkey]
  by_cases hik : i = k
  · rw [if_pos hik]
    simp_rw [show ∀ s : Fin p, (s = j ∧ i = k) ↔ (s = j) from
      fun s => ⟨And.left, fun h => ⟨h, hik⟩⟩]
    rw [Finset.sum_ite_eq' Finset.univ j (fun s => D s l)]
    simp
  · rw [if_neg hik]
    simp_rw [show ∀ s : Fin p, (s = j ∧ i = k) ↔ False from
      fun s => ⟨fun h => hik h.2, False.elim⟩]
    simp

/-- **Row-wise Jacobian decomposition** — proved (VJP.md follow-up D).

    For a row-independent function `M ↦ (r ↦ g (M r))`, the (i,j,k,l)
    Jacobian entry is `pdiv g (A i) j l` when `i = k` and `0` otherwise.

    Requires `Differentiable ℝ g`: without it, the flattened Pi-valued
    function may be non-differentiable at `Mat.flatten A` (per
    `differentiable_pi`'s coordinate-wise condition), making `fderiv = 0`
    junk and breaking the per-row decomposition. -/
theorem pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by
  unfold pdivMat pdiv
  set F : Vec (m * n) → Vec (m * p) :=
    fun v => Mat.flatten ((fun M : Mat m n => fun r => g (M r)) (Mat.unflatten v))
    with hF
  set rowProj : Fin m → (Vec (m * n) →L[ℝ] Vec n) := fun k' =>
    reindexCLM (fun j' : Fin n => finProdFinEquiv (k', j'))
  -- Coord decomposition: F's (k', l') coord equals (g · l') ∘ rowProj k'.
  have h_coord : ∀ (k' : Fin m) (l' : Fin p),
      (fun v : Vec (m * n) => F v (finProdFinEquiv (k', l'))) =
      (fun w : Vec n => g w l') ∘ (rowProj k') := by
    intro k' l'
    funext v
    show Mat.flatten ((fun M : Mat m n => fun r => g (M r)) (Mat.unflatten v))
        (finProdFinEquiv (k', l')) = g ((rowProj k') v) l'
    unfold Mat.flatten
    simp only [Equiv.symm_apply_apply]
    show g (Mat.unflatten v k') l' = g ((rowProj k') v) l'
    rfl
  have h_g_l : ∀ (l' : Fin p) (w : Vec n),
      DifferentiableAt ℝ (fun w => g w l') w :=
    fun l' w => differentiableAt_pi.mp (h_g_diff w) l'
  have h_coord_diff : ∀ (k' : Fin m) (l' : Fin p) (v : Vec (m * n)),
      DifferentiableAt ℝ (fun v' : Vec (m * n) => F v' (finProdFinEquiv (k', l'))) v := by
    intro k' l' v
    rw [h_coord k' l']
    exact (h_g_l l' _).comp v (rowProj k').differentiableAt
  have h_F_diff : DifferentiableAt ℝ F (Mat.flatten A) := by
    rw [(differentiableAt_pi : DifferentiableAt ℝ F (Mat.flatten A) ↔ _)]
    intro idx
    have h_idx : finProdFinEquiv (finProdFinEquiv.symm idx) = idx :=
      Equiv.apply_symm_apply _ _
    have h_idx' : idx = finProdFinEquiv
        ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2) := by
      conv_lhs => rw [← h_idx]
    rw [h_idx']
    exact h_coord_diff _ _ (Mat.flatten A)
  -- Convert coord (fPF (k,l)) of fderiv F to fderiv of the (k,l)-coord function.
  have h_swap :
      fderiv ℝ F (Mat.flatten A) (basisVec (finProdFinEquiv (i, j))) (finProdFinEquiv (k, l)) =
      fderiv ℝ (fun v : Vec (m * n) => F v (finProdFinEquiv (k, l))) (Mat.flatten A)
        (basisVec (finProdFinEquiv (i, j))) := by
    rw [fderiv_apply h_F_diff (finProdFinEquiv (k, l))]
    rfl
  rw [h_swap]
  rw [h_coord k l]
  rw [fderiv_comp _ (h_g_l l _) (rowProj k).differentiableAt]
  rw [(rowProj k).fderiv]
  have h_row_A : (rowProj k) (Mat.flatten A) = A k := by
    funext j'
    show Mat.flatten A (finProdFinEquiv (k, j')) = A k j'
    show A (finProdFinEquiv.symm (finProdFinEquiv (k, j'))).1
            (finProdFinEquiv.symm (finProdFinEquiv (k, j'))).2 = A k j'
    simp
  rw [h_row_A]
  rw [fderiv_apply (h_g_diff _) l]
  -- Evaluate the comp chain so the inner rowProj is exposed.
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply]
  by_cases hik : i = k
  · subst hik
    rw [if_pos rfl]
    have h_basis : (rowProj i) (basisVec (finProdFinEquiv (i, j))) = basisVec j := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (i, j')) = basisVec j j'
      simp only [basisVec_apply]
      by_cases hjj : j' = j
      · subst hjj; simp
      · rw [if_neg hjj, if_neg ?_]
        intro heq
        apply hjj
        exact (Prod.mk.inj (finProdFinEquiv.injective heq.symm)).2.symm
    rw [h_basis]
  · rw [if_neg hik]
    have h_basis : (rowProj k) (basisVec (finProdFinEquiv (i, j))) = (0 : Vec n) := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (k, j')) = (0 : ℝ)
      simp only [basisVec_apply]
      rw [if_neg]
      intro heq
      apply hik
      exact (Prod.mk.inj (finProdFinEquiv.injective heq)).1.symm
    rw [h_basis]
    simp

/-- **Row-wise lifting of a `HasVJP`** (Phase 8, Tensor-level).

    Given any `g : Vec n → Vec p` with a proved `HasVJP`, applying `g`
    independently to each row of a matrix `A : Mat m n` gives a
    `HasVJPMat` on `Mat m n → Mat m p`. The backward is just `g.backward`
    applied per row. Generalizes `rowSoftmax_has_vjp_mat`: any per-token
    operation (LayerNorm, GELU, dense, activation) lifts to a per-sequence
    matrix operation via this one helper. -/
noncomputable def rowwise_has_vjp_mat {m n p : Nat} {g : Vec n → Vec p}
    (hg : HasVJP g) (hg_diff : Differentiable ℝ g) :
    HasVJPMat (fun A : Mat m n => fun r => g (A r)) where
  backward := fun A dY => fun r c => hg.backward (A r) (dY r) c
  correct := by
    intro A dY i j
    -- Replace pdivMat of the row-independent fn with its row/vector form.
    simp_rw [pdivMat_rowIndep g hg_diff]
    -- Push the *dY through the if-else, then pull the if-else out of the inner sum.
    have h : ∀ k : Fin m,
        (∑ l : Fin p, (if i = k then pdiv g (A i) j l else 0) * dY k l) =
        if i = k then ∑ l : Fin p, pdiv g (A i) j l * dY k l else 0 := by
      intro k
      by_cases hik : i = k
      · simp [hik]
      · simp [hik]
    simp_rw [h]
    rw [Finset.sum_ite_eq Finset.univ i
        (fun k => ∑ l : Fin p, pdiv g (A i) j l * dY k l)]
    simp only [Finset.mem_univ, if_true]
    exact hg.correct (A i) (dY i) j

-- ════════════════════════════════════════════════════════════════
-- § Column-slab independence (vmap over a column-axis partition)
-- ════════════════════════════════════════════════════════════════

/-! ## Per-head / per-slab column independence

Multi-head attention applies the same per-head function to each of `heads`
column slabs of width `d_in` from a `Mat n (heads * d_in)` input. The
column-slab analog of `rowwise_has_vjp_mat` factors that vmap-over-heads
structure: each head's output depends only on its own slab of the input,
so the matrix Jacobian is block-diagonal across the head axis. -/

/-- Apply `g : Mat n d_in → Mat n d_out` to each of the `heads` column
    slabs of width `d_in` in a `Mat n (heads * d_in)` input, producing
    a `Mat n (heads * d_out)` output. Output column `(h, j_out)` is
    column `j_out` of `g (slab h M)`, where `slab h M` extracts the
    `d_in`-wide column block at head index `h`. -/
noncomputable def colSlabApply {n heads d_in d_out : Nat}
    (g : Mat n d_in → Mat n d_out) : Mat n (heads * d_in) → Mat n (heads * d_out) :=
  fun M => fun r hj =>
    g (fun r' j_in => M r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_in)))
      r (finProdFinEquiv.symm hj).2

/-- **Column-slab independence Jacobian** — column-axis analog of
    `pdivMat_rowIndep`. For a slab-applied function `colSlabApply g`,
    the Jacobian is block-diagonal across the `heads` axis: zero unless
    the input slab `h_j` matches the output slab `h_l`, otherwise equal
    to `pdivMat g` on that slab.

    Requires `Differentiable ℝ (flat g)` for the same reason as
    `pdivMat_rowIndep`: the Pi-valued flat form must be differentiable
    everywhere so `fderiv` doesn't fall back to junk-default 0. -/
theorem pdivMat_colIndep {n heads d_in d_out : Nat} (g : Mat n d_in → Mat n d_out)
    (h_g_diff : Differentiable ℝ
                  (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v))))
    (A : Mat n (heads * d_in))
    (i : Fin n) (h_j : Fin heads) (j' : Fin d_in)
    (k : Fin n) (h_l : Fin heads) (j'' : Fin d_out) :
    pdivMat (colSlabApply g) A
            i (finProdFinEquiv (h_j, j'))
            k (finProdFinEquiv (h_l, j'')) =
    (if h_j = h_l then
      pdivMat g (fun r' j_in => A r' (finProdFinEquiv (h_l, j_in))) i j' k j''
     else 0) := by
  unfold pdivMat pdiv
  -- Slab-projection CLM: extracts head h's d_in columns as a flattened Vec.
  set slabProj : Fin heads → (Vec (n * (heads * d_in)) →L[ℝ] Vec (n * d_in)) := fun h' =>
    reindexCLM (fun idx : Fin (n * d_in) =>
      finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                       finProdFinEquiv (h', (finProdFinEquiv.symm idx).2)))
    with hSlabProj_def
  -- Coord `(k', encode(h_l', j_out))` of the flat function factors as
  -- `(g-coord-fn) ∘ slabProj h_l'`.
  have h_coord : ∀ (k' : Fin n) (h_l' : Fin heads) (j_out : Fin d_out),
      (fun v : Vec (n * (heads * d_in)) =>
         Mat.flatten (colSlabApply g (Mat.unflatten v))
           (finProdFinEquiv (k', finProdFinEquiv (h_l', j_out)))) =
      (fun w : Vec (n * d_in) =>
         Mat.flatten (g (Mat.unflatten w)) (finProdFinEquiv (k', j_out))) ∘ (slabProj h_l') := by
    intro k' h_l' j_out
    funext v
    show Mat.flatten (colSlabApply g (Mat.unflatten v))
            (finProdFinEquiv (k', finProdFinEquiv (h_l', j_out))) =
         Mat.flatten (g (Mat.unflatten ((slabProj h_l') v))) (finProdFinEquiv (k', j_out))
    unfold Mat.flatten colSlabApply
    simp only [Equiv.symm_apply_apply]
    show g (fun r' j_in => Mat.unflatten v r' (finProdFinEquiv (h_l', j_in))) k' j_out =
         g (Mat.unflatten ((slabProj h_l') v)) k' j_out
    congr 1
    funext r' j_in
    show Mat.unflatten v r' (finProdFinEquiv (h_l', j_in)) =
         Mat.unflatten ((slabProj h_l') v) r' j_in
    unfold Mat.unflatten
    show v (finProdFinEquiv (r', finProdFinEquiv (h_l', j_in))) =
         (slabProj h_l') v (finProdFinEquiv (r', j_in))
    show _ = v (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv (r', j_in))).1,
               finProdFinEquiv (h_l', (finProdFinEquiv.symm (finProdFinEquiv (r', j_in))).2)))
    rw [Equiv.symm_apply_apply]
  -- Differentiability of each scalar coord function.
  have h_g_coord_diff : ∀ (k' : Fin n) (j_out : Fin d_out) (w : Vec (n * d_in)),
      DifferentiableAt ℝ
        (fun w : Vec (n * d_in) =>
          Mat.flatten (g (Mat.unflatten w)) (finProdFinEquiv (k', j_out))) w :=
    fun k' j_out w => differentiableAt_pi.mp (h_g_diff w) _
  have h_F_coord_diff : ∀ (k' : Fin n) (h_l' : Fin heads) (j_out : Fin d_out)
      (v : Vec (n * (heads * d_in))),
      DifferentiableAt ℝ
        (fun v' : Vec (n * (heads * d_in)) =>
          Mat.flatten (colSlabApply g (Mat.unflatten v'))
            (finProdFinEquiv (k', finProdFinEquiv (h_l', j_out)))) v := by
    intro k' h_l' j_out v
    rw [h_coord k' h_l' j_out]
    exact (h_g_coord_diff k' j_out _).comp v (slabProj h_l').differentiableAt
  -- Full flat function is differentiable: every Pi-coord is.
  have h_F_diff : DifferentiableAt ℝ
      (fun v : Vec (n * (heads * d_in)) =>
        Mat.flatten (colSlabApply g (Mat.unflatten v))) (Mat.flatten A) := by
    rw [(differentiableAt_pi : DifferentiableAt ℝ _ _ ↔ _)]
    intro idx
    have h_eq1 : idx = finProdFinEquiv (finProdFinEquiv.symm idx) :=
      (Equiv.apply_symm_apply _ _).symm
    set p := finProdFinEquiv.symm idx with hp
    have h_eq2 : p.2 = finProdFinEquiv (finProdFinEquiv.symm p.2) :=
      (Equiv.apply_symm_apply _ _).symm
    rw [h_eq1]
    show DifferentiableAt ℝ
      (fun v' => Mat.flatten (colSlabApply g (Mat.unflatten v'))
        (finProdFinEquiv (p.1, p.2))) (Mat.flatten A)
    rw [h_eq2]
    exact h_F_coord_diff _ _ _ _
  -- fderiv of the (k, encode(h_l, j'')) coord = fderiv of g-coord ∘ slabProj h_l.
  rw [show fderiv ℝ (fun v : Vec (n * (heads * d_in)) =>
              Mat.flatten (colSlabApply g (Mat.unflatten v))) (Mat.flatten A)
            (basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j'))))
            (finProdFinEquiv (k, finProdFinEquiv (h_l, j''))) =
          fderiv ℝ (fun v : Vec (n * (heads * d_in)) =>
              Mat.flatten (colSlabApply g (Mat.unflatten v))
                (finProdFinEquiv (k, finProdFinEquiv (h_l, j'')))) (Mat.flatten A)
            (basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))) from by
    rw [fderiv_apply h_F_diff (finProdFinEquiv (k, finProdFinEquiv (h_l, j'')))]
    rfl]
  rw [h_coord k h_l j'']
  rw [fderiv_comp _ (h_g_coord_diff k j'' _) (slabProj h_l).differentiableAt]
  rw [(slabProj h_l).fderiv]
  -- slabProj h_l (Mat.flatten A) = Mat.flatten (slab-fn at h_l of A).
  have h_slab_A : (slabProj h_l) (Mat.flatten A) =
      Mat.flatten (fun r' j_in => A r' (finProdFinEquiv (h_l, j_in))) := by
    funext idx
    show Mat.flatten A
          (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                           finProdFinEquiv (h_l, (finProdFinEquiv.symm idx).2))) = _
    unfold Mat.flatten
    simp only [Equiv.symm_apply_apply]
  rw [h_slab_A]
  rw [fderiv_apply (h_g_diff _) (finProdFinEquiv (k, j''))]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply]
  -- Compute (slabProj h_l) (basisVec (encode(i, encode(h_j, j')))) — one of two cases.
  by_cases hhh : h_j = h_l
  · subst hhh
    rw [if_pos rfl]
    have h_basis : (slabProj h_j) (basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))) =
                   (basisVec (finProdFinEquiv (i, j')) : Vec (n * d_in)) := by
      funext idx
      show basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))
            (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                             finProdFinEquiv (h_j, (finProdFinEquiv.symm idx).2))) =
           basisVec (finProdFinEquiv (i, j')) idx
      simp only [basisVec_apply]
      by_cases hii : idx = finProdFinEquiv (i, j')
      · subst hii
        simp [Equiv.symm_apply_apply]
      · rw [if_neg hii, if_neg]
        intro heq
        apply hii
        have step1 := finProdFinEquiv.injective heq
        have h_inner := finProdFinEquiv.injective (Prod.mk.inj step1).2
        rw [show idx = finProdFinEquiv (finProdFinEquiv.symm idx) from
              (Equiv.apply_symm_apply _ _).symm]
        exact congrArg finProdFinEquiv (Prod.ext (Prod.mk.inj step1).1 (Prod.mk.inj h_inner).2)
    rw [h_basis]
  · rw [if_neg hhh]
    have h_basis : (slabProj h_l) (basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))) =
                   (0 : Vec (n * d_in)) := by
      funext idx
      show basisVec (finProdFinEquiv (i, finProdFinEquiv (h_j, j')))
            (finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                             finProdFinEquiv (h_l, (finProdFinEquiv.symm idx).2))) =
           (0 : ℝ)
      simp only [basisVec_apply]
      rw [if_neg]
      intro heq
      apply hhh
      have step1 := finProdFinEquiv.injective heq
      have h_inner := finProdFinEquiv.injective (Prod.mk.inj step1).2
      exact ((Prod.mk.inj h_inner).1).symm
    rw [h_basis]
    simp

/-- **Lift `HasVJPMat g` to column-slab vmap** — column-axis analog of
    `rowwise_has_vjp_mat`. Given `g : Mat n d_in → Mat n d_out` with a
    matrix VJP, applying `g` independently to each of `heads`-many column
    slabs gives a `HasVJPMat` for `colSlabApply g`. The backward applies
    `g.backward` per slab. -/
noncomputable def colSlabwise_has_vjp_mat {n heads d_in d_out : Nat}
    {g : Mat n d_in → Mat n d_out}
    (hg : HasVJPMat g)
    (hg_diff : Differentiable ℝ
                 (fun v : Vec (n * d_in) => Mat.flatten (g (Mat.unflatten v)))) :
    HasVJPMat (colSlabApply g (heads := heads)) where
  backward := fun M dY r hj =>
    hg.backward (fun r' j_in => M r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_in)))
                (fun r' j_out => dY r' (finProdFinEquiv ((finProdFinEquiv.symm hj).1, j_out)))
                r (finProdFinEquiv.symm hj).2
  correct := by
    intro M dY i jj
    -- jj : Fin (heads * d_in). Decompose into (h_j, j').
    set p_jj := finProdFinEquiv.symm jj with hp_jj_def
    have hjj_eq : jj = finProdFinEquiv (p_jj.1, p_jj.2) := (Equiv.apply_symm_apply _ _).symm
    -- Reshape the LHS structure field via direct computation: p_jj.1, p_jj.2.
    show hg.backward
            (fun r' j_in => M r' (finProdFinEquiv (p_jj.1, j_in)))
            (fun r' j_out => dY r' (finProdFinEquiv (p_jj.1, j_out)))
            i p_jj.2 =
         ∑ k : Fin n, ∑ l : Fin (heads * d_out),
            pdivMat (colSlabApply g) M i jj k l * dY k l
    -- Replace jj on the RHS with finProdFinEquiv (p_jj.1, p_jj.2).
    rw [hjj_eq]
    -- Reindex the sum over l = encode(h_l, j'') via Fintype.sum_equiv + sum_prod_type.
    have h_reindex : ∀ k : Fin n,
        (∑ l : Fin (heads * d_out),
            pdivMat (colSlabApply g) M i (finProdFinEquiv (p_jj.1, p_jj.2)) k l * dY k l) =
        ∑ h_l : Fin heads, ∑ j'' : Fin d_out,
            pdivMat (colSlabApply g) M i (finProdFinEquiv (p_jj.1, p_jj.2))
              k (finProdFinEquiv (h_l, j'')) *
            dY k (finProdFinEquiv (h_l, j'')) := by
      intro k
      rw [Fintype.sum_equiv finProdFinEquiv.symm
            (fun l : Fin (heads * d_out) =>
              pdivMat (colSlabApply g) M i (finProdFinEquiv (p_jj.1, p_jj.2)) k l * dY k l)
            (fun p : Fin heads × Fin d_out =>
              pdivMat (colSlabApply g) M i (finProdFinEquiv (p_jj.1, p_jj.2))
                k (finProdFinEquiv p) *
              dY k (finProdFinEquiv p))
            (fun l => by simp only [Equiv.apply_symm_apply])]
      rw [Fintype.sum_prod_type]
    simp_rw [h_reindex]
    -- Apply pdivMat_colIndep per term: contributes 0 unless h_l = p_jj.1.
    simp_rw [pdivMat_colIndep g hg_diff]
    -- Collapse the outer sum over h_l using the Kronecker if.
    have h_collapse : ∀ k : Fin n,
        (∑ h_l : Fin heads, ∑ j'' : Fin d_out,
          (if p_jj.1 = h_l then
            pdivMat g (fun r' j_in => M r' (finProdFinEquiv (h_l, j_in))) i p_jj.2 k j''
           else 0) * dY k (finProdFinEquiv (h_l, j''))) =
        ∑ j'' : Fin d_out,
          pdivMat g (fun r' j_in => M r' (finProdFinEquiv (p_jj.1, j_in))) i p_jj.2 k j'' *
          dY k (finProdFinEquiv (p_jj.1, j'')) := by
      intro k
      have h_inner : ∀ h_l : Fin heads,
          (∑ j'' : Fin d_out,
            (if p_jj.1 = h_l then
              pdivMat g (fun r' j_in => M r' (finProdFinEquiv (h_l, j_in))) i p_jj.2 k j''
             else 0) * dY k (finProdFinEquiv (h_l, j''))) =
          (if p_jj.1 = h_l then
            ∑ j'' : Fin d_out,
              pdivMat g (fun r' j_in => M r' (finProdFinEquiv (h_l, j_in))) i p_jj.2 k j'' *
              dY k (finProdFinEquiv (h_l, j''))
           else 0) := by
        intro h_l
        by_cases hh : p_jj.1 = h_l
        · simp [hh]
        · simp [hh]
      simp_rw [h_inner]
      rw [Finset.sum_ite_eq Finset.univ p_jj.1
          (fun h_l => ∑ j'' : Fin d_out,
            pdivMat g (fun r' j_in => M r' (finProdFinEquiv (h_l, j_in))) i p_jj.2 k j'' *
            dY k (finProdFinEquiv (h_l, j'')))]
      simp only [Finset.mem_univ, if_true]
    simp_rw [h_collapse]
    -- Now the goal is exactly hg.correct on the slab.
    exact hg.correct (fun r' j_in => M r' (finProdFinEquiv (p_jj.1, j_in)))
                     (fun r' j_out => dY r' (finProdFinEquiv (p_jj.1, j_out)))
                     i p_jj.2

-- ════════════════════════════════════════════════════════════════
-- § Ternary VJP for matrix functions (HasVJPMat3)
-- ════════════════════════════════════════════════════════════════

/-! ## Ternary matrix VJP

For ternary-input functions like SDPA `(Q, K, V) ↦ out`, package the
three per-input VJPs as a single structure analogous to `HasVJPMat`.
The backward returns the triple of per-input gradients; correctness
holds independently for each input (with the others fixed). -/

/-- VJP structure for `Mat × Mat × Mat → Mat` functions where all
    three inputs share the same shape `Mat n d_in` and the output is
    `Mat n d_out`. Backward returns the triple of per-input gradients;
    `correct_{1,2,3}` ensure each gradient matches the partial derivative
    treating the other two inputs as constants. -/
structure HasVJPMat3 {n d_in d_out : Nat}
    (F : Mat n d_in → Mat n d_in → Mat n d_in → Mat n d_out) where
  backward : Mat n d_in → Mat n d_in → Mat n d_in → Mat n d_out →
             (Mat n d_in × Mat n d_in × Mat n d_in)
  correct_1 : ∀ A B C dY i j,
    (backward A B C dY).1 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun A' => F A' B C) A i j k l * dY k l
  correct_2 : ∀ A B C dY i j,
    (backward A B C dY).2.1 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun B' => F A B' C) B i j k l * dY k l
  correct_3 : ∀ A B C dY i j,
    (backward A B C dY).2.2 i j =
    ∑ k : Fin n, ∑ l : Fin d_out,
      pdivMat (fun C' => F A B C') C i j k l * dY k l

/-- **Scalar-scale Jacobian** — theorem, derived from `pdiv_mul` +
    `pdiv_const` + `pdiv_id` via the flatten bijection.
    `∂(s · A')_{kl} / ∂A'_{ij} = s · δ_{ik,jl}`. -/
theorem pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 := by
  unfold pdivMat
  -- Step 1: the flattened scalar-scale function simplifies to `fun v k' => s * v k'`.
  -- This uses Mat.unflatten_flatten roundtrip pointwise.
  have h_reduces :
      (fun v : Vec (m * n) =>
        Mat.flatten ((fun M : Mat m n => fun r c => s * M r c) (Mat.unflatten v))) =
      (fun v : Vec (m * n) => fun k' : Fin (m * n) => s * v k') := by
    funext v k'
    show s * Mat.unflatten v (finProdFinEquiv.symm k').1 (finProdFinEquiv.symm k').2 = s * v k'
    unfold Mat.unflatten
    -- Goal: s * v (fPF ((fPF.symm k').1, (fPF.symm k').2)) = s * v k'
    rw [show ((finProdFinEquiv.symm k').1, (finProdFinEquiv.symm k').2) = finProdFinEquiv.symm k'
        from rfl]
    rw [Equiv.apply_symm_apply]
  rw [h_reduces]
  -- Step 2: rewrite as a product of (constant s) and (identity).
  have h_product :
      (fun v : Vec (m * n) => fun k' : Fin (m * n) => s * v k') =
      (fun v k' =>
        (fun (_ : Vec (m * n)) (_ : Fin (m * n)) => s) v k' *
        (fun (w : Vec (m * n)) => w) v k') := rfl
  rw [h_product]
  -- Step 3: apply pdiv_mul. Both factors are Differentiable everywhere.
  have h_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec (m * n)) (_ : Fin (m * n)) => s) (Mat.flatten A) :=
    differentiableAt_const _
  have h_id_diff : DifferentiableAt ℝ
      (fun (w : Vec (m * n)) => w) (Mat.flatten A) :=
    differentiableAt_id
  rw [pdiv_mul (fun _ _ => s) (fun w => w) _ h_const_diff h_id_diff]
  -- Step 4: pdiv_const for the constant factor, pdiv_id for identity.
  -- The constant function `fun _ _ => s` has Vec m = Vec (m*n) → Vec n = Vec (m*n) shape;
  -- we need to treat the inner constant as `fun _ => (fun _ => s)` for pdiv_const.
  have h_const :
      pdiv (fun _ : Vec (m * n) => fun _ : Fin (m * n) => s) (Mat.flatten A)
        (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l)) = 0 :=
    pdiv_const (fun _ : Fin (m * n) => s) (Mat.flatten A)
      (finProdFinEquiv (i, j)) (finProdFinEquiv (k, l))
  rw [h_const, pdiv_id]
  -- Goal after simp: collapses both sides via the bijection injectivity.
  simp only [zero_mul, zero_add, mul_ite, mul_one, mul_zero]
  -- Now: (if fPF(i,j) = fPF(k,l) then s else 0) = if i = k ∧ j = l then s else 0
  by_cases hij : i = k ∧ j = l
  · obtain ⟨hi, hj⟩ := hij; subst hi; subst hj; simp
  · have hne : finProdFinEquiv (i, j) ≠ finProdFinEquiv (k, l) := by
      intro heq
      apply hij
      have := finProdFinEquiv.injective heq
      exact ⟨(Prod.mk.inj this).1, (Prod.mk.inj this).2⟩
    rw [if_neg hij, if_neg hne]

/-- **Transpose Jacobian** — theorem, derived from `pdiv_reindex` via
    the flatten bijection.  `∂A^T_{kl} / ∂A_{ij} = δ_{l=i, k=j}`. -/
theorem pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 := by
  unfold pdivMat
  -- Step 1: flatten(transpose(unflatten v)) is a gather:
  --   at output idx, returns v at the index obtained by swapping components.
  have h_reduces :
      (fun v : Vec (m * n) =>
        Mat.flatten ((fun M : Mat m n => Mat.transpose M) (Mat.unflatten v))) =
      (fun v : Vec (m * n) => fun idx : Fin (n * m) =>
        v (finProdFinEquiv
              ((finProdFinEquiv.symm idx).2, (finProdFinEquiv.symm idx).1))) := by
    funext v idx
    show Mat.transpose (Mat.unflatten v)
           (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = _
    unfold Mat.transpose Mat.unflatten
    rfl
  rw [h_reduces, pdiv_reindex]
  -- Step 2: collapse the index condition.
  -- Goal: (if fPF(i,j) = σ(fPF(k,l)) then 1 else 0) = (if j = k ∧ i = l then 1 else 0)
  -- where σ(idx) = fPF((fPF.symm idx).2, (fPF.symm idx).1).
  -- At fPF(k,l): σ(fPF(k,l)) = fPF(l, k).
  -- So condition: fPF(i,j) = fPF(l, k) ⟺ (i, j) = (l, k) ⟺ i = l ∧ j = k.
  simp only [Equiv.symm_apply_apply]
  by_cases h : j = k ∧ i = l
  · obtain ⟨hjk, hil⟩ := h
    subst hjk; subst hil
    simp
  · have hne : finProdFinEquiv (i, j) ≠ finProdFinEquiv (l, k) := by
      intro heq
      apply h
      have := finProdFinEquiv.injective heq
      exact ⟨(Prod.mk.inj this).2, (Prod.mk.inj this).1⟩
    rw [if_neg hne, if_neg h]

/-- **Matmul with right factor varying, left factor fixed** — proved.

    `f : Mat p q → Mat m q`,  `f B' = C · B'`.
    Backward: `dB' = C^T · dY`. -/
noncomputable def matmul_left_const_has_vjp {m p q : Nat} (C : Mat m p) :
    HasVJPMat (fun B' : Mat p q => Mat.mul C B') where
  backward := fun _B dY => fun i j => ∑ k : Fin m, C k i * dY k j
  correct := by
    intro B dY i j
    simp_rw [pdivMat_matmul_left_const]
    -- Σ k Σ l, (if l = j then C k i else 0) * dY k l = Σ k, C k i * dY k j
    congr 1; ext k
    -- Inner sum over l: collapse if-else via sum_ite_eq
    have h : ∀ l : Fin q,
        (if l = j then C k i else 0) * dY k l =
        if l = j then C k i * dY k j else 0 := by
      intro l; by_cases hlj : l = j
      · simp [hlj]
      · simp [hlj]
    simp_rw [h]
    rw [Finset.sum_ite_eq' Finset.univ j (fun _ => C k i * dY k j)]
    simp

/-- **Matmul with left factor varying, right factor fixed** — proved.

    `f : Mat m p → Mat m q`,  `f A' = A' · D`.
    Backward: `dA' = dY · D^T`. -/
noncomputable def matmul_right_const_has_vjp {m p q : Nat} (D : Mat p q) :
    HasVJPMat (fun A' : Mat m p => Mat.mul A' D) where
  backward := fun _A dY => fun i j => ∑ l : Fin q, dY i l * D j l
  correct := by
    intro A dY i j
    simp_rw [pdivMat_matmul_right_const]
    -- Σ k Σ l, (if i = k then D j l else 0) * dY k l = Σ l, dY i l * D j l
    have h : ∀ k : Fin m, ∀ l : Fin q,
        (if i = k then D j l else 0) * dY k l =
        if i = k then D j l * dY i l else 0 := by
      intro k l; by_cases hik : i = k
      · simp [hik]
      · simp [hik]
    simp_rw [h]
    rw [Finset.sum_comm]
    have hinner : ∀ l : Fin q,
        ∑ k : Fin m, (if i = k then D j l * dY i l else 0) = D j l * dY i l := by
      intro l
      rw [Finset.sum_ite_eq Finset.univ i (fun _ => D j l * dY i l)]
      simp
    simp_rw [hinner]
    congr 1; ext l; ring

/-- **Scalar-scale VJP** — proved.  Backward: `dA = s · dY`. -/
noncomputable def scalarScale_has_vjp {m n : Nat} (s : ℝ) :
    HasVJPMat (fun M : Mat m n => fun r c => s * M r c) where
  backward := fun _A dY => fun i j => s * dY i j
  correct := by
    intro A dY i j
    simp_rw [pdivMat_scalarScale]
    -- Σ k Σ l, (if i=k ∧ j=l then s else 0) * dY k l = s * dY i j
    have h : ∀ k : Fin m, ∀ l : Fin n,
        (if i = k ∧ j = l then s else 0) * dY k l =
        (if i = k then (if j = l then s * dY k l else 0) else 0) := by
      intro k l
      by_cases hik : i = k <;> by_cases hjl : j = l <;> simp [hik, hjl]
    simp_rw [h]
    rw [Finset.sum_eq_single i (by intro k _ hne; simp [Ne.symm hne]) (by simp)]
    simp only [if_true]
    rw [Finset.sum_eq_single j (by intro l _ hne; simp [Ne.symm hne]) (by simp)]
    simp

/-- **Transpose VJP** — proved.  Backward: `dA = (dY)^T`. -/
noncomputable def transpose_has_vjp {m n : Nat} :
    HasVJPMat (fun M : Mat m n => Mat.transpose M) where
  backward := fun _A dY => fun i j => dY j i
  correct := by
    intro A dY i j
    simp_rw [pdivMat_transpose]
    -- Σ k : Fin n, Σ l : Fin m, (if j=k ∧ i=l then 1 else 0) * dY k l = dY j i
    have h : ∀ k : Fin n, ∀ l : Fin m,
        (if j = k ∧ i = l then (1 : ℝ) else 0) * dY k l =
        (if j = k then (if i = l then dY k l else 0) else 0) := by
      intro k l
      by_cases hjk : j = k <;> by_cases hil : i = l <;> simp [hjk, hil]
    simp_rw [h]
    rw [Finset.sum_eq_single j (by intro k _ hne; simp [Ne.symm hne]) (by simp)]
    simp only [if_true]
    rw [Finset.sum_eq_single i (by intro l _ hne; simp [Ne.symm hne]) (by simp)]
    simp

-- ════════════════════════════════════════════════════════════════
-- § 3D Tensor VJP Framework (for CNN / Depthwise)
-- ════════════════════════════════════════════════════════════════

/-- A 3D feature map: channels × height × width (single sample). -/
abbrev Tensor3 (c h w : Nat) := Fin c → Fin h → Fin w → ℝ

namespace Tensor3

/-- Row-major flatten: `Tensor3 c h w → Vec (c * h * w)`. Two nested
    `finProdFinEquiv` calls: first bundle `(ci, hi)` into `Fin (c*h)`,
    then bundle with `wi` into `Fin ((c*h)*w) = Fin (c*h*w)`. -/
noncomputable def flatten {c h w : Nat} (T : Tensor3 c h w) : Vec (c * h * w) :=
  fun k =>
    let ch_w := finProdFinEquiv.symm k      -- : Fin (c*h) × Fin w
    let c_h := finProdFinEquiv.symm ch_w.1  -- : Fin c × Fin h
    T c_h.1 c_h.2 ch_w.2

/-- Row-major unflatten: inverse of `flatten`. -/
noncomputable def unflatten {c h w : Nat} (v : Vec (c * h * w)) : Tensor3 c h w :=
  fun ci hi wi => v (finProdFinEquiv (finProdFinEquiv (ci, hi), wi))

theorem unflatten_flatten {c h w : Nat} (T : Tensor3 c h w) :
    unflatten (flatten T) = T := by
  funext ci hi wi
  unfold unflatten flatten
  simp [Equiv.symm_apply_apply]

theorem flatten_unflatten {c h w : Nat} (v : Vec (c * h * w)) :
    flatten (unflatten v) = v := by
  funext k
  change v (finProdFinEquiv
    (finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm k).1),
     (finProdFinEquiv.symm k).2)) = v k
  rw [Equiv.apply_symm_apply]
  -- Now: v (finProdFinEquiv ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2)) = v k
  rw [show ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2) = finProdFinEquiv.symm k
        from rfl]
  rw [Equiv.apply_symm_apply]

end Tensor3

/-- **3D partial derivative** — now a definition via the triple-nested
    flatten bijection, no longer an axiom. The four structural rules
    (comp / add / id) follow as theorems. Local Jacobian axioms
    (`pdiv3_conv2d_vjp`, `pdiv3_maxPool2_vjp`, `pdiv3_depthwise_vjp`)
    remain — those state specific Jacobian values, not framework. -/
noncomputable def pdiv3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (co : Fin c₂) (ho : Fin h₂) (wo : Fin w₂) : ℝ :=
  pdiv (fun v : Vec (c₁ * h₁ * w₁) =>
          Tensor3.flatten (f (Tensor3.unflatten v)))
    (Tensor3.flatten x)
    (finProdFinEquiv (finProdFinEquiv (ci, hi), wi))
    (finProdFinEquiv (finProdFinEquiv (co, ho), wo))

/-- **Chain rule for 3D partial derivatives** — theorem, via `pdiv_comp`
    and two applications of `Fintype.sum_equiv + sum_prod_type`. Requires
    the flattened forms of `f` and `g` to be differentiable at the
    relevant points. -/
theorem pdiv3_comp {c₁ h₁ w₁ c₂ h₂ w₂ c₃ h₃ w₃ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (g : Tensor3 c₂ h₂ w₂ → Tensor3 c₃ h₃ w₃)
    (x : Tensor3 c₁ h₁ w₁)
    (hf_diff : DifferentiableAt ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
      (Tensor3.flatten x))
    (hg_diff : DifferentiableAt ℝ
      (fun u : Vec (c₂ * h₂ * w₂) => Tensor3.flatten (g (Tensor3.unflatten u)))
      (Tensor3.flatten (f x)))
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (ck : Fin c₃) (hk : Fin h₃) (wk : Fin w₃) :
    pdiv3 (g ∘ f) x ci hi wi ck hk wk =
    ∑ cj : Fin c₂, ∑ hj : Fin h₂, ∑ wj : Fin w₂,
      pdiv3 f x ci hi wi cj hj wj * pdiv3 g (f x) cj hj wj ck hk wk := by
  unfold pdiv3
  -- Flatten turns 3D composition into Vec composition (unflatten ∘ flatten = id).
  have h_compose :
      (fun v : Vec (c₁ * h₁ * w₁) =>
        Tensor3.flatten ((g ∘ f) (Tensor3.unflatten v))) =
      (fun u : Vec (c₂ * h₂ * w₂) => Tensor3.flatten (g (Tensor3.unflatten u))) ∘
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v))) := by
    funext v
    simp [Function.comp, Tensor3.unflatten_flatten]
  have h_mid :
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
        (Tensor3.flatten x) = Tensor3.flatten (f x) := by
    simp [Tensor3.unflatten_flatten]
  have hg_diff' : DifferentiableAt ℝ
      (fun u : Vec (c₂ * h₂ * w₂) => Tensor3.flatten (g (Tensor3.unflatten u)))
      ((fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
        (Tensor3.flatten x)) := by
    rw [h_mid]; exact hg_diff
  rw [h_compose, pdiv_comp _ _ _ hf_diff hg_diff']
  simp_rw [h_mid]
  -- Two-stage collapse of the Fin ((c₂*h₂)*w₂) sum into ∑ cj ∑ hj ∑ wj.
  -- Abbreviate the double-indexed summand as `F r`:
  set F : Fin (c₂ * h₂ * w₂) → ℝ := fun r =>
    pdiv (fun v => Tensor3.flatten (f (Tensor3.unflatten v))) (Tensor3.flatten x)
      (finProdFinEquiv (finProdFinEquiv (ci, hi), wi)) r *
    pdiv (fun u => Tensor3.flatten (g (Tensor3.unflatten u))) (Tensor3.flatten (f x))
      r (finProdFinEquiv (finProdFinEquiv (ck, hk), wk)) with hF
  -- Stage 1: split Fin((c₂*h₂)*w₂) → Fin(c₂*h₂) × Fin w₂ via finProdFinEquiv.
  rw [Fintype.sum_equiv finProdFinEquiv.symm F
      (fun pw : Fin (c₂ * h₂) × Fin w₂ => F (finProdFinEquiv pw))
      (fun r => by
        show F r = F (finProdFinEquiv (finProdFinEquiv.symm r))
        rw [Equiv.apply_symm_apply])]
  rw [Fintype.sum_prod_type]
  -- Goal now: ∑ p : Fin(c₂*h₂), ∑ wj : Fin w₂, F (fPF (p, wj)) = ∑ cj, ∑ hj, ∑ wj, F (...)
  -- Stage 2: split outer Fin(c₂*h₂) → Fin c₂ × Fin h₂ via finProdFinEquiv.
  rw [Fintype.sum_equiv finProdFinEquiv.symm
      (fun p : Fin (c₂ * h₂) => ∑ wj : Fin w₂, F (finProdFinEquiv (p, wj)))
      (fun ch : Fin c₂ × Fin h₂ =>
        ∑ wj : Fin w₂, F (finProdFinEquiv (finProdFinEquiv ch, wj)))
      (fun p => by
        show (∑ wj : Fin w₂, F (finProdFinEquiv (p, wj))) =
             (∑ wj : Fin w₂, F (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv.symm p), wj)))
        rw [Equiv.apply_symm_apply])]
  rw [Fintype.sum_prod_type]

/-- VJP for 3D→3D functions. -/
structure HasVJP3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂) where
  backward : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂ → Tensor3 c₁ h₁ w₁
  correct : ∀ (x : Tensor3 c₁ h₁ w₁) (dy : Tensor3 c₂ h₂ w₂)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁),
    backward x dy ci hi wi =
    ∑ co : Fin c₂, ∑ ho : Fin h₂, ∑ wo : Fin w₂,
      pdiv3 f x ci hi wi co ho wo * dy co ho wo

/-- **Chain rule for 3D VJPs** — proved, no sorry. Requires the
    flattened forms of `f` and `g` to be differentiable everywhere. -/
noncomputable def vjp3_comp {c₁ h₁ w₁ c₂ h₂ w₂ c₃ h₃ w₃ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (g : Tensor3 c₂ h₂ w₂ → Tensor3 c₃ h₃ w₃)
    (hf_diff : Differentiable ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v))))
    (hg_diff : Differentiable ℝ
      (fun u : Vec (c₂ * h₂ * w₂) => Tensor3.flatten (g (Tensor3.unflatten u))))
    (hf : HasVJP3 f) (hg : HasVJP3 g) :
    HasVJP3 (g ∘ f) where
  backward := fun x dy => hf.backward x (hg.backward (f x) dy)
  correct := by
    intro x dy ci hi wi
    rw [hf.correct]; simp_rw [hg.correct]
    -- Goal: ∑∑∑ pdiv3_f * (∑∑∑ pdiv3_g * dy) = ∑∑∑ pdiv3_(g∘f) * dy
    -- Expand RHS: pdiv3_comp → triple sum, then distribute
    have hf_diff_at : DifferentiableAt ℝ
        (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
        (Tensor3.flatten x) := hf_diff (Tensor3.flatten x)
    have hg_diff_at : DifferentiableAt ℝ
        (fun u : Vec (c₂ * h₂ * w₂) => Tensor3.flatten (g (Tensor3.unflatten u)))
        (Tensor3.flatten (f x)) := hg_diff (Tensor3.flatten (f x))
    conv_rhs =>
      arg 2; ext ck; arg 2; ext hk; arg 2; ext wk
      rw [show pdiv3 (g ∘ f) x ci hi wi ck hk wk * dy ck hk wk =
          (∑ cj : Fin c₂, ∑ hj : Fin h₂, ∑ wj : Fin w₂,
            pdiv3 f x ci hi wi cj hj wj * pdiv3 g (f x) cj hj wj ck hk wk) * dy ck hk wk
        from by rw [← pdiv3_comp _ _ _ hf_diff_at hg_diff_at]]
    -- Distribute, pack triples → swap → unpack. (Credit: Lean Zulip)
    simp_rw [Finset.sum_mul, mul_assoc, Finset.mul_sum]
    show ∑ cj, ∑ hj, ∑ wj, ∑ ck, ∑ hk, ∑ wk, _ = ∑ ck, ∑ hk, ∑ wk, ∑ cj, ∑ hj, ∑ wj, _
    calc _ = ∑ jj ∈ Finset.univ ×ˢ Finset.univ ×ˢ Finset.univ,
             ∑ kk ∈ Finset.univ ×ˢ Finset.univ ×ˢ Finset.univ,
             pdiv3 f x ci hi wi jj.1 jj.2.1 jj.2.2 *
               (pdiv3 g (f x) jj.1 jj.2.1 jj.2.2 kk.1 kk.2.1 kk.2.2 *
               dy kk.1 kk.2.1 kk.2.2) := by simp_rw [Finset.sum_product]
         _ = _ := Finset.sum_comm
         _ = _ := by simp_rw [Finset.sum_product]

/-- **Identity Jacobian for Tensor3** — theorem, via `pdiv_id` and
    injectivity of the nested `finProdFinEquiv`. -/
theorem pdiv3_id {c h w : Nat} (x : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w)
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 (fun (t : Tensor3 c h w) => t) x ci hi wi co ho wo =
      if ci = co ∧ hi = ho ∧ wi = wo then 1 else 0 := by
  unfold pdiv3
  -- flatten ∘ id ∘ unflatten = id on Vec (c*h*w)
  have h_id : (fun v : Vec (c * h * w) =>
                Tensor3.flatten (Tensor3.unflatten v)) =
              (fun v : Vec (c * h * w) => v) := by
    funext v; exact Tensor3.flatten_unflatten v
  rw [h_id, pdiv_id]
  -- Goal: (if A = B then 1 else 0) = if C then 1 else 0
  -- where A, B are doubly-nested finProdFinEquiv outputs.
  by_cases h : ci = co ∧ hi = ho ∧ wi = wo
  · obtain ⟨hc, hh, hw⟩ := h
    subst hc; subst hh; subst hw; simp
  · rw [if_neg h, if_neg]
    intro heq
    apply h
    -- heq : finProdFinEquiv (fPF (ci, hi), wi) = finProdFinEquiv (fPF (co, ho), wo)
    have step1 := finProdFinEquiv.injective heq
    have hw_eq : wi = wo := (Prod.mk.inj step1).2
    have step2 := finProdFinEquiv.injective (Prod.mk.inj step1).1
    exact ⟨(Prod.mk.inj step2).1, (Prod.mk.inj step2).2, hw_eq⟩

def identity3_has_vjp (c h w : Nat) : HasVJP3 (fun (x : Tensor3 c h w) => x) where
  backward := fun _x dy => dy
  correct := by
    intro x dy ci hi wi
    -- Don't unfold pdiv3_id yet — work directly with the sum
    -- Rewrite each term under the sum
    show dy ci hi wi = _
    have : ∀ (co : Fin c) (ho : Fin h) (wo : Fin w),
        pdiv3 (fun (t : Tensor3 c h w) => t) x ci hi wi co ho wo * dy co ho wo =
        if ci = co then (if hi = ho then (if wi = wo then dy co ho wo else 0) else 0) else 0 := by
      intro co ho wo; rw [pdiv3_id]
      by_cases hc : ci = co <;> by_cases hh : hi = ho <;> by_cases hw : wi = wo <;> simp [*]
    simp_rw [this]
    -- Each sum is: ∑ x, if a = x then f x else 0
    -- Use Finset.sum_eq_single to collapse
    rw [Finset.sum_eq_single ci (by intro co _ hne; simp [Ne.symm hne]) (by simp)]
    simp only [eq_self_iff_true, ite_true]
    rw [Finset.sum_eq_single hi (by intro ho _ hne; simp [Ne.symm hne]) (by simp)]
    simp only [eq_self_iff_true, ite_true]
    rw [Finset.sum_eq_single wi (by intro wo _ hne; simp [Ne.symm hne]) (by simp)]
    simp

/-- **Sum rule for Tensor3 partial derivatives** — theorem, via `pdiv_add`. -/
theorem pdiv3_add {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁)
    (hf_diff : DifferentiableAt ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
      (Tensor3.flatten x))
    (hg_diff : DifferentiableAt ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (g (Tensor3.unflatten v)))
      (Tensor3.flatten x))
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (co : Fin c₂) (ho : Fin h₂) (wo : Fin w₂) :
    pdiv3 (fun y c h w => f y c h w + g y c h w) x ci hi wi co ho wo
    = pdiv3 f x ci hi wi co ho wo + pdiv3 g x ci hi wi co ho wo := by
  unfold pdiv3
  have h_flat : (fun v : Vec (c₁ * h₁ * w₁) =>
                  Tensor3.flatten ((fun y c h w => f y c h w + g y c h w)
                    (Tensor3.unflatten v))) =
                (fun v k => (fun w => Tensor3.flatten (f (Tensor3.unflatten w))) v k +
                            (fun w => Tensor3.flatten (g (Tensor3.unflatten w))) v k) := by
    funext v k
    unfold Tensor3.flatten
    rfl
  rw [h_flat, pdiv_add _ _ _ hf_diff hg_diff]

@[reducible] noncomputable def biPath3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂) :
    Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂ :=
  fun x c h w => f x c h w + g x c h w

noncomputable def biPath3_has_vjp {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (hf_diff : Differentiable ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v))))
    (hg_diff : Differentiable ℝ
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (g (Tensor3.unflatten v))))
    (hf : HasVJP3 f) (hg : HasVJP3 g) :
    HasVJP3 (biPath3 f g) where
  backward := fun x dy ci hi wi => hf.backward x dy ci hi wi + hg.backward x dy ci hi wi
  correct := by
    intro x dy ci hi wi
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext co
    rw [← Finset.sum_add_distrib]
    congr 1; ext ho
    rw [← Finset.sum_add_distrib]
    congr 1; ext wo
    rw [pdiv3_add _ _ _ (hf_diff (Tensor3.flatten x)) (hg_diff (Tensor3.flatten x))]
    ring

end Proofs
