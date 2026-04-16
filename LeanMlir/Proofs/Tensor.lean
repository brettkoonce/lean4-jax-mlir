import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.Ring

/-!
# Tensor Algebra for VJP Proofs

Vectors, matrices, and operations over `ℝ`, using Mathlib's `Finset.sum`.

Partial derivatives (`pdiv`) and their composition rules (chain rule,
linearity, product rule) are axiomatized — they are theorems of real
analysis. Everything else is proved.
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
-- § Differentiation (axiomatized)
-- ════════════════════════════════════════════════════════════════

axiom pdiv {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) : ℝ

axiom pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k

axiom pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j

axiom pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j

axiom pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0

/-- **Partial derivative of a constant function is zero.**

    For any `c : Vec n` and any input `x`, the function `fun _ => c`
    has zero Jacobian. Standard calculus; axiomatized to stay inside
    our `pdiv` framework. (Mathlib equivalent: `fderiv_const`.) -/
axiom pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0

-- ════════════════════════════════════════════════════════════════
-- § VJP Framework
-- ════════════════════════════════════════════════════════════════

structure HasVJP {m n : Nat} (f : Vec m → Vec n) where
  backward : Vec m → Vec n → Vec m
  correct : ∀ (x : Vec m) (dy : Vec n) (i : Fin m),
    backward x dy i = ∑ j : Fin n, pdiv f x i j * dy j

/-- **Chain rule for VJPs** — proved, no sorry. -/
noncomputable def vjp_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
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
    rw [pdiv_comp]
    simp_rw [← mul_assoc]
    rw [← Finset.sum_mul]

/-- **Additive fan-in** — proved, no sorry. -/
@[reducible] noncomputable def biPath {m n : Nat} (f g : Vec m → Vec n) : Vec m → Vec n :=
  fun x i => f x i + g x i

noncomputable def biPath_has_vjp {m n : Nat}
    (f g : Vec m → Vec n) (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (biPath f g) where
  backward := fun x dy i => hf.backward x dy i + hg.backward x dy i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j; rw [pdiv_add]; ring

/-- **Multiplicative fan-in** — proved, no sorry. -/
@[reducible] noncomputable def elemwiseProduct {n : Nat}
    (f g : Vec n → Vec n) : Vec n → Vec n :=
  fun x i => f x i * g x i

noncomputable def elemwiseProduct_has_vjp {n : Nat}
    (f g : Vec n → Vec n) (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (elemwiseProduct f g) where
  backward := fun x dy i =>
    hf.backward x (fun j => g x j * dy j) i +
    hg.backward x (fun j => f x j * dy j) i
  correct := by
    intro x dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j
    rw [pdiv_mul]; ring

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
    (A : Mat a b) (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by
  unfold pdivMat
  -- Step 1: the flattened composition equals the composition of flatteneds,
  -- because `unflatten ∘ flatten = id`.
  have h_compose :
      (fun v : Vec (a * b) => Mat.flatten ((G ∘ F) (Mat.unflatten v))) =
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) ∘
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) := by
    funext v
    simp [Function.comp, Mat.unflatten_flatten]
  rw [h_compose, pdiv_comp]
  -- Step 2: inside the resulting sum, rewrite `F' (flatten A)` to `flatten (F A)`
  -- (by unflatten ∘ flatten = id), so the "middle point" matches pdivMat's form.
  have h_mid :
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A)
      = Mat.flatten (F A) := by
    simp [Mat.unflatten_flatten]
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

/-- **Sum rule for `pdivMat`** — theorem, via `pdiv_add`. -/
theorem pdivMat_add {a b c d : Nat}
    (F G : Mat a b → Mat c d) (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) :
    pdivMat (fun M r s => F M r s + G M r s) A i j k l
    = pdivMat F A i j k l + pdivMat G A i j k l := by
  unfold pdivMat
  -- flatten of a pointwise sum = pointwise sum of flattens.
  have h_flat : (fun v : Vec (a * b) =>
                  Mat.flatten ((fun M r s => F M r s + G M r s) (Mat.unflatten v))) =
                (fun v k => (fun w => Mat.flatten (F (Mat.unflatten w))) v k +
                            (fun w => Mat.flatten (G (Mat.unflatten w))) v k) := by
    funext v k
    unfold Mat.flatten
    rfl
  rw [h_flat, pdiv_add]

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
    (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (G ∘ F) where
  backward := fun A dY => hF.backward A (hG.backward (F A) dY)
  correct := by
    intro A dY i j
    rw [hF.correct]
    simp_rw [hG.correct]
    -- Goal: ∑∑ pdivMat F A · (∑∑ pdivMat G (F A) · dY) = ∑∑ pdivMat (G∘F) A · dY
    -- Expand RHS via pdivMat_comp, then swap sums.
    conv_rhs =>
      arg 2; ext k; arg 2; ext l
      rw [show pdivMat (G ∘ F) A i j k l * dY k l =
          (∑ p : Fin c, ∑ q : Fin d,
            pdivMat F A i j p q * pdivMat G (F A) p q k l) * dY k l
        from by rw [← pdivMat_comp]]
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
    (F G : Mat a b → Mat c d) (hF : HasVJPMat F) (hG : HasVJPMat G) :
    HasVJPMat (biPathMat F G) where
  backward := fun A dY i j => hF.backward A dY i j + hG.backward A dY i j
  correct := by
    intro A dY i j
    rw [hF.correct, hG.correct, ← Finset.sum_add_distrib]
    congr 1; ext k
    rw [← Finset.sum_add_distrib]
    congr 1; ext l
    rw [pdivMat_add]; ring

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
numerically gradient-checked in `check_axioms.py`. -/

axiom pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0

axiom pdivMat_matmul_right_const {m p q : Nat} (A : Mat m p) (D : Mat p q)
    (i : Fin m) (j : Fin p) (k : Fin m) (l : Fin q) :
    pdivMat (fun A' : Mat m p => Mat.mul A' D) A i j k l =
    if i = k then D j l else 0

axiom pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0

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
  -- Step 3: apply pdiv_mul.
  rw [pdiv_mul (fun _ _ => s) (fun w => w)]
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

/-- **Transpose Jacobian**: `∂A^T_{kl} / ∂A_{ij} = δ_{l=i, k=j}`. -/
axiom pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0

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
    and two applications of `Fintype.sum_equiv + sum_prod_type`. -/
theorem pdiv3_comp {c₁ h₁ w₁ c₂ h₂ w₂ c₃ h₃ w₃ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (g : Tensor3 c₂ h₂ w₂ → Tensor3 c₃ h₃ w₃)
    (x : Tensor3 c₁ h₁ w₁)
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
  rw [h_compose, pdiv_comp]
  -- Substitute the "middle point" F' (flatten x) = flatten (f x).
  have h_mid :
      (fun v : Vec (c₁ * h₁ * w₁) => Tensor3.flatten (f (Tensor3.unflatten v)))
        (Tensor3.flatten x) = Tensor3.flatten (f x) := by
    simp [Tensor3.unflatten_flatten]
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

/-- **Chain rule for 3D VJPs** — proved, no sorry. -/
noncomputable def vjp3_comp {c₁ h₁ w₁ c₂ h₂ w₂ c₃ h₃ w₃ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (g : Tensor3 c₂ h₂ w₂ → Tensor3 c₃ h₃ w₃)
    (hf : HasVJP3 f) (hg : HasVJP3 g) :
    HasVJP3 (g ∘ f) where
  backward := fun x dy => hf.backward x (hg.backward (f x) dy)
  correct := by
    intro x dy ci hi wi
    rw [hf.correct]; simp_rw [hg.correct]
    -- Goal: ∑∑∑ pdiv3_f * (∑∑∑ pdiv3_g * dy) = ∑∑∑ pdiv3_(g∘f) * dy
    -- Expand RHS: pdiv3_comp → triple sum, then distribute
    conv_rhs =>
      arg 2; ext ck; arg 2; ext hk; arg 2; ext wk
      rw [show pdiv3 (g ∘ f) x ci hi wi ck hk wk * dy ck hk wk =
          (∑ cj : Fin c₂, ∑ hj : Fin h₂, ∑ wj : Fin w₂,
            pdiv3 f x ci hi wi cj hj wj * pdiv3 g (f x) cj hj wj ck hk wk) * dy ck hk wk
        from by rw [← pdiv3_comp]]
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
  rw [h_flat, pdiv_add]

@[reducible] noncomputable def biPath3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂) :
    Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂ :=
  fun x c h w => f x c h w + g x c h w

noncomputable def biPath3_has_vjp {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
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
    congr 1; ext wo; rw [pdiv3_add]; ring

end Proofs
