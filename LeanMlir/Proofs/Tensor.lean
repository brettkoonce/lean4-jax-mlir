import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.BigOperators
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

axiom sdiv {m : Nat} (f : Vec m → ℝ) (x : Vec m) (i : Fin m) : ℝ

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
-- § Matrix-level partial derivative
-- ════════════════════════════════════════════════════════════════

/-! `pdivMat` is the matrix analogue of `pdiv`: the partial derivative of a
matrix-valued function of a matrix. Introduced here (Phase 1) purely to give
multi-matrix-input functions like SDPA a way to *state* correctness claims.
The full composition framework (chain rule / sum rule / `HasVJPMat`) is
Phase 2 work — this minimal axiom is enough to stop using a vacuous
type-only axiom for `sdpa`. -/
axiom pdivMat {a b c d : Nat} (f : Mat a b → Mat c d) (A : Mat a b)
    (i : Fin a) (j : Fin b) (k : Fin c) (l : Fin d) : ℝ

-- ════════════════════════════════════════════════════════════════
-- § 3D Tensor VJP Framework (for CNN / Depthwise)
-- ════════════════════════════════════════════════════════════════

/-- A 3D feature map: channels × height × width (single sample). -/
abbrev Tensor3 (c h w : Nat) := Fin c → Fin h → Fin w → ℝ

/-- Partial derivative of a 3D→3D function, indexed by (input, output) triples. -/
axiom pdiv3 {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (co : Fin c₂) (ho : Fin h₂) (wo : Fin w₂) : ℝ

/-- Chain rule for 3D functions. -/
axiom pdiv3_comp {c₁ h₁ w₁ c₂ h₂ w₂ c₃ h₃ w₃ : Nat}
    (f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (g : Tensor3 c₂ h₂ w₂ → Tensor3 c₃ h₃ w₃)
    (x : Tensor3 c₁ h₁ w₁)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (ck : Fin c₃) (hk : Fin h₃) (wk : Fin w₃) :
    pdiv3 (g ∘ f) x ci hi wi ck hk wk =
    ∑ cj : Fin c₂, ∑ hj : Fin h₂, ∑ wj : Fin w₂,
      pdiv3 f x ci hi wi cj hj wj * pdiv3 g (f x) cj hj wj ck hk wk

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

/-- **Identity VJP for Tensor3** — proved. -/
axiom pdiv3_id {c h w : Nat} (x : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w)
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 (fun (t : Tensor3 c h w) => t) x ci hi wi co ho wo =
      if ci = co ∧ hi = ho ∧ wi = wo then 1 else 0

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

/-- **Additive fan-in for Tensor3** — proved. -/
axiom pdiv3_add {c₁ h₁ w₁ c₂ h₂ w₂ : Nat}
    (f g : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂)
    (x : Tensor3 c₁ h₁ w₁)
    (ci : Fin c₁) (hi : Fin h₁) (wi : Fin w₁)
    (co : Fin c₂) (ho : Fin h₂) (wo : Fin w₂) :
    pdiv3 (fun y c h w => f y c h w + g y c h w) x ci hi wi co ho wo
    = pdiv3 f x ci hi wi co ho wo + pdiv3 g x ci hi wi co ho wo

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
