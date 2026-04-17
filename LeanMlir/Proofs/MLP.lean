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

axiom pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j

/-- **Jacobian of dense wrt W** (new, Phase 7).

    `∂ dense(W, b, x)_j / ∂ W_{i', j'} = x_{i'} · δ(j, j')`

    Stated over the `Mat.flatten` bijection so we can reuse the `pdiv`
    framework on `Vec (m*n)`. Symmetric counterpart to `pdiv_dense`. -/
axiom pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0

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
  rw [hDec, pdiv_add, pdiv_const, pdiv_id]
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

axiom pdiv_relu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0

/-- ReLU VJP — proved. -/
noncomputable def relu_has_vjp (n : Nat) : HasVJP (relu n) where
  backward := fun x dy i => if x i > 0 then dy i else 0
  correct := by
    intro x dy i
    simp [pdiv_relu]

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

/-- **Cross-entropy-with-softmax scalar gradient**.

    `∂(-log softmax(z)[label])/∂z_j = softmax(z)_j - onehot(label)_j`

    Stated using `pdiv` on a `Vec 1`-valued wrapper (cross-entropy is
    naturally scalar, but `pdiv` is defined for `Vec → Vec`; we just
    take the only output index).  This lets us get away without a
    separate `sdiv` primitive. -/
axiom softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j

-- ════════════════════════════════════════════════════════════════
-- § MLP Composition
-- ════════════════════════════════════════════════════════════════

noncomputable def mlpForward {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    Vec d₀ → Vec d₃ :=
  dense W₂ b₂ ∘ relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀

noncomputable def mlp_has_vjp {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) := by
  unfold mlpForward
  have h1 := vjp_comp (dense W₀ b₀) (relu d₁) (dense_has_vjp W₀ b₀) (relu_has_vjp d₁)
  have h2 := vjp_comp _ (dense W₁ b₁) h1 (dense_has_vjp W₁ b₁)
  have h3 := vjp_comp _ (relu d₂) h2 (relu_has_vjp d₂)
  exact vjp_comp _ (dense W₂ b₂) h3 (dense_has_vjp W₂ b₂)

end Proofs
