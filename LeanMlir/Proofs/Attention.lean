import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN          -- needed for Kernel4 in patchEmbed
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

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
-- Diff axioms further down where they're introduced.
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
    `geluScalar` is `C^∞` (composition of polynomials, `Real.tanh`,
    `Real.sqrt` of a positive constant), but Mathlib's `Real.tanh`
    Differentiable lemma isn't tagged for `fun_prop` in this toolchain
    snapshot. **Axiomatized** for the same reason as `rowSoftmax_flat_diff`
    and `bnIstdBroadcast_diff` — Mathlib has the calculus, the project
    hasn't built the bridge yet. -/
axiom gelu_per_token_flat_diff (N D : Nat) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => gelu D (X n))
                   (Mat.unflatten v)))

/-- Differentiability of the flattened per-token LayerNorm map.
    `layerNormForward = bnForward`, which is `bnAffine ∘ bnNormalize`
    where `bnNormalize` involves `1/√(bnVar+ε)`. Smoothness depends
    on the `bnIstdBroadcast_diff` axiom (Stage 2d) plus the analogous
    composition; per-token-flat lift is **axiomatized** for the same
    reason as `bnIstdBroadcast_diff` itself. Takes `(hε : 0 < ε)`. -/
axiom layerNorm_per_token_flat_diff (N D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormForward D ε γ β (X n))
                   (Mat.unflatten v)))

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

    Standard calculus; axiomatized to stay in our `pdiv` framework. -/
axiom pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i)

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

/-- **Smoothness of `rowSoftmax`** — axiomatized.

    `rowSoftmax M r c = exp(M r c) / Σⱼ exp(M r j)`. The denominator
    is everywhere positive, so the function is C^∞ via Mathlib's
    `Real.exp` calculus. Formal derivation deferred — axiomatize the
    Differentiable claim so vjpMat_comp can compose through it. -/
axiom rowSoftmax_flat_diff (m n : Nat) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m n))

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
    simp_rw [pdivMat_rowIndep]
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

Previously this section ended with a single `axiom sdpa_has_vjp` whose
type was just `(... functions) × (... functions) × (... functions)`.
That's **vacuous as a correctness claim** — a triple of zero functions
satisfies it. Phase 1 replaces that with:

1. **Concrete definitions** of `sdpa_back_Q`, `sdpa_back_K`, `sdpa_back_V`
   transcribed from the step-by-step derivation above.
2. **Honest correctness axioms** stated in terms of `pdivMat` (the
   matrix-level partial derivative primitive from `Tensor.lean`).

The correctness axioms are **still axioms** — proving them requires the
matrix-level VJP composition framework (Phase 2). But now they *say*
something: each backward equals the pdivMat-contracted cotangent, which
is the definition of being a correct VJP.

The concrete formulas here are numerically gradient-checked in
`check_axioms.py` (`test_sdpa_back_Q/K/V`), so the axioms are credible
up to floating-point precision even before the formal proof lands.
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
nested formula) and invoke its `.correct` to discharge the axiom. -/

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

-- ════════════════════════════════════════════════════════════════
-- § 3. Multi-Head wrapping (Phase 8 — was hand-waved, now axiomatized)
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
Phase 8 closes that gap: we *define* `mhsa_layer` concretely in Lean
(Q/K/V projections → per-head slice → sdpa-per-head → concat → Wo
projection) and *axiomatize* its `HasVJPMat` in one bundled step. The
bundled axiom is the "per-head vmap" fact — formalizing it at the
framework level would need a Mat-level `rowIndep` generalized to the
column axis plus a ternary input primitive, which is more bureaucracy
than the book's pedagogy calls for. The formula is numerically
gradient-checked in `check_axioms.py`, so the axiom is credible up to
floating-point precision. -/

/-- Multi-head SDPA on a single sequence: `Mat N (heads·d_head) → Mat N (heads·d_head)`.

    Concretely defined (not opaque):
    1. Q, K, V projections (each a per-token dense with its own Wq/Wk/Wv).
    2. For each head `h : Fin heads`, extract the `(N, d_head)` slice of Q/K/V
       by indexing `finProdFinEquiv (h, k)` in the combined axis.
    3. Run `sdpa` on each slice.
    4. Concatenate the head outputs back along the feature axis.
    5. Output projection Wo · concat + bo (per-token dense).

    The bundled VJP axiom below packages the correctness of this
    whole thing — equivalent to composing dense Jacobians, the per-head
    SDPA jacobians (we already proved `sdpa_back_{Q,K,V}_correct`),
    and the reshape/unreshape `pdiv_reindex` facts, with the "per-head
    independence" fact as the one primitive that doesn't factor through
    existing axioms. -/
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

/-- **Multi-head SDPA VJP — bundled axiom (Phase 8).**

    The one honest axiom needed to lift single-head SDPA (already proved
    in `sdpa_back_{Q,K,V}_correct`) to the full multi-head layer. The
    axiom asserts existence of a correct backward function; numerically
    gradient-checked in `check_axioms.py`.

    Why an axiom here rather than a theorem: formalizing the "apply SDPA
    independently to each of the `heads` column-slices" requires either a
    column-indep analog of `pdivMat_rowIndep` plus a ternary-input VJP
    framework, or heavy reshape gymnastics. Both are bureaucracy that
    distract from the pedagogy. The mathematical content of the axiom —
    block-diagonality of the Jacobian across heads — is exactly the
    "multi-head is just parallel SDPA" claim the book makes in prose. -/
axiom mhsa_has_vjp_mat (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo)

/-- Differentiability of the flattened multi-head SDPA layer.
    Composition of per-token dense projections, per-head SDPA (which
    factors through `rowSoftmax_flat_diff`), and a final per-token dense.
    **Axiomatized** for the same reason as `mhsa_has_vjp_mat`: formalizing
    the column-axis `pdivMat_rowIndep` analog to factor the per-head
    independence is project-wide bureaucracy. Sibling of `mhsa_has_vjp_mat`. -/
axiom mhsa_layer_flat_diff (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo
                     (Mat.unflatten v)))

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
  rowwise_has_vjp_mat (layerNorm_has_vjp D ε γ β hε)

/-- Per-token dense projection across a sequence.
    `Q = X · W + b`, row-by-row dense with shared weights. -/
noncomputable def dense_per_token_has_vjp_mat (N inD outD : Nat)
    (W : Mat inD outD) (b : Vec outD) :
    HasVJPMat (fun X : Mat N inD => fun n => dense W b (X n)) :=
  rowwise_has_vjp_mat (dense_has_vjp W b)

/-- Per-token GELU across a sequence. Elementwise activation,
    so diagonal Jacobian both across rows and within a row. -/
noncomputable def gelu_per_token_has_vjp_mat (N D : Nat) :
    HasVJPMat (fun X : Mat N D => fun n => gelu D (X n)) :=
  rowwise_has_vjp_mat (gelu_has_vjp D)

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
`biPathMat_has_vjp`. No new axioms beyond `mhsa_has_vjp_mat`. -/

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
  -- only obstacle (handled by `gelu_per_token_flat_diff`'s axiom). We
  -- factor through the gelu helper rather than reproving it inline.
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
    correct VJP, composed from proved building blocks plus the bundled
    axioms (`mhsa_has_vjp_mat` and its flat-diff sibling for attention;
    the per-token LN/MLP smoothness Diff axioms introduced by the
    foundation flip). -/
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

**Three calculus axioms do all the structural work:**

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
cannot find a sixth trick or a fourth calculus axiom anywhere in the
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

/-! ## Patch embedding — the one new axiom for Phase 10

The patch embedding takes an image `Tensor3 ic H W` and produces
`Mat (N+1) D` where N = num_patches (= (H/patchSize) * (W/patchSize)):

1. Conv projection with stride = patchSize: `Tensor3 ic H W → Tensor3 D H' W'`
   where H' = H/patchSize, W' = W/patchSize. (Uses `conv2d` under the hood.)
2. Reshape spatial `(D, H', W')` to tokens `(N, D)` — a pure permutation.
3. Prepend learnable CLS token at row 0 → `(N+1, D)`.
4. Add learnable positional embedding matrix → `(N+1, D)`.

Each step is either an opaque forward (conv) or a `pdiv_reindex`/
`pdiv_add` operation. The composite is stated on flattened endpoints
`Vec (ic*H*W) → Vec ((N+1)*D)` and bundled as one `HasVJP` axiom —
mirror of `conv2d_weight_grad_has_vjp` and `mhsa_has_vjp_mat`. Formalizing
each step's rank-crossing would require a unified rank-polymorphic VJP
framework; bundling here keeps the finale clean and honest. -/

/-- Patch embedding forward on flattened endpoints.

    Opaque: the actual computation (conv + reshape + CLS prepend + pos
    embed) is implemented in the codegen. We axiomatize that this forward
    function has a well-defined VJP; the closed-form backward equals the
    step-by-step sum of conv's weight/input gradients + CLS and positional
    gradients + reshape index permutation. Gradient-checkable. -/
noncomputable opaque patchEmbed_flat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D)

/-- **Patch embedding VJP — Phase 10 bundled axiom.** -/
axiom patchEmbed_flat_has_vjp
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    HasVJP (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed)

/-- **Patch embedding Differentiability — bundled axiom (sibling of
    `patchEmbed_flat_has_vjp`).** Same justification: the actual
    computation lives in MLIR; we axiomatize that the forward function
    is `Differentiable` so it composes through `vjp_comp` chains. -/
axiom patchEmbed_flat_diff
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Differentiable ℝ (patchEmbed_flat ic H W patchSize N D
                       W_conv b_conv cls_token pos_embed)

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
    discharged by the per-stage Diff lemmas/axioms above. -/
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
