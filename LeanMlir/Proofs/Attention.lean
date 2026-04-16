import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
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
-- § 0. Matrix transpose (needed for SDPA)
-- ════════════════════════════════════════════════════════════════

namespace Mat

/-- Matrix transpose: swap rows and columns. -/
def transpose (A : Mat m n) : Mat n m :=
  fun j i => A i j

end Mat

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

/-- **Correctness of `sdpa_back_Q`.** The backward w.r.t. Q equals the
    contraction of `pdivMat (fun Q' => sdpa n d Q' K V) Q` against `dOut`.
    Axiomatized pending Phase 2. Numerically gradient-checked. -/
axiom sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l

/-- **Correctness of `sdpa_back_K`.** Axiomatized pending Phase 2. -/
axiom sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l

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
-- § 3. Multi-Head wrapping
-- ════════════════════════════════════════════════════════════════

/-! ## Multi-head: parallelism over a partition

Multi-head attention is:

  1. Split the feature dim `d` into `h` "heads" of size `d_h = d/h`.
  2. Run SDPA independently on each head.
  3. Concatenate the head outputs.
  4. Apply one more dense projection `W_o`.

In the MLIR (`emitMHSAForward`):
    reshape (B, N, D) -> (B, N, H, D_h)
    transpose -> (B, H, N, D_h)
    [SDPA per head, using batching_dims = [0, 1]]
    transpose -> (B, N, H, D_h)
    reshape -> (B, N, D)
    dense projection (the "output projection" `Wo`)

**No new VJP math.** Reshape and transpose are just index permutations
— their Jacobians are permutation matrices, and their VJPs are just
inverse reshapes/transposes. The `h` independent SDPAs run in parallel;
their VJPs are independent (like a batch dimension).

If you wanted to prove `mhsa_has_vjp` in the framework, you'd:
- Define reshape/transpose as functions with trivial VJPs (permute
  indices -> VJP is the inverse permutation)
- Apply the SDPA backward trio (`sdpa_back_{Q,K,V}` + their
  `*_correct` axioms) per head (parallel over a new "head" axis)
- Compose with the output projection via `dense_has_vjp`

All mechanical. The insight is already captured in the SDPA backward
trio above; multi-head is an orchestration layer. -/

-- ════════════════════════════════════════════════════════════════
-- § 4. Transformer Block
-- ════════════════════════════════════════════════════════════════

/-! ## A transformer encoder block

From `emitTransformerBlockForward` (line 796):

    block(x) = x + MLP(LN(x + MHSA(LN(x))))

Expanding:

    h1 = x + MHSA(LN1(x))       -- attention sub-layer with residual
    h2 = h1 + MLP(LN2(h1))      -- MLP sub-layer with residual

where `MLP` is `Dense -> GELU -> Dense`.

Every piece has a `HasVJP` in the book:
- `LN1`, `LN2` — `layerNorm_has_vjp` (`LayerNorm.lean`)
- `MHSA` — `sdpa_back_{Q,K,V}` + multi-head wrapping (this file)
- `MLP` — `dense_has_vjp` composed with `gelu_has_vjp` composed with `dense_has_vjp` (via `vjp_comp`)
- `+` residual connections — `biPath_has_vjp` with identity (`Residual.lean`)

So the whole transformer block assembles from the chain rule and
`biPath_has_vjp`, applied to previously-proved `HasVJP` instances. No
new calculus axioms. No new structural moves. Just composition.

This is the capstone observation of the book: **a transformer block
is built from exactly the same tools as a ResNet block.** The five
structural primitives (add, multiply, compose, softmax closed-form,
dense-with-batch-dim) are sufficient for the whole modern architecture
zoo. Everything else is orchestration.
-/

-- ════════════════════════════════════════════════════════════════
-- § 5. The end of the road
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

**Stated concretely, correctness axiomatized (Phase 2 target):**
- Scaled dot-product attention backwards `sdpa_back_{Q,K,V}` — the
  formulas are written out and numerically gradient-checked;
  formal correctness awaits the matrix-level VJP framework.

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

end Proofs
