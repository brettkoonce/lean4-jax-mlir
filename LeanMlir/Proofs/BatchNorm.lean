import LeanMlir.Proofs.Tensor
import Mathlib.Data.Real.Sqrt

/-!
# Batch Normalization VJP

This is the first layer where the casual "stare and differentiate" approach
breaks down. In dense and conv layers, every output cell `yⱼ` depends on
its input independently of the other inputs. In batch norm, every output
depends on **every input** through the mean and variance reductions, so
the Jacobian is dense and the chain rule has to do real work.

The famous result we'll derive: the input gradient collapses to a single
**three-term closed form** that doesn't expose the individual contributions
from `mean` and `variance`. This is the "consolidated" BN backward formula
that every ML framework hard-codes (because deriving it on the fly is a
pain). It's what `MlirCodegen.lean` emits at line 799:

    %cbg_t5 = istd * (N * d_xhat - sum(d_xhat) - xhat * sum(d_xhat * xhat))
    %cbg_dconv = (1/N) * %cbg_t5

This file:
1. Defines BN forward step by step (mean → var → istd → xhat → affine).
2. States the **easy** parameter gradients (γ, β).
3. Walks through the derivation of the **hard** input gradient and states
   the consolidated formula.

## A note on shapes

The actual implementation reduces over `[batch, h, w]` per channel. For
clarity, this file works on a single 1D `Vec n` (think of `n` as
`B · H · W` flattened, for one channel). The math is identical; only the
indexing changes when you go to 4D.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Forward pass — defined incrementally
-- ════════════════════════════════════════════════════════════════

/-- Population mean: `μ = (1/N) Σᵢ xᵢ` -/
noncomputable def bnMean (n : Nat) (x : Vec n) : ℝ :=
  (∑ i : Fin n, x i) / (n : ℝ)

/-- Population variance: `σ² = (1/N) Σᵢ (xᵢ − μ)²` -/
noncomputable def bnVar (n : Nat) (x : Vec n) : ℝ :=
  let μ := bnMean n x
  (∑ i : Fin n, (x i - μ) * (x i - μ)) / (n : ℝ)

/-- Inverse standard deviation: `istd = 1 / √(σ² + ε)` -/
noncomputable def bnIstd (n : Nat) (x : Vec n) (ε : ℝ) : ℝ :=
  1 / Real.sqrt (bnVar n x + ε)

/-- Normalized output: `x̂ᵢ = (xᵢ − μ) · istd`

    `x̂` has mean 0 and variance 1 (up to ε-correction). It's the
    "centered, unit-scaled" version of `x`. -/
noncomputable def bnXhat (n : Nat) (ε : ℝ) (x : Vec n) : Vec n :=
  fun i => (x i - bnMean n x) * bnIstd n x ε

/-- The full BN forward: `yᵢ = γ · x̂ᵢ + β`

    `γ` and `β` are learnable per-channel parameters that restore the
    network's representational freedom that normalization took away.
    Without them, BN would force every layer's output to have mean 0,
    variance 1 — too constraining.

    MLIR (`MlirCodegen.lean` lines 723–728):
      %cbn_g_bc  = broadcast %g
      %cbn_gn    = multiply %cbn_norm, %cbn_g_bc
      %cbn_bt_bc = broadcast %bt
      %cbn_pre   = add %cbn_gn, %cbn_bt_bc
-/
noncomputable def bnForward (n : Nat) (ε γ β : ℝ) (x : Vec n) : Vec n :=
  fun i => γ * bnXhat n ε x i + β

-- ════════════════════════════════════════════════════════════════
-- § Parameter gradients (the easy part)
-- ════════════════════════════════════════════════════════════════

/-- **γ gradient**: `dγ = Σᵢ dyᵢ · x̂ᵢ`

    `γ` is a scalar that multiplies each `x̂ᵢ`. By the product rule:
    `∂yᵢ/∂γ = x̂ᵢ`. Summing over the output cotangent `dy`:
    `dγ = Σᵢ dyᵢ · x̂ᵢ`.

    This is just an inner product of `dy` with `x̂` — no mean/variance
    chain-rule trickery, because γ doesn't enter the reduction.

    MLIR (`MlirCodegen.lean` lines 766–768):
      %cbg_gn = multiply %effGrad, %cbn_norm
      %d_g    = reduce add %cbg_gn across dimensions = [0, 2, 3]
-/
noncomputable def bn_grad_gamma (n : Nat) (ε : ℝ) (x : Vec n) (dy : Vec n) : ℝ :=
  ∑ i : Fin n, dy i * bnXhat n ε x i

/-- **β gradient**: `dβ = Σᵢ dyᵢ`

    `β` is added to every output, so `∂yᵢ/∂β = 1` and the gradient is
    just the sum of the output cotangents. Even simpler than dγ.

    MLIR (line 770):
      %d_bt = reduce add %effGrad across dimensions = [0, 2, 3]
-/
noncomputable def bn_grad_beta (n : Nat) (dy : Vec n) : ℝ := ∑ i : Fin n, dy i

-- ════════════════════════════════════════════════════════════════
-- § Input gradient — the derivation
-- ════════════════════════════════════════════════════════════════

/-! ## Why the input gradient is hard

The output `yⱼ` depends on `xᵢ` through **three** paths:

  (a) Directly: `xⱼ` appears in `(xⱼ − μ)` (only when `i = j`).
  (b) Via `μ`:    `μ` is `(1/N) Σₖ xₖ`, so changing `xᵢ` changes `μ`
                  by `1/N`, which shifts every `(xⱼ − μ)`.
  (c) Via `σ²`:   `σ²` is `(1/N) Σₖ (xₖ − μ)²`, so changing `xᵢ`
                  changes `σ²`, which changes `istd`, which scales
                  every `x̂ⱼ`.

So `∂yⱼ/∂xᵢ ≠ 0` for **every** `(i, j)` pair — the Jacobian is dense.
Naively, the VJP costs O(N²); the consolidated form turns it into O(N)
by collapsing the cancellations algebraically.

## The derivation

Strip off the affine layer first: let `dx̂ᵢ := γ · dyᵢ`. Then we need
the VJP of `bnXhat` (the normalize step) at the cotangent `dx̂`.

For `x̂ⱼ = (xⱼ − μ) · istd`, the chain rule gives:

    ∂x̂ⱼ/∂xᵢ = (∂xⱼ/∂xᵢ − ∂μ/∂xᵢ) · istd + (xⱼ − μ) · ∂istd/∂xᵢ

We need three sub-derivatives:

    ∂xⱼ/∂xᵢ  = δᵢⱼ                               (identity)
    ∂μ/∂xᵢ   = 1/N                                (mean is linear in x)
    ∂σ²/∂xᵢ  = (2/N) · (xᵢ − μ) · (1 − 1/N)
              ≈ (2/N) · (xᵢ − μ)                  (the (1−1/N) term
                                                   eats into a Σ that
                                                   sums to zero, so it
                                                   doesn't survive)
    ∂istd/∂xᵢ = (−1/2) · istd³ · ∂σ²/∂xᵢ
              = −istd³ · (xᵢ − μ) / N
              = −istd · x̂ᵢ / N                    (since x̂ᵢ = (xᵢ−μ)·istd)

Substituting:

    ∂x̂ⱼ/∂xᵢ = (δᵢⱼ − 1/N) · istd − (xⱼ − μ) · istd · x̂ᵢ / N
            = istd · (δᵢⱼ − 1/N − x̂ⱼ · x̂ᵢ / N)
            = (istd / N) · (N · δᵢⱼ − 1 − x̂ᵢ · x̂ⱼ)

Now contract with `dx̂` to get the input cotangent of the normalize step:

    dxᵢ = Σⱼ (∂x̂ⱼ/∂xᵢ) · dx̂ⱼ
        = (istd / N) · Σⱼ (N · δᵢⱼ − 1 − x̂ᵢ · x̂ⱼ) · dx̂ⱼ
        = (istd / N) · (N · dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ · Σⱼ x̂ⱼ · dx̂ⱼ)

This is the consolidated formula — three terms, two scalar reductions
(`Σⱼ dx̂ⱼ` and `Σⱼ x̂ⱼ · dx̂ⱼ`), one elementwise broadcast. O(N) work
instead of O(N²). And it's exactly what the MLIR emits.
-/

/-- **The consolidated BN input gradient.**

      dxᵢ = (1/N) · istd · (N · dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ · Σⱼ x̂ⱼ · dx̂ⱼ)

    where `dx̂ᵢ = γ · dyᵢ` (gradient pulled back through the affine
    layer first).

    This matches `MlirCodegen.lean` lines 794–801:
      %cbg_t1 = N * d_xhat
      %cbg_t2 = %cbg_t1 - sum(d_xhat)              -- subtract mean
      %cbg_t3 = xhat * sum(xhat * d_xhat)
      %cbg_t4 = %cbg_t2 - %cbg_t3                  -- the three-term combo
      %cbg_t5 = istd * %cbg_t4
      %cbg_dconv = (1/N) * %cbg_t5
-/
noncomputable def bn_grad_input
    (n : Nat) (ε γ : ℝ) (x : Vec n) (dy : Vec n) : Vec n :=
  let xh : Vec n := bnXhat n ε x
  let dxhat : Vec n := fun i => γ * dy i
  let invN : ℝ := 1 / (n : ℝ)
  let s : ℝ := bnIstd n x ε
  let sumDxhat : ℝ := ∑ i : Fin n, dxhat i
  let sumXhatDxhat : ℝ := ∑ i : Fin n, xh i * dxhat i
  fun i =>
    invN * s * ((n : ℝ) * dxhat i - sumDxhat - xh i * sumXhatDxhat)

-- ════════════════════════════════════════════════════════════════
-- § Correctness statements
-- ════════════════════════════════════════════════════════════════

/- **A note on the parameter gradients**

   `bn_grad_gamma` and `bn_grad_beta` are scalar-valued derivatives w.r.t.
   scalar (per-channel) parameters, which doesn't fit our `pdiv` /
   `HasVJP` framework cleanly (everything in `Tensor.lean` is sized over
   `Vec`). The mathematical content of "these are the correct gradients"
   is just the product rule applied to `γ · x̂ᵢ + β`:

       ∂(γ · x̂ᵢ + β)/∂γ = x̂ᵢ        →  dγ = Σᵢ dyᵢ · x̂ᵢ
       ∂(γ · x̂ᵢ + β)/∂β ​​= 1          →  dβ = Σᵢ dyᵢ

   We state these as the *definitions* `bn_grad_gamma` and `bn_grad_beta`
   above; the sum-over-i is the bookkeeping that turns "per-output
   gradient" into "per-parameter gradient."
-/

-- See `bn_input_grad_correct` below `bn_has_vjp` for the headline correctness theorem.

-- ════════════════════════════════════════════════════════════════
-- § Decomposition: bn = affine ∘ xhat
-- ════════════════════════════════════════════════════════════════

/-! ## Cleaner view: BN as a composition

The BN forward is really two steps glued together:

  1. **Normalize** (`bnXhat`): the hard part with mean/var/istd
     reductions. `Vec n → Vec n`, no parameters.
  2. **Affine** (`fun v i => γ · vᵢ + β`): elementwise scale-and-shift.
     The parameters γ, β live here.

If we had a `HasVJP` instance for each, we could compose them with
`vjp_comp` from `Tensor.lean` and get the full BN VJP "for free."

The affine VJP is trivial:
  ∂(γ · vᵢ + β)/∂vⱼ = γ · δᵢⱼ
  → back(v, dy)ᵢ = γ · dyᵢ

The normalize VJP is the consolidated three-term formula above (with
`γ = 1`, since the affine has been factored out).

We state both as `HasVJP` instances. Their composition (via `vjp_comp`)
gives the full BN input gradient — and the parameter gradients are
collected at the affine layer alongside.
-/

/-- The normalize step as a function `Vec n → Vec n` (no params except ε). -/
noncomputable def bnNormalize (n : Nat) (ε : ℝ) : Vec n → Vec n :=
  bnXhat n ε

/-- The affine step as a function `Vec n → Vec n` (γ, β as constants). -/
noncomputable def bnAffine (n : Nat) (γ β : ℝ) : Vec n → Vec n :=
  fun v i => γ * v i + β

/-- BN as the composition of normalize and affine. -/
theorem bnForward_eq_compose (n : Nat) (ε γ β : ℝ) :
    bnForward n ε γ β = bnAffine n γ β ∘ bnNormalize n ε := by
  funext x i; rfl

/-- The affine Jacobian is diagonal: `∂(γ·vᵢ + β)/∂vⱼ = γ · δᵢⱼ`.

    Proved from foundation rules: `bnAffine` decomposes as
    `(γ · v) + (constant β)`, where the linear term factors further
    as `(constant γ) * (identity)` for `pdiv_mul`. The pieces collapse
    via `pdiv_add` + `pdiv_mul` + `pdiv_const` + `pdiv_id`. -/
theorem pdiv_bnAffine (n : Nat) (γ β : ℝ)
    (v : Vec n) (i j : Fin n) :
    pdiv (bnAffine n γ β) v i j =
      if i = j then γ else 0 := by
  unfold bnAffine
  -- Decompose `γ * v i + β` as `(γ · v) + (const β)`.
  rw [show (fun v : Vec n => fun i : Fin n => γ * v i + β) =
        (fun v i =>
          (fun (y : Vec n) (k : Fin n) => γ * y k) v i +
          (fun (_ : Vec n) (_ : Fin n) => β) v i) from rfl]
  have h_lin_diff : DifferentiableAt ℝ
      (fun (y : Vec n) (k : Fin n) => γ * y k) v := by fun_prop
  have h_const_β_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (_ : Fin n) => β) v :=
    differentiableAt_const _
  rw [pdiv_add _ _ _ h_lin_diff h_const_β_diff]
  -- Constant term: pdiv = 0.
  rw [show pdiv (fun (_ : Vec n) (_ : Fin n) => β) v i j = 0
      from pdiv_const (fun _ : Fin n => β) v i j]
  -- Linear term: factor as (constant γ) * identity, apply pdiv_mul.
  rw [show (fun y : Vec n => fun k : Fin n => γ * y k) =
        (fun y k =>
          (fun (_ : Vec n) (_ : Fin n) => γ) y k *
          (fun (y' : Vec n) => y') y k) from rfl]
  have h_const_γ_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (_ : Fin n) => γ) v :=
    differentiableAt_const _
  have h_id_diff : DifferentiableAt ℝ
      (fun (y' : Vec n) => y') v := differentiableAt_id
  rw [pdiv_mul _ _ _ h_const_γ_diff h_id_diff]
  rw [show pdiv (fun (_ : Vec n) (_ : Fin n) => γ) v i j = 0
      from pdiv_const (fun _ : Fin n => γ) v i j]
  rw [pdiv_id]
  by_cases h : i = j
  · rw [if_pos h, if_pos h]; ring
  · rw [if_neg h, if_neg h]; ring

-- ════════════════════════════════════════════════════════════════
-- § The hard Jacobian: `pdiv_bnNormalize` — now derived
-- ════════════════════════════════════════════════════════════════

/-! The consolidated three-term formula used to be axiomatized directly.
Now it's a theorem: we factor `bnXhat` as the elementwise product of
the centered input and the broadcast `istd`, apply `pdiv_mul`, and
collapse via `ring` using the `x̂ᵢ = (xᵢ - μ) · istd` identity.

Only **two** elementary calculus facts remain axiomatized:

1. `pdiv_bnCentered` — ∂(xⱼ - μ(x))/∂xᵢ = δᵢⱼ - 1/n.
   Equivalent to Mathlib's `HasDerivAt.sub` applied to `id` and `(const_mul) ∘ (Finset.sum)`.

2. `pdiv_bnIstdBroadcast` — ∂istd(x,ε)/∂xᵢ = -istd³ · (xᵢ - μ) / n.
   Equivalent to `Real.hasDerivAt_sqrt` + `HasDerivAt.inv` + chain rule
   against `bnVar`, whose derivative is `(2/n)·(xᵢ - μ)` by the same
   product-rule trick as `pdiv_bnCentered`.

The three-term formula falls out by ring manipulation alone. -/

/-- Centered input: `(x - μ(x))` as a `Vec n → Vec n` function. -/
noncomputable def bnCentered (n : Nat) : Vec n → Vec n :=
  fun x j => x j - bnMean n x

/-- Broadcast inverse-stddev: `istd(x,ε)` as a `Vec n → Vec n` function
    (constant in the output index, just lifted for `pdiv_mul`). -/
noncomputable def bnIstdBroadcast (n : Nat) (ε : ℝ) : Vec n → Vec n :=
  fun x _ => bnIstd n x ε

/-- `bnXhat` factors as `bnCentered · bnIstdBroadcast` (elementwise product). -/
theorem bnXhat_eq_product (n : Nat) (ε : ℝ) (x : Vec n) :
    bnXhat n ε x = fun j => bnCentered n x j * bnIstdBroadcast n ε x j := by
  funext j
  unfold bnXhat bnCentered bnIstdBroadcast
  rfl

/-- **Centered-input Jacobian** — proved from foundation rules.

    `∂(xⱼ - μ(x))/∂xᵢ = δᵢⱼ - 1/n`

    Decomposition: `bnCentered y k = y k - (∑ s, y s)/n` factors as
    `(id y) k + (-(1/n)) * (∑ s, y s)`. The first half collapses via
    `pdiv_id`; the second factors as `(constant) * (sum)` and uses
    `pdiv_mul` + `pdiv_const` + `pdiv_finset_sum` + `pdiv_reindex` to
    yield `-1/n`. -/
theorem pdiv_bnCentered (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (bnCentered n) x i j =
      (if i = j then (1 : ℝ) else 0) - 1 / (n : ℝ) := by
  -- Step 1: rewrite bnCentered as `id + (-(∑ ·)/n)`.
  rw [show (bnCentered n : Vec n → Vec n) =
        (fun y k =>
          (fun (y' : Vec n) => y') y k +
          (fun (y' : Vec n) (_ : Fin n) => -((∑ s : Fin n, y' s) / (n : ℝ))) y k) from by
    funext y k
    unfold bnCentered bnMean
    ring]
  have h_id_diff : DifferentiableAt ℝ (fun y' : Vec n => y') x := differentiableAt_id
  have h_negMean_diff : DifferentiableAt ℝ
      (fun (y' : Vec n) (_ : Fin n) => -((∑ s : Fin n, y' s) / (n : ℝ))) x := by fun_prop
  rw [pdiv_add _ _ _ h_id_diff h_negMean_diff, pdiv_id]
  -- Step 2: factor the negMean term as (constant -1/n) * (sum).
  rw [show (fun (y' : Vec n) (_ : Fin n) => -((∑ s : Fin n, y' s) / (n : ℝ))) =
        (fun y' k =>
          (fun (_ : Vec n) (_ : Fin n) => -(1 / (n : ℝ))) y' k *
          (fun (z : Vec n) (_ : Fin n) => ∑ s : Fin n, z s) y' k) from by
    funext y' k
    ring]
  have h_neg_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (_ : Fin n) => -(1 / (n : ℝ))) x :=
    differentiableAt_const _
  have h_sum_diff : DifferentiableAt ℝ
      (fun (z : Vec n) (_ : Fin n) => ∑ s : Fin n, z s) x := by fun_prop
  rw [pdiv_mul _ _ _ h_neg_const_diff h_sum_diff]
  rw [show pdiv (fun (_ : Vec n) (_ : Fin n) => -(1 / (n : ℝ))) x i j = 0
      from pdiv_const (fun _ : Fin n => -(1 / (n : ℝ))) x i j]
  -- Step 3: pdiv of `∑ s, z s` via pdiv_finset_sum + pdiv_reindex.
  rw [show (fun (z : Vec n) (_ : Fin n) => ∑ s : Fin n, z s) =
        (fun z k => ∑ s : Fin n,
          (fun (z' : Vec n) (_ : Fin n) => z' s) z k) from rfl]
  have h_proj_diff : ∀ s ∈ (Finset.univ : Finset (Fin n)),
      DifferentiableAt ℝ (fun (z' : Vec n) (_ : Fin n) => z' s) x := by
    intro s _
    exact (reindexCLM (fun _ : Fin n => s)).differentiableAt
  rw [pdiv_finset_sum _ _ _ h_proj_diff]
  have h_term : ∀ s : Fin n,
      pdiv (fun (z' : Vec n) (_ : Fin n) => z' s) x i j =
        if i = s then (1 : ℝ) else 0 := by
    intro s
    rw [show (fun (z' : Vec n) (_ : Fin n) => z' s) =
          (fun z' => fun k' : Fin n => z' ((fun _ : Fin n => s) k')) from rfl]
    rw [pdiv_reindex (fun _ : Fin n => s)]
  simp_rw [h_term]
  rw [Finset.sum_ite_eq Finset.univ i (fun _ : Fin n => (1 : ℝ))]
  simp
  ring

/-- **Smoothness of `bnIstdBroadcast`** — axiomatized.

    `bnIstdBroadcast n ε x = 1/√(σ²(x) + ε)` is C^∞ when `ε > 0`
    (since `σ²(x) + ε ≥ ε > 0`, so `Real.sqrt` and reciprocal are
    smooth at every point of the domain). The Mathlib derivation
    via `Real.sqrt`'s `hasDerivAt` and `HasDerivAt.inv` is doable
    but not in scope here — we axiomatize alongside the existing
    `pdiv_bnIstdBroadcast` so `pdiv_mul` and `vjp_comp` calls in
    the BN chain can discharge their `Differentiable` hypotheses. -/
axiom bnIstdBroadcast_diff (n : Nat) (ε : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (bnIstdBroadcast n ε)

/-- **Broadcast inverse-stddev Jacobian** — axiomatized elementary fact.

    `∂istd(x,ε)/∂xᵢ = -istd³(x,ε) · (xᵢ - μ(x)) / n`

    Derivation sketch (standard calculus):
    - `istd = 1/√(σ²+ε)` → chain rule through `Real.sqrt` and `x ↦ 1/x`:
        `∂istd/∂σ² = -(1/2) · istd³`
    - `∂σ²/∂xᵢ = (2/n) · (xᵢ - μ)`  (product rule on `(xⱼ - μ)²` summed,
      using `Σⱼ (xⱼ - μ) = 0` to cancel a `(1 - 1/n)` factor)
    - Chain together: `∂istd/∂xᵢ = -istd³ · (xᵢ - μ) / n`.

    **Mathlib correspondence**: `Real.hasDerivAt_sqrt` + `HasDerivAt.inv`
    + product/sum rules on the inner `bnVar`. Axiomatized here to avoid
    the fderiv-bridge plumbing. -/
axiom pdiv_bnIstdBroadcast (n : Nat) (ε : ℝ) (x : Vec n) (i j : Fin n) :
    pdiv (bnIstdBroadcast n ε) x i j =
      -(bnIstd n x ε)^3 * (x i - bnMean n x) / (n : ℝ)

/-- **The BN normalize Jacobian — derived, no longer axiomatized.**

    `pdiv (bnNormalize n ε) x i j = (istd / n) · (n · δᵢⱼ − 1 − x̂ᵢ · x̂ⱼ)`

    Proof: factor `bnXhat = bnCentered · bnIstdBroadcast`, apply
    `pdiv_mul`, substitute the two elementary Jacobians, then expand
    `x̂ₖ = (xₖ - μ) · istd` and collapse with `ring`. -/
theorem pdiv_bnNormalize (n : Nat) (ε : ℝ) (hε : 0 < ε)
    (x : Vec n) (i j : Fin n) :
    pdiv (bnNormalize n ε) x i j =
      bnIstd n x ε / (n : ℝ) *
        ((n : ℝ) * (if i = j then 1 else 0) - 1 - bnXhat n ε x i * bnXhat n ε x j) := by
  -- Step 1: rewrite bnNormalize as the elementwise product of bnCentered and bnIstdBroadcast.
  have hfactor : bnNormalize n ε =
                 (fun y : Vec n => fun k : Fin n => bnCentered n y k * bnIstdBroadcast n ε y k) := by
    funext y
    exact bnXhat_eq_product n ε y
  rw [show bnNormalize n ε = bnNormalize n ε from rfl, hfactor]
  -- Step 2: apply pdiv_mul. Both factors are Differentiable: bnCentered is
  -- linear (proved via fun_prop), bnIstdBroadcast is smooth when ε > 0
  -- (axiomatized as bnIstdBroadcast_diff).
  have h_centered_diff : DifferentiableAt ℝ (bnCentered n) x := by
    unfold bnCentered bnMean; fun_prop
  have h_istd_diff : DifferentiableAt ℝ (bnIstdBroadcast n ε) x :=
    (bnIstdBroadcast_diff n ε hε) x
  rw [pdiv_mul _ _ _ h_centered_diff h_istd_diff]
  -- Step 3: substitute the two elementary Jacobians.
  rw [pdiv_bnCentered, pdiv_bnIstdBroadcast]
  -- Step 4: expand x̂ on the RHS and collapse with `ring`.
  -- Existence of `i : Fin n` gives us `n ≠ 0`, so `↑n · (↑n)⁻¹ = 1`.
  have hn : (n : ℝ) ≠ 0 := by
    have hpos : 0 < n := Nat.pos_of_ne_zero fun hz =>
      absurd i.isLt (by simp [hz])
    exact_mod_cast hpos.ne'
  unfold bnXhat bnIstdBroadcast bnCentered
  -- Both sides are now polynomial in (x_i - μ), (x_j - μ), istd, n (with `(↑n)⁻¹`).
  -- Handle the `if i = j` branches, then `field_simp` + `ring` closes both.
  by_cases hij : i = j
  · subst hij; simp; field_simp; ring
  · simp [hij]; field_simp; ring

/-- **Affine VJP** (the easy half): `back(v, dy)ᵢ = γ · dyᵢ`.

    Each input enters one output multiplied by `γ`; the gradient comes
    back scaled by `γ`. -/
noncomputable def bnAffine_has_vjp (n : Nat) (γ β : ℝ) :
    HasVJP (bnAffine n γ β) where
  backward := fun _v dy => fun i => γ * dy i
  correct := by
    intro x dy i
    simp [pdiv_bnAffine]

/-- **Normalize VJP** (the hard half): the consolidated formula with γ = 1.

    `back(x, dx̂)ᵢ = (1/N) · istd · (N · dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ · Σⱼ x̂ⱼ · dx̂ⱼ)` -/
noncomputable def bnNormalize_has_vjp (n : Nat) (ε : ℝ) (hε : 0 < ε) :
    HasVJP (bnNormalize n ε) where
  backward := fun x dxhat =>
    let xh := bnXhat n ε x
    let invN : ℝ := 1 / (n : ℝ)
    let s : ℝ := bnIstd n x ε
    let sumDx := ∑ j : Fin n, dxhat j
    let sumXhatDx := ∑ j : Fin n, xh j * dxhat j
    fun i =>
      invN * s * ((n : ℝ) * dxhat i - sumDx - xh i * sumXhatDx)
  correct := by
    intro x dxhat i
    simp_rw [pdiv_bnNormalize n ε hε x i]
    set s := bnIstd n x ε
    set xh := bnXhat n ε x
    -- LHS: (1/n) * s * (n * dxhat i - Σ dxhat - xh i * Σ(xh·dxhat))
    -- RHS: ∑ j, s/n * (n*δᵢⱼ - 1 - xh i * xh j) * dxhat j
    -- Step 1: rewrite each summand to separate the three contributions
    have hterm : ∀ j : Fin n,
        s / ↑n * (↑n * (if i = j then (1:ℝ) else 0) - 1 - xh i * xh j) * dxhat j =
        s / ↑n * (↑n * (if i = j then dxhat j else 0) - dxhat j - xh i * (xh j * dxhat j)) := by
      intro j
      by_cases h : i = j
      · subst h; simp; ring
      · simp [h]; ring
    simp_rw [hterm]
    -- Step 2: factor s/n out, distribute the sum
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_sub_distrib]
    -- Step 3: Kronecker delta
    rw [show ∑ j : Fin n, ↑n * (if i = j then dxhat j else 0) = ↑n * dxhat i from by
      simp [Finset.sum_ite_eq', Finset.mem_univ]]
    -- Step 4: factor xh i out
    rw [show ∑ j : Fin n, xh i * (xh j * dxhat j) =
        xh i * ∑ j : Fin n, xh j * dxhat j from by rw [← Finset.mul_sum]]
    ring

/-- **The BN VJP from the composition** — chain rule glues affine ∘ normalize.

    This is the structural payoff: once `bnNormalize_has_vjp` and
    `bnAffine_has_vjp` are in hand, the full BN input gradient comes
    from one application of `vjp_comp`. The chain rule mechanically
    threads `dy → dx̂ → dx`:

        dx̂ᵢ = γ · dyᵢ                           (from bnAffine_has_vjp)
        dxᵢ = (1/N · istd) · (N · dx̂ᵢ − …)     (from bnNormalize_has_vjp)

    The composition is exactly the two-step backward pass that the
    MLIR emits: lines 773 (`d_norm = grad * gamma_bc`) followed by
    lines 794–801 (the consolidated three-term formula).
-/
noncomputable def bn_has_vjp (n : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (bnForward n ε γ β) := by
  rw [bnForward_eq_compose]
  have h_normalize_diff : Differentiable ℝ (bnNormalize n ε) := by
    rw [show bnNormalize n ε =
          (fun y : Vec n => fun k : Fin n =>
            bnCentered n y k * bnIstdBroadcast n ε y k) from by
      funext y; exact bnXhat_eq_product n ε y]
    have h_centered : Differentiable ℝ (bnCentered n) := by
      have h_eq : (bnCentered n : Vec n → Vec n) =
                  fun x => fun j => x j - (∑ i, x i) * ((n : ℝ))⁻¹ := by
        funext x j; unfold bnCentered bnMean; ring
      rw [h_eq]; fun_prop
    exact h_centered.mul (bnIstdBroadcast_diff n ε hε)
  have h_affine_diff : Differentiable ℝ (bnAffine n γ β) := by
    unfold bnAffine; fun_prop
  exact vjp_comp (bnNormalize n ε) (bnAffine n γ β)
    h_normalize_diff h_affine_diff
    (bnNormalize_has_vjp n ε hε) (bnAffine_has_vjp n γ β)

/-- The standalone end-to-end theorem: `bn_grad_input` is the correct VJP
    of `bnForward`. Follows from `bn_has_vjp` by definitional unfolding. -/
theorem bn_input_grad_correct (n : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    bn_grad_input n ε γ x dy i =
    ∑ j : Fin n, pdiv (bnForward n ε γ β) x i j * dy j := by
  exact (bn_has_vjp n ε γ β hε).correct x dy i

end Proofs
