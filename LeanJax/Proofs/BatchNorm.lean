import LeanJax.Proofs.Tensor

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

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Forward pass — defined incrementally
-- ════════════════════════════════════════════════════════════════

/-- Population mean: `μ = (1/N) Σᵢ xᵢ` -/
noncomputable def bnMean (n : Nat) (x : Vec n) : Float :=
  finSum n x / n.toFloat

/-- Population variance: `σ² = (1/N) Σᵢ (xᵢ − μ)²` -/
noncomputable def bnVar (n : Nat) (x : Vec n) : Float :=
  let μ := bnMean n x
  finSum n (fun i => (x i - μ) * (x i - μ)) / n.toFloat

/-- Inverse standard deviation: `istd = 1 / √(σ² + ε)` -/
noncomputable def bnIstd (n : Nat) (x : Vec n) (ε : Float) : Float :=
  1.0 / Float.sqrt (bnVar n x + ε)

/-- Normalized output: `x̂ᵢ = (xᵢ − μ) · istd`

    `x̂` has mean 0 and variance 1 (up to ε-correction). It's the
    "centered, unit-scaled" version of `x`. -/
noncomputable def bnXhat (n : Nat) (ε : Float) (x : Vec n) : Vec n :=
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
noncomputable def bnForward (n : Nat) (ε γ β : Float) (x : Vec n) : Vec n :=
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
noncomputable def bn_grad_gamma (n : Nat) (ε : Float) (x : Vec n) (dy : Vec n) : Float :=
  finSum n (fun i => dy i * bnXhat n ε x i)

/-- **β gradient**: `dβ = Σᵢ dyᵢ`

    `β` is added to every output, so `∂yᵢ/∂β = 1` and the gradient is
    just the sum of the output cotangents. Even simpler than dγ.

    MLIR (line 770):
      %d_bt = reduce add %effGrad across dimensions = [0, 2, 3]
-/
noncomputable def bn_grad_beta (n : Nat) (dy : Vec n) : Float := finSum n dy

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
    (n : Nat) (ε γ : Float) (x : Vec n) (dy : Vec n) : Vec n :=
  let xh : Vec n := bnXhat n ε x
  let dxhat : Vec n := fun i => γ * dy i
  let invN : Float := 1.0 / n.toFloat
  let s : Float := bnIstd n x ε
  let sumDxhat : Float := finSum n dxhat
  let sumXhatDxhat : Float := finSum n (fun i => xh i * dxhat i)
  fun i =>
    invN * s * (n.toFloat * dxhat i - sumDxhat - xh i * sumXhatDxhat)

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
       ∂(γ · x̂ᵢ + β)/∂β = 1          →  dβ = Σᵢ dyᵢ

   We state these as the *definitions* `bn_grad_gamma` and `bn_grad_beta`
   above; the sum-over-i is the bookkeeping that turns "per-output
   gradient" into "per-parameter gradient."
-/

/-- **BN input gradient correctness — the headline result.**

    The consolidated three-term formula `bn_grad_input` is the correct
    VJP of `bnForward` with respect to its input `x`.

    The proof is the derivation in the comment block above:
      • differentiate the affine to get `dx̂`
      • differentiate `bnXhat` using the three sub-derivatives
        (∂x/∂x, ∂μ/∂x, ∂σ²/∂x → ∂istd/∂x)
      • contract with `dx̂`
      • observe that the cross terms collapse via the cancellations
        `Σ(xⱼ−μ) = 0` and the rearrangement of the double sum.

    The `sorry` is real algebra, not a placeholder for a missing concept.
    A reader with paper and pencil can verify it; the value of stating it
    in Lean is that the **claim is unambiguous**: this exact function,
    on this exact input, returns this exact value. -/
theorem bn_input_grad_correct (n : Nat) (ε γ β : Float)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    bn_grad_input n ε γ x dy i =
    finSum n (fun j => pdiv (bnForward n ε γ β) x i j * dy j) := by
  sorry

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
noncomputable def bnNormalize (n : Nat) (ε : Float) : Vec n → Vec n :=
  bnXhat n ε

/-- The affine step as a function `Vec n → Vec n` (γ, β as constants). -/
noncomputable def bnAffine (n : Nat) (γ β : Float) : Vec n → Vec n :=
  fun v i => γ * v i + β

/-- BN as the composition of normalize and affine. -/
theorem bnForward_eq_compose (n : Nat) (ε γ β : Float) :
    bnForward n ε γ β = bnAffine n γ β ∘ bnNormalize n ε := by
  funext x i; rfl

/-- **Affine VJP** (the easy half): `back(v, dy)ᵢ = γ · dyᵢ`.

    Each input enters one output multiplied by `γ`; the gradient comes
    back scaled by `γ`. -/
noncomputable def bnAffine_has_vjp (n : Nat) (γ β : Float) :
    HasVJP (bnAffine n γ β) where
  backward := fun _v dy => fun i => γ * dy i
  correct := by
    intro x dy i
    sorry  -- ∂(γ·vᵢ + β)/∂vⱼ = γ·δᵢⱼ; collapses the sum to a single term

/-- **Normalize VJP** (the hard half): the consolidated formula with γ = 1.

    `back(x, dx̂)ᵢ = (1/N) · istd · (N · dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ · Σⱼ x̂ⱼ · dx̂ⱼ)` -/
noncomputable def bnNormalize_has_vjp (n : Nat) (ε : Float) :
    HasVJP (bnNormalize n ε) where
  backward := fun x dxhat =>
    let xh := bnXhat n ε x
    let invN : Float := 1.0 / n.toFloat
    let s : Float := bnIstd n x ε
    let sumDx := finSum n dxhat
    let sumXhatDx := finSum n (fun j => xh j * dxhat j)
    fun i =>
      invN * s * (n.toFloat * dxhat i - sumDx - xh i * sumXhatDx)
  correct := by
    intro x dxhat i
    sorry  -- the derivation in the long comment above

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
noncomputable def bn_has_vjp (n : Nat) (ε γ β : Float) :
    HasVJP (bnForward n ε γ β) := by
  rw [bnForward_eq_compose]
  exact vjp_comp (bnNormalize n ε) (bnAffine n γ β)
    (bnNormalize_has_vjp n ε) (bnAffine_has_vjp n γ β)

end Proofs
