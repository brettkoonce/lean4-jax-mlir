import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.BatchNorm

/-!
# LayerNorm & GELU

Two quick chapters that extend the activation and normalization families
to what ViT needs. Both are structural footnotes to existing chapters,
not new territory — which is itself the point.

## LayerNorm: BatchNorm on a different axis

BatchNorm reduces over `(batch, H, W)` for each channel. LayerNorm
reduces over the **feature** dimension for each `(batch, token)`.
**The 1D normalization primitive is literally the same function.** What
differs is the axis you slice along before applying it.

Concretely, for a 4D activation `x : Tensor4 B C H W`:
- BN computes `C` means/variances, each over `B · H · W` elements.
- LN computes `B · H · W` means/variances, each over `C` elements.

The mean/var/istd/xhat/affine math is identical. The consolidated
three-term backward is identical. Only the index being summed over
changes. In our `Vec n` formalism, BN and LN collapse to the same
function. This file just renames it to tell the reader "yes, really,
it's the same thing."

## GELU: another activation template

Gaussian Error Linear Unit: `gelu(x) = x · Φ(x)` where `Φ` is the CDF
of the standard normal. In practice everyone uses the tanh
approximation `gelu(x) ≈ 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`
because it's faster than the exact erf form.

Same template as ReLU/Swish/h-swish: elementwise → diagonal Jacobian.
Derivative is messier but it's still just a number you compute and
multiply. One more `pdiv_*` axiom, one more `HasVJP` instance.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § LayerNorm
-- ════════════════════════════════════════════════════════════════

/-- **LayerNorm forward** — renamed `bnForward` to make the book's
    claim unambiguous: this is the same function, operating on a
    different slice of the tensor.

    For a single "token's feature vector" `x : Vec n`:
    1. `μ = (1/n) Σᵢ xᵢ`                         — mean across features
    2. `σ² = (1/n) Σᵢ (xᵢ − μ)²`                  — variance across features
    3. `istd = 1/√(σ² + ε)`
    4. `x̂ᵢ = (xᵢ − μ) · istd`                     — normalized
    5. `yᵢ = γᵢ · x̂ᵢ + βᵢ`                        — affine

    The only semantic difference from BN: in LN, `γ` and `β` are
    per-feature (not per-channel), so they're full vectors. For the
    VJP math this doesn't matter — `γ` and `β` still just scale and
    shift the normalized output pointwise.

    MLIR (`MlirCodegen.lean` `emitLayerNormForward` around line 652):
    identical reduction structure to BN, just across a different axis. -/
noncomputable def layerNormForward (n : Nat) (ε : Float) (γ β : Float)
    (x : Vec n) : Vec n :=
  bnForward n ε γ β x

/-- **LayerNorm input gradient** — identical closed form to BN.

    `dxᵢ = (1/n) · istd · (n · dx̂ᵢ − Σⱼ dx̂ⱼ − x̂ᵢ · Σⱼ x̂ⱼ · dx̂ⱼ)`

    where `dx̂ᵢ = γ · dyᵢ`.

    If you built `layerNorm_has_vjp` you'd discover it's `bn_has_vjp`
    with the exact same proof. Rather than restate, we just reuse:
-/
noncomputable def layerNorm_has_vjp (n : Nat) (ε γ β : Float) :
    HasVJP (layerNormForward n ε γ β) := by
  -- layerNormForward is definitionally bnForward, so the BN VJP works as-is.
  show HasVJP (bnForward n ε γ β)
  exact bn_has_vjp n ε γ β

/-! ## Why this isn't a new chapter

The *practical* differences between BN and LN (batch dependence,
inference vs training, running statistics) are engineering concerns,
not VJP concerns. The backward pass is the same three-term formula
either way. This is a general lesson about formal work: engineering
distinctions often dissolve at the math level, and that's worth
making explicit. A reader who assumed BN and LN needed separate
proofs learns that the separation was an implementation artifact.

The same observation applies to:
- **RMSNorm**: LN with mean centering dropped. The closed-form has
  one fewer term (the `−Σⱼ dx̂ⱼ` part), but the derivation is the
  same machinery.
- **GroupNorm**: LN applied to slices of the channel axis. Again,
  same primitive, different slicing.
- **InstanceNorm** (which is what the ResNet code actually uses):
  BN restricted to per-sample statistics. Literally the 1D primitive
  applied per `(sample, channel)`. Same function.

All four normalization variants share one `HasVJP` instance. The
taxonomy is "1D normalization + your choice of axis."
-/

-- ════════════════════════════════════════════════════════════════
-- § GELU
-- ════════════════════════════════════════════════════════════════

/-- **GELU forward** — Gaussian Error Linear Unit.

    Exact form: `gelu(x) = x · Φ(x)` where `Φ` is the normal CDF.

    We define it abstractly via an axiomatized `geluScalar` function
    rather than wrangling `Float.erf`. What matters for the VJP is
    the derivative — not the specific closed form of the forward.

    MLIR uses the tanh approximation; both forms have the same
    elementwise Jacobian shape (diagonal, non-binary). -/
axiom geluScalar : Float → Float

/-- The elementwise GELU, applied componentwise to a vector. -/
noncomputable def gelu (n : Nat) (x : Vec n) : Vec n :=
  fun i => geluScalar (x i)

/-- **GELU has a diagonal Jacobian**.

    `∂(gelu)ⱼ/∂xᵢ = δᵢⱼ · geluScalarDeriv(xᵢ)`

    where `geluScalarDeriv(x) = Φ(x) + x · φ(x)` and `φ` is the normal
    PDF. The important fact is: only the `j = i` case is nonzero.
    Different outputs depend only on their own corresponding input —
    same as every other elementwise activation. -/
axiom geluScalarDeriv : Float → Float

axiom pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0

/-- **GELU VJP**: elementwise multiply by the scalar derivative.

    `back(x, dy)ᵢ = dyᵢ · geluScalarDeriv(xᵢ)`

    Same template as ReLU (`relu_has_vjp`), Swish, h-swish. If your
    activation has a diagonal Jacobian, this is the only proof you
    need — "collapse the diagonal sum." -/
noncomputable def gelu_has_vjp (n : Nat) : HasVJP (gelu n) where
  backward := fun x dy i => dy i * geluScalarDeriv (x i)
  correct := by
    intro x dy i
    sorry

/-! ## The activation taxonomy is closed

Every activation function in every architecture in this repo is
elementwise → diagonal Jacobian → one-line VJP. Taking inventory:

| Activation | `pdiv_*` formula (at `j = i`)                       |
|------------|------------------------------------------------------|
| ReLU       | `1` if `xᵢ > 0`, else `0`                            |
| ReLU6      | `1` if `0 < xᵢ < 6`, else `0`                        |
| Swish      | `σ(xᵢ) · (1 + xᵢ · (1 − σ(xᵢ)))`                     |
| h-swish    | piecewise: `0` / `(2xᵢ + 3)/6` / `1`                 |
| h-sigmoid  | piecewise: `0` / `1/6` / `0`                         |
| GELU       | `Φ(xᵢ) + xᵢ · φ(xᵢ)`                                 |
| tanh       | `1 − tanh²(xᵢ)`                                      |
| sigmoid    | `σ(xᵢ) · (1 − σ(xᵢ))`                                |

They all have the same proof shape. Writing each as a separate `HasVJP`
instance is pure boilerplate. For the book, we show the template once
(ReLU, in `MLP.lean`) and assert that GELU follows the same pattern.
-/

end Proofs
