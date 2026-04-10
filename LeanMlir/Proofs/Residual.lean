import LeanMlir.Proofs.Tensor

/-!
# Residual Connections — Gradient Accumulation

The first chapter where backprop has to **accumulate** gradients from
multiple paths into the same input. So far every layer has been a
straight-line composition (chain rule), but residual blocks introduce
fan-out: one input feeds two paths whose outputs are added.

The math is trivial — it's the *pattern* that matters. Once you see
"two backwards add", you'll see it everywhere: residuals, attention,
SE blocks, multi-head outputs, anywhere a tensor is consumed by more
than one downstream op.

This file:
1. Adds `pdiv_add` (linearity of partial derivatives) as an axiom.
2. Defines `biPath f g x = f x + g x` and proves its VJP.
3. Specializes to `residual f x = f x + x` and the projected variant
   `residualProj proj f x = proj x + f x`.
4. Comments on how this matches the ResNet skip connection in the
   MLIR (`MlirCodegen.lean` residual block emission).
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Calculus axiom: derivatives are linear
-- ════════════════════════════════════════════════════════════════

/-- **Linearity of partial derivatives**: ∂(f + g)ⱼ/∂xᵢ = ∂fⱼ/∂xᵢ + ∂gⱼ/∂xᵢ

    This is the additive half of "differentiation is a linear operation."
    A standard fact of real analysis; we take it as an axiom because
    `pdiv` is itself axiomatized. -/
axiom pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j

-- ════════════════════════════════════════════════════════════════
-- § The bi-path VJP: y = f(x) + g(x)
-- ════════════════════════════════════════════════════════════════

/-- Two functions added pointwise: `(biPath f g)(x)ᵢ = f(x)ᵢ + g(x)ᵢ`. -/
noncomputable def biPath {m n : Nat} (f g : Vec m → Vec n) : Vec m → Vec n :=
  fun x i => f x i + g x i

/-- **Bi-path VJP**: backward gradients from two parallel paths add.

    `back_biPath(x, dy) = f.back(x, dy) + g.back(x, dy)`

    The cotangent `dy` is sent backward through **both** paths, and the
    resulting input cotangents are summed. Each path "sees" the full `dy`
    (no splitting) — this is a consequence of the linearity of derivatives.

    **Proof sketch** (sorry'd, structure shown):

      back_biPath(x, dy)ᵢ = f.back(x, dy)ᵢ + g.back(x, dy)ᵢ          (defn)
                          = Σⱼ ∂fⱼ/∂xᵢ · dyⱼ + Σⱼ ∂gⱼ/∂xᵢ · dyⱼ      (hf, hg correct)
                          = Σⱼ (∂fⱼ/∂xᵢ + ∂gⱼ/∂xᵢ) · dyⱼ              (finSum_add, distrib)
                          = Σⱼ ∂(f + g)ⱼ/∂xᵢ · dyⱼ                    (pdiv_add)
                          = RHS                                         ✓
-/
noncomputable def biPath_has_vjp {m n : Nat}
    (f g : Vec m → Vec n) (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (biPath f g) where
  backward := fun x dy i => hf.backward x dy i + hg.backward x dy i
  correct := by
    intro x dy i
    sorry

-- ════════════════════════════════════════════════════════════════
-- § Identity has a (trivial) VJP
-- ════════════════════════════════════════════════════════════════

/-- ∂(id)ⱼ/∂xᵢ = δᵢⱼ — the identity Jacobian is, well, the identity matrix. -/
axiom pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0

/-- **Identity VJP**: gradient passes through unchanged.

    `back_id(x, dy) = dy`

    Trivial but worth stating: it's the base case for the residual VJP. -/
def identity_has_vjp (n : Nat) : HasVJP (fun (x : Vec n) => x) where
  backward := fun _x dy => dy
  correct := by
    intro x dy i
    -- dy i = Σⱼ (if i = j then 1 else 0) · dyⱼ = dyᵢ (only j = i contributes)
    sorry

-- ════════════════════════════════════════════════════════════════
-- § Residual block: y = f(x) + x
-- ════════════════════════════════════════════════════════════════

/-- A basic residual block: output = sub-network output + identity.

    `residual f x = f(x) + x`

    The "skip connection" lets gradients flow directly from output back
    to input without going through `f`. This is why ResNets train: even
    if `f` has near-zero gradients (vanishing), the identity path keeps
    the signal alive. -/
noncomputable def residual {n : Nat} (f : Vec n → Vec n) : Vec n → Vec n :=
  biPath f (fun x => x)

/-- **Residual VJP**: `dx = f.back(x, dy) + dy`.

    The skip's contribution is just `dy` (identity backward). The block's
    contribution is `f.back(x, dy)`. They add. This is **why** ResNets
    are easier to train: the gradient floor is `dy` itself, so it can
    never get smaller than the loss gradient at this layer.

    MLIR (`MlirCodegen.lean` residual block backward, around line 1107):
      The "skip grad" is added to the first convBn of the block — exactly
      `f.back(x, dy) + dy_skip`, where `dy_skip = dy` here. -/
noncomputable def residual_has_vjp {n : Nat}
    (f : Vec n → Vec n) (hf : HasVJP f) :
    HasVJP (residual f) :=
  biPath_has_vjp f (fun x => x) hf (identity_has_vjp n)

-- ════════════════════════════════════════════════════════════════
-- § Projected residual: y = proj(x) + f(x)
-- ════════════════════════════════════════════════════════════════

/-- Projected residual block: when input and output shapes don't match
    (e.g. when stride > 1 downsamples), the skip is not identity but a
    1×1 projection conv.

    `residualProj proj f x = proj(x) + f(x)`

    Both paths now have nontrivial backwards. The gradient still adds at
    the input — neither path is privileged. -/
noncomputable def residualProj {m n : Nat}
    (proj f : Vec m → Vec n) : Vec m → Vec n :=
  biPath proj f

/-- **Projected residual VJP**: `dx = proj.back(x, dy) + f.back(x, dy)`.

    Both backwards run on the same `dy` and their results sum at `x`.
    This is the truly general "fan-out → backward fan-in" pattern.

    MLIR: ResNets with stride > 1 use this — see `emitConvBnBackward`
    where the projection's VJP is emitted alongside the main block's,
    and both gradients accumulate into the same incoming-grad SSA. -/
noncomputable def residualProj_has_vjp {m n : Nat}
    (proj f : Vec m → Vec n) (hproj : HasVJP proj) (hf : HasVJP f) :
    HasVJP (residualProj proj f) :=
  biPath_has_vjp proj f hproj hf

-- ════════════════════════════════════════════════════════════════
-- § The pattern, in plain English
-- ════════════════════════════════════════════════════════════════

/-! ## Why this matters beyond ResNets

The fan-out/backward-add pattern is the **structural building block** for
every modern architecture:

  • **ResNets** — `y = f(x) + x` (this file).
  • **DenseNets** — `y = concat(f(x), x)`. The concat splits dy and each
    half goes back through its respective path. Same pattern, different
    glue (split instead of add).
  • **Squeeze-and-Excitation** — `y = x · gate(x)`. The product rule
    introduces a different kind of bi-path: the gate's gradient gets
    `x ⊙ dy` and the main path's gradient gets `gate(x) ⊙ dy`. See
    `SE.lean` for that derivation.
  • **Multi-head attention** — concatenated heads. Same structure.
  • **Two-tower models** — independent encoders → joint loss. Even more
    extreme fan-out.

If you understand `biPath_has_vjp`, you understand backprop through any
DAG. Composition (chain rule) handles the "linear" part; bi-path handles
the joins. Together they're enough for any computation graph.
-/

end Proofs
