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

This file builds on the proved foundations in `Tensor.lean`:
- `biPath f g` and `biPath_has_vjp` (additive fan-in, proved)
- `identity_has_vjp` (identity VJP, proved)
- `pdiv_add` and `pdiv_id` (calculus facts, proved from Mathlib's `fderiv`)

With those in hand, the residual definitions are one-liners — no sorry's.

1. Defines `residual f x = f x + x` via `biPath f id` and its VJP.
2. Defines `residualProj proj f x = proj x + f x` via `biPath proj f`
   and its VJP.
3. Comments on how this matches the ResNet skip connection in the
   MLIR (`MlirCodegen.lean` residual block emission).
-/

open Finset BigOperators

namespace Proofs

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
      `f.back(x, dy) + dy_skip`, where `dy_skip = dy` here.

    Proof: immediate from `biPath_has_vjp` and `identity_has_vjp`,
    both proved in `Tensor.lean`. -/
noncomputable def residual_has_vjp {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f) :
    HasVJP (residual f) :=
  biPath_has_vjp f (fun x => x) hf_diff differentiable_id hf (identity_has_vjp n)

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
    and both gradients accumulate into the same incoming-grad SSA.

    Proof: immediate from `biPath_has_vjp`, proved in `Tensor.lean`. -/
noncomputable def residualProj_has_vjp {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f) :
    HasVJP (residualProj proj f) :=
  biPath_has_vjp proj f hproj_diff hf_diff hproj hf

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

/-- **Public correctness theorem for `residual_has_vjp`**: skip-connection
backward equals the `pdiv`-contracted Jacobian of `f + id`. -/
theorem residual_has_vjp_correct {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (residual_has_vjp f hf_diff hf).backward x dy i =
    ∑ j : Fin n, pdiv (residual f) x i j * dy j :=
  (residual_has_vjp f hf_diff hf).correct x dy i

/-- **Public correctness theorem for `residualProj_has_vjp`**: same as
`residual_has_vjp_correct` but for the projected variant where the skip
isn't identity. -/
theorem residualProj_has_vjp_correct {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (residualProj_has_vjp proj f hproj_diff hf_diff hproj hf).backward x dy i =
    ∑ j : Fin n, pdiv (residualProj proj f) x i j * dy j :=
  (residualProj_has_vjp proj f hproj_diff hf_diff hproj hf).correct x dy i

end Proofs
