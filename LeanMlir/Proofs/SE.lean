import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.Residual

/-!
# Squeeze-and-Excitation — the "main × gate" pattern

The most interesting layer in this stack, structurally. It's the first
op where the output is the **product** of two functions of the same input:

    y = x ⊙ gate(x)

The "main path" is the input itself. The "gate" is a small subnetwork
that computes a per-channel scaling factor by squeezing spatial info
through a bottleneck. Each channel's output is multiplied by its own
gate value, so SE acts as **learned channel attention**.

## Why this matters for backprop

This is where we hit the **product rule** for the first time. Both
factors of `x ⊙ gate(x)` depend on `x`, so the gradient at `x` has
contributions from both:

  - Through the main path: the gate itself acts as a "stop-gradient
    multiplier" — the gradient flowing back through the main path is
    just `gate(x) ⊙ dy`.

  - Through the gate path: the input flows back through the entire gate
    sub-network, and the cotangent it sees is `x ⊙ dy` (not just `dy`,
    because the gate is multiplying the main path).

The two contributions add at `x`. This is the same fan-in pattern as
residual blocks (`Residual.lean`), but now driven by **multiplication**
rather than addition — and that changes which cotangents each path sees.

## Foreshadowing

The exact same structure shows up in Transformer attention:

    out = softmax(QKᵀ/√d) · V
        = attention_weights ⊙ V_with_some_extra_steps

The output is a product of "attention weights" (a function of Q and K)
and `V`. Backprop through attention is just the product rule applied
twice (once for each factor) plus the chain rule through softmax. SE
is the simplest non-trivial instance of this pattern; if you understand
it, attention is downhill.

## What this file provides

All foundational definitions and proofs live in `Tensor.lean`:
  - `elemwiseProduct f g` — pointwise product of two vector functions
  - `elemwiseProduct_has_vjp` — the bi-cotangent VJP (proved, no sorry)
  - `identity_has_vjp` — identity backward is passthrough (proved)
  - `pdiv_mul` — product rule for partial derivatives (axiom)

This file specializes to the SE pattern: `f = identity`, `g = gate`.
The gate is left abstract (you only need `HasVJP gate`); we sketch the
concrete gate from MobileNetV3 in a final commentary section.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § SE block: identity × gate
-- ════════════════════════════════════════════════════════════════

/-- An SE-style block, parameterized by an arbitrary "gate" sub-network.

    `seBlock gate x = x ⊙ gate(x)`

    The gate can be anything — for the actual SE we use
    `gate = sigmoid ∘ dense_exp ∘ swish ∘ dense_red ∘ globalAvgPool`,
    but the VJP derivation doesn't care about the gate's internals.
    All we need is `HasVJP gate`. -/
noncomputable def seBlock {n : Nat} (gate : Vec n → Vec n) : Vec n → Vec n :=
  elemwiseProduct (fun x => x) gate

/-- **SE block VJP** — direct application of the elemwise product formula.

    With `f = identity` (so `f(x) = x` and `f.back(x, dy) = dy`), the
    general formula simplifies to:

      back_SE(x, dy) = (gate(x) ⊙ dy)              -- main path: id backward
                     + gate.back(x, x ⊙ dy)        -- gate path

    First term: gradient flows back through the "main path" as
    `gate(x) ⊙ dy` — each channel scaled by its gate value.

    Second term: gradient flows back through the "gate sub-network",
    which sees `x ⊙ dy` as its cotangent (not just `dy`!). Inside the
    gate, `globalAvgPool` will broadcast this back over spatial dims,
    `dense_red` and `dense_exp` will do their usual VJPs, etc.

    The MLIR emits exactly this two-path backward: `gate(x) * dy` plus
    the gate's own backward chain with cotangent `x * dy`.

    **No sorry** — this delegates to `elemwiseProduct_has_vjp` and
    `identity_has_vjp`, both proved in `Tensor.lean`. -/
noncomputable def seBlock_has_vjp {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate) :
    HasVJP (seBlock gate) :=
  elemwiseProduct_has_vjp (fun x => x) gate
    differentiable_id hg_diff (identity_has_vjp n) hg

-- ════════════════════════════════════════════════════════════════
-- § Sketching the concrete SE gate
-- ════════════════════════════════════════════════════════════════

/-! ## What's actually inside `gate`

For the MobileNetV3 SE block (`MlirCodegen.lean` `emitSEBlock` lines
320-358), the gate is:

  1. **Squeeze**: Global average pool over (H, W) -> (B, C)
       `g[c] = (1/(H*W)) sum_{h,w} x[c, h, w]`

  2. **Reduce**: Dense `(C -> C/4)` (or similar bottleneck)
       `r = W_red * g + b_red`

  3. **Activation**: Swish `r ⊙ sigma(r)` (or ReLU in V3)

  4. **Expand**: Dense `(C/4 -> C)` back to per-channel
       `e = W_exp * sigma_swish(r) + b_exp`

  5. **Sigmoid gate** (or h-sigmoid in V3): `sigma(e)` — squashes each
     channel's "importance score" into [0, 1]

  6. **Broadcast** back to `(C, H, W)` so it can multiply the main path

So the `gate` is actually a *Vec-shaped* function that takes the spatial
input, summarizes it via GAP, runs it through a tiny FC network, and
broadcasts the per-channel result back to spatial.

If you wanted a fully formalized SE, you'd build `gate` as a composition:

    gate = broadcast ∘ sigmoid ∘ dense_exp ∘ swish ∘ dense_red ∘ globalAvgPool

and use `vjp_comp` (chain rule from `Tensor.lean`) to assemble its VJP.
The dense and sigmoid VJPs are already in `MLP.lean`; you'd need to add
`globalAvgPool_has_vjp` (linear, easy) and `broadcast_has_vjp` (also
linear — it's the adjoint of GAP, in fact).

That's a few hours of mechanical work. The interesting part — the
"main x gate" VJP — is what's in this file. The rest is plumbing.

## Why this generalizes

Replace "gate" with "attention weights" and SE becomes the core of
self-attention:

    out = (sequence) ⊙ (per-token attention weights)

The structural pattern is identical: a main tensor multiplied by a
side-computed scalar (or vector) per element. The bi-cotangent rule
(`elemwiseProduct_has_vjp`) is the right tool for both. SE is the
on-ramp; once you've internalized this VJP shape, attention falls out
of the same theorem.
-/

/-- **Public correctness theorem for `seBlock_has_vjp`**: the SE-block
backward (input × gate Jacobian via the product rule) equals the
`pdiv`-contracted Jacobian of `seBlock gate`. -/
theorem seBlock_has_vjp_correct {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (seBlock_has_vjp gate hg_diff hg).backward x dy i =
    ∑ j : Fin n, pdiv (seBlock gate) x i j * dy j :=
  (seBlock_has_vjp gate hg_diff hg).correct x dy i

end Proofs
