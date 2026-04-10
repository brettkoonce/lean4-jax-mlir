import LeanJax.Proofs.Tensor
import LeanJax.Proofs.Residual

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

  • Through the main path: the gate itself acts as a "stop-gradient
    multiplier" — the gradient flowing back through the main path is
    just `gate(x) ⊙ dy`.

  • Through the gate path: the input flows back through the entire gate
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

This file:
1. Adds `pdiv_mul` (product rule for partial derivatives) as an axiom.
2. Defines `elemwiseProduct f g x = f(x) ⊙ g(x)` — the abstract pattern.
3. Proves the bi-cotangent VJP for it: each path sees the **other**
   function's value Hadamard-multiplied with `dy`.
4. Specializes to SE: `f = identity`, `g = gate`. The gate is left
   abstract (you only need its `HasVJP`); we sketch the concrete gate
   from MobileNetV3 in a final commentary section.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Calculus axiom: product rule for partial derivatives
-- ════════════════════════════════════════════════════════════════

/-- **Product rule for partial derivatives** of an elementwise product.

    For `h(x)ⱼ = f(x)ⱼ · g(x)ⱼ`:

      ∂hⱼ/∂xᵢ = (∂fⱼ/∂xᵢ) · g(x)ⱼ + f(x)ⱼ · (∂gⱼ/∂xᵢ)

    Standard calculus; axiomatized because `pdiv` is itself axiomatized. -/
axiom pdiv_mul {n : Nat} (f g : Vec n → Vec n) (x : Vec n) (i j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j

-- ════════════════════════════════════════════════════════════════
-- § Elementwise product as a function
-- ════════════════════════════════════════════════════════════════

/-- Elementwise product of two vector-valued functions:
    `(elemwiseProduct f g)(x)ᵢ = f(x)ᵢ · g(x)ᵢ`

    Both `f` and `g` are `Vec n → Vec n`. -/
noncomputable def elemwiseProduct {n : Nat}
    (f g : Vec n → Vec n) : Vec n → Vec n :=
  fun x i => f x i * g x i

-- ════════════════════════════════════════════════════════════════
-- § The "main × gate" VJP — the centerpiece
-- ════════════════════════════════════════════════════════════════

/-- **Elementwise product VJP** — the magic formula for "f times g":

      back(x, dy) = f.back(x, g(x) ⊙ dy) + g.back(x, f(x) ⊙ dy)

    Read this carefully. Each backward path runs on the **full** `dy`,
    but Hadamard-multiplied with the **other** function's forward value:

      • `f`'s backward sees `g(x) ⊙ dy`. The "scale" applied to `f` in
        the forward pass becomes the "weight" on its cotangent in the
        backward pass.
      • `g`'s backward sees `f(x) ⊙ dy`. Symmetrically.

    The two resulting input cotangents are summed at `x` (because both
    paths fan in from the same input — same as residual blocks).

    **Proof sketch** (sorry'd; structure shown):

      back(x, dy)ᵢ
        = f.back(x, g(x)⊙dy)ᵢ + g.back(x, f(x)⊙dy)ᵢ          (defn)
        = Σⱼ ∂fⱼ/∂xᵢ · (g(x)ⱼ · dyⱼ)
        + Σⱼ ∂gⱼ/∂xᵢ · (f(x)ⱼ · dyⱼ)                          (hf, hg correct)
        = Σⱼ (∂fⱼ/∂xᵢ · g(x)ⱼ + f(x)ⱼ · ∂gⱼ/∂xᵢ) · dyⱼ        (combine, distrib)
        = Σⱼ ∂(f·g)ⱼ/∂xᵢ · dyⱼ                                (pdiv_mul)
        = RHS                                                  ✓
-/
noncomputable def elemwiseProduct_has_vjp {n : Nat}
    (f g : Vec n → Vec n) (hf : HasVJP f) (hg : HasVJP g) :
    HasVJP (elemwiseProduct f g) where
  backward := fun x dy =>
    -- f sees the gate's forward value ⊙ dy
    let dy_for_f : Vec n := fun j => g x j * dy j
    -- g sees the main path's forward value ⊙ dy
    let dy_for_g : Vec n := fun j => f x j * dy j
    fun i => hf.backward x dy_for_f i + hg.backward x dy_for_g i
  correct := by
    intro x dy i
    sorry

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
    the gate's own backward chain with cotangent `x * dy`. -/
noncomputable def seBlock_has_vjp {n : Nat}
    (gate : Vec n → Vec n) (hg : HasVJP gate) :
    HasVJP (seBlock gate) :=
  elemwiseProduct_has_vjp (fun x => x) gate (identity_has_vjp n) hg

-- ════════════════════════════════════════════════════════════════
-- § Sketching the concrete SE gate
-- ════════════════════════════════════════════════════════════════

/-! ## What's actually inside `gate`

For the MobileNetV3 SE block (`MlirCodegen.lean` `emitSEBlock` lines
320–358), the gate is:

  1. **Squeeze**: Global average pool over (H, W) → (B, C)
       `g[c] = (1/(H·W)) Σ_{h,w} x[c, h, w]`

  2. **Reduce**: Dense `(C → C/4)` (or similar bottleneck)
       `r = W_red · g + b_red`

  3. **Activation**: Swish `r ⊙ σ(r)` (or ReLU in V3)

  4. **Expand**: Dense `(C/4 → C)` back to per-channel
       `e = W_exp · σ_swish(r) + b_exp`

  5. **Sigmoid gate** (or h-sigmoid in V3): `σ(e)` — squashes each
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
"main × gate" VJP — is what's in this file. The rest is plumbing.

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

end Proofs
