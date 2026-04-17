import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP

/-!
# CNN VJP Proofs

VJP correctness for the convolutional and pooling layers used in
`mlir_poc/hand_cnn_train_step.mlir`. The architecture there is:

    x(1,28,28) → Conv(1→32) → ReLU → Conv(32→32) → ReLU → MaxPool
              → Flatten → Dense(6272→512) → ReLU → Dense(512→512)
              → ReLU → Dense(512→10) → logits

The dense and ReLU layers are inherited from `MLP.lean`. This file
adds the new operations: conv2d, max-pool, and flatten.

## The big idea: conv backward IS conv

The most pedagogically valuable result is that the VJP of a convolution
is itself expressible as **convolutions** — with appropriate kernel
reversal and axis transposition. This is why conv layers train
efficiently: there's no special backward operator. The same primitive
runs in both directions.

Two specific tricks appear in the MLIR:

1. **Input-gradient via reversed kernel**: `dx = conv(dy, reverse(Wᵀ))`
2. **Weight-gradient via the transpose trick**: `dW = conv(xᵀ, dyᵀ)`,
   where the spatial dims of the gradient become the "kernel".

We state the VJP formulas as axioms (the proofs are standard matrix
calculus on cross-correlations), and the commentary explains why each
formula has the form it does.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Tensor types for CNN
-- ════════════════════════════════════════════════════════════════

-- Tensor3 is imported from Tensor.lean

/-- A conv kernel: out_channels × in_channels × kH × kW.
    This is the OIHW layout used by StableHLO and IREE. -/
abbrev Kernel4 (oc ic kh kw : Nat) :=
  Fin oc → Fin ic → Fin kh → Fin kw → ℝ

namespace Kernel4

/-! `Kernel4 oc ic kH kW` and `Vec (oc * ic * kH * kW)` are in bijection
    by row-major flattening — mirrors `Mat.flatten` / `Tensor3.flatten`.
    We need this so that the weight-gradient VJP can be stated as a plain
    `HasVJP` (Vec → Vec) on the flattened kernel, reusing the existing
    framework instead of introducing a parallel 4D machinery.

    Nat multiplication associates left, so `oc * ic * kH * kW` parses as
    `((oc * ic) * kH) * kW` — three nested `finProdFinEquiv` calls. -/

/-- Row-major flatten: `Kernel4 oc ic kH kW → Vec (oc * ic * kH * kW)`. -/
noncomputable def flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Vec (oc * ic * kH * kW) :=
  fun k =>
    let ockH_kW := finProdFinEquiv.symm k         -- : Fin (oc*ic*kH) × Fin kW
    let ocic_kH := finProdFinEquiv.symm ockH_kW.1 -- : Fin (oc*ic) × Fin kH
    let oc_ic   := finProdFinEquiv.symm ocic_kH.1 -- : Fin oc × Fin ic
    W oc_ic.1 oc_ic.2 ocic_kH.2 ockH_kW.2

/-- Row-major unflatten: inverse of `flatten`. -/
noncomputable def unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : Kernel4 oc ic kH kW :=
  fun o c kh kw =>
    v (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw))

theorem unflatten_flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : unflatten (flatten W) = W := by
  funext o c kh kw
  unfold unflatten flatten
  simp [Equiv.symm_apply_apply]

theorem flatten_unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : flatten (unflatten v) = v := by
  funext k
  -- After `change`, Lean's Prod struct-eta already collapses the innermost
  -- pair `(c.1, c.2)` to `c = fPF.symm (..).1`, so we start by collapsing
  -- that innermost `fPF (fPF.symm ...)` directly. Two more round-trips follow,
  -- each needing an explicit Prod-eta `show` + another `Equiv.apply_symm_apply`.
  change v (finProdFinEquiv
    (finProdFinEquiv
      (finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1),
       (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2),
     (finProdFinEquiv.symm k).2)) = v k
  rw [Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1,
            (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2) =
           finProdFinEquiv.symm (finProdFinEquiv.symm k).1 from rfl,
      Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2) =
           finProdFinEquiv.symm k from rfl,
      Equiv.apply_symm_apply]

end Kernel4

-- ════════════════════════════════════════════════════════════════
-- § Conv2d
-- ════════════════════════════════════════════════════════════════

/-- **Conv2d forward** (SAME padding, stride 1).

    `y[o, h, w] = (Σ_{c, kh, kw} x[c, h+kh−p, w+kw−p] · W[o, c, kh, kw]) + b[o]`

    where `p = (kH−1)/2` is the padding offset and out-of-bounds reads
    return 0 (zero padding). The output spatial size equals the input.

    Note: this is technically *cross-correlation*, not convolution in the
    strict signal-processing sense. ML literature uses "convolution" loosely;
    the difference (kernel flipping) only matters when comparing against
    classical signal-processing references.

    MLIR (`hand_cnn_train_step.mlir`):
      %cv0 = "stablehlo.convolution"(%x, %W0) {
        padding = dense<[[1, 1], [1, 1]]>, ...
      }
      %h0pre = stablehlo.add %cv0, broadcast(%b0) -/
axiom conv2d {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) : Tensor3 oc h w

/-- **Conv2d input-VJP** — one axiom bundling the backward function and its
    correctness.  A `HasVJP3` record carries both the backward function
    and a proof that it equals the `pdiv3`-contracted cotangent.

    The backward function (accessed as `(conv2d_has_vjp3 W b).backward`, or
    via the named `conv2d_input_grad` abbrev below) implements the standard
    "reversed-kernel, transposed-I/O" formula:

      `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] ·
                                   dy[o, h+kh−p, w+kw−p]`

    Two transformations on `W` make conv's backward *itself* a convolution:
    - **Reverse spatial dims** (`kh ↦ kH−1−kh`): conv backward "looks the
      other way" along the spatial axes — each output cell influenced each
      input cell at a *negated* offset.
    - **Swap I/O channels** (`c ↔ o`): the weight tensor is "transposed" so
      it now maps from the gradient (oc channels) back to the input (ic).

    MLIR emits exactly this structure:
      %W1_t   = stablehlo.transpose %W1, dims = [1, 0, 2, 3]   -- swap oc↔ic
      %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3]         -- flip spatial
      %d_h0   = "stablehlo.convolution"(%d_h1pre, %W1_rev) ...
-/
axiom conv2d_has_vjp3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP3 (conv2d W b : Tensor3 ic h w → Tensor3 oc h w)

/-- Named accessor for the conv2d input backward — aligns with MLIR
    codegen (`stablehlo.convolution` in the backward pass). -/
noncomputable abbrev conv2d_input_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  (conv2d_has_vjp3 W b).backward x dy

/-! ### Weight gradient (Phase 7 — now axiomatized)

The conv weight gradient implements the **transpose trick**:

    `dW[o, c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[o, h, w]`

Here's the slick observation: this *is* a convolution, with the input
and gradient playing the roles of "input" and "kernel" respectively.

- View the input `x : (ic, H, W)` as `(ic, 1, H, W)` (treat channels as batch).
- View the gradient `dy : (oc, H, W)` as `(oc, 1, H, W)` (same trick).
- Now do a standard convolution: input shape `(ic, 1, H, W)`, kernel
  shape `(oc, 1, H, W)`. The "spatial" dims of the kernel are H×W (the
  whole image), so the output is the small `(ic, oc, kH, kW)` weight
  gradient — produced by sliding the gradient as a giant kernel.
- Transpose the output `(ic, oc, kH, kW) → (oc, ic, kH, kW)` to match
  the kernel layout.

This avoids needing a separate "convolution-with-funny-dimension-numbers"
op; we use the same forward conv operator, just with shapes reinterpreted.
Critical for backends like IREE that don't accept non-standard
`dimension_numbers` (see `iree-org/iree#21955`).

MLIR (Conv 1 backward — exactly this trick):
    %x_t      = stablehlo.transpose %x, dims = [1, 0, 2, 3]      -- (1,128,28,28)
    %dh0p_t   = stablehlo.transpose %d_h0pre, dims = [1, 0, 2, 3] -- (32,128,28,28)
    %d_W0_raw = "stablehlo.convolution"(%x_t, %dh0p_t) ...        -- (1,32,3,3)
    %d_W0     = stablehlo.transpose %d_W0_raw, dims = [1, 0, 2, 3] -- (32,1,3,3)

**Framework.** `HasVJP3` covered only input→output VJPs. For the
weight gradient we reuse the plain `HasVJP` on `Vec` by flattening
both the kernel (`Kernel4.flatten : Kernel4 → Vec (oc*ic*kH*kW)`) and
the output (`Tensor3.flatten : Tensor3 → Vec (oc*h*w)`). The axiom
asserts existence of a correct backward for the flattened function;
the user-facing `conv2d_weight_grad` wrapper does the flatten / unflatten
housekeeping so callers see the natural `Kernel4` type.

Numerical validation: `check_axioms.py:test_conv2d_weight_grad`
gradient-checks the transpose-trick formula against finite differences. -/

/-- **Conv2d weight-VJP** — bundled axiom on the `W`-flattened function.

    View `fun W => conv2d W b x` through the Kernel4 ↔ Vec bijection and
    the output's Tensor3 ↔ Vec bijection; the resulting `Vec → Vec` function
    has a plain `HasVJP`. Its `.backward` is the transpose-trick formula
    documented above, gradient-checked numerically. -/
axiom conv2d_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v) b x))

/-- Named accessor for the conv2d weight backward — aligns with MLIR
    codegen (the "transpose trick" `stablehlo.convolution` in the backward
    pass). Unwraps the flattening so callers see `Kernel4 → Kernel4`. -/
noncomputable def conv2d_weight_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Kernel4 oc ic kH kW :=
  Kernel4.unflatten
    ((conv2d_weight_grad_has_vjp b x).backward
      (Kernel4.flatten W) (Tensor3.flatten dy))

/-- **Conv2d bias gradient** — sum the output cotangent over all spatial cells.

    `db[o] = Σ_{h, w} dy[o, h, w]`

    Each output cell adds the same `b[o]`, so its gradient accumulates
    the contributions from every spatial position.

    MLIR:
      %d_b1 = stablehlo.reduce(%d_h1pre) applies stablehlo.add
                across dimensions = [0, 2, 3]
    (dim 0 is batch, dims 2 and 3 are spatial.) -/
noncomputable def conv2d_bias_grad {oc h w : Nat} (dy : Tensor3 oc h w) : Vec oc :=
  fun o => ∑ y : Fin h, ∑ x : Fin w, dy o y x

-- ════════════════════════════════════════════════════════════════
-- § MaxPool
-- ════════════════════════════════════════════════════════════════

/-- **MaxPool 2×2 stride 2 forward**.

    Each output cell is the maximum of a 2×2 window of input cells:
    `y[c, h, w] = max{ x[c, 2h+a, 2w+b] : a, b ∈ {0,1} }`

    MLIR:
      %pool = "stablehlo.reduce_window"(%h1, %neginf) ({
        ^bb0(%a, %b): stablehlo.return (stablehlo.maximum %a, %b)
      }) {window_dimensions = [1, 1, 2, 2], window_strides = [1, 1, 2, 2]} -/
axiom maxPool2 {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) : Tensor3 c h w

/-- **MaxPool2 input-VJP** — gradient routes only to the argmax positions.

    The backward function implements:

      `dx[c, 2h+a, 2w+b] = dy[c, h, w] · 𝟙[(a,b) is the argmax of the window]`

    Conceptually, max-pool is a piecewise selection: each output is one
    specific input. So the Jacobian is a sparse 0/1 matrix and the VJP
    just routes the gradient to the chosen input.

    MLIR uses `stablehlo.select_and_scatter` to implement this directly:
      %d_h1 = "stablehlo.select_and_scatter"(%h1, %d_pool, %zf) ({
        -- selector: pick the GE element
        ^bb0(%a, %b): stablehlo.return (stablehlo.compare GE, %a, %b)
      }, {
        -- scatter: accumulate by addition (no overlap with stride = window)
        ^bb0(%a, %b): stablehlo.return (stablehlo.add %a, %b)
      }) {window_dimensions = [1,1,2,2], window_strides = [1,1,2,2]} -/
axiom maxPool2_has_vjp3 {c h w : Nat} :
    HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)

/-- Named accessor for the maxPool2 input backward — aligns with MLIR
    `stablehlo.select_and_scatter` in codegen. -/
noncomputable abbrev maxPool2_input_grad {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w) : Tensor3 c (2*h) (2*w) :=
  maxPool2_has_vjp3.backward x dy

-- ════════════════════════════════════════════════════════════════
-- § Flatten
-- ════════════════════════════════════════════════════════════════

/-! ## Reshape (flatten / unflatten)

Flatten is a permutation of indices, so its VJP is just the inverse
permutation. No gradient computation needed.

MLIR:
    %flat = stablehlo.reshape %pool
            : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>

The flatten / unflatten bijection is **already defined** in
`Tensor.lean` as `Tensor3.flatten` / `Tensor3.unflatten` (used by the
`pdiv3` derivation in Phase 5). We reuse those here rather than
duplicating — see `Tensor3.flatten_unflatten` / `unflatten_flatten`
for the mutual-inverse proofs. -/

-- ════════════════════════════════════════════════════════════════
-- § The full CNN backward pass
-- ════════════════════════════════════════════════════════════════

/-- **Walking through the CNN backward pass**.

    Unlike the MLP, where the chain rule (`vjp_comp`) gave us the whole
    backward pass in one go, here the layer types vary (Tensor3 ↔ Vec
    via flatten) so a uniform `HasVJP`-style composition would need a
    type family. For pedagogical clarity, we instead trace the backward
    pass step-by-step, matching `hand_cnn_train_step.mlir`.

    Forward:
        x ────conv W₀── h₀pre ──relu── h₀ ──conv W₁── h₁pre ──relu── h₁
          ──maxPool── pool ──flatten── d₀in ──dense W₂── d₀pre ──relu── d₀
          ──dense W₃── d₁pre ──relu── d₁ ──dense W₄── logits

    Backward (each step labeled with which lemma justifies it):

        d_logits  = softmax_ce_grad logits label                  [softmaxCE_grad]
        d_W4      = outer d₁ d_logits                             [dense_weight_grad]
        d_b4      = d_logits                                      [dense_bias_grad]
        d_d₁      = mulVec W₄ d_logits                            [dense_has_vjp]
        d_d₁pre   = relu_back d₁pre d_d₁                          [relu_has_vjp]
        d_W3      = outer d₀ d_d₁pre                              [dense_weight_grad]
        d_b3      = d_d₁pre                                       [dense_bias_grad]
        d_d₀      = mulVec W₃ d_d₁pre                             [dense_has_vjp]
        d_d₀pre   = relu_back d₀pre d_d₀                          [relu_has_vjp]
        d_W2      = outer d₀in d_d₀pre                            [dense_weight_grad]
        d_b2      = d_d₀pre                                       [dense_bias_grad]
        d_d₀in    = mulVec W₂ d_d₀pre                             [dense_has_vjp]
        d_pool    = unflatten d_d₀in                              [flatten VJP = unflatten]
        d_h₁      = maxPool2_input_grad h₁ d_pool                 [maxPool2_input_grad]
        d_h₁pre   = relu_back h₁pre d_h₁                          [relu_has_vjp, lifted to T3]
        d_W1      = conv2d_weight_grad W₁ b₁ h₀ d_h₁pre           [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b1      = conv2d_bias_grad d_h₁pre                      [conv2d_bias_grad]
        d_h₀      = conv2d_input_grad W₁ b₁ h₀ d_h₁pre            [conv2d_has_vjp3]     ← reversed kernel
        d_h₀pre   = relu_back h₀pre d_h₀                          [relu_has_vjp, lifted to T3]
        d_W0      = conv2d_weight_grad W₀ b₀ x d_h₀pre            [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b0      = conv2d_bias_grad d_h₀pre                      [conv2d_bias_grad]

    Each line of the backward pass corresponds to a single line in
    `hand_cnn_train_step.mlir` (lines 134–272). The backward pass is just
    the forward layers walked in reverse, replacing each forward operation
    with its VJP. The MLIR is the literal compiled-down version of this
    derivation.

    The novelty over the MLP is in the conv layers, where the VJP turns
    out to be — itself — a convolution, just with reversed/transposed
    kernels (`conv2d_input_grad`) or swapped axes (`conv2d_weight_grad`'s
    transpose trick). Once you accept those two tricks, the entire CNN
    backprop fits in a page.
-/
example : True := trivial  -- anchor for the docstring above

/-! ## Summary of axioms in this file

- `conv2d`, `maxPool2` — forward operations (black-box forward).
- `conv2d_has_vjp3`, `maxPool2_has_vjp3` — the input-path VJPs, each
  packaging both the backward function and its correctness into a
  single `HasVJP3` axiom.
- `conv2d_weight_grad_has_vjp` — Phase 7: the weight-path VJP, bundled
  as a plain `HasVJP` on the Kernel4-flattened function. Numerically
  gradient-checked against the transpose-trick formula in
  `check_axioms.py:test_conv2d_weight_grad`.

Derived (not axioms):
- `conv2d_input_grad`, `maxPool2_input_grad`, `conv2d_weight_grad` —
  named accessors, defined as `.backward` (plus flatten / unflatten
  housekeeping for `conv2d_weight_grad`) of the corresponding VJP.
- `conv2d_bias_grad` — concrete sum-over-spatial formula (no VJP
  framework claim; conv is opaque wrt `b`, so this is documented as
  the "correct formula" and cross-checked numerically rather than
  formally tied to a `pdiv`).
- 3D reshape (`Tensor3.flatten` / `Tensor3.unflatten`) imported from
  `Tensor.lean`; 4D reshape (`Kernel4.flatten` / `Kernel4.unflatten`)
  defined here, both proved bijections. -/

end Proofs
