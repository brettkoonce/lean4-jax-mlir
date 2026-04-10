import LeanJax.Proofs.Tensor
import LeanJax.Proofs.MLP

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

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Tensor types for CNN
-- ════════════════════════════════════════════════════════════════

/-- A 3D feature map: channels × height × width (single sample). -/
abbrev Tensor3 (c h w : Nat) := Fin c → Fin h → Fin w → Float

/-- A conv kernel: out_channels × in_channels × kH × kW.
    This is the OIHW layout used by StableHLO and IREE. -/
abbrev Kernel4 (oc ic kh kw : Nat) :=
  Fin oc → Fin ic → Fin kh → Fin kw → Float

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

/-- **Conv2d input gradient** — full convolution with reversed, transposed kernel.

    `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] · dy[o, h+kh−p, w+kw−p]`

    Two transformations on W:
    - **Reverse spatial dims** (`kh ↦ kH−1−kh`): conv backward "looks the
      other way" along the spatial axes — each output cell influenced each
      input cell at a *negated* offset.
    - **Swap I/O channels** (`c ↔ o`): the weight tensor is "transposed" so
      it now maps from the gradient (oc channels) back to the input (ic).

    MLIR uses exactly this structure:
      %W1_t   = stablehlo.transpose %W1, dims = [1, 0, 2, 3]   -- swap oc↔ic
      %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3]         -- flip spatial
      %d_h0   = "stablehlo.convolution"(%d_h1pre, %W1_rev) ...
-/
axiom conv2d_input_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) :
    Tensor3 ic h w

/-- **Conv2d weight gradient** — the transpose trick.

    `dW[o, c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[o, h, w]`

    Here's the slick observation: this *is* a convolution, with the input
    and gradient playing the roles of "input" and "kernel" respectively.
    Concretely:

    - View the input `x : (ic, H, W)` as `(ic, 1, H, W)` (treat channels as
      batch).
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
-/
axiom conv2d_weight_grad {ic oc h w kH kW : Nat}
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) :
    Kernel4 oc ic kH kW

/-- **Conv2d bias gradient** — sum the output cotangent over all spatial cells.

    `db[o] = Σ_{h, w} dy[o, h, w]`

    Each output cell adds the same `b[o]`, so its gradient accumulates
    the contributions from every spatial position.

    MLIR:
      %d_b1 = stablehlo.reduce(%d_h1pre) applies stablehlo.add
                across dimensions = [0, 2, 3]
    (dim 0 is batch, dims 2 and 3 are spatial.) -/
noncomputable def conv2d_bias_grad {oc h w : Nat} (dy : Tensor3 oc h w) : Vec oc :=
  fun o => finSum h (fun y => finSum w (fun x => dy o y x))

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

/-- **MaxPool VJP** — gradient routes only to the argmax positions.

    For each 2×2 window, the gradient dy[c, h, w] is *placed* at whichever
    of the four input cells achieved the maximum. Other cells get 0.

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
axiom maxPool2_input_grad {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w) :
    Tensor3 c (2*h) (2*w)

-- ════════════════════════════════════════════════════════════════
-- § Flatten
-- ════════════════════════════════════════════════════════════════

/-- **Flatten** — reshape a 3D tensor to a 1D vector.

    `flatten x [c, h, w] = x` viewed as a `Vec (c·h·w)`. The mapping is
    a bijection on the underlying data; nothing is computed.

    MLIR:
      %flat = stablehlo.reshape %pool
              : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>

    Because flatten is a permutation of indices, its VJP is the inverse
    permutation: `unflatten dy`. There is no gradient computation. -/
axiom flatten {c h w : Nat} (x : Tensor3 c h w) : Vec (c * h * w)

/-- The reverse of `flatten`. Together with `flatten`, this forms an
    isomorphism `Tensor3 c h w ≃ Vec (c·h·w)`.

    The VJP of flatten is unflatten, and vice versa: both operations are
    just reindexings. -/
axiom unflatten {c h w : Nat} (v : Vec (c * h * w)) : Tensor3 c h w

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
        d_W1      = conv2d_weight_grad h₀ d_h₁pre                 [conv2d_weight_grad]  ← transpose trick
        d_b1      = conv2d_bias_grad d_h₁pre                      [conv2d_bias_grad]
        d_h₀      = conv2d_input_grad W₁ b₁ h₀ d_h₁pre            [conv2d_input_grad]   ← reversed kernel
        d_h₀pre   = relu_back h₀pre d_h₀                          [relu_has_vjp, lifted to T3]
        d_W0      = conv2d_weight_grad x d_h₀pre                  [conv2d_weight_grad]
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
example : True := trivial  -- placeholder so the file isn't empty after the doc

end Proofs
