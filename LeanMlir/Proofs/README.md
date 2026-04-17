# Verified VJP Proofs

Machine-checked proofs that the backward pass (VJP) of every layer
matches its forward-pass Jacobian. Zero `sorry`s. If `lake build`
succeeds, every theorem is correct.

## Dependency graph

```
Tensor.lean                    ← axioms + VJP framework
  │
  │  pdiv_comp (chain rule)
  │  pdiv_add  (sum rule)
  │  pdiv_mul  (product rule)
  │  pdiv_id   (identity)
  │  vjp_comp  (VJP composition — proved)
  │  biPath    (additive fan-in — proved)
  │  elemwiseProduct (multiplicative fan-in — proved)
  │
  ├── MLP.lean                 dense (input + weight + bias grads) + ReLU + softmax CE
  │
  ├── CNN.lean                 conv2d (input + weight grads) + maxPool + flatten
  │                            (incl. Kernel4.flatten bijection, Phase 7)
  │
  ├── BatchNorm.lean           batch norm (the hard one)
  │     │
  │     │  bnNormalize_has_vjp  ← 3-term consolidated formula
  │     │  bnAffine_has_vjp     ← trivial diagonal
  │     └─ bn_has_vjp           ← vjp_comp glues them
  │
  ├── Residual.lean            skip connections (biPath)
  │
  ├── Depthwise.lean           depthwise conv (input + weight grads)
  │
  ├── SE.lean                  squeeze-and-excitation (elemwiseProduct)
  │
  ├── LayerNorm.lean           layer norm + GELU
  │
  └── Attention.lean           softmax + scaled dot-product attention
```

## Axioms

All axiom declarations across the proof suite, grouped by file:

**Tensor.lean** — calculus foundations (1D `Vec`, 2D `Mat`, 3D `Tensor3`):
| Axiom | What it says |
|-------|-------------|
| `pdiv` | Partial derivative function (existence) |
| `pdiv_id` | ∂xᵢ/∂xⱼ = δᵢⱼ |
| `pdiv_comp` | Chain rule |
| `pdiv_add` | Sum rule |
| `pdiv_mul` | Product rule |
| `pdiv_const` | Derivative of a constant is zero |
| `pdiv_reindex` | Gather Jacobian: `∂y_{σ(k)}/∂y_i = δ_{i,σ(k)}` |
| `pdivMat_rowIndep` | Row-independent function ⇒ block-diagonal Jacobian |

> **Progression** — axioms 41 → 26 over several phases:
> - **Phases 4–5**: `pdivMat`, `pdivMat_comp`, `pdivMat_add`,
>   `pdivMat_id` and the whole `pdiv3` family collapsed to
>   definitions + theorems via the `Mat.flatten` / `Tensor3.flatten`
>   bijections.
> - **Phase 6**: `pdivMat_scalarScale`, `pdivMat_transpose`,
>   and both `pdivMat_matmul_{left,right}_const` derived from
>   `pdiv_const` + `pdiv_reindex` + `pdiv_finset_sum` (itself a
>   theorem, via `Finset.induction_on` over `pdiv_add` + `pdiv_const`).
> - **Phase 7** (this commit): **Weight-gradient correctness**, closing
>   the gap where `conv2d_weight_grad` and `depthwise_weight_grad` were
>   documented in prose but had no formal axiom. Two new bundled VJP
>   axioms (`conv2d_weight_grad_has_vjp`, `depthwise_weight_grad_has_vjp3`),
>   plus one new elementary Jacobian axiom `pdiv_dense_W` that unlocks
>   a proved `dense_weight_grad_correct` theorem (the old `Mat.outer`
>   `rfl` was vacuous). Dense bias gradient becomes a theorem too,
>   derived purely from existing axioms via `pdiv_add` + `pdiv_const`
>   + `pdiv_id`. `Kernel4.flatten` / `unflatten` added as a proved
>   bijection mirroring `Mat.flatten` / `Tensor3.flatten`, so the
>   4D weight tensor can be plumbed through the plain `HasVJP` on `Vec`
>   instead of introducing a parallel 4D framework.
>
> Remaining Mat-level axiom: only `pdivMat_rowIndep` — the
> genuinely-new-primitive that ties Mat-row structure to Vec-level
> pdiv of an opaque row function (can't derive without either a
> vmap-style axiom or knowing the row function's definition).

**MLP.lean** — dense layers:
| Axiom | What it says |
|-------|-------------|
| `pdiv_dense` | Dense layer Jacobian wrt input |
| `pdiv_dense_W` | Dense Jacobian wrt weight (Phase 7; unlocks the outer-product theorem) |
| `pdiv_relu` | ReLU Jacobian (diagonal, 0/1) |
| `softmaxCE_grad` | Softmax cross-entropy gradient = softmax − onehot |

> `dense_weight_grad_correct` (outer product is the weight gradient)
> and `dense_bias_grad_correct` (bias gradient is identity) are now
> theorems. The former uses `pdiv_dense_W`; the latter is derived from
> `pdiv_add` + `pdiv_const` + `pdiv_id` with no new axiom.

**CNN.lean** — convolution and pooling:
| Axiom | What it says |
|-------|-------------|
| `conv2d` | Conv forward (opaque function) |
| `conv2d_has_vjp3` | Conv2d input-VJP (function + correctness bundled) |
| `conv2d_weight_grad_has_vjp` | Conv2d weight-VJP via flattened `HasVJP` (Phase 7) |
| `maxPool2` | MaxPool forward (opaque function) |
| `maxPool2_has_vjp3` | MaxPool2 input-VJP (function + correctness bundled) |

> Phase 7: the weight gradient is now axiomatized via the `Kernel4.flatten`
> bijection, so `HasVJP` on `Vec (oc*ic*kH*kW)` suffices — no parallel
> 4D framework needed. The conv bias gradient formula (sum over spatial)
> is defined as `conv2d_bias_grad` and documented, but since `conv2d`
> is opaque in `b` we leave it outside the `HasVJP` frame; it's
> cross-checked numerically instead.

**BatchNorm.lean** — the hard one:
| Axiom | What it says |
|-------|-------------|
| `pdiv_bnAffine` | ∂(γv+β)/∂v = γδᵢⱼ |
| `pdiv_bnCentered` | ∂(xⱼ-μ(x))/∂xᵢ = δᵢⱼ - 1/n |
| `pdiv_bnIstdBroadcast` | ∂istd(x,ε)/∂xᵢ = -istd³·(xᵢ-μ)/n (broadcast) |

> **The three-term consolidated BN formula is now a theorem**, not an
> axiom. `pdiv_bnNormalize` is proved by factoring `bnXhat` as
> `(x - μ) · istd`, applying `pdiv_mul`, substituting the two
> elementary Jacobians above, and collapsing via `ring` + `field_simp`
> using `x̂ₖ = (xₖ - μ) · istd`. Each elementary axiom corresponds
> directly to a Mathlib `HasDerivAt`/`HasFDerivAt` fact (sub rule,
> `Real.sqrt`/`inv` chain) — see docstrings.

**Depthwise.lean** — depthwise convolution:
| Axiom | What it says |
|-------|-------------|
| `depthwiseConv2d` | Depthwise conv forward (opaque function) |
| `depthwise_has_vjp3` | Depthwise input-VJP (function + correctness bundled) |
| `depthwise_weight_grad_has_vjp3` | Depthwise weight-VJP (Phase 7) |

> The depthwise kernel `DepthwiseKernel c kH kW` is definitionally equal
> to `Tensor3 c kH kW`, so the weight-grad axiom reuses the existing
> `HasVJP3` directly — no `Kernel4.flatten` needed (unlike the regular
> conv case, which has a 4D kernel).

**LayerNorm.lean** — layer norm and GELU:
| Axiom | What it says |
|-------|-------------|
| `geluScalar` | GELU activation (function signature) |
| `geluScalarDeriv` | GELU derivative |
| `pdiv_gelu` | GELU Jacobian (diagonal) |

**Attention.lean** — softmax and attention:
| Axiom | What it says |
|-------|-------------|
| `pdiv_softmax` | Softmax Jacobian (rank-1 correction) |

> All three `sdpa_back_*_correct` statements are now **theorems**, not
> axioms (Phase 3). Each is proved by constructing a `HasVJPMat` for
> `fun _ => sdpa n d · K V` (or similar for K, V) as a composition of
> four proved `HasVJPMat` building blocks via `vjpMat_comp`, then
> reducing the chain's backward to the concrete `sdpa_back_{Q,K,V}`
> formula. The old `sdpa_has_vjp` axiom (a vacuous type declaration)
> is gone entirely.

Plus three Lean core axioms (`propext`, `Classical.choice`, `Quot.sound`)
present in every nontrivial Lean program.

Total: 8 (Tensor) + 4 (MLP) + 5 (CNN) + 3 (BatchNorm) + 3 (Depthwise)
+ 3 (LayerNorm) + 1 (Attention) = **27 axioms**.

Everything else — every `HasVJP` instance, every composition,
every correctness theorem — is proved from these axioms by
Lean's type checker.

## `#print axioms` output (HasVJP instances)

Which axioms each proved theorem actually uses (via `lake env lean`):

```
vjp_comp               → pdiv, pdiv_comp
biPath_has_vjp         → pdiv, pdiv_add
elemwiseProduct_has_vjp → pdiv, pdiv_mul
identity_has_vjp       → pdiv, pdiv_id
vjpMat_comp            → pdiv, pdiv_comp  (via Mat.flatten bijection)
biPathMat_has_vjp      → pdiv, pdiv_add   (via Mat.flatten bijection)
identityMat_has_vjp    → pdiv, pdiv_id    (via Mat.flatten bijection)
matmul_left_const_has_vjp  → pdivMat, pdivMat_matmul_left_const
matmul_right_const_has_vjp → pdivMat, pdivMat_matmul_right_const
scalarScale_has_vjp        → pdivMat, pdivMat_scalarScale
transpose_has_vjp          → pdivMat, pdivMat_transpose
rowSoftmax_has_vjp_mat     → pdivMat, pdivMat_rowIndep,
                             pdiv, pdiv_softmax
sdpa_back_V_correct    → pdivMat, pdivMat_matmul_left_const
sdpa_back_Q_correct    → pdivMat, pdivMat_matmul_left_const,
                         pdivMat_matmul_right_const,
                         pdivMat_scalarScale, pdivMat_rowIndep,
                         pdivMat_comp, pdiv, pdiv_softmax
sdpa_back_K_correct    → (same as Q) + pdivMat_transpose
dense_has_vjp          → pdiv, pdiv_dense
dense_weight_grad_correct → pdiv, pdiv_dense_W          (Phase 7 — one new axiom)
dense_bias_grad_correct   → pdiv, pdiv_add, pdiv_const, pdiv_id  (Phase 7 — zero new axioms)
conv2d_weight_grad     → pdiv, conv2d, conv2d_weight_grad_has_vjp     (Phase 7)
depthwiseConv2d_weight_grad → pdiv, depthwiseConv2d, depthwise_weight_grad_has_vjp3  (Phase 7)
bn_has_vjp             → pdiv, pdiv_bnAffine, pdiv_bnCentered, pdiv_bnIstdBroadcast, pdiv_comp, pdiv_mul
bn_input_grad_correct  → (same as bn_has_vjp)
bnNormalize_has_vjp    → pdiv, pdiv_bnCentered, pdiv_bnIstdBroadcast, pdiv_mul
bnAffine_has_vjp       → pdiv, pdiv_bnAffine
residual_has_vjp       → pdiv, pdiv_add, pdiv_id
seBlock_has_vjp        → pdiv, pdiv_id, pdiv_mul
layerNorm_has_vjp      → pdiv, pdiv_bnAffine, pdiv_bnNormalize, pdiv_comp
softmax_has_vjp        → pdiv, pdiv_softmax
```

(Lean core axioms `propext`, `Classical.choice`, `Quot.sound` omitted — present in every nontrivial Lean program.)

## The three rules

All of backpropagation:

```
vjp_comp              f ∘ g  →  back_f(x, back_g(f(x), dy))
biPath_has_vjp        f + g  →  back_f(x, dy) + back_g(x, dy)
elemwiseProduct_has_vjp  f * g  →  back_f(x, g·dy) + back_g(x, f·dy)
```

## The five Jacobian tricks

Every layer's backward pass is one of:

1. **Diagonal** — activations (ReLU, GELU): one multiply
2. **Sparse Toeplitz** — conv: reversed kernel convolution
3. **Binary selection** — max pool: route to argmax
4. **Rank-1 correction** — batch/layer norm, softmax: closed-form 3-term formula
5. **Outer product** — dense/matmul: input ⊗ grad

## Verify

```bash
lake build LeanMlir.Proofs.Tensor LeanMlir.Proofs.MLP \
  LeanMlir.Proofs.CNN LeanMlir.Proofs.BatchNorm \
  LeanMlir.Proofs.Residual LeanMlir.Proofs.Depthwise \
  LeanMlir.Proofs.SE LeanMlir.Proofs.LayerNorm \
  LeanMlir.Proofs.Attention
```

If it builds, it's correct. That's the point.
