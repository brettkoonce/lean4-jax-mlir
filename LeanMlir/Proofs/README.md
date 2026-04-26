# Verified VJP Proofs

Machine-checked proofs that the backward pass (VJP) of every layer
matches its forward-pass Jacobian. Zero `sorry`s. If `lake build`
succeeds, every theorem is correct.

## Foundation: Mathlib's `fderiv`

Earlier drafts of this suite axiomatized the entire calculus
foundation — chain rule, sum rule, product rule, identity, reindex —
as eight opaque facts. The current foundation **flips that**:
`pdiv` is *defined* in terms of Mathlib's Fréchet derivative
`fderiv`, the structural rules are *theorems* proved from Mathlib's
API, and every downstream chapter threads a `Differentiable`
hypothesis through its compositions.

Many later-chapter axioms have been pruned the same way. Where
earlier drafts axiomatized `conv2d`, `maxPool2`, `depthwiseConv2d`,
or `geluScalar` as opaque functions with stated Jacobians, the
current version defines them concretely and proves their
gradient-related lemmas from the foundation.

The diff-threading branch closed out every remaining "provable but
deferred" Jacobian: `pdivMat_rowIndep`, `pdiv_softmax`,
`softmaxCE_grad`, `pdiv_gelu`, `pdiv_bnIstdBroadcast`, the BN
inverse-stddev smoothness, the row-wise softmax smoothness, and all
seven transformer-level composition chains.

The progression: **30 → 4 axioms.** See `VJP.md` at the repo root
for the full elimination history.

## Dependency graph

```
Tensor.lean                    ← pdiv (def via fderiv) + VJP framework
  │
  │  pdiv_comp (chain rule)         ← theorem
  │  pdiv_add  (sum rule)           ← theorem
  │  pdiv_mul  (product rule)       ← theorem
  │  pdiv_id   (identity)           ← theorem
  │  vjp_comp  (VJP composition)    ← theorem
  │  biPath    (additive fan-in)    ← theorem
  │  elemwiseProduct                ← theorem
  │  pdivMat_rowIndep               ← theorem (was the last surviving Mat-axiom)
  │
  ├── MLP.lean                 dense (proved both sides) + ReLU (subgradient axiom)
  │                            + softmax CE (proved, lives in Attention.lean)
  │
  ├── CNN.lean                 conv2d (def) + maxPool (def) + weight/bias grads (theorems)
  │                            input-side VJPs stay axiomatic (padding boundary / argmax)
  │
  ├── BatchNorm.lean           BN (every axiom proved from foundation)
  │
  ├── Residual.lean            skip connections (biPath; zero new axioms)
  │
  ├── Depthwise.lean           depthwise conv (def) + weight/bias grads (theorems)
  │                            input-side VJP stays axiomatic
  │
  ├── SE.lean                  squeeze-and-excitation (elemwiseProduct; zero new axioms)
  │
  ├── LayerNorm.lean           LayerNorm (proved) + GELU (gelu Jacobian proved)
  │
  └── Attention.lean           softmax (proved) + SDPA (proved) + MHSA (proved)
                               + ViT body chains (proved) + patchEmbed (proved)
```

## Axioms (4 total)

The 4 surviving axioms are at genuine non-smoothness boundaries
(ReLU and MaxPool subgradient conventions), not deferred proofs.
Going below 4 requires `HasVJP.correct` weakening to "smooth subset
only" — a project-wide rewrite, separate multi-week effort.
Grouped by file:

**Tensor.lean** — calculus foundation: **0 axioms.** `pdiv` is a
`noncomputable def` over `fderiv`; every structural rule is a
theorem. `pdivMat_rowIndep` (the last surviving Mat-level axiom in
prior drafts) is now a theorem proved via the row-projection
`ContinuousLinearMap` and the chain rule, given a `Differentiable`
hypothesis on the per-row function.

**MLP.lean** — dense layers: **3 axioms.**
| Axiom | What it says |
|-------|-------------|
| `pdiv_relu` | ReLU Jacobian (guarded subgradient — `(∀ k, x k ≠ 0)`) |
| `relu_has_vjp` | ReLU bundled VJP, existence at non-smooth points |
| `mlp_has_vjp` | MLP composition shortcut (composes through ReLU) |

> `pdiv_dense`, `pdiv_dense_W`, `dense_weight_grad_correct`, and
> `dense_bias_grad_correct` are now **theorems** proved from the
> foundation. `softmaxCE_grad` is also a theorem (relocated to
> `Attention.lean` next to `pdiv_softmax`).

**CNN.lean** — convolution and pooling: **1 axiom.**
| Axiom | What it says |
|-------|-------------|
| `maxPool2_has_vjp3` | MaxPool2 input-VJP (argmax routing convention) |

> `conv2d` and `maxPool2` are now concrete `def`s. The
> weight-grad and bias-grad VJPs (`conv2d_weight_grad_has_vjp`,
> `conv2d_bias_grad_has_vjp`) are theorems proved from foundation
> via `unfold + fun_prop`. `conv2d_has_vjp3` is now a theorem
> (Phase 1, Apr 2026) — proved via `pdiv_finset_sum` × 3 +
> `pdiv_const_mul_pi_pad_eval` per-summand + Σ_(c, kh, kw) collapse.

**BatchNorm.lean** — the hard one: **0 axioms.**

> Every BN Jacobian is now a theorem. `pdiv_bnAffine` and
> `pdiv_bnCentered` were proved in Stage 1; `pdiv_bnIstdBroadcast`
> and the smoothness lemma `bnIstdBroadcast_diff` were the last to
> fall in the diff-threading branch — the centering CLM,
> `HasFDerivAt.sqrt` (under `bnVar + ε > 0`), and
> `(hasDerivAt_inv).comp_hasFDerivAt` close the chain. Every BN
> proof now carries a `(hε : 0 < ε)` hypothesis.

**Residual.lean** — skip connections: **0 axioms.** Pure composition
over `biPath_has_vjp` + `identity_has_vjp` from `Tensor.lean`.

**Depthwise.lean** — depthwise conv: **0 axioms.**

> `depthwiseConv2d` is now a concrete `def`, weight and bias
> gradients are theorems via `unfold + fun_prop`. `depthwise_has_vjp3`
> is now a theorem (Phase 2, Apr 2026) — same recipe as conv2d with
> one fewer Σ level (no cross-channel mixing in depthwise).

**SE.lean** — squeeze-and-excitation: **0 axioms.** Pure composition
over `elemwiseProduct_has_vjp` + `dense_has_vjp` + `identity_has_vjp`.

**LayerNorm.lean** — layer norm and GELU: **0 axioms.**

> `geluScalar` and `geluScalarDeriv` are now concrete `def`s using
> the standard `tanh`-approximation formula. `pdiv_gelu` is a
> theorem proved via `fderiv_apply` + chain rule with
> `geluScalar ∘ ContinuousLinearMap.proj j`, then
> `fderiv_eq_smul_deriv` to convert scalar `fderiv` ↔ `deriv`. A
> new `Real.differentiable_tanh` `@[fun_prop]` lemma (derived from
> `Real.tanh_eq_sinh_div_cosh` + `Real.cosh_pos`) carries the
> smoothness through. `layerNorm_has_vjp` reuses the BN proof
> template on a different axis.

**Attention.lean** — softmax, attention, ViT: **0 axioms.**

> `pdiv_softmax`, `softmaxCE_grad`, the three `sdpa_back_*_correct`
> theorems, `rowSoftmax_flat_diff`, and **every** transformer-level
> chain (`transformerMlp_has_vjp_mat`,
> `transformerAttnSublayer_has_vjp_mat`,
> `transformerMlpSublayer_has_vjp_mat`,
> `transformerBlock_has_vjp_mat`,
> `transformerTower_has_vjp_mat`, `vit_body_has_vjp_mat`,
> `mhsa_has_vjp_mat`, `mhsa_layer_flat_diff`,
> `classifier_flat_has_vjp`, `vit_full_has_vjp`) are theorems.
> `patchEmbed_flat`, `patchEmbed_flat_diff`, and
> `patchEmbed_flat_has_vjp` were the last to fall: Phase 6a (Apr 2026)
> de-opaqued the forward and proved Diff via `differentiableAt_pad_eval`;
> Phase 6b (Apr 2026) proved the closed-form input-VJP via the same
> recipe used for `conv2d_has_vjp3`/`depthwise_has_vjp3`, with one new
> wrinkle: split `Σ n : Fin (N+1)` into the n=0 (CLS row, zero img-grad
> contribution) and `Σ p : Fin N` (n = p.succ, conv projection).

Plus three Lean core axioms (`propext`, `Classical.choice`,
`Quot.sound`) present in every nontrivial Lean program.

**Total: 0 (Tensor) + 3 (MLP) + 1 (CNN) + 0 (BatchNorm) + 0
(Residual) + 0 (Depthwise) + 0 (SE) + 0 (LayerNorm) + 0 (Attention)
= 4 axioms.**

Five of nine content modules add zero new axioms.

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

## Numerical gradient checks

`check_axioms.py` runs 25 finite-difference checks. 7 of the 10
surviving axioms are FD-tested directly (`pdiv_relu` at non-zero
points, `mlp_has_vjp` by full-network composition, the conv /
maxPool / depthwise input VJPs, bundled `mhsa_has_vjp_mat`, and
`patchEmbed_flat_has_vjp`); the remaining 18 belt-and-suspender the
proved Jacobian theorems. `relu_has_vjp` is redundant with
`pdiv_relu`; the two `_diff` siblings are smoothness claims with no
FD purchase. Typical max-error is ~1e-11 in float64.

## Verify

```bash
lake build LeanMlir.Proofs.Tensor LeanMlir.Proofs.MLP \
  LeanMlir.Proofs.CNN LeanMlir.Proofs.BatchNorm \
  LeanMlir.Proofs.Residual LeanMlir.Proofs.Depthwise \
  LeanMlir.Proofs.SE LeanMlir.Proofs.LayerNorm \
  LeanMlir.Proofs.Attention
```

If it builds, it's correct. That's the point.
