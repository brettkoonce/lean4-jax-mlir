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

The progression: **30 → 0 project axioms.** See `VJP.md` (foundation
flip and per-chapter migration) and `pdiv.md` (final 4-axiom retirement)
at the repo root for the full elimination history.

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
  ├── MLP.lean                 dense (proved both sides) + ReLU (pdiv_relu proved,
  │                            relu/mlp _has_vjp = canonical-witness defs)
  │                            + softmax CE (proved, lives in Attention.lean)
  │
  ├── CNN.lean                 conv2d (def) + maxPool (def) + weight/bias grads (theorems)
  │                            conv2d_has_vjp3 (theorem); maxPool2_has_vjp3
  │                            (canonical-witness def — codegen substitutes argmax)
  │
  ├── BatchNorm.lean           BN (every axiom proved from foundation)
  │
  ├── Residual.lean            skip connections (biPath; zero new axioms)
  │
  ├── Depthwise.lean           depthwise conv (def) + weight/bias grads (theorems)
  │                            depthwise_has_vjp3 (theorem)
  │
  ├── SE.lean                  squeeze-and-excitation (elemwiseProduct; zero new axioms)
  │
  ├── LayerNorm.lean           LayerNorm (proved) + GELU (gelu Jacobian proved)
  │
  └── Attention.lean           softmax (proved) + SDPA (proved) + MHSA (proved)
                               + ViT body chains (proved) + patchEmbed (proved)
```

## Axioms (0 project)

Pure-Mathlib closure on every theorem. `#print axioms vit_full_has_vjp`
shows only `propext`, `Classical.choice`, `Quot.sound` (Lean core).

The earlier 4-axiom floor was retired in Phase 7 (Apr 2026):

- `relu_has_vjp`, `mlp_has_vjp`, `maxPool2_has_vjp3` — converted from
  `axiom` to `noncomputable def` with the canonical pdiv-derived
  witness. `HasVJP.correct` holds by `rfl` since `pdiv` is a `def`
  over `fderiv` (post-foundation-flip). At non-smooth points the
  canonical backward is `fderiv`'s junk default of `0`; the codegen
  substitutes the standard subgradient/argmax convention — see
  "Codegen trust boundary" below.
- `pdiv_relu` — proved via local-diagonal-CLM transport
  (~80 LOC). At a smooth point (`∀ k, x k ≠ 0`), ReLU agrees with the
  diagonal indicator CLM `Π k, (if x k > 0 then proj k else 0)` on
  `Metric.ball x (min |x k|)` (every coordinate keeps its sign).
  `HasFDerivAt.congr_of_eventuallyEq` transports the CLM's self-fderiv
  to ReLU; direct evaluation at `basisVec i` reads off the entry.

**Tensor.lean** — calculus foundation: **0 axioms.** `pdiv` is a
`noncomputable def` over `fderiv`; every structural rule is a
theorem. `pdivMat_rowIndep` (the last surviving Mat-level axiom in
prior drafts) is now a theorem proved via the row-projection
`ContinuousLinearMap` and the chain rule, given a `Differentiable`
hypothesis on the per-row function.

**MLP.lean** — dense layers: **0 axioms.**

> `pdiv_dense`, `pdiv_dense_W`, `dense_weight_grad_correct`,
> `dense_bias_grad_correct`, and `pdiv_relu` are theorems.
> `relu_has_vjp` and `mlp_has_vjp` are `def`s over the canonical
> pdiv-derived witness. `softmaxCE_grad` is a theorem (relocated to
> `Attention.lean` next to `pdiv_softmax`).

**CNN.lean** — convolution and pooling: **0 axioms.**

> `conv2d` and `maxPool2` are concrete `def`s. The weight-grad and
> bias-grad VJPs (`conv2d_weight_grad_has_vjp`,
> `conv2d_bias_grad_has_vjp`) are theorems proved from foundation
> via `unfold + fun_prop`. `conv2d_has_vjp3` is a theorem (Phase 1,
> Apr 2026) — proved via `pdiv_finset_sum` × 3 +
> `pdiv_const_mul_pi_pad_eval` per-summand + Σ_(c, kh, kw) collapse.
> `maxPool2_has_vjp3` is a `def` over the canonical pdiv-derived
> witness.

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

**Total: 0 project axioms across all nine content modules.**

## Codegen trust boundary

`HasVJP.correct` certifies the *canonical* backward
`backward x dy i = ∑ j, pdiv f x i j * dy j`. Where `f` is everywhere
differentiable, this is the true Jacobian-vector product (and the
`_diff` theorems on each layer carry that hypothesis through).

For the two non-smooth ops — ReLU at `x i = 0`, MaxPool at argmax
ties — the canonical backward is `fderiv`'s junk default of `0`,
because Mathlib's `fderiv` returns `0` at non-differentiable points
by convention. The codegen (`MlirCodegen.lean`) does **not** emit the
canonical backward at the kinks. Instead it emits the standard ML-
framework subgradient conventions:

- ReLU: `if x > 0 then dy else 0` (the `relu'(0) := 0` convention).
- MaxPool: `stablehlo.select_and_scatter` with GE-selector + add-
  scatter (route the gradient to the argmax position; ties broken
  by the GE comparator).

These match the canonical Lean witness at smooth points and differ
only at the kinks. The verification gap is intrinsic to backward
passes through non-smooth ops — every ML framework lives with the
same gap. The numerical FD checks in `check_jacobians.py` and the
end-to-end oracles in `tests/vjp_oracle/` cover the codegen-emitted
formula at the kinks.

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

`check_jacobians.py` runs 25 finite-difference checks. They cover the
codegen-emitted backward formulas (where the trust gap actually
lives — see "Codegen trust boundary" above), particularly at the
ReLU and MaxPool kinks where the codegen substitutes a subgradient
convention for the canonical Lean witness. Typical max-error is
~1e-11 in float64.

## Independent kernel re-check (comparator)

`tests/comparator/` runs
[leanprover/comparator](https://github.com/leanprover/comparator) on
38 theorems spanning the foundation rules, every chapter's headline
Jacobian, and the public `*_has_vjp_correct` wrappers. comparator
re-runs Lean's kernel typechecker independently
of the elaborator and verifies the transitive axiom closure of each
proof. The configured allowlist is exactly
`{propext, Quot.sound, Classical.choice}` (Lean core); any project
axiom in the closure would fail the run. See
`tests/comparator/README.md` for the prereq + run instructions.

## Verify

```bash
lake build LeanMlir.Proofs.Tensor LeanMlir.Proofs.MLP \
  LeanMlir.Proofs.CNN LeanMlir.Proofs.BatchNorm \
  LeanMlir.Proofs.Residual LeanMlir.Proofs.Depthwise \
  LeanMlir.Proofs.SE LeanMlir.Proofs.LayerNorm \
  LeanMlir.Proofs.Attention
```

If it builds, it's correct. That's the point.
