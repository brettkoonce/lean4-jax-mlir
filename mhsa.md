# mhsa.md — plan for removing the multi-head attention axioms

**Status:** deferred. Two axioms remain on `main` (commit `89c923c`):

1. `mhsa_has_vjp_mat` (Attention.lean:1002) — bundled VJP correctness for
   the multi-head SDPA layer.
2. `mhsa_layer_flat_diff` (Attention.lean:1013) — `Differentiable ℝ` for
   the flattened layer.

Both are honest "vmap-over-heads" deferrals: the math is exactly "multi-head
attention is `heads`-many parallel SDPAs, decoupled by column slabs," and
the framework to formalize that decoupling needs to be built from scratch.

> **One-line summary.** Removing these axioms is a real ~600–800 LOC
> framework-development effort, not a quick proof. Plan it as its own
> branch with the colIndep half landing as a standalone PR first.

---

## Why the previous attempt didn't land

A first pass on `attention-diff-threading` follow-up (the session that
proved `pdiv_softmax`, `softmaxCE_grad`, `pdiv_bnIstdBroadcast`) tried to
extend that momentum to `mhsa`. Bailed for these reasons:

1. **Encoding bureaucracy.** `pdivMat_colIndep` (the column-slab analog of
   `pdivMat_rowIndep`, the precondition for any "vmap over heads"
   formulation) has *nested* `finProdFinEquiv` encodings: outer `(row,
   flat-col)` from `Mat.flatten`, plus inner `(head, intra-head-col)` from
   the slab structure. The `slabProj` CLM has the form
   `reindexCLM (fun idx => finProdFinEquiv (decode_idx.1, finProdFinEquiv
   (h, decode_idx.2)))`, and the `show`-tactic unification for
   "(slab fn at v with h) = Mat.unflatten ((slabProj h) v)" kept failing
   because of partial `Equiv.symm_apply_apply` reductions vs the
   `Function.comp` form. Each error was individually fixable; the combined
   debugging ate hours.
2. **Joint SDPA differentiability isn't free.** The half-measure plan
   (just prove `mhsa_layer_flat_diff`, leave `mhsa_has_vjp_mat`
   axiomatic) underestimated this: the existing `matmul_left_const_flat_diff`
   / `_right_const_flat_diff` / `scalarScale_flat_diff` /
   `transpose_flat_diff` / `rowSoftmax_flat_diff` family treats *one*
   matrix as the variable with the others held constant. To prove
   `Differentiable ℝ (fun v => Mat.flatten (sdpa N d_head Q(v) K(v) V(v)))`
   when Q, K, V all depend on `v`, you need joint differentiability of
   `(Q, K, V) ↦ sdpa Q K V`, which doesn't follow from the existing
   one-variable lemmas. That's a new ~150 LOC of plumbing on its own.
3. **`fun_prop` doesn't carry positivity through SDPA.** The denominator
   in `rowSoftmax`'s row-wise softmax is `Σ_j Real.exp(...)` which is
   positive but `fun_prop` won't auto-discharge the `≠ 0` side condition
   for division. `rowSoftmax_flat_diff` already pays this cost manually;
   the multi-head version would either need to call into it per slab or
   re-prove the positivity chain.

---

## Recommended architecture (Approach A: ternary framework)

After surveying both paths, **the ternary `HasVJPMat3` framework is
cleaner than the unary-with-column-stacking alternative.** Reasons:

- `HasVJPMat3` parallels the existing `HasVJP3` for 3D tensors
  (`Tensor.lean:1253`); it's a natural extension of the project's own
  framework conventions.
- The unary alternative (stack Q | K | V into `Mat N (3D)`) requires a
  case split on whether the column index falls in the Q-slab, K-slab, or
  V-slab — ugly proof, not the clean per-head story.
- The ternary framework can be reused later if other ternary ops appear
  (none in the current chapter list, but a softer cost than dead code).

### Pieces to build (in order)

| # | Piece                              | File          | Est. LOC |
|---|------------------------------------|---------------|----------|
| 1 | `pdivMat_colIndep` (theorem)       | Tensor.lean   | ~120     |
| 2 | `colSlabApply` (def) + `colSlabwise_has_vjp_mat` | Tensor.lean | ~100 |
| 3 | `HasVJPMat3` (struct + comp)       | Tensor.lean   | ~150     |
| 4 | `sdpa_has_vjp_mat3` (combined per-head VJP) | Attention.lean | ~120 |
| 5 | `mhsa_has_vjp_mat` (compose pieces) | Attention.lean | ~80     |
| 6 | `mhsa_layer_flat_diff` (parallel)  | Attention.lean | ~100     |
|   | **Total**                          |               | **~670** |

### Detail sketches

**Piece 1: `pdivMat_colIndep`.** Mirrors `pdivMat_rowIndep`
(Tensor.lean:827–913) but with column slabs in place of rows. Key CLM:

```lean
-- σ : Fin (n * d_in) → Fin (n * (heads * d_in))
-- σ slab_idx = encode(decode(slab_idx).1, encode(h, decode(slab_idx).2))
slabProj h := reindexCLM (fun idx : Fin (n * d_in) =>
  finProdFinEquiv ((finProdFinEquiv.symm idx).1,
                   finProdFinEquiv (h, (finProdFinEquiv.symm idx).2)))
```

The proof structure parallels rowIndep: define `F` as the flat form,
show coord `(k', encode(h_l', j_out))` of `F` factors as
`(g-coord-fn) ∘ slabProj h_l'`, then `fderiv_apply` + `fderiv_comp`. The
final case split is on `h_j = h_l` (whether the input slab matches the
output slab).

**Piece 2: `colSlabApply` + `colSlabwise_has_vjp_mat`.** The lift-by-vmap
analog of `rowwise_has_vjp_mat`. Works for non-uniform input/output
widths (`d_in ≠ d_out`) — required because SDPA per-head takes a
combined Q | K | V slab of width `3 * d_head` and outputs `d_head`.

**Piece 3: `HasVJPMat3`.** Structure:

```lean
structure HasVJPMat3 {a b c d : Nat} (F : Mat a b → Mat a b → Mat a b → Mat c d) where
  backward : Mat a b → Mat a b → Mat a b → Mat c d → (Mat a b × Mat a b × Mat a b)
  correct_1 : ∀ A B C dY i j,
    (backward A B C dY).1 i j =
    ∑ k l, pdivMat (fun A' => F A' B C) A i j k l * dY k l
  correct_2 : ...similar for B...
  correct_3 : ...similar for C...
```

Plus a `vjpMat3_unary_wrap` that converts `HasVJPMat3 F` + dependencies
on a single input `X` (via three projections `g_Q, g_K, g_V : Mat a b →
Mat a b`) into `HasVJPMat (fun X => F (g_Q X) (g_K X) (g_V X))`. This is
the chain-rule fan-in for three intermediates.

**Piece 4: `sdpa_has_vjp_mat3`.** Bundles the existing
`sdpa_back_{Q,K,V}_correct` theorems (Attention.lean:772, 866, 899) into
a single `HasVJPMat3 (sdpa N d_head)`. The backward returns the triple
`(sdpa_back_Q, sdpa_back_K, sdpa_back_V)`; correctness is just unpacking
the existing per-input theorems.

**Piece 5: `mhsa_has_vjp_mat`.** Compose:
- The Q/K/V dense projections give three `HasVJPMat` instances (existing
  `dense` lemmas, lifted per-row).
- `colSlabwise_has_vjp_mat` lifts `sdpa_has_vjp_mat3` to "vmap over
  heads," producing a `HasVJPMat3 (mhsa_inner)` over the full
  `Mat N D × Mat N D × Mat N D → Mat N D`.
- `vjpMat3_unary_wrap` collapses the three-input layer with the dense
  projections into `HasVJPMat (fun X => mhsa_inner (Q X) (K X) (V X))`.
- Final `vjpMat_comp` with output dense gives `HasVJPMat (mhsa_layer ...)`.

**Piece 6: `mhsa_layer_flat_diff`.** Parallel to piece 5 but for
`Differentiable ℝ`. Build a `sdpa_diff_jointly` helper that gives joint
differentiability of `(Q, K, V) ↦ sdpa Q K V` (use product domain
+ `Differentiable.mul`, `.transpose`, `rowSoftmax_flat_diff`,
`scalarScale_flat_diff`). Then chain with dense diffs.

---

## Approach B (rejected): unary with column stacking

For posterity, the alternative architecture and why it lost:

Stack `[Q | K | V]` into `Mat N (3*D)`, reorganize so head `h`'s data is
contiguous (a column permutation), apply a per-slab function
`Mat N (3*d_head) → Mat N d_head` that splits and runs SDPA, then
reorganize and concatenate.

- **Pro**: reuses existing unary `HasVJPMat` framework — no `HasVJPMat3`.
- **Con**: the per-slab function's `HasVJPMat` proof requires a case
  split on whether the input column is in the Q-, K-, or V-third of the
  slab. Each branch unfolds to `sdpa_back_{Q,K,V}_correct`, which is
  fine, but the surrounding `pdivMat` chain rule across the three branches
  is ugly. Also requires a column-permutation reindex CLM that isn't
  used anywhere else.

The total LOC ends up similar (~600), but the per-slab proof is harder to
read and to maintain.

---

## Pitfalls catalog (from the failed attempt)

These tripped the attempt and will trip the next one. Pre-mitigate.

1. **Nested `finProdFinEquiv` encoding.** When the input is `Mat n (heads * d_in)`,
   its flat form is indexed by `Fin (n * (heads * d_in))` which decomposes
   to `(row, col)` then `col` decomposes to `(head, intra-head-col)`. The
   `slabProj` CLM and the basisVec arguments need to agree on which
   encoding goes where. In the failed attempt, mixing
   `Mat.flatten A (finProdFinEquiv (r, finProdFinEquiv (h, j)))`
   form with
   `Mat.flatten (fun r' j_in => A r' (finProdFinEquiv (h, j_in))) idx`
   form caused `show` to fail repeatedly. **Fix:** use explicit `Mat.flatten`
   def expansion early; don't rely on `simp` to bridge the two forms.

2. **`Function.comp` doesn't auto-eta after `simp only [Equiv.symm_apply_apply]`.**
   In the colIndep proof, after the `funext v; show ...; unfold Mat.flatten;
   simp only [Equiv.symm_apply_apply]` chain, the remaining goal had
   `((fun w => g (Mat.unflatten w) k' j_out) ∘ ⇑(slabProj h_l')) v` on the
   RHS. Lean did not reduce this composition for `show`. **Fix:** unfold
   `Function.comp` explicitly or restructure as a `have :=` then `convert`.

3. **`⟨a, b⟩` for `Eq` doesn't work.** Tried `exact ⟨h1, h2⟩` to prove
   `(i, j) = (i', j')` from component-wise equalities. That doesn't
   typecheck — `Eq.refl` doesn't take fields. **Fix:** use `Prod.ext h1
   h2` or split the goal with `constructor`.

4. **`set p := ...` doesn't unfold inside `let p := ...` from a
   `noncomputable def` body.** The `colSlabwise_has_vjp_mat` backward
   used `let p := finProdFinEquiv.symm hj; ...`; in the correctness
   proof the goal had the literal `let`-binding from the structure
   field, while my `set p := finProdFinEquiv.symm jj` only renamed
   occurrences in the goal, not the let-bound copy from the field.
   **Fix:** `simp only [show backward = ... from rfl]` or unfold the
   structure-field name explicitly to expose the let, then `dsimp only`.

5. **Joint differentiability of SDPA isn't `fun_prop`-able.** Even after
   `unfold mhsa_layer sdpa rowSoftmax softmax`, `fun_prop` can't
   discharge the inner `Real.exp / sum-of-exps` because the positivity
   side condition isn't propagated. **Fix:** build a `sdpa_diff_jointly`
   helper that invokes `rowSoftmax_flat_diff` (or `softmax_diff`) at the
   right level of abstraction, then composes with `Differentiable.mul`
   for the matmul ops.

6. **`unfold sdpa_scale` after `unfold sdpa` fails — already inlined.**
   `sdpa` uses `sdpa_scale` definitionally; once `sdpa` is unfolded the
   constant `1 / √d` is already substituted, and trying to `unfold
   sdpa_scale` after errors with "failed to unfold." **Fix:** put
   `sdpa_scale` *before* `sdpa` in the unfold list, or drop it.

---

## Suggested execution order

To minimize the rework risk:

1. **Phase 1 (standalone, ~265 LOC) — ✅ LANDED.** `pdivMat_colIndep` +
   `colSlabApply` + `colSlabwise_has_vjp_mat` on branch
   `colslab-vmap-framework` (commit `6259cae`). Pure-Mathlib closure on
   all three. Doesn't change axiom counts; reusable foundation for
   Phase 3.
2. **Phase 2 (~150 LOC):** `HasVJPMat3` framework + `sdpa_has_vjp_mat3`.
   Standalone too: doesn't depend on Phase 1 directly (Phase 1 is the
   "vmap" half, Phase 2 is the "ternary" half).
3. **Phase 3 (~200 LOC):** combine Phases 1 + 2 into `mhsa_has_vjp_mat`
   and `mhsa_layer_flat_diff`. This is where the integration risk lives;
   the prior two phases having clean APIs makes this manageable.
4. **VJP.md update + commits**, mirroring the
   `attention-diff-threading` discipline (one commit per landed piece).

After Phase 3, axiom count drops 10 → 8. The remaining 8 are pure
"framework convention" axioms (ReLU subgradient cluster + opaque
codegen).

---

## Time budget

- ~~Phase 1: 4–6 hours~~ — actual: ~1 hour (much faster than estimated
  once the pitfalls catalog was applied. The original failure was
  tactic-ordering noise, not a fundamental difficulty).
- Phase 2: 4–6 hours (`HasVJPMat3` is straightforward but verbose; the
  chain rule for ternary inputs needs care).
- Phase 3: 6–8 hours (integration, joint diff plumbing, mhsa_layer
  composition).

**Remaining: ~10–14 focused hours, ~2 working days.**

---

## When to pick this back up

The right time to do this work is when:

1. You have a clean ~3-day window for focused proof work (not interleaved
   with other tasks — the encoding tactics need momentum to debug).
2. You've reviewed `pdivMat_rowIndep` and `rowwise_has_vjp_mat` recently
   so the row-axis pattern is fresh.
3. Mathlib hasn't drifted significantly (the `HasFDerivAt` /
   `Differentiable.*` API is stable but `fun_prop`'s automation
   sometimes changes — pin Mathlib at the start of the branch).

The bad time to do this: as a side-quest during another task. The
encoding bureaucracy is unforgiving when interrupted.

---

## What `mhsa_has_vjp_mat` removal *would* buy us

- 10 → 8 project axioms.
- The book's claim "verified multi-head attention" becomes literal rather
  than "verified modulo this one bundled axiom about head independence."
- Buzzard would stop pointing at it as the obvious next axiom to kill.

The cost (3 working days, ~600 LOC of framework that's only used here)
versus that benefit is the question to answer when scheduling the work.
The current state — 10 axioms, clearly-documented "stays axiomatic"
reasons in VJP.md — is honestly defensible; this is upgrade work, not
correctness work.
