# VJP.md — Foundation Flip Landed; Mechanical Follow-ups Done

**Branch:** `attention-diff-threading` (4 commits past `main` head `2d4e92e`).
Cumulative across the foundation-flip + this branch: **23 → 13 project axioms.**

> **Strategy summary.** Foundation flip (attempt #3, "guarded ReLU") landed
> on `main` and preserved the soundness analysis from attempts #1 and #2
> (still load-bearing; see "Soundness analysis" below). This branch then
> walked down the mechanical follow-ups (A–E from the original VJP.md
> roadmap) and proved everything that wasn't a genuine subgradient/
> opaque-codegen convention.

---

## What landed (cumulative)

### Stage 1 — Foundation (`main`, commit `8290155`)

- `axiom pdiv` → `noncomputable def pdiv f x i j := fderiv ℝ f x (basisVec i) j`.
- `axiom pdiv_id` / `_const` / `_reindex` → unconditional theorems.
- `axiom pdiv_add` / `_mul` / `_comp` → theorems with `DifferentiableAt` hypotheses.
- `pdiv_finset_sum` now requires `∀ s ∈ S, DifferentiableAt ℝ (f s) x`.
- Internal Tensor.lean cascade threaded across `vjp_comp`, `biPath_has_vjp`,
  `elemwiseProduct_has_vjp`, `pdivMat_comp`, `pdivMat_add`, `vjpMat_comp`,
  `biPathMat_has_vjp`, the matmul/scalar/transpose lemmas, and the rank-3 chain.

### Stage 2 — Per-chapter migrations (`main`)

| Chapter | Notable change |
|---|---|
| MLP | Diff threaded through `pdiv_dense*`. **`pdiv_relu` guarded** with `(∀ k, x k ≠ 0)`. **`relu_has_vjp` and `mlp_has_vjp` axiomatized** (subgradient routing). |
| CNN | Diff threaded through `conv2d_weight_grad_has_vjp` and `conv2d_bias_grad_has_vjp` via `unfold + fun_prop`. `conv2d_has_vjp3` and `maxPool2_has_vjp3` stay axiom (non-smooth ops). |
| Depthwise | Same recipe as CNN. |
| BatchNorm | `pdiv_bnAffine` / `pdiv_bnCentered` from foundation. `bn_has_vjp` and friends take `(hε : 0 < ε)`. |
| Residual + SE + LayerNorm | `Differentiable` arguments threaded through `residual_has_vjp`, `residualProj_has_vjp`, `seBlock_has_vjp`, `layerNorm_has_vjp`. |
| Attention | Diff helpers (`matmul_left/right_const_flat_diff`, `scalarScale_flat_diff`, `transpose_flat_diff`) via `fun_prop`. SDPA Q/K/V chains proved via `vjpMat_comp`. |

### Stage 3 — Branch `attention-diff-threading` (this branch)

| Commit | What | Axiom delta |
|---|---|---|
| `581a771` | **Follow-up A.** Thread `Differentiable` through 8 deferred Attention chains: `transformerMlp_has_vjp_mat`, both sublayers, `transformerBlock_has_vjp_mat`, `transformerTower_has_vjp_mat` (induction on k), `vit_body_has_vjp_mat`, `classifier_flat_has_vjp`, `vit_full_has_vjp`. | −4 (8 chain axioms removed; 4 Diff bridges added: `gelu_per_token_flat_diff`, `layerNorm_per_token_flat_diff`, `mhsa_layer_flat_diff`, `patchEmbed_flat_diff`) |
| `5532d15` | **Follow-ups B + C + Real.tanh.** `rowSoftmax_flat_diff` (`exp · (sum-exp)⁻¹`, `Differentiable.inv` with `Finset.sum_pos`); `bnIstdBroadcast_diff` (`Differentiable.sqrt` + `Differentiable.inv` over `bnVar + ε > 0`); `gelu_per_token_flat_diff` promoted via a new `Real.differentiable_tanh` `@[fun_prop]` lemma derived from `Real.tanh_eq_sinh_div_cosh` + `Real.cosh_pos`. | −3 |
| `f8ac5b1` | **Follow-up D + free-rider.** `pdivMat_rowIndep` (was unconditional axiom; **was technically unsound** — counterexample below): now a theorem requiring `Differentiable ℝ g`, proved via `fderiv_apply` + chain rule with `reindexCLM` row-projection + `(rowProj k) (basisVec (i, j)) = basisVec j` if `i = k`, else `0`. Cascade: `rowwise_has_vjp_mat` takes new `hg_diff` hypothesis; new Vec-level `softmax_diff`/`gelu_diff`/`layerNorm_diff`/`dense_diff` lemmas to discharge it. Free-rider: `layerNorm_per_token_flat_diff` axiom → theorem via the same row-projection CLM trick. | −2 |
| `6f65ba3` | **Follow-up E (first half).** `pdiv_gelu`: `gelu n` has diagonal Jacobian by `fderiv_apply` + chain rule with `geluScalar ∘ ContinuousLinearMap.proj j`, then `fderiv_eq_smul_deriv` to convert scalar `fderiv` ↔ `deriv`. Moves `Real.differentiable_tanh` + `gelu_diff` from Attention.lean to LayerNorm.lean (next to `gelu`'s definition). | −1 |

**Net for the branch: −10 axioms (23 → 13).**

---

## Project axiom inventory (13)

**MLP / activations (4):**
- `pdiv_relu` — guarded subgradient axiom (DL convention).
- `relu_has_vjp` — existence at non-smooth points.
- `mlp_has_vjp` — composes through ReLU.
- `softmaxCE_grad` — gradient of cross-entropy loss. Provable via chain rule on `-log ∘ softmax_label` if `pdiv_softmax` is available.

**CNN-family (3):**
- `conv2d_has_vjp3` — input-path VJP through padding boundary.
- `maxPool2_has_vjp3` — argmax routing convention.
- `depthwise_has_vjp3` — input-path VJP, parallel to conv2d.

**Closed-form derivative formulas (2):**
- `pdiv_softmax` — `∂(exp(z j) / Σ exp(z k))/∂z i`. Quotient-rule formula.
- `pdiv_bnIstdBroadcast` — `∂(1/√(σ²+ε))/∂x_i = -istd³ · (xᵢ - μ) / n`.

**Attention bundles (4):**
- `mhsa_has_vjp_mat` — multi-head SDPA bundled.
- `mhsa_layer_flat_diff` — Diff sibling.
- `patchEmbed_flat_has_vjp` — opaque-codegen patch embedding.
- `patchEmbed_flat_diff` — Diff sibling.

---

## What's still tackleable (and what isn't)

### Tackleable but multi-hour (the realistic next session)

**`pdiv_softmax`** — closed-form derivative.

The mathematically-clean approach is the 1D directional-derivative reduction:
1. `fderiv f z (basisVec i) j = fderiv (fun z' => softmax c z' j) z (basisVec i)` via `fderiv_apply`.
2. Match with `HasDerivAt (fun t => softmax c (z + t • basisVec i) j) (rhs) 0` via `HasFDerivAt.comp_hasDerivAt_of_eq` + `HasDerivAt.unique`.
3. The 1D function unfolds to `Real.exp (z j + t · δ_j) / Σ_k Real.exp (z k + t · δ_k)` where `δ_k = if k = i then 1 else 0`.
4. Apply Mathlib's 1D `HasDerivAt.div` (well-supported), `HasDerivAt.fun_sum`, `Real.hasDerivAt_exp`.
5. `Σ_k Real.exp (z k) · δ_k` collapses to `Real.exp (z i)` via `Finset.sum_ite_eq`.
6. Algebra to match `softmax z j · ((if i = j then 1 else 0) - softmax z i)`.

Attempted on this branch (uncommitted, reverted). Tripping points:
- `congr 1` over-reduces in the function-equality step (massaging `fun t => softmax c (z + t • v) j` into the explicit quotient form).
- `Finset.sum_ite_eq` vs `'` direction for collapsing `Σ_k (if k = i then ... else 0)`.
- `show`-statement type unification at the `t = 0` substitution points (`0 * δ k` not auto-reducing inside the goal that survives `convert h_div using 1`).

Each fixable individually; combined proof likely **~80 lines, 2-3 focused hours**.

**`softmaxCE_grad`** — chain rule on `-log ∘ softmax_label`. Provable via `pdiv_comp` on the composition + `Real.deriv_log` at `softmax label > 0` + `pdiv_softmax`. **Estimated 1-2 hours**, conditional on `pdiv_softmax` (theorem) — or it can stay axiom-conditional on the still-axiomatic `pdiv_softmax`.

**`pdiv_bnIstdBroadcast`** — `1/√(σ²+ε)` derivative. Chain through `Real.sqrt`, `Inv.inv`, and `bnVar`'s product-rule derivative `(2/n) · (xᵢ - μ)`. **Estimated 3-4 hours**. Same shape as `pdiv_softmax` (1D directional approach is right) but deeper algebra in the inner `bnVar` step.

**Realistic floor after follow-up E:** 13 → 10 axioms.

### Stays axiomatic (10 of the current 13)

- **Subgradient conventions (3):** `pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`. Could be derived only by weakening `HasVJP.correct` to a "smooth subset only" formulation — project-wide rewrite, separate multi-week effort.
- **Non-smooth/boundary conventions (3):** `conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3`. Same weakening would unlock these, modulo a shared boundary-handling axiom.
- **Bundled vmap-VJP (2):** `mhsa_has_vjp_mat`, `mhsa_layer_flat_diff`. Need a column-axis analog of `pdivMat_rowIndep` plus a ternary-input VJP framework lemma.
- **Opaque-codegen interfaces (2):** `patchEmbed_flat_has_vjp`, `patchEmbed_flat_diff`. The actual computation lives in MLIR; we axiomatize the forward+backward consistency.

Below ~10 axioms, every remaining axiom is "the ML framework treats this op's gradient by convention X" or "this opaque codegen forward and backward are mutually consistent" — neither is provable from standard analysis without weakening the framework.

---

## Soundness analysis (carry-forward, still load-bearing)

Three soundness wells now resolved. Anyone considering a future restructure should re-read these.

### 1. `pdiv_relu` (unguarded) is incompatible with `fderiv`-based `pdiv`

For `n ≥ 2`, take `x = (1, 0) : Vec 2`. The function `relu 2` is **not**
`Differentiable` at `x` — coordinate `y₁ ↦ if y₁ > 0 then y₁ else 0` is
not differentiable at `y₁ = 0`, and `fderiv_pi` says a Pi-valued
function is differentiable iff each coordinate is. So
`fderiv ℝ (relu 2) (1, 0) = 0` (Mathlib junk default), making
`pdiv (relu 2) (1, 0) 0 0 = 0`. The unguarded axiom asserts `1`. **0 ≠ 1.**

Resolution: `pdiv_relu` is guarded by `(∀ k, x k ≠ 0)`. ✅

### 2. Unconditional `pdiv_add` / `_mul` / `_comp` are incompatible with `fderiv`

Counterexample for `pdiv_add`: take `f y = fun _ => abs (y 0)` (not
`DifferentiableAt 0`) and `g y = fun _ => y 0` (= identity). At `x_0 = 0`:

- `f + g` has a kink at 0; `fderiv (f+g)` is junk = 0; LHS = `pdiv (f+g) x 0 0 = 0`.
- `pdiv f x 0 0` is junk = 0; `pdiv g x 0 0 = 1`. RHS = `0 + 1 = 1`.
- Unconditional axiom claims LHS = RHS, i.e., `0 = 1`. ✗

Resolution: bilinear rules require `DifferentiableAt` hypotheses. ✅

### 3. **Unconditional `pdivMat_rowIndep` was technically unsound** (resolved this branch, commit `f8ac5b1`)

Counterexample with `relu : Vec 1 → Vec 1`, `m = 2`, `n = 1`, `p = 1`, take
`A 0 = (0,)` (kink), `A 1 = (1,)`. The flattened row-wise function
`F(v)` is non-differentiable at `flat A` because the 0-coord
(`relu` applied to row 0) is non-differentiable at `relu(0)`.
Therefore `fderiv F (flat A) = 0` (junk), so the LHS
`pdivMat (rowwise relu) A 1 0 1 0 = 0`.

But the RHS — under the unconditional axiom — is
`pdiv relu (A 1) 0 0 = 1` (since `relu` IS differentiable at `(1,)`,
derivative `1`). So **the unconditional axiom asserts `0 = 1`**.

Resolution: `pdivMat_rowIndep` now requires `Differentiable ℝ g`. With
the hypothesis, the Pi-decomposition `F` IS differentiable at every
flat A, and `fderiv_apply` + chain rule with row-projection CLM gives
the correct decomposition. ✅

The unsoundness was harmless in practice because every use of
`rowwise_has_vjp_mat` in the project was with a differentiable function
(softmax, dense, gelu, layerNorm). But the axiom's blanket form was a
landmine for any future caller. The cascade in `f8ac5b1` made the
hypothesis explicit at every call site.

---

## Pitfalls (encountered during the migration + branch)

1. **Lambda-form vs CLM-coercion in `rw`.** Passing
   `(reindexCLM σ).differentiableAt` directly to a `pdiv_*` theorem
   inside a rewrite generates a pattern with the `⇑(reindexCLM σ)`
   coercion that doesn't unify with goals containing the lambda form.
   **Fix:** name the diff hypothesis with an explicit lambda type.

2. **`fun_prop` on division by `Nat`-coerced denominators.** `fun_prop`
   doesn't currently know `HDiv.hDiv` is `Differentiable` in general
   (would need a non-zero denominator hypothesis). Workaround: rewrite
   `x / (↑n : ℝ)` as `x * (↑n)⁻¹` before `fun_prop`.

3. **`fun_prop` on `Real.sqrt` / `Real.exp` chains for `Vec → ℝ` codomain.**
   Doesn't auto-handle the positivity-conditional smoothness of
   `1/√(...)` or `exp(...) / Σ exp(...)`. Use manual `Differentiable.inv`
   / `Differentiable.sqrt` lemmas with explicit `≠ 0` hypotheses, or
   rewrite `c x / d x` as `c x * (d x)⁻¹` and chain.
   `Mathlib.Analysis.Calculus.Deriv.Inv`'s `Differentiable.fun_div` only
   works for `𝕜 → 𝕜'` (scalar domain), NOT for general `Vec → ℝ`.

4. **`Real.tanh` not `@[fun_prop]`-tagged in current Mathlib snapshot.**
   Project bridges via `theorem Real.differentiable_tanh` (LayerNorm.lean,
   commit `5532d15` / `6f65ba3`) derived from
   `Real.tanh_eq_sinh_div_cosh` + `Real.differentiable_sinh` /
   `Real.differentiable_cosh` + `Real.cosh_pos`, then tagged
   `@[fun_prop]` so downstream `geluScalar`/`gelu_diff` auto-discharge.

5. **Dot notation on `DifferentiableAt`/`Differentiable` mishandles
   field names that exist in multiple namespaces.** Workarounds:
   `Differentiable.fun_div` requires the `𝕜` to be a `NontriviallyNormedField`,
   and Lean's dot resolution sometimes binds `𝕜 := domain` instead of
   the codomain field. Use the function-call form `Differentiable.fun_div h₁ h₂ h₃`
   with explicit args, or `(h_num v).fun_div (h_denom v) (h_ne v)` for
   the `DifferentiableAt`-level call.

6. **`HasFDerivAt.comp_hasDerivAt` substitution at `f x` not auto-reducing.**
   When composing `HasFDerivAt l l' z` with `HasDerivAt (fun t => z + t • v) v 0`,
   Lean expects `l` to be `HasFDerivAt` at `f 0 = z + 0 • v`, which
   doesn't reduce to `z` definitionally without prompting. Use
   `HasFDerivAt.comp_hasDerivAt_of_eq h_l h_f (by simp)` to inject the
   reduction.

7. **Big surgery via `sed` on Lean files is risky.** When deleting
   parallel-track blocks, find the correct opening and closing line
   boundaries (look at section headers, not line counts).

8. **Doc-comment + axiom interaction.** Back-to-back `/-- ... --//-- ... -/`
   gives "unexpected token '/--'" — Lean wants exactly one doc-comment
   per declaration.

---

## Discipline notes (what the staged plan got right, branch-flavor)

- **Tree-green at every commit.** All four commits on this branch landed
  with `lake build` passing. Each commit is a self-contained chunk
  (chain threading, then Diff bridge promotions, then `pdivMat_rowIndep`
  + cascade, then `pdiv_gelu`).
- **`#print axioms` after each stage.** Every promoted theorem (the 8
  chains, the 5 Diff bridges, `pdivMat_rowIndep`, `pdiv_gelu`) was
  audited against pure-Mathlib closure (`propext / Classical.choice /
  Quot.sound`) before commit. Catches accidental dependencies that
  raw `^axiom` greps miss.
- **Don't optimize for axiom count alone.** The previous branch's
  cosmetic trivial-form commits (`9c03889`, `f94bc03`) were tried and
  reverted as noise. This branch's commits all add genuine proof
  content. The 4 Diff bridges added in `581a771` (and 2 of them
  promoted to theorems by `5532d15` + `f8ac5b1`) are honest book-keeping
  — the chains they enable being theorems is the substance.
- **Time-box ambitious proofs.** `pdiv_softmax` was attempted on this
  branch via the 1D directional-derivative approach. Math was right;
  Lean elaboration tripped on three things (see Pitfalls #5 + #6 + the
  `congr` over-reduction). Reverted rather than committing a half-done
  proof. The right next step is a focused 2-3 hour session with the
  attempted skeleton as the starting point — not interleaved with
  other work.

---

## Time estimate for remaining follow-up

- **`pdiv_softmax`:** 2-3 hours. Skeleton drafted; pitfalls cataloged.
- **`softmaxCE_grad`:** 1-2 hours, conditional on `pdiv_softmax`
  (theorem) — chain rule on `-log ∘ softmax_label`.
- **`pdiv_bnIstdBroadcast`:** 3-4 hours. Same 1D approach as
  `pdiv_softmax`, plus `bnVar` derivative algebra.

**Total to reach the floor:** ~one focused session of 6-9 hours.

---

## After the floor (10 axioms)

10 axioms, broken down:
- 3 ReLU subgradient conventions (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`).
- 3 non-smooth/boundary conventions (`conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3`).
- 2 vmap-VJP framework bundles (`mhsa_has_vjp_mat`, `mhsa_layer_flat_diff`).
- 2 opaque-codegen interfaces (`patchEmbed_flat_has_vjp`, `patchEmbed_flat_diff`).

Below 10 starts requiring framework-level changes:
- **Weakening `HasVJP.correct`** to a "smooth subset" formulation so the
  ReLU/conv/depthwise/maxpool subgradient axioms can be deduced from
  smaller base axioms.
- **A column-axis `pdivMat_rowIndep` analog** plus a ternary-input VJP
  framework so `mhsa_has_vjp_mat` follows from the single-head SDPA
  proof (`sdpa_back_{Q,K,V}_correct`, already proved as theorems on `main`).

Both are project-wide rewrites — separate multi-week efforts, not
this branch's continuation.
