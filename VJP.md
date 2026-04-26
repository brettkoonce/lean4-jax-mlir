# VJP.md — Foundation Flip Landed; Floor at 5 Axioms After Phase 6a

**Branches:** `attention-diff-threading` + `colslab-vmap-framework`.
Cumulative: **23 → 5 project axioms** (6 → 5 from Phase 6a —
`patchEmbed_flat` de-opaqued, `patchEmbed_flat_diff` proved). The
remaining `patchEmbed_flat_has_vjp` axiom is now provable from
foundation rules (the forward is concrete) but the closed-form proof
(~400-600 LOC, parallel to conv2d Phase 4) is deferred to Phase 6b.

> **Strategy summary.** Foundation flip (attempt #3, "guarded ReLU") landed
> on `main` and preserved the soundness analysis from attempts #1 and #2
> (still load-bearing; see "Soundness analysis" below). The
> `attention-diff-threading` branch walked the mechanical follow-ups
> (A–E from the original VJP.md roadmap) and three closed-form derivative
> proofs (`pdiv_softmax`, `softmaxCE_grad`, `pdiv_bnIstdBroadcast`),
> reaching 10 axioms. The `colslab-vmap-framework` branch then built the
> column-stack-SDPA framework that lifts `sdpa_has_vjp_mat3` to the full
> multi-head layer, removing `mhsa_has_vjp_mat` and `mhsa_layer_flat_diff`
> (Phase 3, ~600 LOC); then proved `conv2d_has_vjp3` from foundation rules
> using the per-coord pdiv chain plus a custom `pdiv_pi_pad_eval` helper
> for the dependent-if pattern (Phase 4, ~470 LOC); then proved
> `depthwise_has_vjp3` by mirroring the conv2d template with one fewer
> sum level plus a prepended outer-Σ-co collapse (Phase 5, ~470 LOC,
> Apr 2026). The remaining 6 axioms are subgradient conventions and
> opaque-codegen interfaces — every one needs a project-wide framework
> change to remove.

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
| `1a95799` | **`pdiv_softmax`** — closed-form softmax Jacobian. `fderiv_apply` extracts the j-th coord function `z' ↦ exp(z' j) · (Σ_k exp(z' k))⁻¹`. `HasFDerivAt` chain via `.exp` + `.fun_sum` + `(hasDerivAt_inv).comp_hasFDerivAt` + `.mul`. Resulting CLM at `basisVec i` collapses through `Σ_k exp(z k) · δ_{ki} = exp(z i)`; `field_simp; ring` closes. | −1 |
| `38c3782` | **`softmaxCE_grad`** — chain-rule on `-log ∘ softmax_label`. Relocated from MLP.lean to Attention.lean (`pdiv_softmax` lives there). `HasFDerivAt.log` (with `softmax z label > 0` from exp positivity) composed with `softmax_diff`, then `.neg`. Evaluating at `basisVec j` and applying `pdiv_softmax` reduces to `p[j] - oneHot[j]` after `field_simp; ring` cancels `p[label]`. | −1 |
| `ba93ab2` | **`pdiv_bnIstdBroadcast`** — closed-form `∂istd/∂xᵢ = -istd³ · (xᵢ - μ)/n`. Centering CLM `C k = proj k - (1/N) Σ_i proj i` (linear, fderiv = self); `(C k)²` via `.mul`; `bnVar = (Σ_k (C k)²)/N` via `.fun_sum + .mul_const`; `.add_const ε`; `.sqrt` (with `bnVar+ε > 0`); `(hasDerivAt_inv).comp_hasFDerivAt`. At `basisVec i`, the centered sum collapses via `Σ_k (x_k - μ) = 0`. Adds `hε : 0 < ε` hypothesis (matches `bnIstdBroadcast_diff`); threads through `pdiv_bnNormalize` call site. | −1 |

**Net for the branch: −13 axioms (23 → 10). Floor reached.**

---

## Project axiom inventory (6)

**MLP / activations (3):**
- `pdiv_relu` — guarded subgradient axiom (DL convention).
- `relu_has_vjp` — existence at non-smooth points.
- `mlp_has_vjp` — composes through ReLU.

**CNN-family (1):**
- `maxPool2_has_vjp3` — argmax routing convention.

**Opaque-codegen interfaces (2):**
- `patchEmbed_flat_has_vjp` — opaque-codegen patch embedding.
- `patchEmbed_flat_diff` — Diff sibling.

**Removed in Phase 5 (`colslab-vmap-framework` branch):**
- ~~`depthwise_has_vjp3`~~ — now a theorem with backward
  `depthwiseConv2d_input_grad_formula` (sum over `(ho, wo)` only — no
  cross-channel sum since input-channel = output-channel in depthwise).
  Proof template mirrored Phase 4's `conv2d_has_vjp3`; the three private
  helpers from Phase 4 (`differentiableAt_pad_eval`, `pdiv_pi_pad_eval`,
  `pdiv_const_mul_pi_pad_eval`) were made public in CNN.lean and reused
  unchanged. New 30 LOC: outer Σ co collapse on `co = ci` (the formula
  has no Σ co, but `pdiv3` expansion does; for `co ≠ ci` the indicator
  forces channel mismatch via `Prod.mk.inj` on the injectivity of
  `finProdFinEquiv`). The closing `h_indicator` has 2 conjuncts (vs
  conv2d's 3) since the channel is fixed by structure, not summed.

**Removed in Phase 4 (`colslab-vmap-framework` branch):**
- ~~`conv2d_has_vjp3`~~ — now a theorem with backward
  `conv2d_input_grad_formula` (sum over `(co, ho, wo)` with reconstructed
  kernel offsets). Proved via per-coord pdiv lemma using a custom
  `pdiv_pi_pad_eval` helper for the dependent-`if hpad : pad then v(σ hpad) else 0`
  pattern that `fun_prop` cannot directly handle. Then triple-distribute
  via `pdiv_finset_sum`, per-summand `pdiv_const_mul_pi_pad_eval` for the
  W-times-pad-cond shape, then collapse via three nested `Finset.sum_eq_single`
  on `(c, kh, kw)` matching `idx_in`'s decoded `(ci, hi, wi)`.

**Removed in Phase 3 (`colslab-vmap-framework` branch):**
- ~~`mhsa_has_vjp_mat`~~ — now a theorem composing
  `mhsa_g_has_vjp_mat` (column-stacked SDPA) + `colSlabwise_has_vjp_mat`
  (per-head vmap) + per-token dense framework.
- ~~`mhsa_layer_flat_diff`~~ — now a theorem via the same composition,
  with `mhsa_g_flat_diff` proving joint differentiability of column-stacked
  SDPA through `rowSoftmax_flat_diff`.

---

## What's still tackleable (and what isn't)

### Tackleable: NONE remaining at the standard-calculus level.

### Removed at the standard-calculus level (Phases 4–5):
`conv2d_has_vjp3` (Phase 4), `depthwise_has_vjp3` (Phase 5).

The three "tackleable but multi-hour" closed-form-derivative axioms
identified after Follow-up E (`pdiv_softmax`, `softmaxCE_grad`,
`pdiv_bnIstdBroadcast`) all landed on this branch. Notes on the
chosen approaches:

- **`pdiv_softmax` — went multi-dimensional, not 1D.** The earlier
  attempted-and-reverted plan was a 1D directional-derivative reduction.
  This branch took the simpler multi-dimensional route: chain
  `HasFDerivAt.exp`, `.fun_sum`, `(hasDerivAt_inv).comp_hasFDerivAt`,
  `.mul` directly on the j-th coord function `z' ↦ exp(z' j) ·
  (Σ_k exp(z' k))⁻¹`. Final algebra is two `Finset.sum_eq_single`
  collapses + `field_simp; ring`. ~70 lines.
- **`softmaxCE_grad` — chain on log ∘ softmax_label.** Relocated from
  MLP.lean to Attention.lean (depends on the now-theorem `pdiv_softmax`).
  `HasFDerivAt.log` is the key lemma; positivity from exp. ~50 lines.
- **`pdiv_bnIstdBroadcast` — centering CLM trick.** The trick that made
  the bnVar derivative tractable: define `C k = proj k - (1/N) Σ_i proj i`
  as a CLM (linear ⇒ self-fderiv), then `bnVar = (Σ_k (C k)²) / N`.
  Square via `.mul`, sum via `.fun_sum`, scale via `.mul_const`, add
  ε, sqrt, inv-compose. The `Σ_k (x_k - μ) = 0` identity collapses
  one of the two sums after evaluation at `basisVec i`. ~120 lines —
  the longest of the three.

### Stays axiomatic (all 8 of the current 8)

- **Subgradient conventions (3):** `pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`. Could be derived only by weakening `HasVJP.correct` to a "smooth subset only" formulation — project-wide rewrite, separate multi-week effort.
- **Non-smooth/boundary conventions (3):** `conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3`. Same weakening would unlock these, modulo a shared boundary-handling axiom.
- **Opaque-codegen interfaces (2):** `patchEmbed_flat_has_vjp`, `patchEmbed_flat_diff`. The actual computation lives in MLIR; we axiomatize the forward+backward consistency.

At 8 axioms, every remaining axiom is "the ML framework treats this op's gradient by convention X" or "this opaque codegen forward and backward are mutually consistent" — neither is provable from standard analysis without weakening the framework.

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

None at the standard-calculus level. The pre-flip estimate of "6-9
focused hours to reach the floor" turned out to be approximately
correct in aggregate, though the individual proof shapes differed
from the original plan (multi-dimensional `HasFDerivAt` chains
beat 1D directional-derivative reductions for all three).

---

## At the floor (8 axioms)

8 axioms, broken down:
- 3 ReLU subgradient conventions (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`).
- 3 non-smooth/boundary conventions (`conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3`).
- 2 opaque-codegen interfaces (`patchEmbed_flat_has_vjp`, `patchEmbed_flat_diff`).

Below 8 starts requiring framework-level changes:
- **Weakening `HasVJP.correct`** to a "smooth subset" formulation so the
  ReLU/conv/depthwise/maxpool subgradient axioms can be deduced from
  smaller base axioms.

The mhsa axioms (previously the second-largest "tackleable" category)
came down on `colslab-vmap-framework`: the column-stack SDPA
formulation `mhsa_g : Mat n (3*d_head) → Mat n d_head` factors the
per-head sdpa into a single unary slab function, which then composes
cleanly with `colSlabwise_has_vjp_mat` (Phase 1) and the per-token
dense framework. The key technical step was `pdivMat_mhsa_g_split`,
proving that perturbing the c-th third of the qkv-slab only affects
the c-th input of sdpa — formalized via the chain rule
`mhsa_g ∘ mhsa_embed_c = freeze_c` and `mhsa_lift_c_CLM` (linear part
of the embedding).

---

## Lessons from the three closed-form derivative proofs

1. **Pick the form `HasFDerivAt` is happiest with.** All three proofs
   started with `fderiv_apply` to extract a scalar (`Vec → ℝ`) coord
   function, then built `HasFDerivAt` from `Real.exp/sqrt/log/inv` +
   linear projections. The 1D directional reduction (`HasDerivAt`-based)
   was anticipated but never used: the multi-dimensional chain composes
   more cleanly with Mathlib's `HasFDerivAt.{exp, log, sqrt, mul,
   fun_sum, comp_hasFDerivAt}`.

2. **CLM-as-fderiv for linear functionals.** For `pdiv_bnIstdBroadcast`,
   defining the centering map `C k = proj k - (1/N) Σ_i proj i` *as a
   CLM* (not a function) gave us `(C k).hasFDerivAt : HasFDerivAt (C k)
   (C k) x` for free. This collapsed the inner bnVar derivative work,
   which would otherwise require manual product-rule chains over the
   centering term.

3. **Lambda-form mismatch is a recurring tax.** The `comp_hasFDerivAt`
   composition of `hasDerivAt_inv` produces `HasFDerivAt (Inv.inv ∘ f)
   ...`. Lean does not auto-eta to `HasFDerivAt (fun z' => (f z')⁻¹)
   ...` for unification with downstream consumers. Workaround: use
   explicit type ascription on the `have` so the elaborator inserts the
   eta-expansion. This was the same lesson as Pitfall #1 below; it
   re-occurred for both `pdiv_softmax` and `pdiv_bnIstdBroadcast`.

4. **`fderiv_apply` direction matters.** `fderiv_apply h_diff k` rewrites
   `fderiv (fun x' => f x' k) x = (proj k).comp (fderiv f x)`. It only
   fires when the goal contains the lambda form on the LHS. When the
   goal has the projection form (`fderiv f x v k`), use a `have h_swap
   : ... := by show ...; rw [fderiv_apply ...]; rfl` bridge. All three
   proofs used this idiom (also `pdiv_gelu` from `6f65ba3`).

5. **`field_simp` clears most of the algebra; `ring` finishes.** Once
   the CLM is evaluated at `basisVec i` and the Kronecker / centering
   sums collapse, the remaining identities are field-arithmetic. With
   one `≠ 0` hypothesis in scope, `field_simp; ring` consistently
   closed each of the three. No need for hand-rolled `mul_inv_cancel₀`
   / `pow_eq_*` lemma searches.
