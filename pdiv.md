# pdiv.md — final 4 axioms: tautology audit and elimination plan

**Status:** floor at 4 axioms after Phase 6b (`patchEmbed_flat_has_vjp`,
commit `3d7fa52`). The four remaining are framed as "subgradient
convention residuals" but an audit shows the framing is more nuanced:
**three are literal tautologies in current Lean** and the fourth is
morally a Mathlib theorem deferred for time. Below 4 honestly-counted
axioms is reachable in ~1-2 hours of work, not a multi-week project-
wide rewrite.

This doc lays out the audit, the trust-boundary that the `axiom`
keyword is currently flagging, and the two-phase plan to land at 0 or
1 axioms with no loss of soundness signal.

---

## Audit summary

| Axiom | File | True status |
|-------|------|-------------|
| `pdiv_relu` | `MLP.lean:290` | Morally a Mathlib theorem (~50-80 LOC) |
| `relu_has_vjp` | `MLP.lean:305` | **Tautology** — `def` with canonical witness |
| `mlp_has_vjp` | `MLP.lean:346` | **Tautology** — `def` with canonical witness |
| `maxPool2_has_vjp3` | `CNN.lean:1499` | **Tautology** — `def` with canonical witness |

### Why three are tautologies

`HasVJP f` is the structure

```lean
structure HasVJP {m n : Nat} (f : Vec m → Vec n) where
  backward : Vec m → Vec n → Vec m
  correct : ∀ (x : Vec m) (dy : Vec n) (i : Fin m),
    backward x dy i = ∑ j : Fin n, pdiv f x i j * dy j
```

For *any* `f`, the structure is inhabited by the canonical witness:

```lean
noncomputable def canonical_has_vjp {m n : Nat} (f : Vec m → Vec n) : HasVJP f where
  backward x dy i := ∑ j : Fin n, pdiv f x i j * dy j
  correct _ _ _ := rfl
```

This works because `pdiv` is now a `def` (`fderiv ℝ f x (basisVec i) j`,
`Tensor.lean:103`), not an axiom — `correct` is satisfied by `rfl`. Same
story for `HasVJP3` (`Tensor.lean:1551`) and `HasVJPMat` (`Tensor.lean:451`).

The earlier feedback memory ("trivial-form replacement is cosmetic noise
because pdiv is still axiomatic") was written **pre-foundation-flip**
when `pdiv` was an `axiom`. The feedback's premise no longer holds:
post-flip, the trivial `def` replacement adds zero new axiomatic
dependencies.

### What the `axiom` keyword is currently flagging

Real content is at the **codegen trust boundary**, not in Lean:

- **In Lean,** the canonical backward (= `∑ j, pdiv f x i j * dy j`) is
  Mathlib's junk default (= 0) at non-smooth points, because `fderiv`
  returns `0` at non-differentiable points by convention.
- **In codegen** (`MlirCodegen.lean:2260, 2426, 3737`), the emitted
  formulas are the standard subgradient conventions — `if x > 0 then
  dy else 0` for ReLU, argmax routing for max-pool. These match the
  canonical Lean witness at smooth points and **disagree** at the
  kinks (where the canonical witness is 0 and the codegen formula is
  whatever the convention picks).

So the `axiom` declaration is doing *documentary* work: it flags
"the formula codegen actually emits at the kink is a convention not
certified by Lean." That's an honest signal but the keyword `axiom`
overstates the gap — Lean's `correct` is satisfied (trivially), it
just isn't the formula codegen uses.

### Why `pdiv_relu` is morally a theorem

```lean
axiom pdiv_relu (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0)
    (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0
```

At `(∀ k, x k ≠ 0)`, ReLU agrees with a fixed diagonal CLM in a
neighborhood (radius `r < min |x k|` keeps every coord on the same
side of zero). Since `Differentiable` is local, `fderiv (relu n) x`
equals that CLM, and `pdiv` reads off the entry directly. **No
project-wide framework change required.**

### Soundness gap from the keyword `axiom`?

Audited the codebase for downstream extraction:

- Only `CNN.lean:1506` reads `.backward` from any of these axioms
  (`maxPool2_input_grad := maxPool2_has_vjp3.backward x dy`), and it's
  a `noncomputable abbrev` — no theorem proves a *value* for that
  alias.
- No `Differentiable ℝ (relu n)` claim exists anywhere in `LeanMlir/`.
  So `vjp_comp` cannot be applied with ReLU; no extracted-backward
  formula propagates into a Lean theorem.
- `tests/vjp_oracle/phase3/MainVjpOracleDenseRelu.lean` is a numerical
  FD smoke test on the codegen-emitted backward, independent of
  Lean's `correct`.

**Conclusion: no soundness bugs hiding.** Switching the three axioms
to `def`s is safe.

---

## Phase 1 — convert tautological axioms to definitions (~30 min)

Three near-identical edits.

### `relu_has_vjp` (MLP.lean:296-305)

Replace:

```lean
axiom relu_has_vjp (n : Nat) : HasVJP (relu n)
```

with:

```lean
/-- **ReLU bundled VJP — canonical (junk-at-kink) witness.**

    `HasVJP.correct` is satisfied by the canonical pdiv-derived backward
    (= `fderiv`'s junk default of 0 at the kinks). The codegen
    (`MlirCodegen.lean`) emits the standard subgradient formula
    `if x > 0 then dy else 0` instead — see "Codegen trust boundary"
    in `LeanMlir/Proofs/README.md`. -/
noncomputable def relu_has_vjp (n : Nat) : HasVJP (relu n) where
  backward x dy i := ∑ j : Fin n, pdiv (relu n) x i j * dy j
  correct _ _ _  := rfl
```

### `mlp_has_vjp` (MLP.lean:337-350)

Same shape:

```lean
noncomputable def mlp_has_vjp {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) where
  backward x dy i :=
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j
  correct _ _ _  := rfl
```

### `maxPool2_has_vjp3` (CNN.lean:1481-1500)

Same shape, with `HasVJP3` and `pdiv3`:

```lean
noncomputable def maxPool2_has_vjp3 {c h w : Nat} :
    HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w) where
  backward x dy ci hi wi :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo
  correct _ _ _ _ _ := rfl
```

`maxPool2_input_grad` (the `noncomputable abbrev` at `CNN.lean:1504`)
keeps its definition — same call site, now resolves to the canonical
backward via the `def` instead of the axiom.

### Verify

```bash
lake build LeanMlir.Proofs.MLP LeanMlir.Proofs.CNN \
           LeanMlir.Proofs.Attention
```

Expected: clean build. No downstream `rw`/`simp` site depends on the
*shape* of these backwards (only `.backward` extraction at
`CNN.lean:1506`, which is opaque to its body).

`#print axioms vit_full_has_vjp` should still show only Lean core
(`propext`, `Classical.choice`, `Quot.sound`).

**Result: 4 → 1 project axioms** (only `pdiv_relu` remains).

---

## Phase 2 — prove `pdiv_relu` from Mathlib (~50-80 LOC, half a day)

### Strategy

At `(∀ k, x k ≠ 0)`, ReLU is locally a fixed diagonal indicator CLM.
Use `Filter.EventuallyEq.fderiv_eq` to transport `fderiv` of the local
CLM (which is itself, since CLMs are their own `fderiv`) to ReLU.

### Proof skeleton

```lean
/-- ReLU's local diagonal indicator CLM at a smooth point `x`. -/
noncomputable def relu_local_CLM (n : Nat) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  ContinuousLinearMap.pi (fun i =>
    if x i > 0 then ContinuousLinearMap.proj i
                else (0 : Vec n →L[ℝ] ℝ))

theorem pdiv_relu (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0)
    (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0 := by
  -- 1. Pick a radius smaller than min |x k|.
  obtain ⟨r, hr_pos, hr_lt⟩ := -- minimum of |x k| over Fin n
    sorry
  -- 2. Show relu agrees with relu_local_CLM on Metric.ball x r.
  have h_local : Set.EqOn (relu n) (relu_local_CLM n x) (Metric.ball x r) := by
    intro y hy
    funext k
    -- |y k - x k| ≤ ‖y - x‖ < r ≤ |x k|, so y k has same sign as x k
    have h_close : |y k - x k| < |x k| := by
      calc |y k - x k| ≤ ‖y - x‖ := norm_apply_le_norm _ _
        _ < r := Metric.mem_ball_iff_norm.mp hy  -- or symm form
        _ ≤ |x k| := hr_lt k
    -- case split on x k > 0 or x k < 0; conclude y k > 0 or y k < 0 resp.
    sorry
  -- 3. Promote EqOn on a ball to EventuallyEq at x.
  have h_evt : (relu n) =ᶠ[𝓝 x] (relu_local_CLM n x) :=
    h_local.eventuallyEq_of_mem (Metric.ball_mem_nhds _ hr_pos)
  -- 4. Transport fderiv.
  have h_fderiv : fderiv ℝ (relu n) x = relu_local_CLM n x := by
    rw [h_evt.fderiv_eq]
    exact (relu_local_CLM n x).fderiv
  -- 5. Evaluate.
  unfold pdiv
  rw [h_fderiv]
  -- relu_local_CLM n x (basisVec i) j = if x j > 0 then (basisVec i) j else 0
  -- Expand via ContinuousLinearMap.pi_apply, basisVec_apply, case-split on i = j.
  sorry
```

### Tactical risks

1. **Step 1 (radius pick):** `min |x k|` over `Fin n` is straightforward
   via `Finset.inf'` on `Finset.univ` (non-empty since either `n = 0`
   trivializes the goal or `Fin n` is nonempty). For `n = 0`, both sides
   are vacuous — handle separately or use `Nat.recAux`.
2. **Step 2 (sign preservation):** the `calc` block uses
   `norm_apply_le_norm` (Mathlib has this for `Pi.normedAddCommGroup`).
   Alternatively `‖y - x‖ ≥ |y k - x k|` via `Pi.norm_def` /
   `Real.norm_eq_abs`. Sign-preservation is one `linarith` after
   `abs_lt.mp h_close`.
3. **Step 4 (transport):** `EventuallyEq.fderiv_eq` is in Mathlib as
   `Filter.EventuallyEq.fderiv_eq` (signature: `f =ᶠ[𝓝 x] g →
   fderiv ℝ f x = fderiv ℝ g x`). The CLM's self-fderiv is
   `ContinuousLinearMap.fderiv`.
4. **Step 5 (CLM evaluation):** mechanical. `pi_apply` to break the
   product, `basisVec_apply` to reduce `(basisVec i) j` to
   `if i = j then 1 else 0`, then case-split.

If any of these snag, fallback is to prove the `n = 1` case first as a
sanity check (single-coord ReLU on `Vec 1`), then generalize.

### Verify

```bash
lake build LeanMlir.Proofs.MLP
```

`#print axioms pdiv_relu` should show only Lean core.

**Result: 1 → 0 project axioms.**

---

## Phase 3 — README updates and codegen trust-boundary section (~20 min)

After Phases 1 and 2:

### `LeanMlir/Proofs/README.md`

Replace the "Axioms (4 total)" section with:

```markdown
## Axioms (0 project)

Pure-Mathlib closure on every theorem. `#print axioms vit_full_has_vjp`
shows only `propext`, `Classical.choice`, `Quot.sound` (Lean core).

The earlier 4-axiom floor was retired in Phase 7 (Apr 2026):
- `pdiv_relu` — proved via the local-diagonal-CLM transport (50-80 LOC).
- `relu_has_vjp`, `mlp_has_vjp`, `maxPool2_has_vjp3` — converted from
  `axiom` to `def` with the canonical pdiv-derived witness. The
  `correct` field holds by `rfl` since `pdiv` is a `def` over `fderiv`.

## Codegen trust boundary

`HasVJP.correct` certifies the *canonical* backward
`∑ j, pdiv f x i j * dy j`. At non-smooth points (ReLU at `x i = 0`,
maxPool at argmax ties), this is `fderiv`'s junk default of 0. The
codegen (`MlirCodegen.lean`) emits the standard subgradient formula
instead — `if x > 0 then dy else 0` for ReLU, argmax routing for
max-pool. These match the canonical Lean witness at smooth points and
differ at the kinks. The trust boundary is at codegen, not within Lean.
The numerical FD checks in `check_axioms.py` (and the `vjp_oracle`
end-to-end tests) cover the codegen-emitted formula.
```

### `pdiv.md` (this doc)

Update the Status header to mark Phases 1 / 2 / 3 as landed with
commit hashes.

### Memory update

Update `project_axiom_elimination.md`:
- Change "floor at 4 axioms" → "0 project axioms; codegen trust
  boundary documented".
- Note: the prior `feedback_axiom_count_metric.md` ("trivial-form
  replacement is cosmetic noise") had its premise change with the
  foundation flip — `pdiv` is now a `def`, so trivial-form `def`s
  add zero new axiomatic dependencies. Add a `Why no longer`: line.

---

## What this plan deliberately does NOT do

- **Does not "weaken `HasVJP.correct` to smooth-subset only".** That's
  the multi-week project-wide rewrite the prior plan flagged. With
  Phase 1's tautological-`def` move, that rewrite is unnecessary —
  the axioms already satisfy `correct` trivially via `rfl`. The
  rewrite would only be needed if we wanted to *prove* the codegen's
  subgradient formula matches `correct`, which is mathematically
  impossible (it doesn't, at the kinks).

- **Does not change codegen.** Codegen continues to emit the
  subgradient/argmax formulas. These are correct ML-framework
  conventions; the verification gap is intrinsic to "differentiation
  of non-smooth functions" and is the same gap PyTorch/JAX/TF live
  with. The honest place to flag it is the README, not an axiom
  declaration.

- **Does not touch FD checks or the `vjp_oracle` suite.** Those
  validate the codegen against finite differences and remain the
  empirical safety net for the kink conventions.

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Phase 1 build break (downstream `simp` matches old axiom shape) | Low — only `.backward` is extracted | If hit: full build with `lake build`, fix any `rfl` proofs that depended on opaque-axiom structure |
| Phase 2 `Filter.EventuallyEq.fderiv_eq` signature mismatch in pinned Mathlib | Low | Mathlib's `Filter.EventuallyEq.fderiv_eq` has been stable since 2023; if missing in pin, derive locally from `HasFDerivAt.congr_of_eventuallyEq` |
| `n = 0` edge case in radius pick | Low | Handle as separate match arm — `Fin 0` is empty, so the `i j` quantifier is vacuous |
| Reviewer pushback ("you're hiding the subgradient gap") | Medium | The README "Codegen trust boundary" section is the load-bearing part of this plan — if it lands clearly, the `def`-vs-`axiom` choice is just style |

---

## Time estimate

- Phase 1: 30 min (3 near-identical edits + verify build)
- Phase 2: 4-6 hours focused (proof + Lean dance + verify pure-Mathlib closure)
- Phase 3: 20 min (README + memory + commit messages)

**Total: half a day to one day** for 4 → 0 axioms with a stronger
honesty story than the current 4-axiom floor.

---

## Commit plan (one per phase, bisect-friendly)

1. `Proofs/MLP, CNN: convert tautological _has_vjp axioms to defs (-3 axioms)`
2. `Proofs/MLP: prove pdiv_relu via local-diagonal-CLM transport (-1 axiom)`
3. `Proofs/README: document codegen trust boundary; pdiv.md plan landed`

After commit 3, drop `pdiv.md` from the repo root or move to
`historical/` alongside the other landed plan docs.
