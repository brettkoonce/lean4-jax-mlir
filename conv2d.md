# conv2d.md — input-VJP elimination plan (Phases 1, 2, 6)

**Phase 1 (conv2d) status:** LANDED (commit `0b03697`).
~470 LOC, pure-Mathlib closure verified. See **"Phase 1 lessons"** below
for the recipe.

**Phase 2 (depthwise) status:** LANDED (commit `50e977a`).
~470 LOC including 30 LOC of new outer-Σ-co collapse logic. The proof
template mirrored conv2d directly — single-shot build, no Lean errors,
in part because the three private helpers from Phase 1 (now public in
CNN.lean) generalized cleanly.

**Phase 6a (patchEmbed forward + diff) status:** LANDED (commit `a0859d7`).
~110 LOC. De-opaqued `patchEmbed_flat` from `noncomputable opaque` to
concrete `noncomputable def`; replaced `axiom patchEmbed_flat_diff`
with a theorem proved via `fun_prop` + `differentiableAt_pad_eval`.

**Phase 6b (patchEmbed HasVJP) status:** LANDED.
~800 LOC including the closed-form input-grad formula and the full
correctness proof. The proof uses the absorb-trick: pack the `n.val = 0`
CLS-row case into the pad guard of `var_F`, so the chain rules
(`pdiv_finset_sum` × 3 + `pdiv_const_mul_pi_pad_eval`) apply uniformly.
The closing splits `Σ n : Fin (N+1)` via `Fin.sum_univ_succ` into the
n=0 row (zero contribution) and `Σ p : Fin N` (n = p.succ), then per-p
collapses Σ c on c=c_in, simplifies the indicator via injectivity of
`finProdFinEquiv`, and reorders Σ d using two `Finset.sum_comm`
applications to align with the formula's `∑ p kh kw, [if h_match then
∑ d, ... else 0]` shape.

Axiom count: 8 → 7 (Phase 1) → 6 (Phase 2) → 5 (Phase 6a) → 4 (Phase 6b).

`#print axioms vit_full_has_vjp` confirms pure-Mathlib closure: only
`propext`, `Classical.choice`, `Quot.sound` (Lean core).

The 4 remaining are pure framework-convention axioms:
- 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`)
- 1 maxPool2 subgradient (`maxPool2_has_vjp3`)

Below 4 requires `HasVJP.correct` weakening to "smooth subset only"
(project-wide rewrite, multi-week, out of scope).

---

## Phase 6b — patchEmbed HasVJP closed-form (planned)

The forward (de-opaqued in Phase 6a) is at `Attention.lean:2746`:

```lean
noncomputable def patchEmbed_flat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D) :=
  fun img =>
    fun idx_out =>
      let n := (finProdFinEquiv.symm idx_out).1
      let d := (finProdFinEquiv.symm idx_out).2
      pos_embed n d +
        (if n.val = 0 then
          cls_token d
         else
          b_conv d +
          ∑ c kh kw, W_conv d c kh kw *
            (let W' := W / patchSize
             let p := n.val - 1
             let h' := p / W'
             let w' := p % W'
             let hh := h' * patchSize + kh.val
             let ww := w' * patchSize + kw.val
             if hpad : hh < H ∧ ww < W then
               img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
             else 0))
```

### Closed-form backward

```lean
noncomputable def patchEmbed_input_grad_formula
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize)
    (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    ∑ p : Fin N, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if h_match : h' * patchSize + kh.val = hh.val ∧
                   w' * patchSize + kw.val = ww.val then
        ∑ d : Fin D, W_conv d c kh kw *
          dy (finProdFinEquiv (⟨p.val + 1, Nat.succ_lt_succ p.isLt⟩, d))
      else 0
```

Why the `(p, kh, kw)` outer loop (not `(h', w', kh, kw)`): the same
reasoning as conv2d's `(co, ho, wo)` — the loop variable `p` directly
matches the patch-index axis of `dy`, no partial bijection needed.

### Proof skeleton

Mirrors conv2d's `correct` proof at `CNN.lean:284-758`. The new wrinkle
is the n=0 (CLS row) case where the function is constant in img.

```lean
noncomputable def patchEmbed_flat_has_vjp ... :
    HasVJP (patchEmbed_flat ...) where
  backward := fun _img dy => patchEmbed_input_grad_formula ic H W patchSize N D W_conv dy
  correct := by
    intro img dy idx_in
    set c, hh, ww via finProdFinEquiv.symm chain.

    -- Step 1: per-(idx_in, idx_out) pdiv. Two cases on n.val = 0.
    have h_pdiv : ∀ idx_out, pdiv f img idx_in idx_out = ... := by
      intro idx_out
      set n := (finProdFinEquiv.symm idx_out).1
      set d := (finProdFinEquiv.symm idx_out).2

      by_cases hn0 : n.val = 0
      · -- n = 0: f at idx_out is `pos_embed n d + cls_token d`, constant in img.
        --   pdiv = 0. RHS at n=0 also = 0 (the formula's `if n=0 then 0 else ...`).
        rw [show (fun img' => patchEmbed_flat ... img' idx_out)
              = (fun _ => pos_embed n d + cls_token d) from by
              funext img'; unfold patchEmbed_flat; simp [hn0]]
        rw [pdiv_const]
        -- Show RHS at hn0 also = 0.
        ...

      · -- n > 0: f at idx_out = pos_embed n d + b_conv d + Σ c' kh kw, W_conv * pad-img.
        -- Same structure as conv2d Phase 1 / depthwise Phase 2.
        -- Decompose constant + linear-in-img, apply pdiv_add + pdiv_const,
        -- triple distribute Σ c' kh kw via pdiv_finset_sum,
        -- per-summand pdiv_const_mul_pi_pad_eval.
        ...

    -- Step 2: closing collapse.
    show patchEmbed_input_grad_formula ... = ∑ idx_out, pdiv * dy idx_out
    -- Reindex Σ idx_out ≃ Σ n × Σ d via two `Fintype.sum_equiv finProdFinEquiv.symm`
    -- + `Fintype.sum_prod_type` (see depthwise_bias_grad_has_vjp at
    -- Depthwise.lean:592-617 for the exact pattern).
    rw [Fintype.sum_equiv finProdFinEquiv.symm ...]
    rw [Fintype.sum_prod_type]

    -- Now: ∑ p kh kw, formula = ∑ n d, pdiv * dy(flat(n, d)).

    -- Outer Σ n collapse: split n=0 vs n>0.
    -- For n=0: contribution is 0 (use h_pdiv at hn0).
    -- For n>0: reindex n ↔ p+1 via Fintype.sum_equiv.
    -- The Fin (N+1) → {0} ⊕ Fin N split or `Finset.sum_eq_zero` + reindex.

    -- Then for each (p, d), unfold h_pdiv, simp Equiv.symm_apply_apply,
    -- pull dy out, congr 1, h_indicator (2-conjunct, similar to depthwise),
    -- by_cases on the back_cond, Σ kh kw collapse via Finset.sum_eq_single twice.
    ...
```

### Pitfalls (carried from Phases 1, 2)

The 7 pitfalls from Phase 1 (see "Phase 1 lessons" below) all apply.
**One new pitfall** specific to patchEmbed:

8. **The n.val = 0 case split must come BEFORE the Σ c kh kw distribution.**
   If you try to apply `pdiv_const_mul_pi_pad_eval` directly without
   first splitting on n.val, the helper assumes the inner is the standard
   `if hpad : pad then v(σ hpad) else 0` shape. But for n = 0, the
   function is `cls_token d` (constant), not the pad-eval pattern. Two
   options:
   (a) Split early via `by_cases hn0 : n.val = 0` at the top of `h_pdiv`.
   (b) Write the forward to absorb n=0 into the pad guard:
       `if hpad : 1 ≤ n.val ∧ hh < H ∧ ww < W then img(σ) else 0`.
       But this changes the forward def, breaking Phase 6a's commit.
   Recommendation: (a) — keep Phase 6a's forward stable.

### Outer Σ n collapse on `n.val = 0` exclusion

The closing collapse needs to convert `∑ n : Fin (N+1), body(n)` (where
body is 0 for n=0) into `∑ p : Fin N, body(p+1)`. Two equivalent
approaches:

**Option A: Use `Fin.sum_univ_succ`.**
```lean
rw [Fin.sum_univ_succ]  -- splits ∑ n : Fin (N+1) into body(0) + ∑ p : Fin N, body(p.succ)
-- body(0) reduces to 0 by hn0.
-- Σ p part is the desired form.
```

**Option B: Use `Fintype.sum_equiv` with `Fin.succAboveEmb`.**
More flexible but more bookkeeping.

**Option A is recommended** — `Fin.sum_univ_succ` is exactly the right
shape. The `body(0)` term should simplify to 0 directly via the n=0
case of `h_pdiv`.

### Reusable helpers from Phase 1 (also used by Phase 2)

The three helpers in CNN.lean:145-242 are public:
- `differentiableAt_pad_eval`
- `pdiv_pi_pad_eval`
- `pdiv_const_mul_pi_pad_eval`

For Phase 6b: instantiate `c_const idx_out := W_conv d c kh kw` (where
d is decoded from idx_out's column-second component) and `σ idx_out hpad`
encoding the input position via the patch decomposition `(c, h'*P+kh, w'*P+kw)`.

### Estimated LOC and time

- **Formula def:** ~12 LOC.
- **Per-coord pdiv lemma (`h_pdiv`):** ~200 LOC.
  - n=0 case: ~30 LOC (constant-in-img argument).
  - n>0 case: ~170 LOC (mirror conv2d, no Σ c collapse needed since
    channel decoded from idx_out directly — like depthwise).
- **Closing collapse:** ~150 LOC.
  - Reindex Σ idx_out → Σ n d: ~20 LOC.
  - Outer Σ n collapse on n=0 exclusion: ~30 LOC (use `Fin.sum_univ_succ`).
  - Per-(p, d): pull dy out, h_indicator, Σ kh kw collapse: ~100 LOC.

**Total: ~400-600 LOC.** Time estimate: 4-6 focused hours.

### When to pick Phase 6b back up

The right time:
- A clean ~half-day window (similar to Phase 1).
- Phases 1-2's CNN.lean / Depthwise.lean recently reviewed.
- Comfortable with Phase 1's pitfalls (1-7 below).

After 6b lands: axiom count drops 5 → 4. The blueprint claim "verified
ViT modulo subgradient conventions" becomes literal.

---

## Phase 2 axiom count summary (post-Phase-6a)

Axiom count: **6 → 5**. The 5 remaining are:
- 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`)
- 1 maxPool2 subgradient (`maxPool2_has_vjp3`)
- 1 patchEmbed HasVJP (`patchEmbed_flat_has_vjp` — provable, deferred)

---

## What's different from conv2d

| Aspect | conv2d | depthwise |
|---|---|---|
| Output dims | `oc` separate from `ic` | output `c` = input `c` |
| Kernel shape | `(oc, ic, kH, kW)` | `(c, kH, kW)` |
| Channel sum | `Σ c kh kw` (3-level) | `Σ kh kw` (2-level, no `Σ c`) |
| Per-summand pdiv | `W o c kh kw * indicator` | `W co kh kw * indicator` |
| Channel match | `c = ci` (extracted via `Prod.mk.inj`) | `co = ci` (the output channel matches input) |

Concrete forward (Depthwise.lean:67):
```lean
depthwiseConv2d W b x ch hi wi
  = b ch + ∑ kh kw, W ch kh kw *
      (if hpad : pad(kh, hi, kw, wi) then x ch ⟨hh-pH, _⟩ ⟨ww-pW, _⟩ else 0)
```

The key observation: the FORWARD output channel `ch` IS the input
channel for `x`. There's no cross-channel mixing. So in `pdiv F v idx_in
idx_out` (with idx_in = (ci, hi_in, wi_in), idx_out = (co, ho, wo)):
- For `co ≠ ci`: pdiv = 0 (the W and indicator both reference `co`, but
  the v-eval is at `co` too — for the indicator to match `idx_in = (ci, ...)`,
  we'd need the v-eval index's first component = ci, but it's co. So `co ≠ ci`
  forces no match).
- For `co = ci`: pdiv = `∑ kh kw, W ci kh kw * (if pad ∧ kh+ho-pH = hi.val ∧
  kw+wo-pW = wi.val then 1 else 0)`.

---

## Reusable helpers from Phase 1 (CNN.lean:145–242)

**Important:** the three helpers are currently `private` in CNN.lean.
For depthwise, either:
- (a) change `private` → public (recommended; they're generic enough to
  live as utility lemmas), or
- (b) move them to `Tensor.lean` near the existing `pdiv_*` rules (better
  long-term home), or
- (c) re-declare them in Depthwise.lean (worst — code duplication).

The helpers (signatures, all `Vec n` / `Vec m` parameterized — no
conv2d-specific types):

```lean
/-- Differentiability of `if hpad : P then v(σ hpad) else 0`. -/
private lemma differentiableAt_pad_eval {n : Nat} (P : Prop) [Decidable P]
    (σ : P → Fin n) (v : Vec n) :
    DifferentiableAt ℝ (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) v

/-- pdiv of a per-output dependent if-eval-or-zero family. -/
private lemma pdiv_pi_pad_eval {n m : Nat}
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            if h : P k' then v' (σ k' h) else (0 : ℝ))
          v idx_in idx_out =
    if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0

/-- pdiv of `c_const * pad-eval` family — the conv2d per-summand pattern. -/
private lemma pdiv_const_mul_pi_pad_eval {n m : Nat}
    (c_const : Fin m → ℝ)
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            c_const k' *
            (if h : P k' then v' (σ k' h) else (0 : ℝ)))
          v idx_in idx_out =
    c_const idx_out *
    (if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0)
```

For depthwise, instantiate `c_const idx_out := W ci kh kw` (or the
appropriate decoded form), and `σ idx_out hpad := finProdFinEquiv (...)`
encoding the input position.

---

## Recommended formula form: `(ho, wo)` loop, NOT `(kh, kw)`

The conv2d formula uses `∑ co ho wo, body` (output-position loop). For
depthwise, drop the `Σ co` (since output channel = input channel = ci):

```lean
noncomputable def depthwiseConv2d_input_grad_formula {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) : Tensor3 c h w :=
  fun ci hi wi =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy ci ho wo
      else 0
```

**Why `(ho, wo)` not `(kh, kw)`:** the `(ho, wo)` form falls out
directly from the pdiv expansion. The `(kh, kw)` form (which the original
conv2d.md proposed) would require a partial bijection between `Fin h`
and `Fin kH` — ~50 LOC of `Finset.sum_bij` bookkeeping to no benefit.
Phase 1 switched mid-proof to `(ho, wo)` after this realization.

---

## Proof skeleton (mirror conv2d's structure)

The full conv2d proof is at `LeanMlir/Proofs/CNN.lean` lines 250–706.
Adapt with these substitutions:

| conv2d | depthwise |
|---|---|
| `Kernel4 oc ic kH kW` | `DepthwiseKernel c kH kW` |
| `oc * ic * kH * kW` flat | (no flat — kernel is 3D, same shape as `Tensor3`) |
| `∑ c kh kw` triple sum | `∑ kh kw` double sum |
| `W o(k') c kh kw` | `W o(k') kh kw` (no separate c index) |
| Per-(c, kh, kw) summand | Per-(kh, kw) summand |
| Channel match: `c = ci` extracted via Prod.mk.inj on inner pair | Channel match: built into the auto-`co = c` since v-eval uses output channel directly |

### Step-by-step

```lean
noncomputable def depthwise_has_vjp3 ... : HasVJP3 (depthwiseConv2d W b) where
  backward := fun _x dy => depthwiseConv2d_input_grad_formula W dy
  correct := by
    intro x dy ci hi wi
    set idx_in : Fin (c * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in

    -- Step 1: per-(idx_in, idx_out) pdiv lemma (UN-collapsed).
    have h_pdiv : ∀ idx_out : Fin (c * h * w),
        pdiv (fun v' : Vec (c * h * w) =>
                Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ kh : Fin kH, ∑ kw : Fin kW,
          W (ohw_o := decoded) kh kw *
            (if hpad : pad(kh, ohw_hi, kw, ohw_wi) then
              (if idx_in = finProdFinEquiv (finProdFinEquiv
                  (ohw_o, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩)
                then 1 else 0)
             else 0) := by
      intro idx_out
      set ohw_wi := ...; set ohw_hi := ...; set ohw_o := ...
      -- Decompose F = b + var_part (same as conv2d).
      rw [show F = (fun v' k' => b_part v' k' + var_part v' k') from ...]
      have h_b_diff := differentiableAt_const _
      have h_lin_diff : ... := by
        rw [differentiableAt_pi]; intro k'
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _ (...) _
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv b_part _ _ _ = 0 from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute over kh-sum.
      rw [show var_part = (fun v'' k' => ∑ kh, summand_kh) from rfl]
      have h_kh_diff : ∀ khh ∈ univ, DifferentiableAt ℝ (summand_kh khh) _ := ...
      rw [pdiv_finset_sum _ _ _ h_kh_diff]
      congr 1; ext khh
      -- Distribute over kw-sum.
      rw [show summand_kh khh = (fun v''' k'' => ∑ kw, summand_kw khh kw) from rfl]
      have h_kw_diff : ∀ kww ∈ univ, DifferentiableAt ℝ (summand_kw khh kww) _ := ...
      rw [pdiv_finset_sum _ _ _ h_kw_diff]
      congr 1; ext kww
      -- Per-(kh, kw) summand: factor as W (constant) * (dite).
      -- Note: σ for depthwise references `ohw_o` (the output channel = input channel).
      rw [show summand_kw khh kww = (fun v k' => 
            (W-factor k') * (if hpad : pad then v(σ(...)) else 0)) from by
        funext v k'; unfold Tensor3.unflatten; rfl]
      rw [pdiv_const_mul_pi_pad_eval ...]

    -- Step 2: closing collapse.
    show depthwiseConv2d_input_grad_formula W dy ci hi wi =
      ∑ co ho wo, pdiv3 (depthwiseConv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold depthwiseConv2d_input_grad_formula pdiv3
    -- Note: LHS has no Σ co. RHS has Σ co ho wo.
    -- Collapse Σ co on co = ci first.
    rw [Finset.sum_eq_single ci ?_ ?_].symm
    rotate_left
    · -- For co ≠ ci: ∑ ho wo, pdiv * dy = 0 because pdiv = 0.
      intro co _ hco_ne
      apply Finset.sum_eq_zero; intro ho _
      apply Finset.sum_eq_zero; intro wo _
      rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (co, ho), wo))]
      simp only [Equiv.symm_apply_apply]
      -- Show ∑ kh kw, W co kh kw * (if pad ∧ idx_in = flat(co, ...) then 1 else 0) = 0.
      -- For co ≠ ci, idx_in = flat(co, ...) is false (first component mismatch),
      -- so all indicators are 0.
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [show ((let pH := ...; if hpad : pad then (if idx_in = ... then 1 else 0) else 0) : ℝ) = 0 from ?_]
      · ring
      by_cases hpad : pad
      · rw [dif_pos hpad, if_neg ?_]
        intro h_eq
        rw [hidx_in] at h_eq
        have h_inj := finProdFinEquiv.injective h_eq
        have h_inj_pair := Prod.mk.inj h_inj
        have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
        have h_inj_inner_pair := Prod.mk.inj h_inj_inner
        exact hco_ne h_inj_inner_pair.1.symm
      · rw [dif_neg hpad]
    · intro hni; exact absurd (Finset.mem_univ ci) hni
    -- Now: LHS = ∑ ho wo, pdiv(at co=ci) * dy ci ho wo
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    -- Per-(ho, wo): same recipe as conv2d (pull dy out, congr 1, h_indicator,
    -- by_cases back_cond, Σ kh kw collapse via Finset.sum_eq_single).
    -- The h_indicator now has 2 conjuncts (kh+ho = hi+pH ∧ kw+wo = wi+pW),
    -- not 3 (no `c = ci` since c isn't a sum index).
    sorry
```

The closing's per-(ho, wo) inner equality is structurally a 2-level
collapse (Σ kh kw) instead of 3-level (Σ c kh kw). Otherwise the same
shape as conv2d's.

---

## Pitfalls (from Phase 1 — pre-mitigate)

1. **`fun_prop` does not handle `dite`.** Use the helpers; do NOT try
   to `fun_prop` directly on the body containing `if hpad : pad then ...`.
   Phase 1 hit `No theorems found for dite` on the very first attempt.

2. **`congr 1` does NOT split Prod equality automatically.** When proving
   `finProdFinEquiv (a, b) = finProdFinEquiv (c, d)`, `congr 1` opens to
   `(a, b) = (c, d)` but a SECOND `congr 1` may not split into `a = c`
   and `b = d`. Use `Prod.mk.inj` on the hypothesis side (extracting from
   `h_inj_pair := Prod.mk.inj h_inj`) and explicit substitution
   `rw [← h_c, ← h_hi, ← h_wi]` on the goal side.

3. **Beta reduction needed before `dif_pos`/`dif_neg`.** After `funext v'`
   the goal has form `(fun v'' k' => ...) v' idx_out = ...`. Lean does NOT
   auto-beta-reduce. Use `show (if h : P idx_out then v' (σ idx_out h) else 0) = ...`
   to force beta, then `rw [dif_pos hP]`.

4. **`Fin.ext_iff.mp` direction.** `h_inj_inner_pair.2 : hi = ⟨..., _⟩`
   gives `Fin.ext_iff.mp this : hi.val = (⟨_⟩).val = kh.val + ho.val - pH`.
   Do NOT add `.symm` if the target is `hi.val = ...`. Phase 1 spent two
   build-error rounds chasing this.

5. **`set` abbreviations don't propagate into bound function bodies.**
   `set ohw_o := ... with hohw_o` rewrites the OUTER goal but NOT the
   `fun v'' k' => ... (decoded k')` lambda body. Inside the lambda, k' is
   bound and the decoding is fresh. For depthwise, set the abbreviations
   (ohw_o, ohw_hi, ohw_wi) but expect to handle decoded forms separately
   inside the function body.

6. **Indicator simplification: prove a non-dependent form first.** Phase 1
   used a sub-lemma `h_indicator` that converts the dependent-if indicator
   to a non-dependent conjunction `c = ci ∧ kh+ho = hi+pH ∧ kw+wo = wi+pW`.
   For depthwise: drop the `c = ci` conjunct (channel is forced by the
   structure). The conjunction becomes 2 conjuncts.

7. **Σ over a Fin not in scope: `intro hni; exact absurd (Finset.mem_univ ci) hni`.**
   `Finset.sum_eq_single` has a side condition "if the chosen point is not
   in the Finset, the result is the default". For `Finset.univ`, this is
   always vacuous; the explicit dispatch is `intro hni; exact absurd
   (Finset.mem_univ ci) hni`.

---

## Estimated LOC and time

- **Formula def:** ~12 LOC.
- **Per-coord pdiv lemma (`h_pdiv`):** ~150 LOC (about 60% of conv2d's
  due to one fewer sum level).
- **Closing collapse:** ~150 LOC (similar to conv2d, plus an outer
  `Σ co` collapse on `co = ci` at the start).
- **Helper migration** (private → public, or move to Tensor.lean): ~5 LOC
  of edits.

**Total: ~300–400 LOC.**

**Time estimate: 4–6 focused hours.** Faster than Phase 1 (12–16 hours)
because:
- The proof template is locked in (mirror conv2d).
- The 3 helpers are reused (no new diff/pdiv infrastructure).
- One fewer sum level (Σ kh kw vs Σ c kh kw).

---

## When to pick this back up

The right time:
- A clean ~half-day window.
- Phase 1's CNN.lean (lines 145–706) recently reviewed.
- You're OK changing 3 `private lemma` → public (or moving them to
  Tensor.lean).

The bad time: as a side-quest. The pitfall #4 alone (Fin.ext_iff
direction) can eat an hour if you're not focused.

---

## What landing this buys

- Axiom count: **7 → 6**.
- The 6 remaining are pure framework-convention axioms:
  - 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`).
  - 1 maxPool2 subgradient (`maxPool2_has_vjp3`).
  - 2 patchEmbed opaque-codegen (`patchEmbed_flat_has_vjp`,
    `patchEmbed_flat_diff`).
- Book's "verified MobileNet" claim becomes literal (no caveat about
  the depthwise input-VJP axiom).

The 4 ReLU/maxPool axioms need a `HasVJP.correct` weakening to
"smooth subset only" — a project-wide rewrite, separate multi-week
effort. The 2 patchEmbed axioms could come down by de-opaquing
`patchEmbed_flat` (~500 LOC). Neither is in scope here.

---

## Phase 1 lessons (the conv2d recipe — preserved here for Phase 2 reference)

The crux insight that made Phase 1 work was the helper trio for the
dependent-if pattern:

1. **`fun_prop` failure is the first wall.** The conv2d body has
   `if hpad : pad then v(σ hpad) else 0` — a `dite` whose then-branch
   uses the `hpad` proof to construct a Fin. `fun_prop` has no `dite`
   rules; it errors with `No theorems found for dite`. Workaround:
   `differentiableAt_pad_eval` does the by_cases manually.

2. **The pi-fderiv extraction.** For `pdiv (fun v k' => g v k')` at
   `(idx_in, idx_out)`, you need `fderiv (fun v => g v idx_out) v
   (basisVec idx_in)`. Phase 1 used `fderiv_apply` (Mathlib lemma) for
   this conversion — same pattern as `LayerNorm.lean:182` for `pdiv_gelu`.
   Then `by_cases` on the pad condition + `ContinuousLinearMap.proj`
   for the eval-CLM in the pad-true branch.

3. **The product structure: `(constant in v) * (dite-in-v)`.** The
   per-summand has this exact shape. `pdiv_const_mul_pi_pad_eval`
   bundles `pdiv_mul` + `pdiv_const(constant) = 0` +
   `pdiv_pi_pad_eval(dite)` into one rule application.

4. **The closing's structure (per-(co, ho, wo) inner equality).** After
   substituting `h_pdiv`, for each `(co, ho, wo)` you need
   `LHS_inner = (∑ c kh kw, W * indicator) * dy co ho wo`. The recipe:
   - Pull dy out of the LHS if-true branch via the lemma
     `(if h then a*b else 0) = (if h then a else 0) * b`.
   - `congr 1` on the `* dy` factor.
   - Convert the indicator (dep-if of Fin equality) to a non-dep
     conjunction via a sub-lemma `h_indicator`.
   - `by_cases back_cond`; in pos case, triple `Finset.sum_eq_single` for
     `(c, kh, kw)`; in neg case, all sums = 0 (via Nat reasoning + omega).

5. **`omega` covers the Nat arithmetic.** The Fin bound proofs and Nat
   subtractions (`hi.val + pH - ho.val < kH` etc.) all close via `omega`,
   given the right hypotheses in context (`kh.isLt`, `hi.isLt`, etc.).

6. **Don't bother with `IsBoundedBilinearMap`.** Conv2d IS bilinear, and
   in principle `IsBoundedBilinearMap.hasFDerivAt` gives joint diff for
   free. But you'd still need to extract the per-input partial derivatives
   AND prove the bilinear bound (a calc on operator norm). Net: comparable
   length to the direct pdiv chain. Phase 1 considered this approach and
   rejected it.
