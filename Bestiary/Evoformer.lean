import LeanMlir

/-! # Evoformer (AlphaFold 2) — Bestiary entry

AlphaFold 2 (Jumper et al., 2021) cracked protein structure prediction.
Its central innovation — the **Evoformer** — is one of the weirder and
more beautiful architectures in the zoo. It doesn't look like anything
else: two parallel tensor representations that update each other
through operations derived from the **triangle inequality**.

## Dual representations

The Evoformer operates on two tensors simultaneously:

- **MSA representation** `(s, r, c_m)` — a Multiple Sequence Alignment:
  `s` evolutionary-related sequences × `r` residues × `c_m = 256`
  feature channels. Captures co-evolutionary signal.
- **Pair representation** `(r, r, c_z)` — `c_z = 128` features per pair
  of residues. Captures geometric / contact hypotheses.

Each of 48 Evoformer blocks updates both representations jointly; then
the Structure Module reads the final first-row (`single`) representation
plus the pair representation and emits 3D coordinates.

## Per-block anatomy

```
  MSA (s, r, c_m)                         Pair (r, r, c_z)
    │                                           │
    ▼                                           │
  MSA row-attention WITH PAIR BIAS  ←───────────┤   ← pair info biases MSA attention
    │                                           │
    ▼                                           │
  MSA column-attention                          │
    │                                           │
    ▼                                           │
  MSA transition (FFN 4×)                       │
    │                                           │
    ├─── outer-product mean ───────────────────►│   ← MSA pairs update the pair tensor
    │                                           │
    │                                           ▼
    │                                    Triangle mul update (out)
    │                                           │
    │                                           ▼
    │                                    Triangle mul update (in)
    │                                           │
    │                                           ▼
    │                                    Triangle self-attn (start)
    │                                           │
    │                                           ▼
    │                                    Triangle self-attn (end)
    │                                           │
    │                                           ▼
    │                                    Pair transition (FFN 4×)
    │                                           │
    ▼                                           ▼
  MSA (s, r, c_m)                         Pair (r, r, c_z)
```

**Triangle multiplicative update**: for pair features `Z_{ij}`, compute
a new `Z_{ij}` by aggregating over all third residues `k`:
`Z_{ij} ← gate · proj(sum_k (a_{ik} ⊙ b_{jk}))`. The "outgoing" and
"incoming" variants differ in which index pairs are summed. This is the
architecture's way of enforcing the triangle inequality as an inductive
bias: if `i` is close to `k` and `k` is close to `j`, then `i` should
be close to `j`.

**Triangle self-attention**: attention over pair features where the
attention pattern is biased by a third index — again, operating on
triangles of residues.

These are the genuinely new primitives. Our bestiary bundles them all
into the single `.evoformerBlock` constructor.

## Structure Module

After 48 Evoformer rounds, the Structure Module takes the `single`
(= first-row MSA) and pair representations and runs `N_struct = 8`
rounds of:

- **Invariant Point Attention (IPA)** — self-attention where query/key
  geometry respects rigid-body motions.
- **Backbone frame update** — predict SE(3) transforms per residue.
- **Side-chain χ-angle head** — predict torsion angles.

Weights are **shared across the 8 rounds** (it's recurrent), so
`nBlocks` affects compute but not parameter count.

## Variants

- `alphaFold2` — canonical paper spec: `c_m = 256`, `c_z = 128`,
  48 Evoformer blocks, 8 Structure Module rounds. ~70M params in
  the evoformer, ~10M in structure module.
- `alphaFold2Mini` — 16 evoformer + 4 structure (useful for fine-tune / ablation).
- `tinyEvoformer` — 4 blocks, `c_m = 64`, `c_z = 32`. Fixture.
-/

-- ════════════════════════════════════════════════════════════════
-- § AlphaFold 2 canonical
-- ════════════════════════════════════════════════════════════════

def alphaFold2 : NetSpec where
  name := "AlphaFold 2 (Evoformer + StructureModule)"
  imageH := 384     -- max residues after cropping (N_res)
  imageW := 1
  layers := [
    -- 48 blocks of dual-representation processing.
    .evoformerBlock 256 128 48,
    -- Structure Module: 8 recurrent IPA rounds (shared weights).
    .structureModule 384 128 8
  ]

-- ════════════════════════════════════════════════════════════════
-- § AlphaFold 2 Mini
-- ════════════════════════════════════════════════════════════════

def alphaFold2Mini : NetSpec where
  name := "AlphaFold 2 Mini (16 + 4)"
  imageH := 256
  imageW := 1
  layers := [
    .evoformerBlock 192 96 16,
    .structureModule 256 96 4
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyEvoformer fixture
-- ════════════════════════════════════════════════════════════════

def tinyEvoformer : NetSpec where
  name := "tiny-Evoformer (4 blocks)"
  imageH := 64
  imageW := 1
  layers := [
    .evoformerBlock 64 32 4,
    .structureModule 128 32 2
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  max residues : {spec.imageH}"
  IO.println s!"  layers       : {spec.layers.length}"
  IO.println s!"  params       : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture :"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate     : OK"
  | some err => IO.println s!"  validate     : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — AlphaFold 2 Evoformer"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Dual-representation architecture: MSA + pair, updated jointly"
  IO.println "  via triangle multiplicative + triangle attention primitives."
  IO.println "  The part of AlphaFold 2 that actually made structure work."

  summarize alphaFold2
  summarize alphaFold2Mini
  summarize tinyEvoformer

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.evoformerBlock (msaCh pairCh nBlocks)` bundles per-block:"
  IO.println "    MSA row-attn (w/ pair bias), MSA col-attn, MSA transition,"
  IO.println "    outer-product mean → pair, triangle-mul update (out + in),"
  IO.println "    triangle self-attention (start + end), pair transition."
  IO.println "    Each one is its own ~200-line implementation in the paper;"
  IO.println "    bundling is honest at this abstraction layer."
  IO.println "  • `.structureModule (singleCh pairCh nBlocks)` is recurrent:"
  IO.println "    weights shared across all nBlocks rounds. Param count"
  IO.println "    does NOT multiply by nBlocks."
  IO.println "  • Dual representation doesn't fit a linear NetSpec: we list"
  IO.println "    evoformerBlock + structureModule linearly but the \"input\""
  IO.println "    is really (MSA, pair) together, and the \"output\" is a set"
  IO.println "    of 3D atom coords. Input embedding (template + extra-MSA"
  IO.println "    stacks) and output heads (distogram, masked-MSA, pLDDT,"
  IO.println "    pTM) are bestiary-scope omissions."
  IO.println "  • The recycling loop (feed outputs back as inputs, up to"
  IO.println "    3 iterations) is a training/inference-time orchestration"
  IO.println "    concern, not architecture."
