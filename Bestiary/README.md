# Bestiary

Companion entries for Part 2 of *Verified Deep Learning with Lean 4*:
a catalogue of famous neural-network architectures expressed as pure
`NetSpec` values, with no training runs attached. Each entry is meant
as a read-only fixture showing "here's what this architecture looks
like in ~20 lines of Lean."

## Entries

| File | Architecture | Variants | Notes |
|------|--------------|----------|-------|
| `AlphaZero.lean` | AlphaZero / AlphaGo Zero | original Go, chess, tiny | Two-headed (policy + value) |
| `Mamba.lean`     | Mamba (selective SSM)     | 130M / 370M / 790M / tiny | Language model, linear-time alternative to attention |
| `SwinT.lean`     | Swin Transformer          | Swin-T / Swin-S / Swin-B / tiny | Hierarchical ViT, windowed + shifted attention |
| `UNet.lean`      | UNet                      | original / RGB / small / tiny | Encoder-decoder with skip connections; diffusion backbone |

## Adding a new entry

1. Create `Bestiary/YourModel.lean`.
2. Declare the architecture as one or more `NetSpec` values (no
   `TrainConfig`, no `main` train loop).
3. Write a print-only `main` that walks through each spec and calls
   `archStr`, `totalParams`, `validate` — the pattern in `AlphaZero.lean`.
4. Register the executable in `lakefile.lean`:
   ```lean
   lean_exe «bestiary-yourmodel» where
     root := `Bestiary.YourModel
   ```
5. Run: `lake build bestiary-yourmodel && .lake/build/bin/bestiary-yourmodel`.

## Why print-only

The bestiary is for showing architecture at the conceptual level. Training
introduces dataset / loss / optimizer / GPU concerns that distract from
"here's the layer layout." A reader can always take any spec and pair it
with a `TrainConfig` to run a real training job — that's one line of code
away, same machinery as the `MainResnetTrain.lean` pattern.

## Limitations acknowledged

The current `NetSpec` is a linear list of layers, so architectures with
multiple output heads (AlphaZero, Siamese nets, multi-task models) appear
as two separate specs sharing prose-level structure rather than a real
shared-body graph. Fixing this would require either:

- A `NetSpec.branch` layer holding a list of sub-specs, or
- A free monad / effect-style spec DSL replacing the `List Layer` form.

Both are real refactors. For bestiary purposes, showing the two forks
side-by-side is both honest and pedagogically clearer than a tree that
might obscure how simple AlphaZero really is.
