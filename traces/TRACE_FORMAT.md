# Training Trace Format

A `.jsonl` (JSON Lines) file capturing a deterministic training run. Both
Phase 2 (Lean → JAX) and Phase 3 (Lean → MLIR → IREE) emit traces in this
format; `tests/diff_traces.py` compares two traces numerically.

The emitted trace is an authoritative artifact: given a `(NetSpec, TrainConfig,
seed)` triple, both phases produce the same trace (within float tolerance), and
the trace can be archived, diffed, and verified years later with a one-line
comparison.

## File structure

One JSON object per line. The first line is a **header record** describing the
run; every subsequent line is a **step record** for one training step.

### Header record

```json
{
  "kind": "header",
  "phase": "phase3",          // "phase2" or "phase3"
  "netspec_name": "mnist-mlp",
  "netspec_hash": "d41d8cd...", // SHA-256 of the NetSpec rendered as text
  "config": {
    "lr": 0.1,
    "batch_size": 128,
    "epochs": 12,
    "use_adam": false,
    "weight_decay": 0.0,
    "cosine": false,
    "warmup_epochs": 0,
    "augment": false,
    "label_smoothing": 0.0,
    "seed": 314159
  },
  "total_params": 669706,
  "dataset": "mnist",
  "emitted_at": "2026-04-20T18:00:00Z",
  "emitter_version": "1"
}
```

### Step record

Required fields on every step record:

```json
{
  "kind": "step",
  "step": 1,              // global step counter, 1-indexed (matches Adam's bias-correction convention)
  "epoch": 0,             // 0-indexed
  "loss": 2.302585,
  "lr": 0.1               // effective lr at this step (after warmup/cosine)
}
```

Optional fields (emitted when the FFI helper is available):

```json
{
  "grad_norm": 4.1205,    // sqrt(sum over all grads of grad^2)
  "param_norm": 23.7612,  // sqrt(sum over all params of p^2)
  "wall_ms": 0.8          // informational — ignored by diff
}
```

The comparator skips optional fields that are missing in either trace
(so a v1 trace without `grad_norm` compares cleanly to another v1 trace);
if both traces have the field, it's enforced under the same numeric
tolerance.

## Comparison rules

`tests/diff_traces.py` compares two traces for agreement:

- **Headers must match identically** on `netspec_name`, `netspec_hash`,
  `config`, `total_params`, `dataset`. `phase` and `emitted_at` are allowed
  to differ (that's the whole point).
- **Step records are compared index-for-index** (phase2.step[0] vs
  phase3.step[0], etc.). Step count must match exactly.
- **Numeric fields** (`loss`, `grad_norm`, `param_norm`, `lr`) are compared
  with `atol=1e-4` and `rtol=1e-3`. Float reduction order differences between
  JAX's XLA and IREE's compilation path make exact equality unrealistic; this
  tolerance is empirical.
- **`wall_ms` is ignored** (not a correctness signal).

## Directory layout

```
traces/
  TRACE_FORMAT.md                   -- this file
  mnist_mlp.phase3.jsonl            -- canonical Phase 3 trace (committed)
  mnist_mlp.phase2.jsonl            -- canonical Phase 2 trace (committed)
  mnist_cnn.phase3.jsonl            -- etc.
```

Committed canonical traces serve as the reproducibility baseline: a reader
runs either phase with the matching seed and `diff_traces.py` against the
committed trace to verify their build matches ours.
