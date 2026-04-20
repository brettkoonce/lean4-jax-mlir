#!/usr/bin/env python3
"""Compare two training-trace .jsonl files for numerical agreement.

Usage:
    python3 tests/diff_traces.py <trace_a.jsonl> <trace_b.jsonl>

Exits 0 on agreement, 1 on any mismatch. Prints a compact summary.

See traces/TRACE_FORMAT.md for the trace contract.
"""
import json
import math
import sys
from pathlib import Path

ATOL = 1e-4
RTOL = 1e-3
# Required numeric fields on every step record.
NUMERIC_REQUIRED = ("loss", "lr")
# Optional numeric fields — compared only when present in BOTH traces.
NUMERIC_OPTIONAL = ("grad_norm", "param_norm")
HEADER_IDENTICAL_FIELDS = (
    "netspec_name", "netspec_hash", "config", "total_params", "dataset",
)


def load_trace(path: Path) -> tuple[dict, list[dict]]:
    """Return (header, [step_records])."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        sys.exit(f"{path}: empty trace")
    header, *steps = records
    if header.get("kind") != "header":
        sys.exit(f"{path}: first record is not a header")
    for i, s in enumerate(steps):
        if s.get("kind") != "step":
            sys.exit(f"{path}: record {i+1} is not a step record")
    return header, steps


def check_headers(ha: dict, hb: dict) -> list[str]:
    """Return list of diff messages; empty if headers match on identity fields."""
    diffs = []
    for key in HEADER_IDENTICAL_FIELDS:
        if ha.get(key) != hb.get(key):
            diffs.append(f"header.{key}: {ha.get(key)!r} vs {hb.get(key)!r}")
    return diffs


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def check_steps(sa: list[dict], sb: list[dict]) -> list[str]:
    diffs = []
    if len(sa) != len(sb):
        diffs.append(f"step count mismatch: {len(sa)} vs {len(sb)}")
        return diffs
    for i, (a, b) in enumerate(zip(sa, sb)):
        # Required fields must be present on both sides.
        for field in NUMERIC_REQUIRED:
            va, vb = a.get(field), b.get(field)
            if va is None or vb is None:
                diffs.append(f"step {i} {field}: missing ({va} vs {vb})")
                continue
            if not close(va, vb):
                delta = abs(va - vb)
                diffs.append(
                    f"step {i} {field}: {va:.6f} vs {vb:.6f} (delta={delta:.2e})"
                )
        # Optional fields compared only when BOTH sides have them.
        for field in NUMERIC_OPTIONAL:
            va, vb = a.get(field), b.get(field)
            if va is None or vb is None:
                continue  # skip — missing from one side is OK
            if not close(va, vb):
                delta = abs(va - vb)
                diffs.append(
                    f"step {i} {field}: {va:.6f} vs {vb:.6f} (delta={delta:.2e})"
                )
    return diffs


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    ha, sa = load_trace(pa)
    hb, sb = load_trace(pb)

    header_diffs = check_headers(ha, hb)
    step_diffs   = check_steps(sa, sb)

    print(f"Comparing:")
    print(f"  A: {pa}  (phase={ha.get('phase')}, steps={len(sa)})")
    print(f"  B: {pb}  (phase={hb.get('phase')}, steps={len(sb)})")
    print(f"Tolerance: atol={ATOL}, rtol={RTOL}")
    print()

    if not header_diffs and not step_diffs:
        print(f"✓ PASS — {len(sa)} steps agree across phases")
        return 0

    if header_diffs:
        print("✗ HEADER MISMATCH:")
        for d in header_diffs:
            print(f"    {d}")
    if step_diffs:
        print(f"✗ STEP MISMATCHES ({len(step_diffs)}):")
        for d in step_diffs[:20]:      # cap output at 20 diffs
            print(f"    {d}")
        if len(step_diffs) > 20:
            print(f"    ...and {len(step_diffs) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
