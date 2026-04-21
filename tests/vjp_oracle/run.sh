#!/usr/bin/env bash
# VJP oracle runner. For each test case, runs phase 3 with init dump +
# NO_SHUFFLE, runs phase 2 with matching init load + NO_SHUFFLE, and
# diffs step-2 loss. Step-2 is the first step whose loss depends on
# the backward pass, so a small Δ there means the Lean hand-derived
# VJP agrees with JAX's value_and_grad.
#
# Usage: tests/vjp_oracle/run.sh [JAX_PLATFORMS=cpu] [case1 case2 ...]
# If JAX_PLATFORMS is set in the environment, it's respected (needed
# on mars — see upstream-issues/2026-04-rocm-miopen-conv-segv/).
set -u
cd "$(dirname "$0")/../.."  # → repo root

# Auto-default IREE_BACKEND and JAX_PLATFORMS on AMD hosts. Needed
# because LeanMlir/Train.lean defaults IREE_BACKEND to "cuda", and
# JAX-ROCm segfaults on conv on gfx1100 (ROCm/MIOpen#3955). User env
# wins. Detect via /dev/kfd (the ROCm kernel-fusion-driver device)
# or /opt/rocm — rocminfo isn't always on PATH.
if [ -z "${IREE_BACKEND:-}" ] && { [ -e /dev/kfd ] || [ -d /opt/rocm ]; }; then
  export IREE_BACKEND=rocm
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

if [ "$#" -gt 0 ]; then
  CASES=("$@")
else
  CASES=(dense dense-relu conv convbn conv-pool)
fi
FAIL=0

for name in "${CASES[@]}"; do
  init_bin=/tmp/vo_${name}.bin
  p3_trace=/tmp/vo_p3_${name}.jsonl
  p2_trace=/tmp/vo_p2_${name}.jsonl
  p3_log=/tmp/vo_p3_${name}.log
  p2_log=/tmp/vo_p2_${name}.log

  # Phase 3
  LEAN_MLIR_INIT_DUMP="$init_bin" \
  LEAN_MLIR_NO_SHUFFLE=1 \
  LEAN_MLIR_TRACE_OUT="$p3_trace" \
    ./.lake/build/bin/vjp-oracle-${name} data > "$p3_log" 2>&1 \
    || { echo "FAIL  ${name}  phase-3 crashed (see $p3_log)"; FAIL=1; continue; }

  # Phase 2 — invoke the binary once to emit the generated Python
  # script (we don't care whether its own Python invocation succeeds;
  # we'll run Python ourselves against the right venv). Then call
  # Python directly from repo root with env vars so `.venv/` resolves.
  # Pass an absolute path so the generated Python still finds data
  # when invoked from repo root.
  ( cd jax && ./.lake/build/bin/vjp-oracle-${name} "$(cd .. && pwd)/data" > /dev/null 2>&1 ) || true
  # Generated Python filename uses underscores; case names use hyphens.
  script=jax/.lake/build/generated_vjp_oracle_${name//-/_}.py
  [ -f "$script" ] || { echo "FAIL  ${name}  phase-2 did not emit $script"; FAIL=1; continue; }
  LEAN_MLIR_INIT_LOAD="$(realpath "$init_bin")" \
  LEAN_MLIR_NO_SHUFFLE=1 \
  LEAN_MLIR_TRACE_OUT="$(realpath -m "$p2_trace")" \
    .venv/bin/python3 "$script" > "$p2_log" 2>&1 \
    || { echo "FAIL  ${name}  phase-2 python crashed (see $p2_log)"; FAIL=1; continue; }

  # Diff. Per-case tolerance: maxPool's argmax tiebreaks disagree
  # between XLA and IREE for tied input elements (MNIST has many
  # zero pixels → many ties), pushing the step-2 Δ above the 1e-4
  # default by ~5×. Not a correctness bug; just a looser noise floor.
  case "$name" in
    conv-pool) tol=1e-3 ;;
    convbn)    tol=1e-4 ;;  # BN variance reductions ≲ 1e-4
    *)         tol=1e-5 ;;  # dense / conv / relu — ULP-tight
  esac
  .venv/bin/python3 tests/vjp_oracle/diff_step.py "$p3_trace" "$p2_trace" "$name" "$tol" || FAIL=1
done

exit $FAIL
