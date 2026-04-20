#!/usr/bin/env bash
# Cross-backend training correctness check for MNIST MLP.
#
# Trains the same NetSpec (same seed, same hyperparameters) through
# the Phase 3 pipeline on two IREE backends — llvm-cpu (reference) and
# rocm (GPU) — then diffs the per-step training traces at tolerance.
#
# What passing means: our codegen + IREE produces numerically
# consistent training dynamics across backends. Any disagreement
# points to either a codegen bug (stablehlo attributes different from
# what we think) or an IREE backend lowering bug.
#
# Requires:
#   lake, python3
#   iree-compile with both llvm-cpu and rocm target backends
#   data/mnist/ populated (`download_mnist.sh`)

set -e
cd "$(dirname "$0")/.."

TRACES_DIR=traces
CPU_TRACE=$TRACES_DIR/mnist_mlp.phase3-cpu.jsonl
ROCM_TRACE=$TRACES_DIR/mnist_mlp.phase3-rocm.jsonl

mkdir -p $TRACES_DIR

echo "▶ Phase 3 training, llvm-cpu backend"
IREE_BACKEND=llvm-cpu \
LEAN_MLIR_TRACE_OUT="$CPU_TRACE" \
  lake exe mnist-mlp-train-f32 data/mnist

echo
echo "▶ Phase 3 training, rocm backend"
IREE_BACKEND=rocm IREE_CHIP="${IREE_CHIP:-gfx1100}" \
LEAN_MLIR_TRACE_OUT="$ROCM_TRACE" \
  lake exe mnist-mlp-train-f32 data/mnist

echo
echo "▶ Comparing traces"
python3 tests/diff_traces.py "$CPU_TRACE" "$ROCM_TRACE"
