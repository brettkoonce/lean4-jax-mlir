# ═══════════════════════════════════════════════════════════════════
# lean4-mlir: MNIST MLP demo (CPU, no GPU required)
#
#   docker build -t lean4-mlir-demo .
#   docker run --rm lean4-mlir-demo
#
# Trains a 3-layer MLP on MNIST from scratch via the Lean 4 → MLIR →
# IREE pipeline. 12 epochs, ~5 min on CPU, expects ~97.9% accuracy.
#
# No GPU, no Python runtime, no PyTorch/JAX — just the Lean binary
# calling IREE's CPU backend directly.
# ═══════════════════════════════════════════════════════════════════

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git gcc g++ cmake ninja-build python3 python3-venv \
    ca-certificates libgmp-dev wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ── 1. Lean 4 ──────────────────────────────────────────────────────
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | \
    sh -s -- -y --default-toolchain leanprover/lean4:v4.29.0
ENV PATH="/root/.elan/bin:${PATH}"

# ── 2. iree-compile (pip) ──────────────────────────────────────────
RUN python3 -m venv /build/venv && \
    /build/venv/bin/pip install --no-cache-dir iree-base-compiler cmake ninja
ENV PATH="/build/venv/bin:${PATH}"

# ── 3. IREE runtime (CPU only) ────────────────────────────────────
RUN git clone --depth 1 https://github.com/iree-org/iree.git /build/iree && \
    cd /build/iree && \
    xargs -a build_tools/scripts/git/runtime_submodules.txt \
      git submodule update --init --depth 1

RUN mkdir -p /build/iree-build && cd /build/iree-build && \
    cmake /build/iree -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DIREE_BUILD_COMPILER=OFF \
      -DIREE_BUILD_TESTS=OFF \
      -DIREE_BUILD_SAMPLES=OFF \
      -DIREE_HAL_DRIVER_DEFAULTS=OFF \
      -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
      -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
      -DBUILD_SHARED_LIBS=OFF && \
    ninja

# ── 4. Clone repo ─────────────────────────────────────────────────
RUN git clone https://github.com/brettkoonce/lean4-mlir.git /build/repo
WORKDIR /build/repo

# ── 5. Build libiree_ffi.so (CPU mode) ────────────────────────────
RUN cd ffi && \
    gcc -fPIC -O2 -c iree_ffi.c \
      -I/build/iree/runtime/src \
      -I/build/iree-build/runtime/src \
      -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
      -DUSE_CPU && \
    gcc -fPIC -O2 -fvisibility=default \
      -c /build/iree/third_party/flatcc/src/runtime/verifier.c \
      -I/build/iree/third_party/flatcc/include \
      -o flatcc_verifier.o && \
    gcc -shared -o libiree_ffi.so iree_ffi.o flatcc_verifier.o \
      -Wl,--whole-archive \
        /build/iree-build/runtime/src/iree/runtime/libiree_runtime_unified.a \
      -Wl,--no-whole-archive \
      -lm -lpthread -ldl && \
    rm -f flatcc_verifier.o iree_ffi.o

# ── 6. Build the MLP trainer ──────────────────────────────────────
RUN lake build mnist-mlp-train-f32

# ── 7. Download MNIST ──────────────────────────────────────────────
RUN bash download_mnist.sh

# ═══════════════════════════════════════════════════════════════════
# Runtime stage — minimal image
# ═══════════════════════════════════════════════════════════════════
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgmp10 python3 && \
    rm -rf /var/lib/apt/lists/*

# The trainer binary
COPY --from=builder /build/repo/.lake/build/bin/mnist-mlp-train-f32 /app/bin/
# IREE FFI shared library
COPY --from=builder /build/repo/ffi/libiree_ffi.so /app/ffi/
# MNIST data
COPY --from=builder /build/repo/data/train-images-idx3-ubyte /app/data/
COPY --from=builder /build/repo/data/train-labels-idx1-ubyte /app/data/
COPY --from=builder /build/repo/data/t10k-images-idx3-ubyte /app/data/
COPY --from=builder /build/repo/data/t10k-labels-idx1-ubyte /app/data/
# iree-compile: copy the native binary + its shared libs.
COPY --from=builder \
  /build/venv/lib/python3.10/site-packages/iree/compiler/_mlir_libs/ \
  /app/iree-libs/
RUN ln -s /app/iree-libs/iree-compile /app/bin/iree-compile

WORKDIR /app
ENV LD_LIBRARY_PATH=/app/ffi:/app/iree-libs
ENV IREE_BACKEND=llvm-cpu
ENV IREE_DEVICE=local-task
ENV PATH="/app/bin:${PATH}"

# First run: generates MLIR → compiles vmfb (~30s) → trains 12 epochs (~5 min)
CMD ["mnist-mlp-train-f32", "data"]
