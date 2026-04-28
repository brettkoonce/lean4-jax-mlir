# ROCm Bootstrap: Running on AMD GPUs (7900 XTX)

Steps to run the Lean → MLIR → IREE training pipeline on an AMD GPU.
The MLIR codegen is backend-agnostic — only the IREE compile flag and
runtime library change.

## Prerequisites

- AMD GPU with ROCm support (7900 XTX = gfx1100)
- ROCm 6.x installed (`/opt/rocm`)
- Lean 4 toolchain (elan)
- Python 3.10+ with pip

## Step 1: Clone and build Lean project

```bash
git clone <repo> lean4-mlir
cd lean4-mlir
lake build LeanMlir
```

## Step 2: Install IREE compiler (pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -f https://iree.dev/pip-release-links.html --pre \
    'iree-base-compiler>=3.12.0rc20260428'
```

The `>=3.12.0rc20260428` pin is the first nightly that includes the two
ROCm/HIP `Distribute`-pass fixes we hit while landing ConvNeXt
([iree-org/iree#24282](https://github.com/iree-org/iree/issues/24282) and
[#24283](https://github.com/iree-org/iree/issues/24283)). Earlier 3.11.0
nightlies fail to compile the LN-NCHW + stacked-`[0,2,3]` reduction
patterns the codegen emits.

Verify:
```bash
.venv/bin/iree-compile --version
```

## Step 3: Generate and compile MLIR for ROCm

The MLIR is identical — only the compile target changes:

```bash
# Build the test binary (generates MLIR + compiles)
lake build test-resnet-residual
```

The backend is now picked via the `IREE_BACKEND` env var — no source edits
needed. Set `IREE_BACKEND=rocm` (default is `cuda`) and everything
downstream (iree-compile flags, target-chip selection, HAL device setup)
routes through that single env var. See `LeanMlir/Train.lean:44`.

Or compile manually:
```bash
# Generate MLIR from Lean
.lake/build/bin/test-resnet-residual  # generates .lake/build/resnet34_train_step.mlir

# Compile for ROCm
.venv/bin/iree-compile \
  .lake/build/resnet34_train_step.mlir \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target-chip=gfx1100 \
  -o .lake/build/resnet34_train_step.vmfb
```

## Step 4: Build IREE runtime with HIP support

The pip package only ships the compiler, not the runtime library. We need
`libiree_ffi.so` built from source with the HIP HAL driver.

```bash
# Clone IREE (if not already)
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init

# Build runtime only (no compiler — we use pip for that)
mkdir -p ../iree-build && cd ../iree-build
cmake ../iree \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_HIP=ON \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DBUILD_SHARED_LIBS=OFF

ninja
```

This builds static `.a` libraries in `iree-build/runtime/src/iree/`.

## Step 5: Build libiree_ffi.so

The FFI wrapper links our C shim against the IREE static libraries:

```bash
cd lean4-mlir/ffi

# Compile the FFI C files
gcc -fPIC -O2 -c iree_ffi.c \
  -I../../iree/runtime/src \
  -I../../iree-build/runtime/src

# Link into shared library with IREE runtime
gcc -shared -o libiree_ffi.so iree_ffi.o \
  -Wl,--whole-archive \
  ../../iree-build/runtime/src/iree/runtime/libiree_runtime_unified.a \
  -Wl,--no-whole-archive \
  -lm -lpthread -ldl
```

Verify it has HIP symbols:
```bash
nm libiree_ffi.so | grep hip_driver
# Should show: t iree_hal_hip_driver_module_register
```

**Important:** The FFI C code references `iree_hal_cuda_driver_module_register`.
For HIP, change this to `iree_hal_hip_driver_module_register` in `iree_ffi.c`:

```c
// Line ~37 in iree_ffi.c — change:
#include "iree/hal/drivers/cuda/registration/driver_module.h"
// to:
#include "iree/hal/drivers/hip/registration/driver_module.h"

// And change the registration call:
iree_hal_cuda_driver_module_register(...)
// to:
iree_hal_hip_driver_module_register(...)

// And the device string (line ~53):
iree_make_cstring_view("cuda")
// to:
iree_make_cstring_view("hip")
```

Or better: use `#ifdef` to support both:
```c
#ifdef USE_HIP
  #include "iree/hal/drivers/hip/registration/driver_module.h"
  #define IREE_REGISTER_DRIVER iree_hal_hip_driver_module_register
  #define IREE_DEVICE_NAME "hip"
#else
  #include "iree/hal/drivers/cuda/registration/driver_module.h"
  #define IREE_REGISTER_DRIVER iree_hal_cuda_driver_module_register
  #define IREE_DEVICE_NAME "cuda"
#endif
```

Then compile with `-DUSE_HIP` for AMD.

## Step 6: Prepare Imagenette data

```bash
# Download
mkdir -p data/imagenette
python3 -c "
import urllib.request, tarfile
urllib.request.urlretrieve(
    'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
    'data/imagenette/imagenette2-320.tgz')
import tarfile
tarfile.open('data/imagenette/imagenette2-320.tgz').extractall('data/imagenette/')
"

# Preprocess to binary format
pip install Pillow numpy
python3 preprocess_imagenette.py data/imagenette/imagenette2-320 data/imagenette
```

## Step 7: Build and run

```bash
# Build the trainer
lake build resnet34-train

# Train. IREE_BACKEND=rocm tells iree-compile to target ROCm/gfx1100;
# HIP_VISIBLE_DEVICES=0 pins to a single GPU (multi-GPU JAX-ROCm has a
# known hang on gfx1100 — see upstream-issues/).
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
  .lake/build/bin/resnet34-train data/imagenette
```

Equivalent via the shell wrapper, which sets the env vars for you:
```bash
./run.sh resnet34 0 rocm
```

The first run compiles the vmfbs (~10-15 min for ResNet-sized models).
Subsequent runs reuse the cached vmfb unless `.lake/build/` is cleared
or the MLIR hash changes.

## Expected performance

The 7900 XTX has ~61 TFLOPS f32 vs ~22 TFLOPS on the 4060 Ti.
With the same batch size 16, expect roughly 2-3x faster per step
(~1s vs ~2.5s), depending on memory bandwidth and IREE's ROCm codegen
maturity.

The 24GB VRAM (vs 16GB) also allows larger batch sizes (32 or 64),
which would further improve throughput and training stability.

## Troubleshooting

**`iree-compile` fails with ROCm target:**
- Ensure ROCm is installed: `rocminfo` should list your GPU
- Check chip name: `rocminfo | grep gfx` → should show `gfx1100`
- Try `--iree-rocm-target-chip=gfx1100` (not gfx11)

**`libiree_ffi.so` segfaults on session create:**
- Verify HIP symbols: `nm libiree_ffi.so | grep hip`
- Check ROCm runtime: `hipInfo` or `rocminfo`
- Ensure `iree_ffi.c` uses `hip` not `cuda` for driver registration

**Performance much slower than expected:**
- Check `rocm-smi` for GPU utilization
- The first step is slow (JIT compilation) — subsequent steps are cached
- Our FFI pushes 220 tensors one-by-one; this is the main overhead

## Architecture notes

The entire pipeline is backend-agnostic up to the IREE compile step:

```
Lean NetSpec → MlirCodegen.generateTrainStep → StableHLO MLIR (same for all backends)
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                                   ▼
                          iree-compile --cuda              iree-compile --rocm
                                    │                                   │
                                    ▼                                   ▼
                              .vmfb (CUDA)                       .vmfb (ROCm)
                                    │                                   │
                              libiree_ffi.so                    libiree_ffi.so
                              (CUDA HAL)                         (HIP HAL)
                                    │                                   │
                                    └───────── same Lean code ──────────┘
```

Zero changes to the Lean code, MLIR codegen, or training loop.
Only the compile flags and runtime library differ.
