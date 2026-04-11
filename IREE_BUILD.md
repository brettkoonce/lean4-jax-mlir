# Building `libiree_ffi.so` (and running mnist-mlp out of the box)

The Lake build links every trainer against `ffi/libiree_ffi.so`, but that
file is **not** checked in — you have to build it once. This page is the
step-by-step. After it, `lake build mnist-mlp-train` should just work.

If you only want to skim: you need (1) `iree-compile` from pip, (2) the
IREE runtime built from source as static archives, (3) one `gcc` invocation
that wraps `ffi/iree_ffi.c` + the runtime archives into `ffi/libiree_ffi.so`.

The narrative version of how this came together (with the gotchas as they
were hit) lives in [`IREE.md`](IREE.md). The ROCm-specific variant is in
[`ROCM.md`](ROCM.md). This file is the consolidated recipe.

## What you need

| Thing | Why | How |
|---|---|---|
| Lean 4.29.0 | builds the trainer | `elan` (see main README §1) |
| `iree-compile` | Lean shells out to it to lower StableHLO → `.vmfb` | `pip install iree-base-compiler` |
| IREE runtime (static `.a`) | linked into `libiree_ffi.so` | build from source, runtime-only |
| GPU toolchain | runtime needs a backend | CUDA toolkit *or* ROCm 6.x |
| `ffi/libiree_ffi.so` | every Lean trainer links `-liree_ffi` | the link command in §4 |
| MNIST data | input | `./download_mnist.sh` |

## 1. Install the IREE compiler

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install iree-base-compiler
iree-compile --version
```

The Lean trainers shell out to `iree-compile` from `$PATH`, so make sure
the venv is active when you run them (or symlink it somewhere on `PATH`).

## 2. Build the IREE runtime from source

A naive `git clone --recursive` of `iree-org/iree` pulls in LLVM via the
torch-mlir / stablehlo submodule chains and balloons past 9 GB. We only
need the **runtime** submodules (~470 MB).

```bash
# Pick a sibling directory — these paths are referenced below.
cd ~/src   # or wherever
git clone https://github.com/iree-org/iree.git
cd iree

# Init only the submodules listed in runtime_submodules.txt
xargs -a build_tools/scripts/git/runtime_submodules.txt \
  git submodule update --init --depth 1
```

Then a runtime-only CMake build:

```bash
mkdir -p ../iree-build && cd ../iree-build

cmake ../iree -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=OFF        `# we use pip's iree-compile` \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_DRIVER_CUDA=ON        `# pick ONE of CUDA / HIP, or both` \
  -DBUILD_SHARED_LIBS=OFF          `# we want static archives`

ninja
```

For AMD/ROCm, swap `-DIREE_HAL_DRIVER_CUDA=ON` for
`-DIREE_HAL_DRIVER_HIP=ON`. You can enable both if you want one
`libiree_ffi.so` that supports either.

Build is ~30 seconds on a modern box. Output sits under
`iree-build/runtime/src/iree/...` as a tree of `.a` files, with
`libiree_runtime_unified.a` containing most of the runtime.

After the build, export the paths so the next step can find them:

```bash
export IREE_SRC=$HOME/src/iree            # adjust to your clone
export IREE_BUILD=$HOME/src/iree-build    # adjust to your build dir
```

## 3. (CUDA only) Pin the compile target if you're on Ada or newer

IREE 3.11's compiler has no GPU target metadata for `sm_89` and later
(see [iree-org/iree#21122](https://github.com/iree-org/iree/issues/21122),
[#22147](https://github.com/iree-org/iree/issues/22147)). The Lean
trainers already pass `--iree-cuda-target=sm_86` for this reason; PTX
JITs forward to Ada at load time. If you change targets, do it in the
`Main*Train.lean` file, not in this doc.

## 4. Build `libiree_ffi.so`

This is the one missing piece. The C source `ffi/iree_ffi.c` already
supports both CUDA and HIP via `#ifdef USE_HIP`, so the only thing you
choose is which driver(s) to compile in.

From the repo root:

```bash
cd ffi

# 4a. Compile the wrapper. Add -DUSE_HIP for AMD/ROCm.
gcc -fPIC -O2 -c iree_ffi.c \
  -I"$IREE_SRC/runtime/src" \
  -I"$IREE_BUILD/runtime/src" \
  -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
  # -DUSE_HIP

# 4b. Link against the static runtime. Note the --start-group / --end-group:
#     flatcc_verify_* lives in libflatcc_parsing.a, the rest in
#     libflatcc_runtime.a, and they reference each other.
gcc -shared -o libiree_ffi.so iree_ffi.o \
  -Wl,--whole-archive \
    "$IREE_BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a" \
  -Wl,--no-whole-archive \
  -Wl,--start-group \
    "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_runtime.a \
    "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_parsing.a \
  -Wl,--end-group \
  -lm -lpthread -ldl

cd ..
```

Notes:
- `--whole-archive` is required around `libiree_runtime_unified.a` so the
  HAL driver registration symbols (which are pulled in by static
  constructors) actually make it into the `.so`.
- `IREE_ALLOCATOR_SYSTEM_CTL` is gated behind a compile-time macro; the
  define above wires it to `iree_allocator_libc_ctl` (gotcha #3 in
  `IREE.md`).
- Flatcc paths can drift between IREE versions — if the `.a` files aren't
  where this command expects, `find "$IREE_BUILD" -name 'libflatcc*.a'`
  and substitute.

## 5. Verify the result

```bash
ls -lh ffi/libiree_ffi.so                  # ~1.4 MB
nm ffi/libiree_ffi.so | grep driver_module_register
# Should print iree_hal_cuda_driver_module_register  (or _hip_, or both)
```

If you skipped `--whole-archive`, the `driver_module_register` symbol
won't be present and session creation will fail at runtime with "no
HAL driver matching 'cuda'/'hip'".

## 6. Build and run mnist-mlp

```bash
./download_mnist.sh                        # → data/*-ubyte
lake build mnist-mlp-train                 # links -liree_ffi from ./ffi
.lake/build/bin/mnist-mlp-train
```

Expected output: 12 epochs, ~16 s/epoch on a modest GPU, final
accuracy ≈ 97.9%.

If you see `error while loading shared libraries: libiree_ffi.so`, it's
an rpath issue — the lakefile sets `-Wl,-rpath,./ffi`, so run the binary
from the repo root (not from `.lake/build/bin/`).

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `cannot find -liree_ffi` at link time | `ffi/libiree_ffi.so` missing | redo §4 |
| `undefined reference to iree_allocator_system` | forgot `-DIREE_ALLOCATOR_SYSTEM_CTL=...` | add the define in §4a |
| `undefined reference to flatcc_verify_*` | flatcc archives missing or wrong order | add both `libflatcc_*.a` inside `--start-group` |
| `no HAL driver matching 'cuda'` at runtime | `--whole-archive` was dropped | redo §4b with the wrap intact |
| `--no-allow-shlib-undefined` errors when linking the Lean trainer | Lean's bundled lld is strict about transitive glibc symbols | already handled — every trainer in `lakefile.lean` passes `-Wl,--allow-shlib-undefined` |
| `iree-compile` not found | venv not active in the shell that runs `lake build` | `source .venv/bin/activate` first |

## What this gets you

Once `libiree_ffi.so` exists in `ffi/`, every other target in
`lakefile.lean` that links `-liree_ffi` (mnist, cifar, resnet, mobilenet,
efficientnet, vit, vgg, …) builds without further setup. The runtime
library is shared across all of them; only the `.vmfb` files differ.
