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
| `iree-compile` | Lean shells out to it to lower StableHLO → `.vmfb` | `pip install 'iree-base-compiler>=3.12.0rc20260428'` |
| IREE runtime (static `.a`) | linked into `libiree_ffi.so` | build from source, runtime-only |
| GPU toolchain | runtime needs a backend | CUDA toolkit *or* ROCm 6.x |
| `ffi/libiree_ffi.so` | every Lean trainer links `-liree_ffi` | the link command in §4 |
| MNIST data | input | `./download_mnist.sh` |

## 1. Install the IREE compiler (plus CMake / Ninja)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -f https://iree.dev/pip-release-links.html --pre \
    'iree-base-compiler>=3.12.0rc20260428' cmake ninja
iree-compile --version
cmake --version
ninja --version
```

The `>=3.12.0rc20260428` pin is the first nightly with two distribute-pass
fixes we hit while landing ConvNeXt
([iree-org/iree#24282](https://github.com/iree-org/iree/issues/24282) and
[#24283](https://github.com/iree-org/iree/issues/24283)). Older 3.11.0
nightlies will fail iree-compile on the LN-NCHW or stacked-`[0,2,3]`
reduction patterns the codegen emits.

`cmake` and `ninja` are only needed for §2 (building the IREE runtime). If
your distro already ships recent versions on `PATH` you can skip them, but
pip-installing inside the venv avoids version-skew surprises — the IREE
runtime-build CMakeLists wants a relatively new CMake.

The Lean trainers shell out to `iree-compile` from `$PATH` (actually from
`./.venv/bin/iree-compile` — see `LeanMlir/Types.lean`), so make sure the
venv is active when you run them.

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

### 4c. Fallback: rebuild flatcc verifier with default visibility (ROCm/HIP)

On at least one ROCm 7.2 / IREE setup, the simple link in §4b produced
a `.so` whose `flatcc_verify_*` symbols ended up as `GLOBAL HIDDEN` in
the dynamic symbol table — meaning they're present in the `.text`
section but absent from `.dynsym`. When the IREE runtime later tries
to verify a `.vmfb` file at session-create time, it errors out with:

```
error while loading: undefined symbol: flatcc_verify_table_as_root
```

If you hit that, the fix is to recompile **just the flatcc verifier
source** with `-fvisibility=default` and link the resulting object
ahead of the archive (so the linker picks the visible version):

```bash
cd ffi

# 4c-i. Compile the verifier source (NOT the prebuilt .a) with default visibility.
gcc -fPIC -O2 -fvisibility=default \
  -c "$IREE_SRC/third_party/flatcc/src/runtime/verifier.c" \
  -I"$IREE_SRC/third_party/flatcc/include" \
  -o flatcc_verifier_visible.o

# 4c-ii. Link as before, but include flatcc_verifier_visible.o BEFORE
#        the archives so its definitions win over the hidden ones.
gcc -shared -o libiree_ffi.so iree_ffi.o flatcc_verifier_visible.o \
  -Wl,--whole-archive \
    "$IREE_BUILD/runtime/src/iree/runtime/libiree_runtime_unified.a" \
  -Wl,--no-whole-archive \
  -Wl,--start-group \
    "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_runtime.a \
    "$IREE_BUILD"/build_tools/third_party/flatcc/libflatcc_parsing.a \
  -Wl,--end-group \
  -lm -lpthread -ldl

rm flatcc_verifier_visible.o
```

After this, `readelf -Ws ffi/libiree_ffi.so | grep flatcc_verify_table_as_root`
should show `GLOBAL DEFAULT` (not `GLOBAL HIDDEN`).

Why this works: the IREE third-party flatcc build compiles its sources
with `-fvisibility=hidden`. The resulting `.o` files in the archives
have `GLOBAL HIDDEN` symbols which are stripped from `.dynsym` when
linked into a shared library. We bypass that by recompiling just
`verifier.c` with default visibility and putting it on the link line
before the archive — the linker resolves to our visible copy.

## 5. Verify the result

```bash
ls -lh ffi/libiree_ffi.so                  # ~1.4 MB
nm ffi/libiree_ffi.so | grep driver_module_register
# Should print iree_hal_cuda_driver_module_register  (or _hip_, or both)
```

If you skipped `--whole-archive`, the `driver_module_register` symbol
won't be present and session creation will fail at runtime with "no
HAL driver matching 'cuda'/'hip'".

## 6. Build and run

Every trainer is self-bootstrapping: it generates its own MLIR,
calls `iree-compile`, and starts training. Just build + run:

```bash
./download_mnist.sh                        # → data/*-ubyte
lake build mnist-mlp-train             # links -liree_ffi from ./ffi
.lake/build/bin/mnist-mlp-train data   # generates vmfbs + trains
```

Expected output: 12 epochs, ~14-16 s/epoch on a modest GPU (or ~90
s/epoch on CPU), final accuracy ≈ 97.9%.

For a bigger smoke test with Imagenette:

```bash
./download_imagenette.sh                   # → data/imagenette/
lake build resnet34-train
./run.sh resnet34                          # sets IREE_BACKEND + GPU
```

The first run spends ~10-15 min in `iree-compile` generating the
train-step vmfb; subsequent runs hit the cache and start training
immediately.

If you see `error while loading shared libraries: libiree_ffi.so`, it's
an rpath issue — the lakefile sets `-Wl,-rpath,./ffi`, so run the binary
from the repo root (not from `.lake/build/bin/`).

**Note:** the historical `mnist-mlp-train` (without the `-f32` suffix)
in `historical/MainMlpTrain.lean` is a pre-codegen artifact that still
uses hand-authored MLIR and a custom FFI. It's kept for reference but
is not the recommended path. Use `mnist-mlp-train` instead — it
uses the same unified `spec.train` loop as every other architecture.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `cannot find -liree_ffi` at link time | `ffi/libiree_ffi.so` missing | redo §4 |
| `undefined reference to iree_allocator_system` | forgot `-DIREE_ALLOCATOR_SYSTEM_CTL=...` | add the define in §4a |
| `undefined reference to flatcc_verify_*` | flatcc archives missing or wrong order | add both `libflatcc_*.a` inside `--start-group` |
| `no HAL driver matching 'cuda'` at runtime | `--whole-archive` was dropped | redo §4b with the wrap intact |
| `--no-allow-shlib-undefined` errors when linking the Lean trainer | Lean's bundled lld is strict about transitive glibc symbols | already handled — every trainer in `lakefile.lean` passes `-Wl,--allow-shlib-undefined` |
| `iree-compile` not found | venv not active in the shell that runs `lake build` | `source .venv/bin/activate` first |
| `undefined symbol: flatcc_verify_table_as_root` *at runtime, not link time* | flatcc symbols built with `-fvisibility=hidden`, stripped from `.dynsym` | use the §4c fallback (rebuild `verifier.c` with `-fvisibility=default`) |
| Trainer crashes with `device target "cuda"` error on a HIP machine (or vice versa) | `IREE_BACKEND` env var not propagated through tmux into the binary | set it via a shell wrapper script (see `run_*.sh` in the repo root for examples) — `IREE_BACKEND=cuda` is the default |
| Eval call fails with `module 'foo_eval' not registered` after epoch 10 | the trainer's eval string is misspelled vs the sanitized spec name | sanitize lowercases and replaces non-alphanum with `_` — "EfficientNet V2-S" → `efficientnet_v2_s_eval`, NOT `efficient_net_v2_s_eval` |

## What this gets you

Once `libiree_ffi.so` exists in `ffi/`, every other target in
`lakefile.lean` that links `-liree_ffi` (mnist, cifar, resnet, mobilenet,
efficientnet, vit, vgg, …) builds without further setup. The runtime
library is shared across all of them; only the `.vmfb` files differ.

## Running a real trainer (operational notes)

Building `libiree_ffi.so` is the hard part; running the bigger trainers
afterwards has a few non-obvious gotchas worth knowing up front.

**1. Use a shell wrapper to set env vars.** The trainers respect
`IREE_BACKEND` (default: `cuda`; set to `rocm` for AMD) and
`HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`. Setting these inline in
`tmux send-keys "FOO=bar binary"` works *most* of the time but we hit
cases where the env didn't propagate cleanly. The repo includes
`run_effnet.sh`, `run_vit.sh`, `run_mnv4.sh` etc as examples — short
shell scripts that `export` then `exec` the binary. Use one for any
trainer you care about.

```bash
# run_resnet34.sh
#!/bin/bash
export HIP_VISIBLE_DEVICES=0      # or CUDA_VISIBLE_DEVICES
export IREE_BACKEND=rocm          # or omit / set to cuda
exec .lake/build/bin/resnet34-train 2>&1 | tee resnet34.log
```

**2. The first run compiles vmfbs.** Each trainer generates its
StableHLO MLIR on launch and shells out to `iree-compile`. For
ResNet-sized models the train step is 500 KB-2 MB of MLIR and IREE
takes ~5-15 minutes to lower it to a `.vmfb`. You'll see:
```
Generating train step MLIR...
  517912 chars
Compiling vmfbs...
  forward compiled
  eval forward compiled
  compiled
```
…and then training starts. The vmfbs are written to `.lake/build/`.
Subsequent runs of the same binary regenerate the MLIR fresh each
time (because the trainer always calls `generateTrainStep`), so
expect the compile delay every launch unless you rip out that step.

**3. First val eval is at epoch 10.** If the eval forward MLIR is
wrong (or the eval call uses a misspelled module name) the trainer
runs for ~10 epochs of training and then crashes. Watch the first
val eval output before walking away from a long run.

**4. Running BN stats are critical for eval accuracy.** Don't be
surprised if epoch-10 val accuracy looks bad initially — running BN
EMA needs ~1 epoch to stabilize. The numbers in `RESULTS.md` are
real.

**5. Big models eat a long time.** EfficientNetV2-S is 38M params
and takes ~9 min/epoch on a single 7900 XTX, so 80 epochs = ~12 hours.
Plan accordingly.
