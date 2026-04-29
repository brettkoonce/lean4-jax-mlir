/-! Float32-in-ByteArray utilities.

    All tensor data (params, images, gradients) stored as raw float32 bytes
    in `ByteArray`. Zero conversion at the FFI boundary — IREE sees the same
    bytes Lean wrote. Lean-side reads widen to `Float` (f64) only when needed
    (loss printing, argmax, debugging).

    Heavy-lift operations (He init, const fill, image loading) are @[extern]
    to C for speed — avoids millions of Lean-level push calls. -/

namespace F32

/-- Number of float32 elements in a ByteArray. -/
def size (ba : ByteArray) : Nat := ba.size / 4

/-- Read a float32 at `idx` (element index, not byte index), widened to Float. -/
@[extern "lean_f32_read"]
opaque read (ba : @& ByteArray) (idx : USize) : Float

/-- Allocate `n` float32 values filled with `v` as a ByteArray. -/
@[extern "lean_f32_const"]
opaque const (n : USize) (v : Float) : IO ByteArray

/-- He-init: `n` float32 values ~ N(0, scale²), packed in ByteArray.
    Uses xorshift + 3-uniform-sum approximation (same as existing randnFA). -/
@[extern "lean_f32_he_init"]
opaque heInit (seed : USize) (n : USize) (scale : Float) : IO ByteArray

/-- Concatenate multiple ByteArrays. Fast (memcpy per chunk). -/
def concat (arrays : Array ByteArray) : ByteArray := Id.run do
  let mut out : ByteArray := .empty
  for a in arrays do out := out.append a
  return out

/-- Slice `count` float32 elements starting at element index `start`. -/
def slice (ba : ByteArray) (start count : Nat) : ByteArray :=
  ba.extract (start * 4) ((start + count) * 4)

/-- Extract the loss (last float32) from a train_step output. -/
def extractLoss (out : ByteArray) (lossIdx : Nat) : Float :=
  read out lossIdx.toUSize

/-- Drop the trailing loss float from train_step output. -/
def dropLoss (out : ByteArray) (nParams : Nat) : ByteArray :=
  out.extract 0 (nParams * 4)

/-- Argmax over 10 float32 values starting at element offset `off`. -/
@[extern "lean_f32_argmax10"]
opaque argmax10 (ba : @& ByteArray) (off : USize) : USize

/-- Load MNIST images from IDX file directly into f32 ByteArray (normalized to [0,1]).
    Returns (images ByteArray, count as Nat). -/
@[extern "lean_f32_load_idx_images"]
opaque loadIdxImages (path : @& String) : IO (ByteArray × Nat)

/-- Load MNIST labels from IDX file into int32 LE ByteArray. -/
@[extern "lean_f32_load_idx_labels"]
opaque loadIdxLabels (path : @& String) : IO (ByteArray × Nat)

/-- Slice a batch of images: `count` images × `pixelsPerImage` floats. Zero-copy. -/
def sliceImages (images : ByteArray) (start count pixelsPerImage : Nat) : ByteArray :=
  images.extract (start * pixelsPerImage * 4) ((start + count) * pixelsPerImage * 4)

/-- Slice a batch of labels: `count` × 4 bytes (int32 LE). -/
def sliceLabels (labels : ByteArray) (start count : Nat) : ByteArray :=
  labels.extract (start * 4) ((start + count) * 4)

/-- Convert a batch of CIFAR-10 raw records to f32 ByteArray.
    `raw` is the concatenated batch file bytes (3073 bytes per record).
    Returns `count × 3072` float32 values normalized to [0,1]. -/
@[extern "lean_f32_cifar_batch"]
opaque cifarBatch (raw : @& ByteArray) (start : USize) (count : USize) : IO ByteArray

/-- Load Imagenette binary file. Returns (images f32 ByteArray, labels i32 ByteArray, count).
    Images are normalized with ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]. -/
@[extern "lean_f32_load_imagenette"]
opaque loadImagenette (path : @& String) : IO (ByteArray × ByteArray × Nat)

/-- Load Imagenette with explicit image size (e.g. 256 for train, 224 for val). -/
@[extern "lean_f32_load_imagenette_sized"]
opaque loadImagenetteSized (path : @& String) (imgSize : USize) : IO (ByteArray × ByteArray × Nat)

/-- Shuffle images and labels in-place (Fisher-Yates). Returns (shuffled images, shuffled labels). -/
@[extern "lean_f32_shuffle"]
opaque shuffle (images : ByteArray) (labels : ByteArray)
    (n : USize) (pixelsPerImage : USize) (seed : USize) : IO (ByteArray × ByteArray)

/-- EMA update: running = (1-momentum)*running + momentum*batch. -/
@[extern "lean_f32_ema"]
opaque ema (running : @& ByteArray) (batch : @& ByteArray) (momentum : Float) : IO ByteArray

/-- Random crop: batch of NCHW images from src_size to crop_size. -/
@[extern "lean_f32_random_crop"]
opaque randomCrop (images : @& ByteArray) (batch : USize) (channels : USize)
    (srcH : USize) (srcW : USize) (cropH : USize) (cropW : USize)
    (seed : USize) : IO ByteArray

/-- Deterministic center crop: same window (y0=x0=max/2) for every image in
    the batch. No RNG. Used as the augment=false preprocessing fallback. -/
@[extern "lean_f32_center_crop"]
opaque centerCrop (images : @& ByteArray) (batch : USize) (channels : USize)
    (srcH : USize) (srcW : USize) (cropH : USize) (cropW : USize) : IO ByteArray

/-- Random horizontal flip for a batch of NCHW images (50% per image). -/
@[extern "lean_f32_random_hflip"]
opaque randomHFlip (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (seed : USize) : IO ByteArray

/-- Mixup (Zhang et al. 2017) — λ ~ Beta(α, α), x_mixed[i] =
    λ·x[i] + (1-λ)·x[π(i)]. Returns the mixed image batch.
    Pair with `mixupSoftLabels` using the SAME seed + alpha. -/
@[extern "lean_f32_mixup_images"]
opaque mixupImages (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (seed : USize) : IO ByteArray

/-- Soft labels for the mixup. Pair with `mixupImages` (same seed + alpha).
    Output shape: [batch, nClasses] f32, with label smoothing applied. -/
@[extern "lean_f32_mixup_soft_labels"]
opaque mixupSoftLabels (intLabels : @& ByteArray) (batch : USize) (nClasses : USize)
    (alpha : Float) (smooth : Float) (seed : USize) : IO ByteArray

/-- CutMix (Yun et al. 2019) — paste a random rectangle from x[π(i)]
    onto x[i]. Pair with `cutmixSoftLabels` (same seed + alpha). -/
@[extern "lean_f32_cutmix_images"]
opaque cutmixImages (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (alpha : Float) (seed : USize) : IO ByteArray

/-- Soft labels for CutMix. λ_actual is recomputed from rectangle area. -/
@[extern "lean_f32_cutmix_soft_labels"]
opaque cutmixSoftLabels (intLabels : @& ByteArray) (batch : USize) (nClasses : USize)
    (height : USize) (width : USize) (alpha : Float) (smooth : Float)
    (seed : USize) : IO ByteArray

/-- Random Erasing (Zhong et al. 2017) — with probability `prob`, fill a
    random rectangle (relative area 2–33%, aspect 0.3–3.3) with N(0,1)
    noise. Per-image independent. Labels unchanged. -/
@[extern "lean_f32_random_erasing"]
opaque randomErasing (images : @& ByteArray) (batch : USize) (channels : USize)
    (height : USize) (width : USize) (prob : Float) (seed : USize) : IO ByteArray

/-- EMA on squared values: out = (1−mom)·running + mom·batch². Used by
    SWAG to maintain a running E[θ²] alongside SWA's running E[θ]. -/
@[extern "lean_f32_ema_sq"]
opaque emaSq (running : @& ByteArray) (batch : @& ByteArray) (momentum : Float) : IO ByteArray

/-- Element-wise subtract: a − b. Used by SWAG for per-epoch deviation
    snapshots `p − swaMean`. -/
@[extern "lean_f32_subtract"]
opaque subtract (a : @& ByteArray) (b : @& ByteArray) : IO ByteArray

/-- SWAG sample weights (Maddox et al. 2019). Given the SWA mean,
    SWA-of-θ² (for diagonal variance), and the K most-recent per-epoch
    deviation snapshots packed row-major as `K × nParams` f32, draw
    one sample from the SWAG posterior `N(μ, ½Σ_diag + ½ Σ_low)`. -/
@[extern "lean_f32_swag_sample"]
opaque swagSample (swaMean : @& ByteArray) (swaSq : @& ByteArray)
    (deviations : @& ByteArray) (nParams : USize) (k : USize)
    (seed : USize) : IO ByteArray

end F32
