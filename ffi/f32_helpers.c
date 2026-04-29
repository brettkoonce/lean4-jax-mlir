// F32 ByteArray helpers for Lean FFI.
// All heavy-lift operations (init, read, argmax, data loading) in C to avoid
// millions of Lean-level push calls.

#include <lean/lean.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---- Read a float32 at element index from ByteArray ----
LEAN_EXPORT double lean_f32_read(b_lean_obj_arg ba, size_t idx) {
    const float* p = (const float*)lean_sarray_cptr(ba);
    return (double)p[idx];
}

// ---- Fill n float32 values with constant v ----
LEAN_EXPORT lean_obj_res lean_f32_const(size_t n, double v, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    float fv = (float)v;
    for (size_t i = 0; i < n; i++) p[i] = fv;
    return lean_io_result_mk_ok(ba);
}

// ---- He init: n float32 values ~ N(0, scale²) ----
LEAN_EXPORT lean_obj_res lean_f32_he_init(size_t seed, size_t n, double scale, lean_obj_arg w) {
    (void)w;
    size_t nbytes = n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    uint64_t s = (uint64_t)seed + 1;
    float fscale = (float)scale;
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int k = 0; k < 3; k++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            acc += (float)s / (float)UINT64_MAX - 0.5f;
        }
        p[i] = acc * 2.0f * fscale;
    }
    return lean_io_result_mk_ok(ba);
}

// ---- Argmax over 10 float32 values at element offset ----
LEAN_EXPORT size_t lean_f32_argmax10(b_lean_obj_arg ba, size_t off) {
    const float* p = (const float*)lean_sarray_cptr(ba);
    size_t best = 0;
    float bestv = p[off];
    for (size_t i = 1; i < 10; i++) {
        if (p[off + i] > bestv) { best = i; bestv = p[off + i]; }
    }
    return best;
}

// ---- Convert a batch of CIFAR-10 raw records to f32 ByteArray ----
// Each record is 3073 bytes (1 label + 3072 pixels). Normalizes to [0,1].
LEAN_EXPORT lean_obj_res lean_f32_cifar_batch(
    b_lean_obj_arg raw_ba, size_t start, size_t count, lean_obj_arg w) {
    (void)w;
    const uint8_t* raw = lean_sarray_cptr(raw_ba);
    size_t npixels = count * 3072;
    size_t nbytes = npixels * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    for (size_t i = 0; i < count; i++) {
        size_t rec_off = (start + i) * 3073;
        for (size_t j = 0; j < 3072; j++) {
            p[i * 3072 + j] = (float)raw[rec_off + 1 + j] / 255.0f;
        }
    }
    return lean_io_result_mk_ok(ba);
}

// ---- Load MNIST IDX images → f32 ByteArray (normalized to [0,1]) ----
static uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

LEAN_EXPORT lean_obj_res lean_f32_load_idx_images(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("cannot open image file")));
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* raw = (uint8_t*)malloc(fsize);
    fread(raw, 1, fsize, f);
    fclose(f);

    uint32_t magic = read_be32(raw);
    if (magic != 2051) { free(raw);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad IDX image magic"))); }
    uint32_t n = read_be32(raw + 4);
    uint32_t rows = read_be32(raw + 8);
    uint32_t cols = read_be32(raw + 12);
    size_t total = (size_t)n * rows * cols;

    size_t nbytes = total * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    float* p = (float*)lean_sarray_cptr(ba);
    for (size_t i = 0; i < total; i++)
        p[i] = (float)raw[16 + i] / 255.0f;
    free(raw);

    // Return (ByteArray, USize) as a pair
    // Prod ByteArray Nat: 2 boxed object fields
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, ba);
    lean_ctor_set(pair, 1, lean_usize_to_nat((size_t)n));
    return lean_io_result_mk_ok(pair);
}

// ---- Load MNIST IDX labels → int32 LE ByteArray ----
LEAN_EXPORT lean_obj_res lean_f32_load_idx_labels(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("cannot open label file")));
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* raw = (uint8_t*)malloc(fsize);
    fread(raw, 1, fsize, f);
    fclose(f);

    uint32_t magic = read_be32(raw);
    if (magic != 2049) { free(raw);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad IDX label magic"))); }
    uint32_t n = read_be32(raw + 4);

    size_t nbytes = (size_t)n * 4;
    lean_object* ba = lean_alloc_sarray(1, nbytes, nbytes);
    uint8_t* out = lean_sarray_cptr(ba);
    for (uint32_t i = 0; i < n; i++) {
        out[i * 4]     = raw[8 + i];
        out[i * 4 + 1] = 0;
        out[i * 4 + 2] = 0;
        out[i * 4 + 3] = 0;
    }
    free(raw);

    // Prod ByteArray Nat: 2 boxed object fields
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, ba);
    lean_ctor_set(pair, 1, lean_usize_to_nat((size_t)n));
    return lean_io_result_mk_ok(pair);
}

// ---- Imagenette binary -> f32 ByteArray (ImageNet mean/std normalized) ----
// Binary: 4-byte count (LE u32), per-sample: 1 byte label + 224*224*3 bytes (CHW, uint8)
// Returns (images ByteArray, labels ByteArray, count Nat)
// Internal loader parameterized by image size
static lean_obj_res load_imagenette_sized(const char* path, size_t img_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open imagenette file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f); return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad header"))); }
    uint32_t count = file_count;
    const size_t pix = 3 * img_size * img_size;
    size_t img_bytes = (size_t)count * pix * 4;
    size_t lbl_bytes = (size_t)count * 4;
    lean_object* img_ba = lean_alloc_sarray(1, img_bytes, img_bytes);
    lean_object* lbl_ba = lean_alloc_sarray(1, lbl_bytes, lbl_bytes);
    float* img = (float*)lean_sarray_cptr(img_ba);
    uint8_t* lbl = lean_sarray_cptr(lbl_ba);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};
    uint8_t* buf = (uint8_t*)malloc(1 + pix);
    for (uint32_t i = 0; i < count; i++) {
        if (fread(buf, 1, 1 + pix, f) != 1 + pix) { free(buf); fclose(f);
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("short read"))); }
        lbl[i*4]=buf[0]; lbl[i*4+1]=0; lbl[i*4+2]=0; lbl[i*4+3]=0;
        float* dst = img + (size_t)i * pix;
        size_t hw = img_size * img_size;
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (size_t j = 0; j < hw; j++)
                dst[ch*hw+j] = (buf[1+ch*hw+j]/255.0f - m) * s;
        }
    }
    free(buf); fclose(f);

    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, lbl_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}

LEAN_EXPORT lean_obj_res lean_f32_load_imagenette(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    return load_imagenette_sized(lean_string_cstr(path_obj), 224);
}

LEAN_EXPORT lean_obj_res lean_f32_load_imagenette_sized(b_lean_obj_arg path_obj, size_t img_size, lean_obj_arg w) {
    (void)w;
    return load_imagenette_sized(lean_string_cstr(path_obj), img_size);
}

// ---- Shuffle images + labels in-place (Fisher-Yates) ----
// images: n * pixels_per * 4 bytes; labels: n * 4 bytes
LEAN_EXPORT lean_obj_res lean_f32_shuffle(lean_obj_arg img_obj, lean_obj_arg lbl_obj,
                                          size_t n, size_t pixels_per, size_t seed,
                                          lean_obj_arg w) {
    (void)w;
    // Ensure exclusive ownership (rc == 1) for in-place mutation
    if (!lean_is_exclusive(img_obj)) img_obj = lean_copy_byte_array(img_obj);
    if (!lean_is_exclusive(lbl_obj)) lbl_obj = lean_copy_byte_array(lbl_obj);
    uint8_t* img = lean_sarray_cptr(img_obj);
    uint8_t* lbl = lean_sarray_cptr(lbl_obj);
    size_t img_stride = pixels_per * 4;
    // Temp buffer for one image
    uint8_t* tmp = (uint8_t*)malloc(img_stride > 4 ? img_stride : 4);
    uint64_t rng = seed ^ 0x5DEECE66DUL;
    for (size_t i = n - 1; i > 0; i--) {
        rng = rng * 6364136223846793005UL + 1442695040888963407UL;
        size_t j = (size_t)((rng >> 16) % (i + 1));
        if (i != j) {
            // Swap images
            memcpy(tmp, img + i * img_stride, img_stride);
            memcpy(img + i * img_stride, img + j * img_stride, img_stride);
            memcpy(img + j * img_stride, tmp, img_stride);
            // Swap labels
            memcpy(tmp, lbl + i * 4, 4);
            memcpy(lbl + i * 4, lbl + j * 4, 4);
            memcpy(lbl + j * 4, tmp, 4);
        }
    }
    free(tmp);
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, img_obj);
    lean_ctor_set(pair, 1, lbl_obj);
    return lean_io_result_mk_ok(pair);
}

// ---- EMA update: running = (1-momentum)*running + momentum*batch ----
LEAN_EXPORT lean_obj_res lean_f32_ema(
    b_lean_obj_arg running_ba, b_lean_obj_arg batch_ba,
    double momentum, lean_obj_arg w) {
    (void)w;
    size_t n = lean_sarray_size(running_ba) / 4;
    size_t nbytes = n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* r = (const float*)lean_sarray_cptr(running_ba);
    const float* b = (const float*)lean_sarray_cptr(batch_ba);
    float* o = (float*)lean_sarray_cptr(out);
    float mom = (float)momentum;
    float omom = 1.0f - mom;
    for (size_t i = 0; i < n; i++) o[i] = omom * r[i] + mom * b[i];
    return lean_io_result_mk_ok(out);
}

// ---- Random crop: batch of NCHW images from src_size to crop_size ----
// Input: batch * C * src_h * src_w floats (already normalized).
// Output: batch * C * crop_h * crop_w floats (random offset per image).
LEAN_EXPORT lean_obj_res lean_f32_random_crop(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t src_h, size_t src_w, size_t crop_h, size_t crop_w,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_pixels = channels * crop_h * crop_w;
    size_t out_nbytes = batch * out_pixels * 4;
    size_t src_pixels = channels * src_h * src_w;
    lean_object* out = lean_alloc_sarray(1, out_nbytes, out_nbytes);
    float* dst = (float*)lean_sarray_cptr(out);
    const float* src = (const float*)lean_sarray_cptr(ba);

    size_t max_y = src_h - crop_h;
    size_t max_x = src_w - crop_w;
    uint64_t s = seed + 1;
    for (size_t i = 0; i < batch; i++) {
        // xorshift64 for random offsets
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t y0 = (max_y > 0) ? (s % (max_y + 1)) : 0;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        size_t x0 = (max_x > 0) ? (s % (max_x + 1)) : 0;

        const float* img_src = src + i * src_pixels;
        float* img_dst = dst + i * out_pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_h; h++) {
                memcpy(img_dst + c * crop_h * crop_w + h * crop_w,
                       img_src + c * src_h * src_w + (y0 + h) * src_w + x0,
                       crop_w * sizeof(float));
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- Deterministic center crop for a batch of NCHW images ----
// Same shape as random_crop but y0/x0 are fixed to (max/2) — same crop
// window for every image, no RNG. Used as the no-augment fallback for
// Imagenette (stored 256x256, model input 224x224) so training gets the
// expected tensor shape even when cfg.augment=false.
LEAN_EXPORT lean_obj_res lean_f32_center_crop(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t src_h, size_t src_w, size_t crop_h, size_t crop_w,
    lean_obj_arg w) {
    (void)w;
    size_t out_pixels = channels * crop_h * crop_w;
    size_t out_nbytes = batch * out_pixels * 4;
    size_t src_pixels = channels * src_h * src_w;
    lean_object* out = lean_alloc_sarray(1, out_nbytes, out_nbytes);
    float* dst = (float*)lean_sarray_cptr(out);
    const float* src = (const float*)lean_sarray_cptr(ba);

    size_t y0 = (src_h > crop_h) ? (src_h - crop_h) / 2 : 0;
    size_t x0 = (src_w > crop_w) ? (src_w - crop_w) / 2 : 0;
    for (size_t i = 0; i < batch; i++) {
        const float* img_src = src + i * src_pixels;
        float* img_dst = dst + i * out_pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < crop_h; h++) {
                memcpy(img_dst + c * crop_h * crop_w + h * crop_w,
                       img_src + c * src_h * src_w + (y0 + h) * src_w + x0,
                       crop_w * sizeof(float));
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ---- Random horizontal flip for a batch of NCHW images (in-place on copy) ----
// pixels_per_image = C * H * W, width = W, 50% chance per image.
LEAN_EXPORT lean_obj_res lean_f32_random_hflip(
    b_lean_obj_arg ba, size_t batch, size_t channels,
    size_t height, size_t width, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels_per_image = channels * height * width;
    size_t nbytes = batch * pixels_per_image * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    memcpy(lean_sarray_cptr(out), lean_sarray_cptr(ba), nbytes);
    float* data = (float*)lean_sarray_cptr(out);

    uint64_t s = seed + 1;
    for (size_t i = 0; i < batch; i++) {
        // xorshift64 for random decision
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        if (s & 1) {
            // Flip this image horizontally: reverse each row in each channel
            float* img = data + i * pixels_per_image;
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    float* row = img + c * height * width + h * width;
                    for (size_t l = 0, r = width - 1; l < r; l++, r--) {
                        float tmp = row[l];
                        row[l] = row[r];
                        row[r] = tmp;
                    }
                }
            }
        }
    }
    return lean_io_result_mk_ok(out);
}

// ============================================================
// Mixup / CutMix / Random Erasing — DeiT-style augmentation pack
// ============================================================
//
// Mixup (Zhang et al. 2017):
//   λ ~ Beta(α, α); pick a permutation π over the batch.
//   x_mixed[i]    = λ·x[i]       + (1-λ)·x[π(i)]
//   y_mixed[i, c] = λ·smooth(y[i], c) + (1-λ)·smooth(y[π(i)], c)
//
// CutMix (Yun et al. 2019):
//   λ ~ Beta(α, α); pick a random rectangle of size √(1-λ);
//   paste rectangle from x[π(i)] onto x[i].
//   λ_actual = 1 - (rect_area / image_area)  (bounded by image edges)
//   y_mixed[i, c] = λ_actual·smooth(y[i], c) + (1-λ_actual)·smooth(y[π(i)], c)
//
// Random Erasing (Zhong et al. 2017): with probability p,
//   pick a rectangle of relative area in [s_min, s_max] and
//   aspect ratio in [r_min, r_max], fill with N(0, 1) noise. Per image,
//   independent. Labels unchanged.
//
// Mixup and CutMix expose two FFI calls each, sharing a seed: `_images`
// computes the mixed-image tensor; `_soft_labels` recomputes λ + π from
// the same seed and emits a `[batch × n_classes]` smoothed soft-label
// tensor. Calling with the same seed in both is critical — the label
// must match the image mix.

// xorshift64 step. Returns a uint64 in (0, 2^64).
static inline uint64_t f32_xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

// Uniform(0, 1) from xorshift state.
static inline double f32_unif01(uint64_t* s) {
    uint64_t x = f32_xorshift64(s);
    return (double)(x >> 11) / (double)(1ULL << 53);
}

// Standard normal via Box-Muller (returns one of the two values).
static inline double f32_randn(uint64_t* s) {
    double u1 = f32_unif01(s);
    double u2 = f32_unif01(s);
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// Marsaglia & Tsang gamma sampler for shape α > 0, scale 1.
// For α < 1, use the boost trick: G(α) = G(α+1) · U^(1/α).
static double f32_gamma_sample(double alpha, uint64_t* s) {
    if (alpha < 1.0) {
        double g1 = f32_gamma_sample(alpha + 1.0, s);
        double u = f32_unif01(s);
        if (u < 1e-300) u = 1e-300;
        return g1 * pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    while (1) {
        double x = f32_randn(s);
        double v = 1.0 + c * x;
        if (v <= 0) continue;
        v = v * v * v;
        double u = f32_unif01(s);
        if (log(u) < 0.5 * x * x + d - d * v + d * log(v)) {
            return d * v;
        }
    }
}

// Beta(α, α) via two Gammas.
static double f32_beta_symmetric(double alpha, uint64_t* s) {
    double g1 = f32_gamma_sample(alpha, s);
    double g2 = f32_gamma_sample(alpha, s);
    return g1 / (g1 + g2);
}

// Fisher-Yates permutation [0, batch). Caller provides storage.
static void f32_permutation(size_t* perm, size_t batch, uint64_t* s) {
    for (size_t i = 0; i < batch; i++) perm[i] = i;
    for (size_t i = batch - 1; i > 0; i--) {
        uint64_t r = f32_xorshift64(s);
        size_t j = (size_t)(r % (i + 1));
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}

// Smoothed onehot value for label `y` at class `c`.
//   smooth = label_smoothing in [0, 1).
//   If c == y: 1 - smooth + smooth/N
//   Else:      smooth/N
static inline float f32_smooth_onehot(int y, size_t c, double smooth, size_t n_classes) {
    double off = smooth / (double)n_classes;
    if ((size_t)y == c) return (float)(1.0 - smooth + off);
    return (float)off;
}

// ----------------------------------------------------------------
// Mixup: images. Returns ByteArray of shape [batch, C, H, W] f32.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_mixup_images(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double alpha, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;  // bias toward keeping main image dominant
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        const float* a = in + i * pixels;
        const float* b = in + perm[i] * pixels;
        float* d = o + i * pixels;
        for (size_t k = 0; k < pixels; k++) d[k] = l * a[k] + l1 * b[k];
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// Mixup: soft labels. Returns [batch, n_classes] f32.
// MUST use same `seed` and `alpha` as the matching mixup_images call.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_mixup_soft_labels(
    b_lean_obj_arg int_labels, size_t batch, size_t n_classes,
    double alpha, double smooth, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_n = batch * n_classes;
    size_t nbytes = out_n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const uint8_t* lbl = (const uint8_t*)lean_sarray_cptr(int_labels);
    uint64_t s = seed ^ 0xa1b2c3d4e5f60718ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    if (lambda < 0.5) lambda = 1.0 - lambda;
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    float l = (float)lambda;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        int32_t y_a, y_b;
        memcpy(&y_a, lbl + i * 4, 4);
        memcpy(&y_b, lbl + perm[i] * 4, 4);
        for (size_t c = 0; c < n_classes; c++) {
            float a = f32_smooth_onehot(y_a, c, smooth, n_classes);
            float b = f32_smooth_onehot(y_b, c, smooth, n_classes);
            o[i * n_classes + c] = l * a + l1 * b;
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// CutMix: images. Returns mixed-image ByteArray.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_cutmix_images(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double alpha, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    const float* in = (const float*)lean_sarray_cptr(images);
    float* o = (float*)lean_sarray_cptr(out);
    memcpy(o, in, nbytes);
    uint64_t s = seed ^ 0xb2c3d4e5f6071829ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    double cut_ratio = sqrt(1.0 - lambda);
    size_t cut_h = (size_t)((double)height * cut_ratio);
    size_t cut_w = (size_t)((double)width * cut_ratio);
    if (cut_h < 1) cut_h = 1; if (cut_w < 1) cut_w = 1;
    size_t cy = (size_t)(f32_unif01(&s) * (double)height);
    size_t cx = (size_t)(f32_unif01(&s) * (double)width);
    size_t y1 = cy > cut_h / 2 ? cy - cut_h / 2 : 0;
    size_t y2 = cy + cut_h / 2; if (y2 > height) y2 = height;
    size_t x1 = cx > cut_w / 2 ? cx - cut_w / 2 : 0;
    size_t x2 = cx + cut_w / 2; if (x2 > width) x2 = width;
    for (size_t i = 0; i < batch; i++) {
        const float* b_img = in + perm[i] * pixels;
        float* d_img = o + i * pixels;
        for (size_t c = 0; c < channels; c++) {
            for (size_t y = y1; y < y2; y++) {
                size_t off = c * height * width + y * width;
                for (size_t x = x1; x < x2; x++) {
                    d_img[off + x] = b_img[off + x];
                }
            }
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// CutMix: soft labels. Recomputes λ_actual from rectangle area.
// MUST use same `seed/alpha` as cutmix_images.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_cutmix_soft_labels(
    b_lean_obj_arg int_labels, size_t batch, size_t n_classes,
    size_t height, size_t width, double alpha, double smooth,
    size_t seed, lean_obj_arg w) {
    (void)w;
    size_t out_n = batch * n_classes;
    size_t nbytes = out_n * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    const uint8_t* lbl = (const uint8_t*)lean_sarray_cptr(int_labels);
    uint64_t s = seed ^ 0xb2c3d4e5f6071829ULL; if (s == 0) s = 1;
    double lambda = f32_beta_symmetric(alpha, &s);
    size_t* perm = (size_t*)malloc(batch * sizeof(size_t));
    f32_permutation(perm, batch, &s);
    double cut_ratio = sqrt(1.0 - lambda);
    size_t cut_h = (size_t)((double)height * cut_ratio);
    size_t cut_w = (size_t)((double)width * cut_ratio);
    if (cut_h < 1) cut_h = 1; if (cut_w < 1) cut_w = 1;
    size_t cy = (size_t)(f32_unif01(&s) * (double)height);
    size_t cx = (size_t)(f32_unif01(&s) * (double)width);
    size_t y1 = cy > cut_h / 2 ? cy - cut_h / 2 : 0;
    size_t y2 = cy + cut_h / 2; if (y2 > height) y2 = height;
    size_t x1 = cx > cut_w / 2 ? cx - cut_w / 2 : 0;
    size_t x2 = cx + cut_w / 2; if (x2 > width) x2 = width;
    double rect_area = (double)((y2 - y1) * (x2 - x1));
    double total_area = (double)(height * width);
    double l_actual = 1.0 - rect_area / total_area;
    float l = (float)l_actual;
    float l1 = 1.0f - l;
    for (size_t i = 0; i < batch; i++) {
        int32_t y_a, y_b;
        memcpy(&y_a, lbl + i * 4, 4);
        memcpy(&y_b, lbl + perm[i] * 4, 4);
        for (size_t c = 0; c < n_classes; c++) {
            float a = f32_smooth_onehot(y_a, c, smooth, n_classes);
            float b = f32_smooth_onehot(y_b, c, smooth, n_classes);
            o[i * n_classes + c] = l * a + l1 * b;
        }
    }
    free(perm);
    return lean_io_result_mk_ok(out);
}

// ----------------------------------------------------------------
// Random Erasing. Per-image, with probability `prob`, fills a random
// rectangle (area 2-33% of image, aspect ratio 0.3-3.3) with N(0,1)
// noise. Labels unchanged — caller can keep using int32 labels.
// ----------------------------------------------------------------
LEAN_EXPORT lean_obj_res lean_f32_random_erasing(
    b_lean_obj_arg images, size_t batch, size_t channels,
    size_t height, size_t width, double prob, size_t seed, lean_obj_arg w) {
    (void)w;
    size_t pixels = channels * height * width;
    size_t nbytes = batch * pixels * 4;
    lean_object* out = lean_alloc_sarray(1, nbytes, nbytes);
    float* o = (float*)lean_sarray_cptr(out);
    memcpy(o, lean_sarray_cptr(images), nbytes);
    uint64_t s = seed ^ 0xc3d4e5f60718293aULL; if (s == 0) s = 1;
    const double s_min = 0.02, s_max = 0.33;
    const double r_min = 0.3, r_max = 3.3;
    for (size_t i = 0; i < batch; i++) {
        if (f32_unif01(&s) >= prob) continue;
        for (int attempt = 0; attempt < 10; attempt++) {
            double area = (double)(height * width);
            double target_area = area * (s_min + (s_max - s_min) * f32_unif01(&s));
            double aspect = exp(log(r_min) + (log(r_max) - log(r_min)) * f32_unif01(&s));
            size_t h_e = (size_t)round(sqrt(target_area * aspect));
            size_t w_e = (size_t)round(sqrt(target_area / aspect));
            if (h_e >= height || w_e >= width || h_e < 1 || w_e < 1) continue;
            size_t y0 = (size_t)(f32_unif01(&s) * (double)(height - h_e));
            size_t x0 = (size_t)(f32_unif01(&s) * (double)(width - w_e));
            for (size_t c = 0; c < channels; c++) {
                for (size_t y = y0; y < y0 + h_e; y++) {
                    size_t off = i * pixels + c * height * width + y * width;
                    for (size_t x = x0; x < x0 + w_e; x++) {
                        o[off + x] = (float)f32_randn(&s);
                    }
                }
            }
            break;
        }
    }
    return lean_io_result_mk_ok(out);
}
