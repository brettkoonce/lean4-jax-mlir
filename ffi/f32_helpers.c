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
LEAN_EXPORT lean_obj_res lean_f32_load_imagenette(b_lean_obj_arg path_obj, lean_obj_arg w) {
    (void)w;
    const char* path = lean_string_cstr(path_obj);
    FILE* f = fopen(path, "rb");
    if (!f) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("cannot open imagenette file")));
    uint32_t file_count;
    if (fread(&file_count, 4, 1, f) != 1) { fclose(f); return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("bad header"))); }
    uint32_t count = file_count;
    const size_t pix = 3 * 224 * 224;
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
        for (int ch = 0; ch < 3; ch++) {
            float m = mean[ch], s = istd[ch];
            for (int j = 0; j < 224*224; j++)
                dst[ch*224*224+j] = (buf[1+ch*224*224+j]/255.0f - m) * s;
        }
    }
    free(buf); fclose(f);
    // ByteArray × ByteArray × Nat = Prod ByteArray (Prod ByteArray Nat)
    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, lbl_ba);
    lean_ctor_set(inner, 1, lean_usize_to_nat((size_t)count));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, img_ba);
    lean_ctor_set(outer, 1, inner);
    return lean_io_result_mk_ok(outer);
}
