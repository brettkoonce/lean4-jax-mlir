// Lean FFI shim for the IREE runtime wrapper.
// Converts between Lean's FloatArray (Float64) and IREE's expected float32.
//
// Exports:
//   lean_iree_session_create(path: String, world) : IO IreeSession
//   lean_iree_mlp_forward(sess, x, W0, b0, W1, b1, W2, b2, batch, world) : IO FloatArray

#include <lean/lean.h>
#include <stdlib.h>
#include <string.h>
#include "iree_ffi.h"

// ---- External class for IreeSession ----
static lean_external_class* g_iree_session_class = NULL;

static void iree_session_finalize(void* p) {
  iree_ffi_session_release((iree_ffi_session_t*)p);
}
static void iree_session_foreach(void* p, b_lean_obj_arg f) { (void)p; (void)f; }

static void ensure_iree_session_class(void) {
  if (!g_iree_session_class) {
    g_iree_session_class = lean_register_external_class(
        iree_session_finalize, iree_session_foreach);
  }
}

// ---- Session create ----
LEAN_EXPORT lean_obj_res lean_iree_session_create(
    b_lean_obj_arg path_obj, lean_obj_arg world) {
  (void)world;
  ensure_iree_session_class();
  const char* path = lean_string_cstr(path_obj);
  iree_ffi_session_t* sess = iree_ffi_session_create(path);
  if (!sess) {
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("iree_ffi_session_create failed (see stderr)")));
  }
  return lean_io_result_mk_ok(
      lean_alloc_external(g_iree_session_class, sess));
}

// ---- Helpers: Float64 FloatArray → float32 staging buffer ----
static float* fa_to_f32(b_lean_obj_arg fa, size_t n) {
  double const* src = lean_float_array_cptr(fa);
  float* dst = (float*)malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
  return dst;
}

// ---- MLP forward ----
// Inputs (Lean-side): Float64 FloatArrays with the expected sizes.
// Hard-coded for the MNIST MLP shape: 784 -> 512 -> 512 -> 10, static batch.
LEAN_EXPORT lean_obj_res lean_iree_mlp_forward(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg x,
    b_lean_obj_arg W0, b_lean_obj_arg b0,
    b_lean_obj_arg W1, b_lean_obj_arg b1,
    b_lean_obj_arg W2, b_lean_obj_arg b2,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);

  // Float64 → Float32 staging buffers.
  float* x_f  = fa_to_f32(x,  batch * 784);
  float* W0_f = fa_to_f32(W0, 784 * 512);
  float* b0_f = fa_to_f32(b0, 512);
  float* W1_f = fa_to_f32(W1, 512 * 512);
  float* b1_f = fa_to_f32(b1, 512);
  float* W2_f = fa_to_f32(W2, 512 * 10);
  float* b2_f = fa_to_f32(b2, 10);

  size_t logits_n = batch * 10;
  float* logits_f = (float*)malloc(logits_n * sizeof(float));

  int32_t ranks[7] = {2, 2, 1, 2, 1, 2, 1};
  int64_t dims[11] = {
      (int64_t)batch, 784,   // x
      784, 512,              // W0
      512,                   // b0
      512, 512,              // W1
      512,                   // b1
      512, 10,               // W2
      10,                    // b2
  };
  const float* inputs[7] = {x_f, W0_f, b0_f, W1_f, b1_f, W2_f, b2_f};
  int64_t out_totals[1] = {(int64_t)logits_n};
  float* outputs[1] = {logits_f};

  int rc = iree_ffi_invoke_f32(sess, "mnist_mlp.forward",
                               7, ranks, dims, inputs,
                               1, out_totals, outputs);

  free(x_f); free(W0_f); free(b0_f); free(W1_f);
  free(b1_f); free(W2_f); free(b2_f);

  if (rc != 0) {
    free(logits_f);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(
            lean_mk_string("iree_ffi_invoke_f32 failed (see stderr)")));
  }

  // Copy float32 back into a Float64 FloatArray.
  lean_object* result = lean_alloc_sarray(sizeof(double), logits_n, logits_n);
  double* rp = lean_float_array_cptr(result);
  for (size_t i = 0; i < logits_n; i++) rp[i] = (double)logits_f[i];
  free(logits_f);

  return lean_io_result_mk_ok(result);
}

// ---- MLP train step ----
// Inputs (Lean-side):
//   sess_obj  — IreeSession
//   params_fa — FloatArray f64, length 669706 (W0|b0|W1|b1|W2|b2 packed)
//   x_fa      — FloatArray f64, length batch*784
//   y_ba      — ByteArray, batch*4 bytes (packed int32 LE labels)
//   lr        — Float (double → f32 at boundary)
//   batch     — USize
// Output: FloatArray f64, length 669707 (new params + loss at index 669706)
LEAN_EXPORT lean_obj_res lean_iree_mlp_train_step(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg params_fa,
    b_lean_obj_arg x_fa,
    b_lean_obj_arg y_ba,
    double lr,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);

  const size_t N_W0 = 784*512, N_b0 = 512;
  const size_t N_W1 = 512*512, N_b1 = 512;
  const size_t N_W2 = 512*10,  N_b2 = 10;
  const size_t N_P  = N_W0+N_b0+N_W1+N_b1+N_W2+N_b2;  // 669706

  // Stage params: f64 → f32
  double const* p_src = lean_float_array_cptr(params_fa);
  float* p_f = (float*)malloc(N_P * sizeof(float));
  for (size_t i = 0; i < N_P; i++) p_f[i] = (float)p_src[i];
  float* W0_p = p_f;
  float* b0_p = W0_p + N_W0;
  float* W1_p = b0_p + N_b0;
  float* b1_p = W1_p + N_W1;
  float* W2_p = b1_p + N_b1;
  float* b2_p = W2_p + N_W2;

  // Stage x: f64 → f32
  size_t N_x = batch * 784;
  double const* x_src = lean_float_array_cptr(x_fa);
  float* x_f = (float*)malloc(N_x * sizeof(float));
  for (size_t i = 0; i < N_x; i++) x_f[i] = (float)x_src[i];

  // y labels: ByteArray → int32*
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);

  // Output buffers
  float* W0o = (float*)malloc(N_W0 * sizeof(float));
  float* b0o = (float*)malloc(N_b0 * sizeof(float));
  float* W1o = (float*)malloc(N_W1 * sizeof(float));
  float* b1o = (float*)malloc(N_b1 * sizeof(float));
  float* W2o = (float*)malloc(N_W2 * sizeof(float));
  float* b2o = (float*)malloc(N_b2 * sizeof(float));
  float loss_f = 0.0f;

  int rc = iree_ffi_train_step_mlp(
      sess, "jit_train_step.main", (int)batch,
      W0_p, b0_p, W1_p, b1_p, W2_p, b2_p,
      x_f, y_ptr, (float)lr,
      W0o, b0o, W1o, b1o, W2o, b2o, &loss_f);

  free(p_f); free(x_f);
  if (rc != 0) {
    free(W0o); free(b0o); free(W1o); free(b1o); free(W2o); free(b2o);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("iree_ffi_train_step_mlp failed")));
  }

  // Pack new params + loss into a single FloatArray (f64).
  size_t N_out = N_P + 1;
  lean_object* result = lean_alloc_sarray(sizeof(double), N_out, N_out);
  double* rp = lean_float_array_cptr(result);
  const float* outs[6] = {W0o, b0o, W1o, b1o, W2o, b2o};
  const size_t sizes[6] = {N_W0, N_b0, N_W1, N_b1, N_W2, N_b2};
  size_t off = 0;
  for (int k = 0; k < 6; k++) {
    for (size_t i = 0; i < sizes[k]; i++) rp[off+i] = (double)outs[k][i];
    off += sizes[k];
  }
  rp[N_P] = (double)loss_f;

  free(W0o); free(b0o); free(W1o); free(b1o); free(W2o); free(b2o);
  return lean_io_result_mk_ok(result);
}

// ---- Generic train step ----
// shapes_ba encodes: [n_params(i32), rank0(i32), d0_0(i32), ..., rank1(i32), d1_0(i32), ...]
// x_shape_ba encodes: [x_rank(i32), xd0(i32), xd1(i32), ...]
LEAN_EXPORT lean_obj_res lean_iree_train_step_packed(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_fa,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_fa,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_ba,
    double lr,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse shapes descriptor
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  // Stage params f64 → f32
  double const* p_src = lean_float_array_cptr(params_fa);
  float* p_f = (float*)malloc(total_params * sizeof(float));
  for (int64_t i = 0; i < total_params; i++) p_f[i] = (float)p_src[i];

  // Parse x shape
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8]; int64_t x_total = 1;
  for (int i = 0; i < x_rank; i++) { x_dims[i] = (int64_t)xsp[1+i]; x_total *= x_dims[i]; }

  // Stage x f64 → f32
  double const* x_src = lean_float_array_cptr(x_fa);
  float* x_f = (float*)malloc(x_total * sizeof(float));
  for (int64_t i = 0; i < x_total; i++) x_f[i] = (float)x_src[i];

  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);
  float* p_out = (float*)malloc(total_params * sizeof(float));
  float loss_f = 0.0f;

  int rc = iree_ffi_train_step_generic(
      sess, fn_name, (int)batch,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, (float)lr,
      p_out, &loss_f);

  free(p_f); free(x_f); free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (rc != 0) {
    free(p_out);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("generic train_step failed")));
  }

  int64_t n_out = total_params + 1;
  lean_object* result = lean_alloc_sarray(sizeof(double), n_out, n_out);
  double* rp = lean_float_array_cptr(result);
  for (int64_t i = 0; i < total_params; i++) rp[i] = (double)p_out[i];
  rp[total_params] = (double)loss_f;
  free(p_out);
  return lean_io_result_mk_ok(result);
}

// ---- Zero-copy f32 generic train step ----
// All tensor data is ByteArray (raw float32 bytes). No f64 conversion.
LEAN_EXPORT lean_obj_res lean_iree_train_step_f32(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_ba,
    double lr,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse shapes (same logic as f64 version)
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  // ZERO COPY: params and x already f32 in ByteArray
  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);

  // Allocate output ByteArray directly: (total_params + 1) * 4 bytes
  size_t n_out_bytes = (total_params + 1) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;

  int rc = iree_ffi_train_step_generic(
      sess, fn_name, (int)batch,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, (float)lr,
      rp, &loss_f);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("f32 train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Adam train step (f32, with step counter t + BN stats output) ----
// bnShapes: packed int32 array [n_bn_layers, oc0, oc1, ...] — each oc appears twice (mean + var)
// Returns: ByteArray of (total_params + 1 + total_bn_stats) floats
LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse param shapes
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  // Parse BN shapes: [n_bn_layers, oc0, oc1, ...]
  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;      // mean size
      bn_sizes[i * 2 + 1] = oc;  // var size
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const int32_t* y_ptr = (const int32_t*)lean_sarray_cptr(y_ba);

  // Output: params + loss + bn_stats
  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam(
      sess, fn_name, (int)batch,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_ptr, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Soft-label train step: y_soft is [batch, n_classes] f32 ----
extern int iree_ffi_train_step_adam_softlabel(
    iree_ffi_session_t* sess, const char* fn_name, int batch, int n_classes,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const float* y_soft, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

LEAN_EXPORT lean_obj_res lean_iree_train_step_adam_f32_softlabel(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    b_lean_obj_arg y_soft_ba,
    double lr, double t,
    b_lean_obj_arg bn_shapes_ba,
    size_t batch, size_t n_classes, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int32_t* param_ranks = (int32_t*)malloc(n_params * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4;
  int64_t* param_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  int64_t* param_sizes = (int64_t*)malloc(n_params * sizeof(int64_t));
  int sp_idx = 1, dims_idx = 0;
  int64_t total_params = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    param_ranks[i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      param_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= param_dims_flat[dims_idx];
      dims_idx++;
    }
    param_sizes[i] = sz;
    total_params += sz;
  }

  const int32_t* bnsp = (const int32_t*)lean_sarray_cptr(bn_shapes_ba);
  int n_bn_layers = bnsp[0];
  int64_t total_bn_stats = 0;
  int64_t* bn_sizes = NULL;
  if (n_bn_layers > 0) {
    bn_sizes = (int64_t*)malloc(n_bn_layers * 2 * sizeof(int64_t));
    for (int i = 0; i < n_bn_layers; i++) {
      int64_t oc = (int64_t)bnsp[1 + i];
      bn_sizes[i * 2] = oc;
      bn_sizes[i * 2 + 1] = oc;
      total_bn_stats += oc * 2;
    }
  }

  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  int64_t x_dims[8];
  for (int i = 0; i < x_rank; i++) x_dims[i] = (int64_t)xsp[1+i];
  const float* x_f = (const float*)lean_sarray_cptr(x_ba);
  const float* y_soft = (const float*)lean_sarray_cptr(y_soft_ba);

  size_t n_out_bytes = (total_params + 1 + total_bn_stats) * 4;
  lean_object* result = lean_alloc_sarray(1, n_out_bytes, n_out_bytes);
  float* rp = (float*)lean_sarray_cptr(result);
  float loss_f = 0.0f;
  float* bn_out = (total_bn_stats > 0) ? rp + total_params + 1 : NULL;

  int rc = iree_ffi_train_step_adam_softlabel(
      sess, fn_name, (int)batch, (int)n_classes,
      n_params, param_ranks, param_dims_flat, param_sizes,
      p_f, x_rank, x_dims, x_f, y_soft, (float)lr, (float)t,
      rp, &loss_f,
      n_bn_layers, bn_sizes, bn_out);

  free(param_ranks); free(param_dims_flat); free(param_sizes);
  if (bn_sizes) free(bn_sizes);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("adam f32 softlabel train_step failed")));
  }
  rp[total_params] = loss_f;
  return lean_io_result_mk_ok(result);
}

// ---- Zero-copy f32 generic forward pass ----
// Pushes x first, then param tensors. Returns logits as ByteArray.
// Forward signature: forward(x, W0, g0, bt0, W1, ...) -> logits
LEAN_EXPORT lean_obj_res lean_iree_forward_f32(
    b_lean_obj_arg sess_obj,
    b_lean_obj_arg fn_name_obj,
    b_lean_obj_arg params_ba,
    b_lean_obj_arg shapes_ba,
    b_lean_obj_arg x_ba,
    b_lean_obj_arg x_shape_ba,
    size_t batch, size_t n_classes, lean_obj_arg world) {
  (void)world;
  iree_ffi_session_t* sess =
      (iree_ffi_session_t*)lean_get_external_data(sess_obj);
  const char* fn_name = lean_string_cstr(fn_name_obj);

  // Parse param shapes
  const int32_t* sp = (const int32_t*)lean_sarray_cptr(shapes_ba);
  int n_params = sp[0];
  int sp_idx = 1;
  int64_t total_params = 0;

  // Count total inputs: 1 (x) + n_params
  int n_inputs = 1 + n_params;
  int32_t* input_ranks = (int32_t*)malloc(n_inputs * sizeof(int32_t));
  size_t max_dims = lean_sarray_size(shapes_ba) / 4 + 8;
  int64_t* input_dims_flat = (int64_t*)malloc(max_dims * sizeof(int64_t));
  const float** input_data = (const float**)malloc(n_inputs * sizeof(float*));

  // First input: x
  const int32_t* xsp = (const int32_t*)lean_sarray_cptr(x_shape_ba);
  int x_rank = xsp[0];
  input_ranks[0] = x_rank;
  int dims_idx = 0;
  for (int i = 0; i < x_rank; i++)
    input_dims_flat[dims_idx++] = (int64_t)xsp[1+i];
  input_data[0] = (const float*)lean_sarray_cptr(x_ba);

  // Remaining inputs: param tensors
  const float* p_f = (const float*)lean_sarray_cptr(params_ba);
  int64_t data_off = 0;
  for (int i = 0; i < n_params; i++) {
    int rank = sp[sp_idx++];
    input_ranks[1+i] = rank;
    int64_t sz = 1;
    for (int d = 0; d < rank; d++) {
      input_dims_flat[dims_idx] = (int64_t)sp[sp_idx++];
      sz *= input_dims_flat[dims_idx];
      dims_idx++;
    }
    input_data[1+i] = p_f + data_off;
    data_off += sz;
    total_params += sz;
  }

  // Output: logits (batch x n_classes)
  int64_t logits_total = (int64_t)(batch * n_classes);
  size_t out_bytes = logits_total * 4;
  lean_object* result = lean_alloc_sarray(1, out_bytes, out_bytes);
  float* logits = (float*)lean_sarray_cptr(result);

  int64_t out_totals[1] = {logits_total};
  float* outputs[1] = {logits};

  int rc = iree_ffi_invoke_f32(sess, fn_name,
      n_inputs, input_ranks, input_dims_flat, input_data,
      1, out_totals, outputs);

  free(input_ranks); free(input_dims_flat); free(input_data);
  if (rc != 0) {
    lean_dec_ref(result);
    return lean_io_result_mk_error(
        lean_mk_io_user_error(lean_mk_string("f32 forward failed")));
  }
  return lean_io_result_mk_ok(result);
}
