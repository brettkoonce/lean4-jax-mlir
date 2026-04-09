// Public C API for the IREE FFI wrapper.
// See iree_ffi.c for implementation.
#ifndef IREE_FFI_H_
#define IREE_FFI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_ffi_session_t iree_ffi_session_t;

// Create a session from a .vmfb path. Returns NULL on failure.
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path);

// Release a session (safe with NULL).
void iree_ffi_session_release(iree_ffi_session_t* sess);

// Invoke a function by name. All tensors float32.
// `input_ranks[i]` gives the rank of input i; `input_dims_flat` has the
// concatenated dimensions (sum of all ranks total entries).
// `output_totals[i]` is the expected element count for output i.
// Returns 0 on success, nonzero on failure (diagnostic printed to stderr).
int iree_ffi_invoke_f32(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,
    const float* const* input_data,
    int n_outputs,
    const int64_t* output_totals,
    float* const* output_data);

// MLP-specific train step.
// Expected .vmfb function: jit_train_step.main
// Inputs:  W0(784x512), b0(512), W1(512x512), b1(512), W2(512x10), b2(10),
//          x(batchx784), y(batch int32), lr(scalar f32)
// Outputs: W0', b0', W1', b1', W2', b2' (same shapes), loss (scalar f32)
// Returns 0 on success.
int iree_ffi_train_step_mlp(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int batch,
    const float* W0, const float* b0,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* x, const int32_t* y, float lr,
    float* W0_new, float* b0_new,
    float* W1_new, float* b1_new,
    float* W2_new, float* b2_new,
    float* loss_out);

// Generic train step for any architecture.
int iree_ffi_train_step_generic(
    iree_ffi_session_t* sess, const char* fn_name, int batch,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr,
    float* packed_params_out, float* loss_out);

// Adam train step: pushes step counter t, pops BN stats after loss.
int iree_ffi_train_step_adam(
    iree_ffi_session_t* sess, const char* fn_name, int batch,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out);

#ifdef __cplusplus
}
#endif

#endif  // IREE_FFI_H_
