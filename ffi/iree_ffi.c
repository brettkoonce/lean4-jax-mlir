// Thin C wrapper around IREE runtime for Lean FFI.
// Narrow API: create session from .vmfb path, invoke a function by name
// with N float32 input tensors and N float32 output tensors.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/runtime/api.h"

// GPU driver selection: compile with -DUSE_HIP for AMD/ROCm,
// -DUSE_CUDA (or neither flag on CUDA systems) for NVIDIA,
// or neither for CPU-only (local-task).
#if defined(USE_HIP)
  #include "iree/hal/drivers/hip/registration/driver_module.h"
  #define IREE_REGISTER_GPU_DRIVER iree_hal_hip_driver_module_register
  #define IREE_DEFAULT_DEVICE "hip"
#elif !defined(USE_CPU)
  #include "iree/hal/drivers/cuda/registration/driver_module.h"
  #define IREE_REGISTER_GPU_DRIVER iree_hal_cuda_driver_module_register
  #define IREE_DEFAULT_DEVICE "cuda"
#else
  // CPU-only: no GPU driver, default to IREE's multi-threaded task system.
  #define IREE_DEFAULT_DEVICE "local-task"
#endif
#include "iree/hal/drivers/local_task/registration/driver_module.h"

// ---- Opaque session handle ----
struct iree_ffi_session_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
};
typedef struct iree_ffi_session_t iree_ffi_session_t;

// ---- Error helper: print status, free it, return NULL / nonzero ----
static void print_status(const char* where, iree_status_t status) {
  fprintf(stderr, "iree_ffi: %s failed:\n", where);
  iree_status_fprint(stderr, status);
  iree_status_free(status);
}

// ---- Create a session from a .vmfb path on CUDA device ----
// Returns NULL on failure. Stderr gets the diagnostic.
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path) {
  iree_ffi_session_t* sess = calloc(1, sizeof(iree_ffi_session_t));
  if (!sess) return NULL;

  // Register drivers (idempotent; only the first session does this).
  static int driver_registered = 0;
  iree_status_t status = iree_ok_status();
  if (!driver_registered) {
#ifdef IREE_REGISTER_GPU_DRIVER
    status = IREE_REGISTER_GPU_DRIVER(
        iree_hal_driver_registry_default());
    if (!iree_status_is_ok(status)) { print_status("gpu_driver_register", status); goto fail; }
#endif
    iree_hal_local_task_driver_module_register(
        iree_hal_driver_registry_default());
    driver_registered = 1;
  }

  // Instance with default options.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(&instance_options,
                                        iree_allocator_system(), &sess->instance);
  if (!iree_status_is_ok(status)) { print_status("instance_create", status); goto fail; }

  // Create device: check IREE_DEVICE env var, default to GPU.
  const char* device_env = getenv("IREE_DEVICE");
  const char* device_name = device_env ? device_env : IREE_DEFAULT_DEVICE;
  status = iree_runtime_instance_try_create_default_device(
      sess->instance, iree_make_cstring_view(device_name), &sess->device);
  if (!iree_status_is_ok(status)) { print_status("device_create", status); goto fail; }

  // Create session.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  status = iree_runtime_session_create_with_device(
      sess->instance, &session_options, sess->device,
      iree_runtime_instance_host_allocator(sess->instance), &sess->session);
  if (!iree_status_is_ok(status)) { print_status("session_create", status); goto fail; }

  // Load the bytecode module from file.
  status = iree_runtime_session_append_bytecode_module_from_file(
      sess->session, vmfb_path);
  if (!iree_status_is_ok(status)) { print_status("append_module", status); goto fail; }

  return sess;

fail:
  if (sess->session) iree_runtime_session_release(sess->session);
  if (sess->device) iree_hal_device_release(sess->device);
  if (sess->instance) iree_runtime_instance_release(sess->instance);
  free(sess);
  return NULL;
}

void iree_ffi_session_release(iree_ffi_session_t* sess) {
  if (!sess) return;
  if (sess->session) iree_runtime_session_release(sess->session);
  if (sess->device) iree_hal_device_release(sess->device);
  if (sess->instance) iree_runtime_instance_release(sess->instance);
  free(sess);
}

// Helper: push a float32 tensor input from host memory.
// `dims` is a length-`rank` array of dimension sizes. `data` points to
// rank-major contiguous float32 values (total: product(dims) elements).
static iree_status_t push_input_f32(
    iree_runtime_call_t* call,
    iree_hal_device_t* device,
    int rank, const int64_t* dims, const float* data) {
  iree_hal_dim_t shape[8];
  if (rank > 8) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "rank > 8");
  iree_host_size_t total = 1;
  for (int i = 0; i < rank; i++) { shape[i] = (iree_hal_dim_t)dims[i]; total *= dims[i]; }

  iree_hal_buffer_view_t* bv = NULL;
  iree_hal_buffer_params_t params = {
    .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device),
      rank, shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params,
      iree_make_const_byte_span(data, total * sizeof(float)),
      &bv));
  iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(call, bv);
  iree_hal_buffer_view_release(bv);
  return status;
}

// Helper: pop a float32 output and copy to host memory.
static iree_status_t pop_output_f32(
    iree_runtime_call_t* call,
    iree_hal_device_t* device,
    int64_t expected_total, float* out_data) {
  iree_hal_buffer_view_t* bv = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_call_outputs_pop_front_buffer_view(call, &bv));
  iree_status_t status = iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(bv), 0, out_data,
      expected_total * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  iree_hal_buffer_view_release(bv);
  return status;
}

// ---- Invoke function by name with N inputs and M outputs. ----
// All tensors are float32 for now. inputs/outputs are flat host arrays.
// Each input i is described by input_ranks[i] and input_dims (packed flat:
// dims for input 0, then dims for input 1, etc).
// Returns 0 on success, nonzero otherwise.
int iree_ffi_invoke_f32(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,  // total length = sum(input_ranks)
    const float* const* input_data,  // array of pointers
    int n_outputs,
    const int64_t* output_totals,    // element count per output
    float* const* output_data) {     // array of pointers

  iree_runtime_call_t call;
  iree_status_t status = iree_runtime_call_initialize_by_name(
      sess->session, iree_make_cstring_view(fn_name), &call);
  if (!iree_status_is_ok(status)) { print_status("call_init", status); return 1; }

  // Push all inputs.
  int dims_offset = 0;
  for (int i = 0; i < n_inputs && iree_status_is_ok(status); i++) {
    status = push_input_f32(&call, sess->device,
                            input_ranks[i], &input_dims_flat[dims_offset], input_data[i]);
    dims_offset += input_ranks[i];
  }
  if (!iree_status_is_ok(status)) { print_status("push_input", status);
    iree_runtime_call_deinitialize(&call); return 2; }

  // Invoke.
  status = iree_runtime_call_invoke(&call, 0);
  if (!iree_status_is_ok(status)) { print_status("invoke", status);
    iree_runtime_call_deinitialize(&call); return 3; }

  // Pop outputs.
  for (int i = 0; i < n_outputs && iree_status_is_ok(status); i++) {
    status = pop_output_f32(&call, sess->device, output_totals[i], output_data[i]);
  }
  if (!iree_status_is_ok(status)) { print_status("pop_output", status);
    iree_runtime_call_deinitialize(&call); return 4; }

  iree_runtime_call_deinitialize(&call);
  return 0;
}

// Generic push_input supporting arbitrary element types and ranks (including 0).
static iree_status_t push_input(
    iree_runtime_call_t* call, iree_hal_device_t* device,
    iree_hal_element_type_t dtype, size_t elem_size,
    int rank, const int64_t* dims, const void* data) {
  iree_hal_dim_t shape[8];
  iree_host_size_t total = 1;
  for (int i = 0; i < rank; i++) { shape[i] = (iree_hal_dim_t)dims[i]; total *= dims[i]; }

  iree_hal_buffer_view_t* bv = NULL;
  iree_hal_buffer_params_t params = {
    .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), rank, shape,
      dtype, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, total * elem_size), &bv));
  iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(call, bv);
  iree_hal_buffer_view_release(bv);
  return status;
}

static iree_status_t pop_output(
    iree_runtime_call_t* call, iree_hal_device_t* device,
    int64_t expected_total, size_t elem_size, void* out_data) {
  iree_hal_buffer_view_t* bv = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_call_outputs_pop_front_buffer_view(call, &bv));
  iree_status_t status = iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(bv), 0, out_data,
      expected_total * elem_size,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  iree_hal_buffer_view_release(bv);
  return status;
}

int iree_ffi_train_step_mlp(
    iree_ffi_session_t* sess, const char* fn_name, int batch,
    const float* W0, const float* b0, const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* x, const int32_t* y, float lr,
    float* W0_new, float* b0_new, float* W1_new, float* b1_new,
    float* W2_new, float* b2_new, float* loss_out) {

  iree_runtime_call_t call;
  iree_status_t s = iree_runtime_call_initialize_by_name(
      sess->session, iree_make_cstring_view(fn_name), &call);
  if (!iree_status_is_ok(s)) { print_status("train_call_init", s); return 1; }

  // Inputs in JAX export order: W0, b0, W1, b1, W2, b2, x, y, lr
  int64_t d_W0[2] = {784, 512}, d_W1[2] = {512, 512}, d_W2[2] = {512, 10};
  int64_t d_b[1]  = {512}, d_b2[1] = {10};
  int64_t d_x[2]  = {batch, 784}, d_y[1] = {batch};
  #define PUSH(dtype, esz, rank, dims, data) \
    if (iree_status_is_ok(s)) s = push_input(&call, sess->device, dtype, esz, rank, dims, data)

  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 2, d_W0, W0);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 1, d_b,  b0);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 2, d_W1, W1);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 1, d_b,  b1);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 2, d_W2, W2);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 1, d_b2, b2);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 2, d_x,  x);
  PUSH(IREE_HAL_ELEMENT_TYPE_INT_32,   4, 1, d_y,  y);
  PUSH(IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4, 0, NULL, &lr);  // rank-0 scalar
  #undef PUSH
  if (!iree_status_is_ok(s)) { print_status("train_push", s);
    iree_runtime_call_deinitialize(&call); return 2; }

  s = iree_runtime_call_invoke(&call, 0);
  if (!iree_status_is_ok(s)) { print_status("train_invoke", s);
    iree_runtime_call_deinitialize(&call); return 3; }

  // Outputs in order: W0', b0', W1', b1', W2', b2', loss
  #define POP(total, out) if (iree_status_is_ok(s)) \
    s = pop_output(&call, sess->device, total, 4, out)
  POP(784*512, W0_new); POP(512, b0_new);
  POP(512*512, W1_new); POP(512, b1_new);
  POP(512*10,  W2_new); POP(10,  b2_new);
  POP(1, loss_out);
  #undef POP
  iree_runtime_call_deinitialize(&call);
  if (!iree_status_is_ok(s)) { print_status("train_pop", s); return 4; }
  return 0;
}

int iree_ffi_train_step_generic(
    iree_ffi_session_t* sess, const char* fn_name, int batch,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const int32_t* y, float lr,
    float* packed_params_out, float* loss_out) {

  iree_runtime_call_t call;
  iree_status_t s = iree_runtime_call_initialize_by_name(
      sess->session, iree_make_cstring_view(fn_name), &call);
  if (!iree_status_is_ok(s)) { print_status("gen_train_init", s); return 1; }

  // Push param tensors
  int dims_off = 0;
  int64_t data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   param_ranks[i], &param_dims_flat[dims_off],
                   packed_params + data_off);
    dims_off += param_ranks[i];
    data_off += param_sizes[i];
  }
  // x (f32)
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   x_rank, x_dims, x);
  // y (int32)
  int64_t d_y[1] = {batch};
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_INT_32, 4,
                   1, d_y, y);
  // lr (scalar f32)
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   0, NULL, &lr);

  if (!iree_status_is_ok(s)) { print_status("gen_train_push", s);
    iree_runtime_call_deinitialize(&call); return 2; }

  s = iree_runtime_call_invoke(&call, 0);
  if (!iree_status_is_ok(s)) { print_status("gen_train_invoke", s);
    iree_runtime_call_deinitialize(&call); return 3; }

  // Pop output params + loss
  data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = pop_output(&call, sess->device, param_sizes[i], 4,
                   packed_params_out + data_off);
    data_off += param_sizes[i];
  }
  if (iree_status_is_ok(s))
    s = pop_output(&call, sess->device, 1, 4, loss_out);

  iree_runtime_call_deinitialize(&call);
  if (!iree_status_is_ok(s)) { print_status("gen_train_pop", s); return 4; }
  return 0;
}

// Adam train step: like generic but also pushes step counter t after lr.
// Also pops n_bn_layers*2 extra BN stat outputs (mean, var pairs) into bn_stats_out.
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
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out) {

  iree_runtime_call_t call;
  iree_status_t s = iree_runtime_call_initialize_by_name(
      sess->session, iree_make_cstring_view(fn_name), &call);
  if (!iree_status_is_ok(s)) { print_status("adam_train_init", s); return 1; }

  int dims_off = 0;
  int64_t data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   param_ranks[i], &param_dims_flat[dims_off],
                   packed_params + data_off);
    dims_off += param_ranks[i];
    data_off += param_sizes[i];
  }
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   x_rank, x_dims, x);
  int64_t d_y[1] = {batch};
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_INT_32, 4,
                   1, d_y, y);
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   0, NULL, &lr);
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   0, NULL, &t);

  if (!iree_status_is_ok(s)) { print_status("adam_train_push", s);
    iree_runtime_call_deinitialize(&call); return 2; }

  s = iree_runtime_call_invoke(&call, 0);
  if (!iree_status_is_ok(s)) { print_status("adam_train_invoke", s);
    iree_runtime_call_deinitialize(&call); return 3; }

  data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = pop_output(&call, sess->device, param_sizes[i], 4,
                   packed_params_out + data_off);
    data_off += param_sizes[i];
  }
  if (iree_status_is_ok(s))
    s = pop_output(&call, sess->device, 1, 4, loss_out);

  // Pop BN stats: n_bn_layers pairs of (mean, var) tensors
  if (bn_stats_out && n_bn_layers > 0) {
    int64_t bn_off = 0;
    for (int i = 0; i < n_bn_layers * 2 && iree_status_is_ok(s); i++) {
      s = pop_output(&call, sess->device, bn_sizes[i], 4,
                     bn_stats_out + bn_off);
      bn_off += bn_sizes[i];
    }
  }

  iree_runtime_call_deinitialize(&call);
  if (!iree_status_is_ok(s)) { print_status("adam_train_pop", s); return 4; }
  return 0;
}

// ============================================================
// Soft-label variant. Same protocol as iree_ffi_train_step_adam,
// but `y_soft` is a `[batch, n_classes]` f32 tensor instead of an
// int32 `[batch]` vector. Used by mixup/cutmix codegen path where
// labels are smoothed + linearly mixed in the dataloader.
// ============================================================
int iree_ffi_train_step_adam_softlabel(
    iree_ffi_session_t* sess, const char* fn_name, int batch, int n_classes,
    int n_params,
    const int32_t* param_ranks,
    const int64_t* param_dims_flat,
    const int64_t* param_sizes,
    const float* packed_params,
    int x_rank, const int64_t* x_dims, const float* x,
    const float* y_soft, float lr, float t,
    float* packed_params_out, float* loss_out,
    int n_bn_layers, const int64_t* bn_sizes, float* bn_stats_out) {

  iree_runtime_call_t call;
  iree_status_t s = iree_runtime_call_initialize_by_name(
      sess->session, iree_make_cstring_view(fn_name), &call);
  if (!iree_status_is_ok(s)) { print_status("adam_train_softlabel_init", s); return 1; }

  int dims_off = 0;
  int64_t data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   param_ranks[i], &param_dims_flat[dims_off],
                   packed_params + data_off);
    dims_off += param_ranks[i];
    data_off += param_sizes[i];
  }
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   x_rank, x_dims, x);
  int64_t d_y[2] = {batch, n_classes};
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   2, d_y, y_soft);
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   0, NULL, &lr);
  if (iree_status_is_ok(s))
    s = push_input(&call, sess->device,
                   IREE_HAL_ELEMENT_TYPE_FLOAT_32, 4,
                   0, NULL, &t);

  if (!iree_status_is_ok(s)) { print_status("adam_train_softlabel_push", s);
    iree_runtime_call_deinitialize(&call); return 2; }

  s = iree_runtime_call_invoke(&call, 0);
  if (!iree_status_is_ok(s)) { print_status("adam_train_softlabel_invoke", s);
    iree_runtime_call_deinitialize(&call); return 3; }

  data_off = 0;
  for (int i = 0; i < n_params && iree_status_is_ok(s); i++) {
    s = pop_output(&call, sess->device, param_sizes[i], 4,
                   packed_params_out + data_off);
    data_off += param_sizes[i];
  }
  if (iree_status_is_ok(s))
    s = pop_output(&call, sess->device, 1, 4, loss_out);
  if (bn_stats_out && n_bn_layers > 0) {
    int64_t bn_off = 0;
    for (int i = 0; i < n_bn_layers * 2 && iree_status_is_ok(s); i++) {
      s = pop_output(&call, sess->device, bn_sizes[i], 4,
                     bn_stats_out + bn_off);
      bn_off += bn_sizes[i];
    }
  }

  iree_runtime_call_deinitialize(&call);
  if (!iree_status_is_ok(s)) { print_status("adam_train_softlabel_pop", s); return 4; }
  return 0;
}
