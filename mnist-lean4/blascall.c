/*
 * blascall.c — thin Lean 4 → cblas_dgemm bridge.
 *
 * FloatArray in Lean 4 is a scalar array (lean_sarray) of doubles.
 * lean_float_array_cptr() returns a double* into the contiguous buffer,
 * so we can pass it directly to CBLAS with zero copying.
 *
 * Ownership: A and B are borrowed (@&), C is consumed and returned
 * (Lean guarantees it has refcount 1 when passed as a non-borrowed arg,
 * so cblas_dgemm writes in-place safely).
 */

#include <lean/lean.h>
#include <cblas.h>
#include <stdint.h>

/* Tell OpenBLAS to use a single thread per call.
 * We parallelise at the Lean task level; BLAS internal threading
 * would cause ~N_workers^2 threads competing for N_workers cores. */
static void __attribute__((constructor)) disable_blas_threads(void) {
    openblas_set_num_threads(1);
}

LEAN_EXPORT lean_obj_res lean_cblas_dgemm(
        uint32_t order, uint32_t transA, uint32_t transB,
        uint64_t M, uint64_t N, uint64_t K, double alpha,
        b_lean_obj_arg A, uint64_t lda,
        b_lean_obj_arg B, uint64_t ldb,
        double beta, lean_obj_arg C, uint64_t ldc)
{
    cblas_dgemm(
        (CBLAS_LAYOUT)order,
        (CBLAS_TRANSPOSE)transA,
        (CBLAS_TRANSPOSE)transB,
        (blasint)M, (blasint)N, (blasint)K,
        alpha,
        lean_float_array_cptr(A), (blasint)lda,
        lean_float_array_cptr(B), (blasint)ldb,
        beta,
        lean_float_array_cptr(C), (blasint)ldc);
    return C;
}
