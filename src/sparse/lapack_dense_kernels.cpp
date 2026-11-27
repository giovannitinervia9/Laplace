// The dense kernels of the supernodal route through the environment's LAPACK and BLAS, by their standard
// Fortran symbols (reference LAPACK: modified BSD; OpenBLAS: BSD-3). The symbols are declared here, so no
// LAPACK header is needed; the trailing size_t arguments are the hidden lengths of the character
// arguments that gfortran passes, which C implementations ignore. The integers are 32-bit (the LP64
// interface): an ILP64 BLAS or LAPACK, whose integers are 64-bit, must not be linked.
#include "detail/dense_kernels.hpp"

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef LAPLACE_FORTRAN_NAME
#define LAPLACE_FORTRAN_NAME(lower, upper) lower##_
#endif

extern "C" {
void LAPLACE_FORTRAN_NAME(dpotrf, DPOTRF)(const char* uplo, const int* n, double* a, const int* lda, int* info,
                                          std::size_t uplo_length);
void LAPLACE_FORTRAN_NAME(dpotri, DPOTRI)(const char* uplo, const int* n, double* a, const int* lda, int* info,
                                          std::size_t uplo_length);
void LAPLACE_FORTRAN_NAME(dpotrs, DPOTRS)(const char* uplo, const int* n, const int* nrhs, const double* a,
                                          const int* lda, double* b, const int* ldb, int* info,
                                          std::size_t uplo_length);
void LAPLACE_FORTRAN_NAME(dsymm, DSYMM)(const char* side, const char* uplo, const int* m, const int* n,
                                        const double* alpha, const double* a, const int* lda, const double* b,
                                        const int* ldb, const double* beta, double* c, const int* ldc,
                                        std::size_t side_length, std::size_t uplo_length);
}

namespace laplace::sparse::detail {

namespace {

// The threads a call may use, from the size of the block: an OpenMP-built BLAS (Rtools' OpenBLAS) takes
// the calling thread's omp_get_max_threads(), and on small blocks more threads cost more than they give.
// Measured with OpenBLAS on 12 threads (note/engine/step2b_factorization.md, sections 3 and 11): dpotri
// at m = 960 took 0.011 s on one thread and 0.046 s on twelve; the whole factorization of a crossed
// model dpotrf at m = 580 took 2.4 ms on one thread and 5.6 ms on twelve.
// Assumes an OpenMP-built BLAS; a pthreads BLAS or MKL with its own threading ignores it. Every call
// site (factorize, compute_selected_inverse and the derivative of supernodal_factor) calls the kernels
// outside any OpenMP region of ours and outside any Kokkos parallel region, where omp_set_num_threads
// changes only the calling thread's setting for the next region.
// ponytail: fixed sizes per kernel; a measured table if other BLAS libraries need it
class block_threads {
 public:
  explicit block_threads(int m, int smallest_threaded) {
#ifdef _OPENMP
    saved_ = omp_get_max_threads();
    if (m < smallest_threaded) omp_set_num_threads(1);
#else
    (void)m, (void)smallest_threaded;
#endif
  }
  ~block_threads() {
#ifdef _OPENMP
    omp_set_num_threads(saved_);
#endif
  }
  block_threads(const block_threads&) = delete;
  block_threads& operator=(const block_threads&) = delete;

 private:
  int saved_ = 1;
};

}  // namespace

int dense_cholesky_lower(int m, double* a, int lda) {
  int info = 0;
  const block_threads threads(m, 1024);
  if (m > 0) LAPLACE_FORTRAN_NAME(dpotrf, DPOTRF)("L", &m, a, &lda, &info, 1);
  return info;
}

int dense_cholesky_inverse_lower(int m, double* a, int lda) {
  int info = 0;
  const block_threads threads(m, 2048);
  if (m > 0) LAPLACE_FORTRAN_NAME(dpotri, DPOTRI)("L", &m, a, &lda, &info, 1);
  return info;
}

void dense_cholesky_solve_lower(int m, int nrhs, const double* a, int lda, double* b, int ldb) {
  int info = 0;
  const block_threads threads(m, 2048);
  if (m > 0 && nrhs > 0) LAPLACE_FORTRAN_NAME(dpotrs, DPOTRS)("L", &m, &nrhs, a, &lda, b, &ldb, &info, 1);
}

void dense_symmetric_product_right(int m, const double* z, int ldz, const double* b, int ldb, double* c, int ldc) {
  const double one = 1.0, zero = 0.0;
  const block_threads threads(m, 128);
  if (m > 0) LAPLACE_FORTRAN_NAME(dsymm, DSYMM)("R", "L", &m, &m, &one, z, &ldz, b, &ldb, &zero, c, &ldc, 1, 1);
}

const char* dense_kernels_realization() { return "lapack"; }

}  // namespace laplace::sparse::detail
