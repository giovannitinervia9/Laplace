#pragma once

// The dense root block of the supernodal route of laplace::sparse::sparse_ldlt (internal). The supernodal
// factor assembles the m x m Schur complement S, then its derived quantities, on the host; this object
// factorizes S = R R', solves with R, forms Z = S^-1 and the product Q = dS Z, either on the host (the
// dense kernels of dense_kernels.hpp) or on the device (cuSOLVER and cuBLAS, cuda_dense_kernels.cpp),
// where R and Z stay resident between calls. Every matrix argument is a host array, m x m column-major.

#include <memory>

namespace laplace::sparse::detail {

class dense_root_block {
 public:
  virtual ~dense_root_block() = default;
  // S = R R' from the lower triangle of S. Returns 0, or k > 0 when the leading minor of order k is not
  // positive definite. Afterwards the diagonal of S holds R_kk (the host block leaves all of R in the
  // lower triangle of S, the device block only its diagonal).
  virtual int factorize(int m, double* S) = 0;
  // x <- S^-1 x for one right-hand side of length m, from the last factorize(m, S).
  virtual void solve(int m, const double* S, double* x) = 0;
  // Z <- S^-1, lower triangle, from the last factorize(m, S). Returns 0, or k > 0 when R_kk is zero.
  virtual int invert(int m, const double* S, double* Z) = 0;
  // Q <- dS Z, with dS given whole (both triangles) and Z from the last invert().
  virtual void symmetric_product(int m, const double* Z, const double* dS, double* Q) = 0;
  // "lapack", "eigen", "cuda", or "<host>, device fallback" when a device block did not fit
  virtual const char* realization() const = 0;
};

// The block for an m x m root: on the device when the build has the device realization, m is at least
// the device threshold (dense_root_block_device_min_width) and the block fits in the free device
// memory; on the host otherwise.
std::unique_ptr<dense_root_block> make_dense_root_block(int m);

// The number of root blocks that the device realization sent to the host because they did not fit (0 in
// a build without it), for tests and benchmarks.
long dense_root_block_device_fallbacks();

// The host block alone, named for reports (host_dense_root_block.cpp).
std::unique_ptr<dense_root_block> make_host_dense_root_block(const char* name);

// A cap on the free device memory that make_dense_root_block() believes it has, in bytes (0: none). For the
// tests of the fallback; no effect in a build without the device realization.
void dense_root_block_device_memory_cap(unsigned long long bytes);

// Whether the build has the device realization.
bool dense_root_block_device_available();

// Sets the width from which a root block goes to the device (the default is the measured threshold in
// cuda_dense_kernels.cpp) and returns the previous one; for the tests, which force the device on small
// blocks. No effect, and 0 returned, in a build without the device realization.
int dense_root_block_set_device_min_width(int width);

}  // namespace laplace::sparse::detail
