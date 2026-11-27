// The dense root block on the host, through the dense kernels of the build (LAPACK or Eigen), and
// make_dense_root_block() for builds without the device realization.
#include "detail/dense_kernels.hpp"
#include "detail/dense_root_block.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace laplace::sparse::detail {

namespace {

class host_dense_root_block final : public dense_root_block {
 public:
  explicit host_dense_root_block(const char* name) : name_(name) {}
  int factorize(int m, double* S) override { return dense_cholesky_lower(m, S, m); }
  void solve(int m, const double* S, double* x) override { dense_cholesky_solve_lower(m, 1, S, m, x, m); }
  int invert(int m, const double* S, double* Z) override {
    std::copy(S, S + static_cast<std::size_t>(m) * m, Z);
    return dense_cholesky_inverse_lower(m, Z, m);
  }
  void symmetric_product(int m, const double* Z, const double* dS, double* Q) override {
    dense_symmetric_product_right(m, Z, m, dS, m, Q, m);
  }
  const char* realization() const override { return name_; }

 private:
  const char* name_;
};

}  // namespace

std::unique_ptr<dense_root_block> make_host_dense_root_block(const char* name) {
  return std::make_unique<host_dense_root_block>(name);
}

#ifndef LAPLACE_DENSE_KERNELS_DEVICE_CUDA
std::unique_ptr<dense_root_block> make_dense_root_block(int) {
  return make_host_dense_root_block(dense_kernels_realization());
}
long dense_root_block_device_fallbacks() { return 0; }
void dense_root_block_device_memory_cap(unsigned long long) {}
bool dense_root_block_device_available() { return false; }
int dense_root_block_set_device_min_width(int) { return 0; }
#endif

}  // namespace laplace::sparse::detail
