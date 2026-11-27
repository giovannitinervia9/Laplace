// The dense root block on the device, through cuSOLVER and cuBLAS (NVIDIA CUDA EULA; linked dynamically,
// nothing of them in the repository). Host C++: compiled by the host compiler, never by nvcc, no Eigen.
// Compiled only when LAPLACE_DENSE_KERNELS_DEVICE is cuda (CMakeLists.txt).
//
// R (the Cholesky factor) and Z = S^-1 stay on the device between calls; S and dS go up, Z and Q come
// down, R's diagonal comes down into the diagonal of the caller's S. The calls run on the default stream
// of the current device (the primary context, shared with Kokkos's CUDA backend), from host code outside
// any parallel region; every copy back is a cudaMemcpy, synchronous with the host.
//
// API (CUDA 13.1 headers): the 64-bit cusolverDnXpotrf and cusolverDnXpotrs; the inverse from the factor
// by the 32-bit cusolverDnDpotri, the 64-bit API having no potri (its int arguments allow m <= 46340,
// beyond the memory check below); the product by cublasDsymm_64. Workspaces are allocated here, as
// cuSOLVER allocates none. A positive info of potrf is a leading minor that is not positive definite and
// is returned as on the host; any other failure throws std::runtime_error naming the routine.
#include "detail/dense_kernels.hpp"
#include "detail/dense_root_block.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <atomic>
#include <new>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace laplace::sparse::detail {

namespace {

// The width from which a root block goes to the device. Measured on 2026-11-27 (RTX 4070 SUPER, 12 host
// threads; prototipi/step2b/bench_w0.cpp, scratchpad time_cuda_dev.txt): the device block against the
// host block of the same MSVC build (Eigen kernels), factorization plus selected inverse plus one
// derivative direction, crossed models. Each device call costs about 0.12 ms however small the block;
// the two cost the same at a width of about 150 (3.1 against 2.8 ms) and the device is faster from
// about 290 (4.4 against 6.0 ms; 31 against 160 ms at 960). Chosen by Giovanni ("Sì, da 256 colonne").
constexpr int device_root_min_width = 256;

std::atomic<long> device_fallbacks{0};
std::atomic<int> device_min_width{device_root_min_width};
std::atomic<unsigned long long> device_memory_cap{0};

void check_cuda(cudaError_t e, const char* what) {
  if (e == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();  // not sticky; clear it
    throw std::bad_alloc();
  }
  if (e != cudaSuccess) throw std::runtime_error(std::string("sparse_ldlt device block: ") + what + ": " + cudaGetErrorString(e));
}
void check_solver(cusolverStatus_t s, const char* what) {
  if (s != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string("sparse_ldlt device block: ") + what + " failed, cusolverStatus " + std::to_string(static_cast<int>(s)));
}
void check_blas(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("sparse_ldlt device block: ") + what + " failed, cublasStatus " + std::to_string(static_cast<int>(s)));
}

// device bytes held by a block of m columns: R, Z, dS, Q, plus room for the workspaces of cuSOLVER and
// cuBLAS (measured below on first use; 256 MB is reserved for them in the check)
unsigned long long device_bytes(int m) {
  return 4ULL * static_cast<unsigned long long>(m) * static_cast<unsigned long long>(m) * sizeof(double) + (256ULL << 20);
}

class device_dense_root_block final : public dense_root_block {
 public:
  // Every device allocation happens here, the workspaces included (sized by the buffer queries), so that
  // a block that does not fit fails at construction (std::bad_alloc, caught by make_dense_root_block,
  // which then takes the host block) and never in the middle of a factorization.
  explicit device_dense_root_block(int m) {
    try {
      check_solver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      check_solver(cusolverDnCreateParams(&params_), "cusolverDnCreateParams");
      check_blas(cublasCreate(&blas_), "cublasCreate");
      const std::size_t bytes = static_cast<std::size_t>(m) * m * sizeof(double);
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&R_), bytes), "cudaMalloc R");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&Z_), bytes), "cudaMalloc Z");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&dS_), bytes), "cudaMalloc dS");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&Q_), bytes), "cudaMalloc Q");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&x_), static_cast<std::size_t>(m) * sizeof(double)), "cudaMalloc x");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&info_), sizeof(int)), "cudaMalloc info");
      std::size_t potrf_dev = 0;
      check_solver(cusolverDnXpotrf_bufferSize(solver_, params_, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, R_, m, CUDA_R_64F,
                                               &potrf_dev, &host_bytes_),
                   "cusolverDnXpotrf_bufferSize");
      int lwork = 0;
      check_solver(cusolverDnDpotri_bufferSize(solver_, CUBLAS_FILL_MODE_LOWER, m, Z_, m, &lwork), "cusolverDnDpotri_bufferSize");
      potri_lwork_ = lwork;
      work_bytes_ = std::max(potrf_dev, static_cast<std::size_t>(lwork) * sizeof(double));
      if (work_bytes_ > 0) check_cuda(cudaMalloc(&work_, work_bytes_), "cudaMalloc workspace");
      host_work_.resize(host_bytes_);
    } catch (...) {
      release();
      throw;
    }
  }
  ~device_dense_root_block() override { release(); }
  device_dense_root_block(const device_dense_root_block&) = delete;
  device_dense_root_block& operator=(const device_dense_root_block&) = delete;

  int factorize(int m, double* S) override {
    const std::size_t bytes = static_cast<std::size_t>(m) * m * sizeof(double);
    check_cuda(cudaMemcpy(R_, S, bytes, cudaMemcpyHostToDevice), "copy of S to the device");
    check_solver(cusolverDnXpotrf(solver_, params_, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, R_, m, CUDA_R_64F, work_,
                                  work_bytes_, host_work_.empty() ? nullptr : host_work_.data(), host_bytes_, info_),
                 "cusolverDnXpotrf");
    const int info = read_info("cusolverDnXpotrf");
    if (info != 0) return info;
    // R's diagonal into the diagonal of the caller's S (stride m + 1 on both sides)
    const std::size_t pitch = static_cast<std::size_t>(m + 1) * sizeof(double);
    check_cuda(cudaMemcpy2D(S, pitch, R_, pitch, sizeof(double), static_cast<std::size_t>(m), cudaMemcpyDeviceToHost),
               "copy of the diagonal of R");
    return 0;
  }

  void solve(int m, const double*, double* x) override {
    check_cuda(cudaMemcpy(x_, x, static_cast<std::size_t>(m) * sizeof(double), cudaMemcpyHostToDevice), "copy of x");
    check_solver(cusolverDnXpotrs(solver_, params_, CUBLAS_FILL_MODE_LOWER, m, 1, CUDA_R_64F, R_, m, CUDA_R_64F, x_, m, info_),
                 "cusolverDnXpotrs");
    if (read_info("cusolverDnXpotrs") != 0) throw std::runtime_error("sparse_ldlt device block: cusolverDnXpotrs info");
    check_cuda(cudaMemcpy(x, x_, static_cast<std::size_t>(m) * sizeof(double), cudaMemcpyDeviceToHost), "copy of x back");
  }

  int invert(int m, const double*, double* Z) override {
    const std::size_t bytes = static_cast<std::size_t>(m) * m * sizeof(double);
    check_cuda(cudaMemcpy(Z_, R_, bytes, cudaMemcpyDeviceToDevice), "copy of R");
    check_solver(cusolverDnDpotri(solver_, CUBLAS_FILL_MODE_LOWER, m, Z_, m, static_cast<double*>(work_), potri_lwork_, info_),
                 "cusolverDnDpotri");
    const int info = read_info("cusolverDnDpotri");
    if (info != 0) return info;
    check_cuda(cudaMemcpy(Z, Z_, bytes, cudaMemcpyDeviceToHost), "copy of Z");
    return 0;
  }

  void symmetric_product(int m, const double*, const double* dS, double* Q) override {
    const std::size_t bytes = static_cast<std::size_t>(m) * m * sizeof(double);
    check_cuda(cudaMemcpy(dS_, dS, bytes, cudaMemcpyHostToDevice), "copy of dS");
    const double one = 1.0, zero = 0.0;
    // Q = dS Z: side right, Z symmetric from its lower triangle (resident since invert())
    check_blas(cublasDsymm_64(blas_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, m, &one, Z_, m, dS_, m, &zero, Q_, m),
               "cublasDsymm_64");
    check_cuda(cudaMemcpy(Q, Q_, bytes, cudaMemcpyDeviceToHost), "copy of Q");
  }

  const char* realization() const override { return "cuda"; }

 private:
  void release() {
    cudaFree(R_), cudaFree(Z_), cudaFree(dS_), cudaFree(Q_), cudaFree(x_), cudaFree(info_), cudaFree(work_);
    R_ = Z_ = dS_ = Q_ = x_ = nullptr, info_ = nullptr, work_ = nullptr;
    if (blas_) cublasDestroy(blas_), blas_ = nullptr;
    if (params_) cusolverDnDestroyParams(params_), params_ = nullptr;
    if (solver_) cusolverDnDestroy(solver_), solver_ = nullptr;
  }
  int read_info(const char* what) {
    int info = 0;
    check_cuda(cudaMemcpy(&info, info_, sizeof(int), cudaMemcpyDeviceToHost), what);
    if (info < 0) throw std::runtime_error(std::string("sparse_ldlt device block: ") + what + " reports an invalid argument");
    return info;
  }

  cusolverDnHandle_t solver_ = nullptr;
  cusolverDnParams_t params_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  double *R_ = nullptr, *Z_ = nullptr, *dS_ = nullptr, *Q_ = nullptr, *x_ = nullptr;
  int* info_ = nullptr;
  void* work_ = nullptr;
  std::size_t work_bytes_ = 0, host_bytes_ = 0;
  int potri_lwork_ = 0;
  std::vector<char> host_work_;
};

}  // namespace

std::unique_ptr<dense_root_block> make_dense_root_block(int m) {
  if (m <= 0 || m < device_min_width.load()) return make_host_dense_root_block(dense_kernels_realization());
  std::size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  unsigned long long available = free_bytes;
  const unsigned long long cap = device_memory_cap.load();
  if (cap > 0 && cap < available) available = cap;
  if (device_bytes(m) > available) {
    ++device_fallbacks;
    return make_host_dense_root_block("host fallback of the cuda block");
  }
  try {
    return std::make_unique<device_dense_root_block>(m);
  } catch (const std::bad_alloc&) {  // the memory went meanwhile, to another process: as if it did not fit
    ++device_fallbacks;
    return make_host_dense_root_block("host fallback of the cuda block");
  }
}

long dense_root_block_device_fallbacks() { return device_fallbacks.load(); }
void dense_root_block_device_memory_cap(unsigned long long bytes) { device_memory_cap.store(bytes); }
bool dense_root_block_device_available() { return true; }
int dense_root_block_set_device_min_width(int width) { return device_min_width.exchange(width); }

}  // namespace laplace::sparse::detail
