// Entry point of the test executable: Kokkos is initialized once around the
// whole doctest run, so every test case can launch kernels on the default
// execution space (OpenMP or CUDA, depending on the build).
//
// On CUDA the per-thread stack is raised from the default 1024 bytes before any
// kernel runs. A distrib's own derivative tensors fit in the default, but the
// generic routes of generic_cdf.hpp and generic_info.hpp keep the integrand's
// tensors of every order on the stack at each node of the quadrature, and that
// grows with the number of parameters: a three-parameter family needs more than
// 3072 bytes (measured on 2026-11-15: 3072 fails with an illegal memory access,
// 4096 works) and the families to come have four and more. The limit is a
// per-thread allocation, so it is paid in device memory times the number of
// resident threads; the figure printed below is the measured price on this
// device. Any host program that launches Laplace kernels with the generic
// routes has to do the same before its first launch, and the size it needs is
//
//   (2 + n_atoms) * sizeof(detail_generic::log_density_partials<P, K>)
//
// for the log-cdf derivative route at order K over P parameters: 3352 bytes at
// P = 4 without atoms, 13384 with six atoms, 48040 at P = 8. The 8192 below
// therefore covers P up to 4 with at most one atom, which is every family and
// wrapper the suite exercises (docs/numerics.md, "The stack the generic
// derivative route needs, and why neither obvious repair closes it").
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdio>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  std::printf("Kokkos execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
#if defined(KOKKOS_ENABLE_CUDA)
  {
    std::size_t before = 0, now = 0, free_before = 0, free_after = 0, total = 0;
    cudaDeviceGetLimit(&before, cudaLimitStackSize);
    cudaMemGetInfo(&free_before, &total);
    const cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, LAPLACE_TEST_CUDA_STACK);
    cudaDeviceGetLimit(&now, cudaLimitStackSize);
    cudaMemGetInfo(&free_after, &total);
    std::printf("CUDA thread stack: %zu -> %zu bytes (%s); device memory free %.0f of %.0f MB, %+.0f MB\n",
                before, now, cudaGetErrorString(e), free_after / 1048576.0, total / 1048576.0,
                (double(free_after) - double(free_before)) / 1048576.0);
  }
#endif
  doctest::Context context(argc, argv);
  return context.run();
}
