// Entry point of the test executable: Kokkos is initialized once around the
// whole doctest run, so every test case can launch kernels on the default
// execution space (OpenMP or CUDA, depending on the build).
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

#include <cstdio>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  std::printf("Kokkos execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
  doctest::Context context(argc, argv);
  return context.run();
}
