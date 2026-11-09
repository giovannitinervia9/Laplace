// A LAPLACE_HD function summed over n points inside a Kokkos kernel on the
// default execution space equals the same sum computed by a host loop, bit for
// bit. The terms 2 i + 1 are integers and every partial sum stays below 2^53,
// so both sums are exact whatever the order of summation.
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>
#include <laplace/core/config.hpp>

namespace {

template <class T>
LAPLACE_HD T twice_plus_one(T x) {
  return T(2) * x + T(1);
}

}  // namespace

TEST_CASE("a LAPLACE_HD function gives the same sum in a kernel and on the host") {
  const long n = 1L << 20;
  laplace::real kernel_sum = 0;
  Kokkos::parallel_reduce(
      "test_config", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const long i, laplace::real& acc) {
        acc += twice_plus_one(static_cast<laplace::real>(i));
      },
      kernel_sum);
  laplace::real host_sum = 0;
  for (long i = 0; i < n; ++i) host_sum += twice_plus_one(static_cast<laplace::real>(i));
  const laplace::real expected = static_cast<laplace::real>(n) * static_cast<laplace::real>(n);
  CHECK(kernel_sum == expected);
  CHECK(host_sum == expected);
}
