// Cost of the Gaussian-lasso integral (include/laplace/math/gaussian_laplace.hpp)
// against the dimension k: one evaluation of gaussian_laplace_log_integral, of
// gaussian_laplace_mean and of gaussian_laplace_cov on one thread, at a deep
// orthant point (one-factor correlation 0.9, t = lam sd = 10, mode at the
// kink), the median of repeated runs. Used for k_max (note: penalties step 6).
//
// Usage: math_gaussian_laplace [k_max = 8]
#include <laplace/math/gaussian_laplace.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace lm = laplace::math;

template <class F>
double median_seconds(F f) {
  std::vector<double> t;
  double total = 0.0;
  while (t.size() < 3 || (total < 2.0 && t.size() < 50)) {
    const auto a = std::chrono::steady_clock::now();
    f();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    t.push_back(s);
    total += s;
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int kmax = argc > 1 ? std::atoi(argv[1]) : 8;
  std::printf("k  orthants  log_integral_s  mean_s  cov_s  (one thread, median)\n");
  for (int k = 1; k <= kmax; ++k) {
    std::vector<double> m(k, 0.0), lam(k, 10.0), S(k * (k + 1) / 2), out(k * (k + 1) / 2);
    for (int p = 0; p < k; ++p)
      for (int q = 0; q <= p; ++q) S[p * (p + 1) / 2 + q] = p == q ? 1.0 : 0.9 * (p % 2 == q % 2 ? 1.0 : -1.0);
    std::vector<double> w(lm::gaussian_laplace_scratch(k, 2));
    volatile double sink = 0.0;
    const double t0 = median_seconds([&] { sink = lm::gaussian_laplace_log_integral(k, m.data(), S.data(), lam.data(), w.data()); });
    const double t1 = median_seconds([&] { sink = lm::gaussian_laplace_mean(k, m.data(), S.data(), lam.data(), out.data(), w.data()); });
    const double t2 = median_seconds([&] { sink = lm::gaussian_laplace_cov(k, m.data(), S.data(), lam.data(), out.data(), w.data()); });
    std::printf("%d  %8d  %14.3e  %6.3e  %6.3e\n", k, 1 << k, t0, t1, t2);
    std::fflush(stdout);
  }
  return 0;
}
