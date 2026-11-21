// The Gaussian-lasso integral and the mean and covariance of its law
// (include/laplace/math/gaussian_laplace.hpp) against the mpmath references of
// tools/reference/gaussian_laplace.py (one-factor Sigma, k = 1..5, t = lam sd to
// about 26, the mode at the kink, random, or off the kink on half the
// coordinates), evaluated in a Kokkos kernel on the default execution space, one
// thread per row and function, the moments on the lattice of N = 16381 (the value's default)
// to keep the case short. The error of log I is absolute; the mean is read in units of
// the law's standard deviations, the covariance in units of their products.
// Bounds: the errors measured on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.

#include <doctest/doctest.h>

#include <laplace/math/gaussian_laplace.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

namespace lm = laplace::math;
constexpr int KM = 5, NC = KM * (KM + 1) / 2;
// columns of a row
constexpr int c_m = 1, c_sd = 1 + KM, c_l = 1 + 2 * KM, c_lam = 1 + 3 * KM, c_hi = 1 + 4 * KM, c_mean = c_hi + 2,
              c_cov = c_mean + KM;

using Rows = Kokkos::View<double**, Kokkos::LayoutRight>;

std::vector<std::vector<double>> read_gl_rows() {
  std::ifstream f(std::string(LAPLACE_REFERENCE_DIR) + "/gaussian_laplace.csv");
  REQUIRE(f.good());
  std::string line;
  std::getline(f, line);
  std::vector<std::vector<double>> v;
  while (std::getline(f, line)) {
    std::vector<double> x;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) x.push_back(std::strtod(tok.c_str(), nullptr));
    if (!x.empty()) v.push_back(x);
  }
  return v;
}

// y: log I (value), log I and mean, log I and covariance
struct GaussianLaplace {
  Rows x, y, w;
  // one team of one thread per (row, function): on CUDA each is its own block, so
  // the rows' divergent routes do not run one after another inside a warp
  KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>::member_type& t) const {
    const int l = t.league_rank(), i = l / 3, f = l % 3;
    const int k = int(x(i, 0));
    double m[KM], lam[KM], S[NC];
    for (int p = 0; p < k; ++p) {
      m[p] = x(i, c_m + p);
      lam[p] = x(i, c_lam + p);
      for (int q = 0; q <= p; ++q)
        S[p * (p + 1) / 2 + q] = x(i, c_sd + p) * x(i, c_sd + q) * (p == q ? 1.0 : x(i, c_l + p) * x(i, c_l + q));
    }
    if (f == 0) y(i, 0) = lm::gaussian_laplace_log_integral(k, m, S, lam, &w(l, 0));
    if (f == 1) y(i, 1) = lm::gaussian_laplace_mean(k, m, S, lam, &y(i, 3), &w(l, 0), lm::mvn_default_lattice);
    if (f == 2) y(i, 2) = lm::gaussian_laplace_cov(k, m, S, lam, &y(i, 3 + KM), &w(l, 0), lm::mvn_default_lattice);
  }
};

}  // namespace

// The kernel sits in a function with external linkage (note/brief_agenti.md).
Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> gaussian_laplace_run(
    const std::vector<std::vector<double>>& v) {
  Rows x("x", v.size(), v[0].size());
  auto h = Kokkos::create_mirror_view(x);
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t j = 0; j < v[i].size(); ++j) h(i, j) = v[i][j];
  Kokkos::deep_copy(x, h);
  Rows y("y", v.size(), 3 + KM + NC), w("w", 3 * v.size(), lm::gaussian_laplace_scratch(KM, 2));
  Kokkos::parallel_for("gaussian_laplace", Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>(3 * int(v.size()), 1),
                       GaussianLaplace{x, y, w});
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
}

TEST_CASE("gaussian_laplace: log integral, mean and covariance against mpmath, k = 1..5" * doctest::test_suite("slow")) {
  const auto rows = read_gl_rows();
  const auto y = gaussian_laplace_run(rows);
  // worst by k: log I (value function), log I (moment functions), mean, covariance
  double worst[KM + 1][4] = {};
  int bad = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    const int k = int(r[0]);
    const double lI = r[c_hi] + r[c_hi + 1];
    double sd[KM];
    for (int p = 0; p < k; ++p) sd[p] = std::sqrt(r[c_cov + p * (p + 1) / 2 + p]);
    double e[4] = {std::fabs(y(i, 0) - lI), std::fmax(std::fabs(y(i, 1) - lI), std::fabs(y(i, 2) - lI)), 0, 0};
    for (int p = 0; p < k; ++p) {
      e[2] = std::fmax(e[2], std::fabs(y(i, 3 + p) - r[c_mean + p]) / sd[p]);
      for (int q = 0; q <= p; ++q) {
        const int o = p * (p + 1) / 2 + q;
        e[3] = std::fmax(e[3], std::fabs(y(i, 3 + KM + o) - r[c_cov + o]) / (sd[p] * sd[q]));
      }
    }
    for (int n = 0; n < 4; ++n) {
      if (!std::isfinite(e[n])) ++bad;
      worst[k][n] = std::fmax(worst[k][n], e[n]);
    }
  }
  CHECK(bad == 0);
  // log I (value), log I (moments), mean, covariance
  const double bound[KM + 1][4] = {{0},
                                   {5.5e-14, 5.5e-14, 2.1e-14, 7.5e-14},
                                   {9.7e-14, 8.3e-14, 3.2e-13, 2.1e-13},
                                   {1.7e-13, 1.7e-13, 2.1e-12, 6.8e-12},
                                   {5.0e-6, 6.1e-11, 5.9e-9, 4.8e-8},
                                   {8.7e-6, 8.7e-6, 1.2e-4, 1.8e-3}};
  for (int k = 1; k <= KM; ++k) {
    INFO("k = ", k, ": log I ", worst[k][0], ", log I (moments) ", worst[k][1], ", mean ", worst[k][2], ", cov ",
         worst[k][3]);
    for (int n = 0; n < 4; ++n) CHECK(worst[k][n] <= bound[k][n]);
  }
}
