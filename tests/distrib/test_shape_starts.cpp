// The starting values of the power exponential laws from the sample kurtosis
// (powerexp_distrib::start_value(y), mvpowerexp1/2_distrib::start_value(y, n,
// theta)): the kurtosis functions at their known values, the inversion at
// exact kurtoses, and the shape recovered from large simulated samples.
// Draws on the host (rng_by_inversion and the laws' own rng are host-device).

#include <doctest/doctest.h>

#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/mvpowerexp1.hpp>
#include <laplace/distrib/mvpowerexp2.hpp>
#include <laplace/distrib/powerexp.hpp>
#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/unstructured.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace shape_starts_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using pe = ld::powerexp_distrib;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;

void powerexp_kurtosis_case() {
  CHECK(pe::kurtosis_of_shape(1.0) == doctest::Approx(6.0).epsilon(1e-14));
  CHECK(pe::kurtosis_of_shape(2.0) == doctest::Approx(3.0).epsilon(1e-14));
  CHECK(pe::kurtosis_of_shape(1e6) == doctest::Approx(1.8).epsilon(1e-5));
  // mvpowerexp's Mardia kurtosis: d (d + 2) at beta = 1, and the univariate kurtosis at d = 1
  namespace dm = ld::detail_mvpowerexp;
  for (int d : {1, 2, 3, 5}) CHECK(dm::mardia_kurtosis_of_shape(d, 1.0) == doctest::Approx(d * (d + 2.0)).epsilon(1e-14));
  for (double p : {0.7, 1.5, 4.0})
    CHECK(dm::mardia_kurtosis_of_shape(1, p / 2) == doctest::Approx(pe::kurtosis_of_shape(p)).epsilon(1e-14));
  // the inversion at exact kurtoses, and its clamps
  for (int d : {1, 2, 3})
    for (double beta : {0.3, 0.75, 1.0, 2.5, 8.0})
      CHECK(dm::shape_from_mardia_kurtosis(d, dm::mardia_kurtosis_of_shape(d, beta)) ==
            doctest::Approx(beta).epsilon(1e-13));
  CHECK(dm::shape_from_mardia_kurtosis(2, 1e3) == 0.25);
  CHECK(dm::shape_from_mardia_kurtosis(2, 1.0) == 10.0);
}

void powerexp_sample_case() {
  double worst = 0.0;
  for (double p : {0.8, 1.5, 2.0, 3.0, 6.0}) {
    const pe truth{1.0, 2.0, p};
    std::vector<double> y(100000);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = ld::rng_by_inversion(truth, 20261119u, i);
    const pe s = ld::distrib_start_value(pe{}, y);
    const auto sm = ld::summarize_sample(y);
    CHECK(s.mu == doctest::Approx(sm.mean).epsilon(1e-14));
    CHECK(s.sigma == doctest::Approx(std::sqrt(sm.variance)).epsilon(1e-12));
    const double rel = std::fabs(s.p - p) / p;
    worst = std::fmax(worst, rel);
    if (print) std::printf("powerexp start, p = %g: p0 = %.6g (relative %.3g), sigma0 = %.6g\n", p, s.p, rel, s.sigma);
    // measured at most 0.0389 (p = 0.8) on n = 1e5 draws (2026-11-19), times 1.5
    CHECK_MESSAGE(rel <= 0.06, "p " << p << ": start " << s.p);
  }
  // a sample too small, or with no spread, starts at the Gaussian
  CHECK(ld::distrib_start_value(pe{}, std::vector<double>{1.0, 2.0, 3.0}).p == 2.0);
  CHECK(ld::distrib_start_value(pe{}, std::vector<double>(10, 1.5)).p == 2.0);
  CHECK(ld::distrib_start_value(pe{}, std::vector<double>(10, 1.5)).sigma > 0.0);
  // the clamps: a two-point sample has kurtosis 1, a sample of one outlier a huge one
  std::vector<double> two(1000);
  for (std::size_t i = 0; i < two.size(); ++i) two[i] = i % 2 ? 1.0 : -1.0;
  CHECK(ld::distrib_start_value(pe{}, two).p == doctest::Approx(20.0).epsilon(1e-15));
  std::vector<double> spike(1000, 0.0);
  spike[0] = 1.0;
  CHECK(ld::distrib_start_value(pe{}, spike).p == doctest::Approx(0.5).epsilon(1e-15));
}

template <class G>
void mv_sample_one(const G& g, const std::vector<double>& th, const char* name, bool unrestricted) {
  const int d = g.dim(), q = g.structure.n_params(), n = 40000;
  std::vector<double> w(g.state_size() + g.scratch_size()), y(std::size_t(n) * d);
  const auto st = g.theta_state(th.data(), w.data());
  const std::vector<double> mu = {0.5, -1.0, 0.25};
  for (int i = 0; i < n; ++i) g.rng(st, mu.data(), 20261119u, std::uint64_t(i), &y[std::size_t(i) * d]);
  std::vector<double> t0(q + 1), cov(lp::packed_size(d)), ws(4096);
  g.start_value(y.data(), n, t0.data());
  const double beta = th[q], rel = std::fabs(t0[q] - beta) / beta;
  // the covariance of the start equals the sample covariance
  const auto sm = ld::summarize_mv_sample(y.data(), n, d);
  g.covariance(t0.data(), cov.data(), ws.data());
  double ec = 0.0, cmax = 0.0;
  for (int k = 0; k < lp::packed_size(d); ++k)
    ec = std::fmax(ec, std::fabs(cov[k] - sm.covariance[k])), cmax = std::fmax(cmax, std::fabs(sm.covariance[k]));
  if (print)
    std::printf("%s d = %d, beta = %g: beta0 = %.6g (relative %.3g), covariance %.3g relative\n", name, d, beta, t0[q],
                rel, ec / cmax);
  // measured at most 0.0062 (log_cholesky, beta = 2.5) on n = 4e4 draws (2026-11-19), times 1.5
  CHECK_MESSAGE(rel <= 0.0095, name << " d " << d << " beta " << beta << ": start " << t0[q]);
  // an unrestricted structure reproduces S exactly; ar1 projects it on its own family
  if (unrestricted) CHECK_MESSAGE(ec <= 1e-12 * cmax, name << ": covariance of the start " << ec / cmax);
}

void mvpowerexp_sample_case() {
  for (double beta : {0.4, 1.0, 2.5}) {
    {
      const log_chol s(3);
      std::vector<double> th = {0.2, 0.3, -0.1, 0.1, 0.4, -0.2, beta};
      mv_sample_one(ld::mvpowerexp1_distrib<log_chol>(s), th, "mvpowerexp1 log_cholesky", true);
      mv_sample_one(ld::mvpowerexp2_distrib<log_chol>(s), th, "mvpowerexp2 log_cholesky", true);
    }
    {
      const lp::ar1_pdmat s(2);
      std::vector<double> th(s.n_params(), 0.3);
      th.push_back(beta);
      mv_sample_one(ld::mvpowerexp1_distrib<lp::ar1_pdmat>(s), th, "mvpowerexp1 ar1", false);
    }
  }
  // too few units for the kurtosis: beta = 1
  const log_chol s(3);
  const ld::mvpowerexp1_distrib<log_chol> g(s);
  const std::vector<double> y = {0.1, 0.2, 0.3, 1.0, 0.0, 2.0, -1.0, 0.5, 0.2, 0.3, 0.3, 0.3};
  std::vector<double> t0(s.n_params() + 1);
  g.start_value(y.data(), 4, t0.data());
  CHECK(t0[s.n_params()] == 1.0);
}

}  // namespace shape_starts_test

TEST_CASE("powerexp and mvpowerexp: the kurtosis of the shape and its inversion") {
  shape_starts_test::powerexp_kurtosis_case();
}

TEST_CASE("powerexp: the starting shape from the sample kurtosis") { shape_starts_test::powerexp_sample_case(); }

TEST_CASE("mvpowerexp: the starting shape from Mardia's kurtosis") { shape_starts_test::mvpowerexp_sample_case(); }
