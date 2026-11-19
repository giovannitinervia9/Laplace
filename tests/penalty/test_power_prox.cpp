// The penalties of the power exponential laws, step 7 (note/penalties.md
// section 5g):
//  - the scalar prox x*(u, a, p) = argmin a|x|^p + (x - u)^2/2
//    (distrib/detail/power_prox.hpp) against the mpmath references of
//    tools/reference/power_prox.py, in a Kokkos kernel; zeros of the bridge
//    (p < 1, below the threshold) exactly; the closed forms p = 1 and p = 2 bit
//    for bit; oddness in u, NaN;
//  - the descriptors and rho of the penalties of powerexp (through
//    univariate_as_multivariate and the independent product) and of
//    mvpowerexp1/2, and the product's rho reading the shape of mvpowerexp.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/distrib/detail/power_prox.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/pdmat/diagonal.hpp>
#include <laplace/pdmat/scaled_identity.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace power_prox_test {

namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
template <class... M>
using product = ld::independent_product_distrib<M...>;

struct power_prox_fn {
  KOKKOS_INLINE_FUNCTION double operator()(double u, double a, double p) const {
    return ld::detail_power_prox::power_prox(u, a, p);
  }
};

// res: [0] closed forms p = 1, 2 bit for bit, [1] oddness, [2] NaN in gives NaN, [3] u = 0 gives 0
void run_special_kernel(double* host) {
  Kokkos::View<double[4]> res("power_prox_special");
  Kokkos::parallel_for(
      "power_prox_special", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        using ld::detail_power_prox::power_prox;
        double bad[4] = {0, 0, 0, 0};
        const double us[6] = {0.0, 0.3, 1.0, 2.5, 1e-9, 1e9}, as[5] = {1e-6, 0.3, 1.0, 2.5, 1e6};
        const double ps[8] = {0.2, 0.5, 0.9, 1.0, 1.3, 2.0, 3.0, 25.0};
        for (double u : us)
          for (double a : as) {
            const double soft = u <= a ? 0.0 : u - a;
            if (power_prox(u, a, 1.0) != soft) bad[0] += 1;
            if (power_prox(u, a, 2.0) != u / (1.0 + 2.0 * a)) bad[0] += 1;
            for (double p : ps)
              if (power_prox(-u, a, p) != -power_prox(u, a, p)) bad[1] += 1;
          }
        const double nan = Kokkos::Experimental::quiet_NaN_v<double>;
        for (double p : ps) {
          if (!Kokkos::isnan(power_prox(nan, 1.0, p))) bad[2] += 1;
          if (power_prox(0.0, 1.0, p) != 0.0) bad[3] += 1;
        }
        for (int i = 0; i < 4; ++i) res(i) = bad[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 4; ++i) host[i] = h(i);
}

// ---- powerexp_distrib::prox_y ----

KOKKOS_INLINE_FUNCTION double pe_objective(const ld::powerexp_distrib& d, double y, double v, double t) {
  return -d.logpdf(y) + (y - v) * (y - v) / (2.0 * t);
}

// the least objective over a grid of 4001 points on the segment [mu, v], refined by golden
// section around the best grid point: a route that knows nothing of thresholds or roots,
// and sees both minima of the bridge
KOKKOS_INLINE_FUNCTION double pe_least(const ld::powerexp_distrib& d, double v, double t) {
  const int n = 4000;
  double best = pe_objective(d, d.mu, v, t);
  int at = 0;
  for (int i = 1; i <= n; ++i) {
    const double f = pe_objective(d, d.mu + (v - d.mu) * i / n, v, t);
    if (f < best) {
      best = f;
      at = i;
    }
  }
  const double h = (v - d.mu) / n, g = 0.38196601125010515;
  double a = d.mu + h * (at - 1 < 0 ? 0 : at - 1), b = d.mu + h * (at + 1 > n ? n : at + 1);
  double c = a + g * (b - a), e = b - g * (b - a), fc = pe_objective(d, c, v, t), fe = pe_objective(d, e, v, t);
  for (int i = 0; i < 200; ++i) {
    if (fc <= fe) {
      b = e, e = c, fe = fc, c = a + g * (b - a), fc = pe_objective(d, c, v, t);
    } else {
      a = c, c = e, fc = fe, e = b - g * (b - a), fe = pe_objective(d, e, v, t);
    }
  }
  return Kokkos::fmin(best, Kokkos::fmin(fc, fe));
}

// res: [0] objective at prox_y above the least found, relative to max(1, |objective|);
// [1] optimality residual |t pen'(y) + y - v| / (max(1, |v - mu|) max(1, t |pen''(y)|)) where
// y != mu; [2] count of y = mu with |v - mu| > 0 for p >= 1 (only the soft threshold may
// return mu then, checked in [0]); [3] NaN in gives NaN out and v = mu gives mu exactly
void run_powerexp_kernel(double* host) {
  Kokkos::View<double[4]> res("powerexp_prox");
  Kokkos::parallel_for(
      "powerexp_prox", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        double r[4] = {0, 0, 0, 0};
        const double ps[9] = {0.3, 0.5, 0.9, 1.0, 1.05, 1.5, 2.0, 3.0, 8.0}, sigmas[2] = {0.4, 2.0};
        const double ts[4] = {1e-3, 0.5, 4.0, 100.0}, us[7] = {1e-3, -1e-3, 0.07, 0.7, -2.5, 30.0, -300.0};
        const double nan = Kokkos::Experimental::quiet_NaN_v<double>;
        for (double p : ps)
          for (double sg : sigmas) {
            const ld::powerexp_distrib d{0.3, sg, p};
            if (!Kokkos::isnan(d.prox_y(nan, 1.0)) || d.prox_y(d.mu, 1.0) != d.mu) r[3] += 1;
            for (double t : ts)
              for (double u : us) {
                const double v = d.mu + u, y = d.prox_y(v, t);
                const double fy = pe_objective(d, y, v, t), fl = pe_least(d, v, t);
                r[0] = Kokkos::fmax(r[0], (fy - fl) / Kokkos::fmax(1.0, Kokkos::fabs(fl)));
                if (y != d.mu) {
                  const double d1 = -d.template logpdf_dy_dk<1, 0>(y)[0], d2 = -d.template logpdf_dy_dk<2, 0>(y)[0];
                  const double sc = Kokkos::fmax(1.0, Kokkos::fabs(u)) * Kokkos::fmax(1.0, t * Kokkos::fabs(d2));
                  r[1] = Kokkos::fmax(r[1], Kokkos::fabs(t * d1 + y - v) / sc);
                }
              }
          }
        for (int i = 0; i < 4; ++i) res(i) = r[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 4; ++i) host[i] = h(i);
}

// ---- mvpowerexp1/2 over scaled_identity_pdmat ----

using iso1 = ld::mvpowerexp1_distrib<laplace::pdmat::scaled_identity_pdmat>;
using iso2 = ld::mvpowerexp2_distrib<laplace::pdmat::scaled_identity_pdmat>;
constexpr int iso_d = 3, iso_w = 640;  // state + scratch of mvpowerexp at d = 3: 12 + 614

// res: [0] beta = 1/2 against the block soft threshold, relative to max(1, |v - mu|);
// [1] optimality residual of the gradient where out != mu, relative as in the scalar case;
// [2] the objective at out above its least along the ray of u (4001 points + golden);
// [3] mvpowerexp2 at q = 1/sigma^2 against mvpowerexp1, relative; [4] u = 0 gives mu
// exactly and the zero region of beta = 1/2 gives mu exactly (counts)
void run_mvpowerexp_kernel(double* host) {
  Kokkos::View<double[5]> res("mvpowerexp_prox");
  Kokkos::parallel_for(
      "mvpowerexp_prox", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        double r[5] = {0, 0, 0, 0, 0};
        double w[iso_w], w2[iso_w];
        const iso1 m1{laplace::pdmat::scaled_identity_pdmat(iso_d)};
        const iso2 m2{laplace::pdmat::scaled_identity_pdmat(iso_d)};
        const double mu[iso_d] = {0.2, -0.1, 0.4}, dirs[2][iso_d] = {{0.6, -0.8, 0.0}, {-0.48, 0.36, 0.8}};
        const double betas[6] = {0.25, 0.45, 0.5, 0.8, 1.0, 2.5}, sigmas[2] = {0.5, 3.0};
        const double ts[3] = {0.01, 0.7, 20.0}, norms[5] = {1e-4, 0.05, 0.9, 4.0, 60.0};
        for (double beta : betas)
          for (double sg : sigmas) {
            const double th1[2] = {Kokkos::log(sg), beta}, th2[2] = {-Kokkos::log(sg), beta};
            const auto st1 = m1.theta_state(th1, w);
            const auto st2 = m2.theta_state(th2, w2);
            double out[iso_d], out2[iso_d], g[iso_d], v[iso_d];
            m1.prox_y(st1, mu, mu, 1.0, out, w + m1.state_size());
            for (int i = 0; i < iso_d; ++i)
              if (out[i] != mu[i]) r[4] += 1;
            for (double t : ts)
              for (const auto& dir : dirs)
                for (double nrm : norms) {
                  for (int i = 0; i < iso_d; ++i) v[i] = mu[i] + nrm * dir[i];
                  m1.prox_y(st1, v, mu, t, out, w + m1.state_size());
                  m2.prox_y(st2, v, mu, t, out2, w2 + m2.state_size());
                  double on = 0.0, e3 = 0.0;
                  for (int i = 0; i < iso_d; ++i) {
                    on += (out[i] - mu[i]) * (out[i] - mu[i]);
                    e3 = Kokkos::fmax(e3, Kokkos::fabs(out2[i] - out[i]));
                  }
                  on = Kokkos::sqrt(on);
                  r[3] = Kokkos::fmax(r[3], e3 / Kokkos::fmax(1.0, nrm));
                  if (beta == 0.5) {
                    const double k = Kokkos::fmax(0.0, 1.0 - t / (2.0 * sg * nrm));
                    for (int i = 0; i < iso_d; ++i)
                      r[0] = Kokkos::fmax(r[0], Kokkos::fabs(out[i] - (mu[i] + k * nrm * dir[i])) / Kokkos::fmax(1.0, nrm));
                    if (k == 0.0 && on != 0.0) r[4] += 1;
                  }
                  // the objective along the ray mu + s dir, s in [0, nrm]
                  auto obj = [&](double s) {
                    double y[iso_d], q = 0.0;
                    for (int i = 0; i < iso_d; ++i) {
                      y[i] = mu[i] + s * dir[i];
                      q += (y[i] - v[i]) * (y[i] - v[i]);
                    }
                    return -m1.logpdf(st1, y, mu) + q / (2.0 * t);
                  };
                  double fo = 0.0;
                  {
                    double q = 0.0;
                    for (int i = 0; i < iso_d; ++i) q += (out[i] - v[i]) * (out[i] - v[i]);
                    fo = -m1.logpdf(st1, out, mu) + q / (2.0 * t);
                  }
                  const int n = 4000;
                  double best = obj(0.0);
                  int at = 0;
                  for (int i = 1; i <= n; ++i) {
                    const double f = obj(nrm * i / n);
                    if (f < best) best = f, at = i;
                  }
                  const double gg = 0.38196601125010515, h = nrm / n;
                  double a = h * (at > 0 ? at - 1 : 0), b = h * (at < n ? at + 1 : n);
                  double c = a + gg * (b - a), e = b - gg * (b - a), fc = obj(c), fe = obj(e);
                  for (int it = 0; it < 200; ++it) {
                    if (fc <= fe) b = e, e = c, fe = fc, c = a + gg * (b - a), fc = obj(c);
                    else a = c, c = e, fc = fe, e = b - gg * (b - a), fe = obj(e);
                  }
                  best = Kokkos::fmin(best, Kokkos::fmin(fc, fe));
                  r[2] = Kokkos::fmax(r[2], (fo - best) / Kokkos::fmax(1.0, Kokkos::fabs(best)));
                  if (on != 0.0) {
                    m1.template logpdf_dy_dk<1, 0>(st1, out, mu, nullptr, g, w + m1.state_size());
                    // pen'' along the ray scales the conditioning, as in the scalar case
                    const double p2 = 2.0 * beta, cc = sg;
                    const double d2 = 0.5 * p2 * (p2 - 1.0) * Kokkos::pow(on / cc, p2 - 2.0) / (cc * cc);
                    const double sc = Kokkos::fmax(1.0, nrm) * Kokkos::fmax(1.0, t * Kokkos::fabs(d2));
                    for (int i = 0; i < iso_d; ++i) r[1] = Kokkos::fmax(r[1], Kokkos::fabs(-t * g[i] + out[i] - v[i]) / sc);
                  }
                }
          }
        for (int i = 0; i < 5; ++i) res(i) = r[i];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 5; ++i) host[i] = h(i);
}

}  // namespace power_prox_test

TEST_CASE("power prox: powerexp_distrib::prox_y, objective and optimality, in a kernel") {
  using namespace power_prox_test;
  double r[4];
  run_powerexp_kernel(r);
  MESSAGE("powerexp prox: objective excess ", r[0], ", optimality ", r[1]);
  CHECK(r[0] <= 4e-15);  // measured 2.51e-15 on gcc and CUDA
  CHECK(r[1] <= 9e-16);  // measured 5.68e-16 on gcc and CUDA
  CHECK(r[3] == 0.0);
}

TEST_CASE("power prox: mvpowerexp1/2 over a scaled identity, in a kernel") {
  using namespace power_prox_test;
  static_assert(lpen::distrib_penalty<iso1>::prox == lpen::penalty_prox_kind::closed_form);
  static_assert(lpen::distrib_penalty<product<iso2>>::prox == lpen::penalty_prox_kind::closed_form);
  static_assert(lpen::distrib_penalty<ld::mvpowerexp1_distrib<laplace::pdmat::diagonal_pdmat>>::prox ==
                lpen::penalty_prox_kind::none);
  const iso1 m1{laplace::pdmat::scaled_identity_pdmat(iso_d)};
  const iso2 m2{laplace::pdmat::scaled_identity_pdmat(iso_d)};
  REQUIRE(m1.state_size() + m1.scratch_size() <= iso_w);
  REQUIRE(m2.state_size() + m2.scratch_size() <= iso_w);
  double r[5];
  run_mvpowerexp_kernel(r);
  MESSAGE("mvpowerexp prox: block soft threshold ", r[0], ", optimality ", r[1], ", objective excess ", r[2],
          ", mvpowerexp2 vs 1 ", r[3]);
  CHECK(r[0] <= 4e-16);  // measured 2.22e-16 on gcc and CUDA
  CHECK(r[1] <= 4e-16);  // measured 2.22e-16 gcc, 2.37e-16 CUDA
  CHECK(r[2] <= 7e-16);  // measured 3.99e-16 gcc, 4.25e-16 CUDA
  CHECK(r[3] == 0.0);    // the same arithmetic: 0 on gcc and CUDA
  CHECK(r[4] == 0.0);
}

TEST_CASE("power prox: the group lasso product and the powerexp product forward prox_y copy by copy") {
  using namespace power_prox_test;
  // two groups of 2 coefficients, mvpowerexp2 with Q = q_g I, beta_g; theta param-major
  // (theta_1, theta_2, beta_1, beta_2)
  const product<iso2> gm(ld::independent_product_component<iso2>{iso2(laplace::pdmat::scaled_identity_pdmat(2)), 2, 2, nullptr});
  const lpen::distrib_penalty<product<iso2>> gp(gm);
  const double theta[4] = {0.3, -0.7, 0.5, 1.7}, v[4] = {0.9, -0.2, 0.05, 0.02};
  std::vector<double> w(gp.state_size() + gp.scratch_size()), w1(4096);
  const auto st = gp.theta_state(theta, w.data());
  for (const double t : {0.1, 1.0, 5.0}) {
    double out[4];
    gp.prox_beta(st, v, t, out, w.data() + gp.state_size());
    for (int g = 0; g < 2; ++g) {
      const iso2 one{laplace::pdmat::scaled_identity_pdmat(2)};
      const double th[2] = {theta[g], theta[2 + g]}, zero[2] = {0.0, 0.0};
      const auto s1 = one.theta_state(th, w1.data());
      double o1[2];
      one.prox_y(s1, v + 2 * g, zero, t, o1, w1.data() + one.state_size());
      CHECK(out[2 * g] == o1[0]);
      CHECK(out[2 * g + 1] == o1[1]);
    }
  }
  // powerexp coefficients with a shared sigma and p by group, prior location mu_0
  const int groups[3] = {0, 1, 0};
  using pe_pen = lpen::distrib_penalty<product<uam<ld::powerexp_distrib>>>;
  const double mu0[3] = {0.1, -0.2, 0.0}, pt[4] = {0.8, 1.5, 0.6, 2.5}, pv[3] = {1.0, -0.1, -3.0};
  const pe_pen pp(product<uam<ld::powerexp_distrib>>(ld::independent_product_component<uam<ld::powerexp_distrib>>{{}, 3, 2, groups}), mu0);
  std::vector<double> pw(pp.state_size() + pp.scratch_size());
  const auto pst = pp.theta_state(pt, pw.data());
  for (const double t : {0.1, 1.0, 5.0}) {
    double out[3];
    pp.prox_beta(pst, pv, t, out, pw.data() + pp.state_size());
    for (int j = 0; j < 3; ++j)
      CHECK(out[j] == ld::powerexp_distrib{mu0[j], pt[groups[j]], pt[2 + groups[j]]}.prox_y(pv[j], t));
  }
}

TEST_CASE("scaled_identity_pdmat: against e^(2 theta) I and diagonal_pdmat with equal parameters") {
  namespace lp = laplace::pdmat;
  const int d = 4, n = lp::packed_size(d);
  const lp::scaled_identity_pdmat s(d);
  const lp::diagonal_pdmat dg(d);
  for (const double th : {-1.3, 0.0, 0.45, 2.0}) {
    const double t1[1] = {th}, td[4] = {th, th, th, th};
    std::vector<double> a(n), b(n);
    s.matrix(t1, a.data(), nullptr);
    dg.matrix(td, b.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(a[i] == b[i]);
    s.inverse(t1, a.data(), nullptr);
    dg.inverse(td, b.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(a[i] == b[i]);
    s.cholesky(t1, a.data(), nullptr);
    dg.cholesky(td, b.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(a[i] == b[i]);
    CHECK(s.log_det(t1, nullptr) == doctest::Approx(dg.log_det(td, nullptr)).epsilon(1e-15));
    // derivatives in the one parameter: c^K e^(c theta) on the diagonal, against central differences
    const int i1[1] = {0}, i2[2] = {0, 0};
    const double h = 1e-5, tp[1] = {th + h}, tm[1] = {th - h};
    std::vector<double> d1(n), d2(n), mp(n), mm(n);
    s.matrix_d1(t1, i1, d1.data(), nullptr);
    s.matrix_d2(t1, i2, d2.data(), nullptr);
    s.matrix(tp, mp.data(), nullptr);
    s.matrix(tm, mm.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(d1[i] == doctest::Approx((mp[i] - mm[i]) / (2 * h)).epsilon(1e-9));
    s.matrix_d1(tp, i1, mp.data(), nullptr);
    s.matrix_d1(tm, i1, mm.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(d2[i] == doctest::Approx((mp[i] - mm[i]) / (2 * h)).epsilon(1e-9));
    s.inverse_d1(t1, i1, d1.data(), nullptr);
    s.inverse(tp, mp.data(), nullptr);
    s.inverse(tm, mm.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(d1[i] == doctest::Approx((mp[i] - mm[i]) / (2 * h)).epsilon(1e-9));
    s.cholesky_d1(t1, i1, d1.data(), nullptr);
    s.cholesky(tp, mp.data(), nullptr);
    s.cholesky(tm, mm.data(), nullptr);
    for (int i = 0; i < n; ++i) CHECK(d1[i] == doctest::Approx((mp[i] - mm[i]) / (2 * h)).epsilon(1e-9));
    CHECK(s.log_det_d1(t1, i1, nullptr) == 2.0 * d);
    CHECK(s.log_det_d2(t1, i2, nullptr) == 0.0);
    // params_from_matrix recovers theta
    double back[1];
    s.matrix(t1, a.data(), nullptr);
    s.params_from_matrix(a.data(), back);
    CHECK(back[0] == doctest::Approx(th).epsilon(1e-15));
  }
}

// The error is read in ulps of x* divided by the condition number of the problem,
// kappa = |u| / (|x*| phi'(x*)) with phi'(x) = 1 + a p (p - 1) |x|^(p-2), the
// relative change of x* per relative change of u (and of a, to the same order):
// rounding the data alone moves x* by kappa ulps. kappa is large near p = 1 when
// x* is tiny (it tends to 1/(p - 1)) and, for p < 1, just above the threshold,
// where phi' vanishes at the edge of the branch. The plain ulp error is printed too.
TEST_CASE("power prox: against mpmath, in a kernel") {
  using namespace power_prox_test;
  const auto r = bench::read_ref(std::string(LAPLACE_REFERENCE_DIR) + "/power_prox.csv");
  REQUIRE(r.x.size() > 2000);
  const auto y = bench::eval_kernel(r.x, power_prox_fn{});
  double worst_cond = 0.0, worst_plain = 0.0;
  std::size_t at = 0, wrong_zeros = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double u = r.x[i][0], a = r.x[i][1], p = r.x[i][2], x = r.hi[i];
    if (x == 0.0 || y[i] == 0.0) {
      if (x != y[i]) ++wrong_zeros;
      continue;
    }
    const double ax = std::fabs(x), kappa = std::fabs(u) / (ax * (1.0 + a * p * (p - 1.0) * std::pow(ax, p - 2.0)));
    const double e = std::fabs(bench::error_in(y[i], r.hi[i], r.lo[i], x));
    worst_plain = std::fmax(worst_plain, e);
    if (e / std::fmax(1.0, kappa) > worst_cond) {
      worst_cond = e / std::fmax(1.0, kappa);
      at = i;
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("power_prox: %zu rows, max %.4g ulp, max %.4g ulp/kappa at (u, a, p) = (%.17g, %.17g, %.17g), wrong zeros %zu\n",
                y.size(), worst_plain, worst_cond, r.x[at][0], r.x[at][1], r.x[at][2], wrong_zeros);
  INFO("max ", worst_cond, " ulp/kappa at (", r.x[at][0], ", ", r.x[at][1], ", ", r.x[at][2], ")");
  CHECK(wrong_zeros == 0);
  CHECK(worst_cond <= 11.0);  // measured 7.17 gcc, 7.27 CUDA (plain 164 and 192 ulp)
}

TEST_CASE("power prox: closed forms, oddness, NaN, in a kernel") {
  using namespace power_prox_test;
  double r[4];
  power_prox_test::run_special_kernel(r);
  CHECK(r[0] == 0.0);
  CHECK(r[1] == 0.0);
  CHECK(r[2] == 0.0);
  CHECK(r[3] == 0.0);
}

TEST_CASE("power exponential penalties: descriptors and rho") {
  using namespace power_prox_test;
  using pe_pen = lpen::distrib_penalty<product<uam<ld::powerexp_distrib>>>;
  static_assert(lpen::penalty<pe_pen>);
  static_assert(pe_pen::normalization == lpen::penalty_normalization::proper);
  static_assert(pe_pen::beta_regularity == 0 && pe_pen::kinks == lpen::penalty_kink_geometry::per_coordinate);
  static_assert(pe_pen::structure == lpen::penalty_structure::separable);
  static_assert(pe_pen::scale_mixture == lpen::penalty_scale_mixture::none);
  using mv1 = ld::mvpowerexp1_distrib<laplace::pdmat::diagonal_pdmat>;
  using mv2 = ld::mvpowerexp2_distrib<laplace::pdmat::diagonal_pdmat>;
  using mv1_pen = lpen::distrib_penalty<mv1>;
  using group_pen = lpen::distrib_penalty<product<mv2>>;
  static_assert(lpen::penalty<mv1_pen> && lpen::penalty<group_pen>);
  static_assert(mv1_pen::kinks == lpen::penalty_kink_geometry::radial && mv1_pen::beta_regularity == 0);
  static_assert(mv1_pen::structure == lpen::penalty_structure::coupled);
  static_assert(group_pen::kinks == lpen::penalty_kink_geometry::radial &&
                group_pen::structure == lpen::penalty_structure::grouped);
  static_assert(mv1_pen::normalization == lpen::penalty_normalization::proper);

  // powerexp: rho = 0 for p >= 1, infinity below; theta param-major (sigma_1, sigma_2, p_1, p_2)
  const int groups[3] = {0, 1, 0};
  const pe_pen pe(product<uam<ld::powerexp_distrib>>(ld::independent_product_component<uam<ld::powerexp_distrib>>{{}, 3, 2, groups}));
  const double th_convex[4] = {1.0, 2.0, 1.0, 3.5}, th_bridge[4] = {1.0, 2.0, 1.5, 0.7};
  CHECK(pe.weak_convexity(th_convex) == 0.0);
  CHECK(std::isinf(pe.weak_convexity(th_bridge)));
  // mvpowerexp: the shape sits after the structure's d parameters; rho = 0 for beta >= 1/2
  const mv1_pen m1{mv1(laplace::pdmat::diagonal_pdmat(3))};
  const double t1[4] = {0.1, 0.2, 0.3, 0.5}, t1b[4] = {0.1, 0.2, 0.3, 0.45};
  CHECK(m1.weak_convexity(t1) == 0.0);
  CHECK(std::isinf(m1.weak_convexity(t1b)));
  // the product passes each copy's distribution, so its rho reads the shape of mvpowerexp
  const group_pen gp(product<mv2>(ld::independent_product_component<mv2>{mv2(laplace::pdmat::diagonal_pdmat(2)), 2, 2, nullptr}));
  // param-major: theta_1 of set 1, of set 2, theta_2 of set 1, of set 2, beta_1, beta_2
  const double tg[6] = {0.0, 0.0, 0.1, 0.1, 0.5, 2.0}, tgb[6] = {0.0, 0.0, 0.1, 0.1, 0.5, 0.3};
  CHECK(gp.weak_convexity(tg) == 0.0);
  CHECK(std::isinf(gp.weak_convexity(tgb)));
}
