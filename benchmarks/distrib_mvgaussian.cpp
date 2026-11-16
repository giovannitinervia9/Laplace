// Speed of mvgaussian1_distrib and mvgaussian2_distrib (include/laplace/distrib/)
// over unstructured_pdmat with log_cholesky, d = 2, 3, 5, 10, n = 1e5 units, on
// the default execution space: ns per unit for
//   order 0  the log-density;
//   order 1  the gradient, mu (A = 1, K = 0) and theta (A = 0, K = 1, each j);
//   order 2  the Hessian, (mu, mu) (A = 2, K = 0), (mu, theta) (A = 1, K = 1,
//            each j) and (theta, theta) (A = 0, K = 2, each pair);
// with theta shared by the units (the state computed once, outside the timing)
// and with theta per unit (each unit computes its own state). The accuracy is
// measured by tests/distrib/test_mvgaussian.cpp.
//
//   distrib_mvgaussian [--kokkos-num-threads=n]
#include <laplace/distrib/mvgaussian1.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/pdmat/unstructured.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using Exec = Kokkos::DefaultExecutionSpace;
using Mat = Kokkos::View<double**, Kokkos::LayoutRight>;

constexpr int max_d = 10;

template <class D>
KOKKOS_INLINE_FUNCTION double unit_work(const D& g, const typename D::state_type& st, const double* y,
                                        const double* mu, int order, double* ws) {
  const int d = g.dim(), q = g.n_params();
  double out[lp::packed_size(max_d)];
  if (order == 0) return g.logpdf(st, y, mu);
  double s = 0.0;
  if (order == 1) {
    g.template logpdf_dmu_dk<1, 0>(st, y, mu, nullptr, out, ws);
    for (int k = 0; k < d; ++k) s += out[k];
    for (int j = 0; j < q; ++j) {
      const int t[1] = {j};
      if (g.template logpdf_dmu_dk<0, 1>(st, y, mu, t, out, ws)) s += out[0];
    }
    return s;
  }
  g.template logpdf_dmu_dk<2, 0>(st, y, mu, nullptr, out, ws);
  s += out[0];
  for (int j = 0; j < q; ++j) {
    const int t[1] = {j};
    if (g.template logpdf_dmu_dk<1, 1>(st, y, mu, t, out, ws)) s += out[0];
  }
  for (int a = 0; a < q; ++a)
    for (int b = a; b < q; ++b) {
      const int t[2] = {a, b};
      if (g.template logpdf_dmu_dk<0, 2>(st, y, mu, t, out, ws)) s += out[0];
    }
  return s;
}

// ns per unit; shared: one state for all units, else each unit its own
template <class D>
double time_case(const D& g, const Kokkos::View<double*>& theta, const Mat& thetas, const Mat& y, const Mat& mu,
                 bool shared, int order) {
  using St = typename D::state_type;
  const int n = int(y.extent(0)), ss = g.state_size(), W = ss + g.scratch_size();
  const int chunk = std::min<long long>(n, (1500LL << 20) / (8LL * W));
  Kokkos::View<double*> sbuf("sbuf", W);
  Kokkos::View<St> st("st");
  Kokkos::parallel_for(
      "state", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) { st() = g.theta_state(theta.data(), sbuf.data()); });
  Mat work("work", chunk, W);
  double best = 1e300, sink = 0.0;
  for (int rep = 0; rep < 4; ++rep) {
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    for (int c0 = 0; c0 < n; c0 += chunk) {
      const int m = std::min(chunk, n - c0);
      double part = 0.0;
      Kokkos::parallel_reduce(
          "units", Kokkos::RangePolicy<Exec>(0, m),
          KOKKOS_LAMBDA(int c, double& acc) {
            const int i = c0 + c;
            double* w = &work(c, 0);
            if (shared) {
              acc += unit_work(g, st(), &y(i, 0), &mu(i, 0), order, w);
            } else {
              const auto own = g.theta_state(&thetas(i, 0), w);
              acc += unit_work(g, own, &y(i, 0), &mu(i, 0), order, w + ss);
            }
          },
          part);
      sink += part;
    }
    Kokkos::fence();
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  if (sink == 12345.678) std::printf("%g", sink);
  return best * 1e9 / n;
}

template <template <class> class G>
void run(const char* name, int d, int n) {
  const log_chol s(d);
  const G<log_chol> g(s);
  const int q = g.n_params();
  std::mt19937_64 rng(20261115 + d);
  std::normal_distribution<double> nd(0.0, 1.0);
  Kokkos::View<double*> theta("theta", q);
  Mat thetas("thetas", n, q), y("y", n, d), mu("mu", n, d);
  auto ht = Kokkos::create_mirror_view(theta);
  auto hts = Kokkos::create_mirror_view(thetas);
  auto hy = Kokkos::create_mirror_view(y);
  auto hm = Kokkos::create_mirror_view(mu);
  for (int k = 0; k < q; ++k) ht(k) = 0.3 * nd(rng);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < q; ++k) hts(i, k) = ht(k) + 0.1 * nd(rng);
    for (int k = 0; k < d; ++k) hy(i, k) = nd(rng), hm(i, k) = 0.2 * nd(rng);
  }
  Kokkos::deep_copy(theta, ht);
  Kokkos::deep_copy(thetas, hts);
  Kokkos::deep_copy(y, hy);
  Kokkos::deep_copy(mu, hm);
  for (int shared = 1; shared >= 0; --shared) {
    std::printf("%s log_cholesky d = %2d, q = %2d, theta %-8s |", name, d, q, shared ? "shared" : "per unit");
    for (int order = 0; order <= 2; ++order)
      std::printf(" order %d %10.1f", order, time_case(g, theta, thetas, y, mu, shared, order));
    std::printf("  ns/unit (%s)\n", Exec::name());
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int n = 100000;
    for (int d : {2, 3, 5, 10}) {
      run<ld::mvgaussian1_distrib>("mvgaussian1", d, n);
      run<ld::mvgaussian2_distrib>("mvgaussian2", d, n);
    }
  }
  Kokkos::finalize();
}
