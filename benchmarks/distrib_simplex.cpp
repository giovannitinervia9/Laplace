// Speed of dirichlet_distrib and multinomial_distrib (include/laplace/distrib/), d = 3,
// 5, 10, n = 2e4 units (each with its own eta and y; phi = 30, size 50 shared), on the
// default execution space: ns per unit for
//   order 0  the log-density (log-mass);
//   order 1  the gradient: eta (A = 1) and, for the Dirichlet, phi (K = 1);
//   order 2  the Hessian: (eta, eta) (A = 2) and, for the Dirichlet, (eta, phi)
//            (A = 1, K = 1) and (phi, phi) (K = 2).
// The units' scratch (scratch_size() doubles each) is allocated once, outside the
// timing. The accuracy is measured by tests/distrib/test_simplex.cpp.
//
//   distrib_simplex [--kokkos-num-threads=n]
#include <laplace/distrib/dirichlet.hpp>
#include <laplace/distrib/multinomial.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

namespace ld = laplace::distrib;
using Exec = Kokkos::DefaultExecutionSpace;
using Mat = Kokkos::View<double**, Kokkos::LayoutRight>;

constexpr int n_units = 20000;

struct Work {
  int law, d, order;
  Mat x, y, w;
  Kokkos::View<double*> sink;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double out[64];
    const double* xi = &x(i, 0);
    const double* yi = &y(i, 0);
    double* wi = &w(i, 0);
    double s = 0.0;
    const int m = d - 1;
    if (law == 1) {
      const ld::dirichlet_distrib D(d);
      const double th[1] = {30.0};
      const auto st = D.theta_state(th, wi);
      if (order == 0) {
        s = D.logpdf(st, yi, xi);
      } else if (order == 1) {
        D.template logpdf_dk<0, 1>(st, yi, xi, nullptr, out, wi);
        for (int a = 0; a < m; ++a) s += out[a];
        const int t[1] = {0};
        D.template logpdf_dk<1, 0>(st, yi, xi, t, out, wi);
        s += out[0];
      } else {
        D.template logpdf_dk<0, 2>(st, yi, xi, nullptr, out, wi);
        s += out[0];
        const int t[2] = {0, 0};
        D.template logpdf_dk<1, 1>(st, yi, xi, t, out, wi);
        s += out[0];
        D.template logpdf_dk<2, 0>(st, yi, xi, t, out, wi);
        s += out[0];
      }
    } else {
      const ld::multinomial_distrib M(d, 50.0);
      const auto st = M.theta_state(nullptr, wi);
      if (order == 0) {
        s = M.logpmf(st, yi, xi);
      } else if (order == 1) {
        M.template logpmf_dk<0, 1>(st, yi, xi, nullptr, out, wi);
        for (int a = 0; a < m; ++a) s += out[a];
      } else {
        M.template logpmf_dk<0, 2>(st, yi, xi, nullptr, out, wi);
        s += out[0];
      }
    }
    sink(i) = s;
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("dirichlet and multinomial, %d units, %s\n", n_units, Exec::name());
    std::mt19937_64 gen(20261117);
    std::normal_distribution<double> nd(0.0, 0.5);
    for (int d : {3, 5, 10}) {
      Mat x("x", n_units, d - 1), y1("y1", n_units, d), y2("y2", n_units, d);
      auto xh = Kokkos::create_mirror_view(x);
      auto y1h = Kokkos::create_mirror_view(y1);
      auto y2h = Kokkos::create_mirror_view(y2);
      for (int i = 0; i < n_units; ++i) {
        double s = 1.0;
        for (int a = 0; a + 1 < d; ++a) s += std::exp(xh(i, a) = nd(gen));
        double left = 50.0, ys = 0.0;
        for (int j = 0; j < d; ++j) {
          const double mu = (j + 1 < d ? std::exp(xh(i, j)) : 1.0) / s;
          y1h(i, j) = mu;  // a point of the simplex near the mean
          ys += mu;
          const double c = j + 1 < d ? std::floor(50.0 * mu) : left;
          y2h(i, j) = c;
          left -= c;
        }
        for (int j = 0; j < d; ++j) y1h(i, j) /= ys;
      }
      Kokkos::deep_copy(x, xh);
      Kokkos::deep_copy(y1, y1h);
      Kokkos::deep_copy(y2, y2h);
      const int ws = std::max(ld::dirichlet_distrib(d).scratch_size(), ld::multinomial_distrib(d, 50.0).scratch_size());
      Mat w("w", n_units, ws);
      Kokkos::View<double*> sink("sink", n_units);
      for (int law = 1; law <= 2; ++law) {
        std::printf("  %-11s d = %2d:", law == 1 ? "dirichlet" : "multinomial", d);
        for (int order = 0; order <= 2; ++order) {
          const Work k{law, d, order, x, law == 1 ? y1 : y2, w, sink};
          Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n_units), k);  // warm-up
          Kokkos::fence();
          double best = 1e300;
          for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(0, n_units), k);
            Kokkos::fence();
            const double t = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            best = t < best ? t : best;
          }
          std::printf("  order %d %9.1f ns/unit", order, best / n_units);
        }
        std::printf("\n");
      }
    }
  }
  Kokkos::finalize();
}
