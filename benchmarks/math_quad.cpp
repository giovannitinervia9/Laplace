// Accuracy, evaluations and speed of the quadrature rules
// (include/laplace/math/quadrature.hpp) on the integrals of
// tools/reference/quadrature.py, per integrand and rule.
//
// Usage: math_quad <reference dir> [stop ...]
#include "quad_common.hpp"

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: math_quad <reference dir> [stop ...]\n");
    return 2;
  }
  bench::print_spaces();
  const auto r = qbench::read(std::string(argv[1]) + "/quadrature.csv");
  std::vector<double> stops;
  for (int i = 2; i < argc; ++i)
    if (argv[i][0] != '-') stops.push_back(std::atof(argv[i]));
  if (stops.empty()) stops.push_back(1e-9);
  static const char* names[] = {"beta", "gamma", "gauss_cos", "student", "laplace", "log2", "skewnormal", "huber", "sech"};
  for (double stop : stops) {
    std::printf("stop = %g\n", stop);
    for (int m = 0; m < qbench::NMETH; ++m) {
      Kokkos::fence();
      const auto t0 = std::chrono::steady_clock::now();
      const auto o = qbench::eval(r, m, stop);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      for (int id = 0; id < 9; ++id) {
        double worst = 0, wp1 = 0, wp2 = 0, ev = 0;
        int cnt = 0, nconv = 0, bad = 0;
        for (std::size_t i = 0; i < r.id.size(); ++i) {
          if (r.id[i] != id || o.value[i] != o.value[i]) continue;
          ++cnt;
          ev += o.evals[i];
          if (!o.conv[i]) ++nconv;
          const double e = std::fabs((o.value[i] - r.hi[i]) - r.lo[i]) / bench::ulp_of(r.hi[i]);
          if (!std::isfinite(e)) {
            ++bad;
            continue;
          }
          if (e > worst) worst = e, wp1 = r.p1[i], wp2 = r.p2[i];
        }
        if (cnt == 0) continue;
        std::printf("  %-10s %-10s max %10.4g ulp (p %.4g, %.4g) | evals mean %6.1f | not converged %d | non-finite %d\n",
                    qbench::method_name(m), names[id], worst, wp1, wp2, ev / cnt, nconv, bad);
      }
      std::size_t napp = 0;
      for (double v : o.value) napp += v == v;
      std::printf("  %-10s %zu applicable rows: %.2f us per integral (kernel and transfers)\n", qbench::method_name(m),
                  napp, 1e6 * secs / (napp ? napp : 1));
    }
  }
  // throughput: the rows repeated 128 times, the default stop
  {
    qbench::Ref big;
    for (int k = 0; k < 128; ++k)
      for (std::size_t i = 0; i < r.id.size(); ++i) {
        big.id.push_back(r.id[i]), big.p1.push_back(r.p1[i]), big.p2.push_back(r.p2[i]);
        big.hi.push_back(r.hi[i]), big.lo.push_back(r.lo[i]);
      }
    std::printf("throughput on %zu integrals (the rows 128 times):\n", big.id.size());
    for (int m = 0; m < qbench::NMETH; ++m) {
      qbench::eval(big, m, 1e-9);  // warm up
      Kokkos::fence();
      const auto t0 = std::chrono::steady_clock::now();
      const auto o = qbench::eval(big, m, 1e-9);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::size_t napp = 0;
      for (double v : o.value) napp += v == v;
      std::printf("  %-10s %.3f us per integral\n", qbench::method_name(m), 1e6 * secs / (napp ? napp : 1));
    }
  }
  return 0;
}
