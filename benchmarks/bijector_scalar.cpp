// Speed of the scalar bijectors (include/laplace/bijector/scalar.hpp), every
// map, on one host thread (Serial) and on the default execution space (OpenMP
// in the gcc build, CUDA in the CUDA build): ns per call of the main member
// functions, at 2^16 points drawn inside the domain of each direction, with
// typical parameters. Accuracy is measured by the tests.
//
//   bijector_scalar [map ...]     (all maps by default)
#include "measure.hpp"

#include <laplace/bijector/scalar.hpp>

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace lb = laplace::bijector;
using map_t = lb::scalar_bijector_map;

namespace {

// the functions timed, by number
const char* fname[] = {"forward",         "forward_d1",           "forward_d6",           "inverse",
                       "inverse_d1",      "inverse_d6",           "fwd_log_jacobian",     "inv_log_jacobian",
                       "inv_log_jac_d6",  "inverse_partials_d1",  "inverse_partials_d6",  "inverse_distance_upper"};
constexpr int nf = 12;
// whether the function takes the argument of g (v) or of h (u)
const bool on_v[] = {true, true, true, false, false, false, true, false, false, false, false, false};

template <int K>
KOKKOS_INLINE_FUNCTION double sum_partials(const lb::bijector_partials<K>& r) {
  double s = 0;
  for (int i = 0; i < lb::bijector_partials<K>::size; ++i) s += r.c[i];
  return s;
}

struct Call {
  lb::scalar_bijector b;
  int f;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (f) {
      case 0: return b.forward(x);
      case 1: return b.forward_d1(x);
      case 2: return b.forward_d6(x);
      case 3: return b.inverse(x);
      case 4: return b.inverse_d1(x);
      case 5: return b.inverse_d6(x);
      case 6: return b.forward_log_jacobian(x);
      case 7: return b.inverse_log_jacobian(x);
      case 8: return b.inverse_log_jacobian_d6(x);
      case 9: return sum_partials(b.inverse_partials_d1(x));
      case 10: return sum_partials(b.inverse_partials_d6(x));
      default: return b.inverse_distance_upper(x);
    }
  }
};

struct Named {
  const char* name;
  lb::scalar_bijector b;
};

std::vector<Named> all_maps() {
  return {{"identity", {map_t::identity}},
          {"log", {map_t::log}},
          {"logit", {map_t::logit}},
          {"probit", {map_t::probit}},
          {"cauchit", {map_t::cauchit}},
          {"cloglog", {map_t::cloglog}},
          {"loglog", {map_t::loglog}},
          {"inverse", {map_t::inverse}},
          {"inverse_sq", {map_t::inverse_sq}},
          {"sqrt", {map_t::sqrt}},
          {"atanh", {map_t::atanh}},
          {"asinh", {map_t::asinh}},
          {"affine", lb::scalar_bijector::affine(1.0, 2.0)},
          {"power", lb::scalar_bijector::power(1.7)},
          {"softplus", lb::scalar_bijector::softplus(1.0)},
          {"bounded", lb::scalar_bijector::bounded(-1.0, 3.0)},
          {"lower_bounded", lb::scalar_bijector::lower_bounded(2.0)},
          {"upper_bounded", lb::scalar_bijector::upper_bounded(5.0)},
          {"box_cox", lb::scalar_bijector::box_cox(0.3)},
          {"yeo_johnson", lb::scalar_bijector::yeo_johnson(0.6)}};
}

// a point inside [lo, hi], away from the bounds
double draw(std::mt19937_64& g, double lo, double hi) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double t = u(g);
  if (std::isfinite(lo) && std::isfinite(hi)) return lo + (hi - lo) * (0.02 + 0.96 * t);
  if (std::isfinite(lo)) return lo + std::exp(-4.0 + 8.0 * t);
  if (std::isfinite(hi)) return hi - std::exp(-4.0 + 8.0 * t);
  return -10.0 + 20.0 * t;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    bench::print_spaces();
    std::vector<std::string> want;
    for (int i = 1; i < argc; ++i)
      if (argv[i][0] != '-') want.push_back(argv[i]);
    std::printf("%-14s %-24s %12s %14s\n", "map", "function", "Serial ns", (std::string(bench::Exec::name()) + " ns").c_str());
    for (const auto& m : all_maps()) {
      if (!want.empty() && std::find(want.begin(), want.end(), std::string(m.name)) == want.end()) continue;
      for (int side = 0; side < 2; ++side) {
        const double lo = side == 0 ? m.b.forward_domain_lower() : m.b.inverse_domain_lower();
        const double hi = side == 0 ? m.b.forward_domain_upper() : m.b.inverse_domain_upper();
        std::mt19937_64 g(20261112 + side);
        const auto t = bench::make_time_args(1 << 16, [&] { return draw(g, lo, hi); });
        for (int f = 0; f < nf; ++f) {
          if (on_v[f] != (side == 0)) continue;
          const Call c{m.b, f};
          const double ns1 = bench::time_ns<bench::HostOne>(t.host, c, 5);
          const double nsd = bench::time_ns<bench::Exec>(t.dev, c, 7);
          std::printf("%-14s %-24s %12.2f %14.3f\n", m.name, fname[f], ns1, nsd);
        }
      }
    }
  }
  Kokkos::finalize();
}
