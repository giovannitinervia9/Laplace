// Accuracy and speed of the von Mises map from the circular standard deviation
// s to the concentration kappa, with its derivatives to order six
// (include/laplace/distrib/vonmises_scale.hpp).
//
// What it answers. The map is built from three routes, a series in v = s^2 for
// a concentrated law, the inverse of the Bessel ratio read from whichever of
// rho and its complement is small, and a series in rho for a nearly uniform
// one. Two questions follow: where the switch points belong, and what one call
// costs, since the whole von Mises derivative tensor is this map plus
// elementary trigonometry. So each route is measured over the whole range and
// not only inside its own band, which is what shows where it stops being worth
// using, and the joint routine is timed against the six separate calls.
//
// Usage: distrib_vonmises_scale <reference dir> <output dir> [name filter]
#include "measure.hpp"

#include <laplace/distrib/vonmises_scale.hpp>

#include <random>

namespace {

namespace ld = laplace::distrib;
namespace dv = laplace::distrib::detail_vonmises;

// The chosen map: the value and one derivative order, through the public
// per-order functions.
struct Chosen {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    switch (n) {
      case 0: return ld::vonmises_kappa_from_s(s);
      case 1: return ld::vonmises_kappa_from_s_dk<1>(s);
      case 2: return ld::vonmises_kappa_from_s_dk<2>(s);
      case 3: return ld::vonmises_kappa_from_s_dk<3>(s);
      case 4: return ld::vonmises_kappa_from_s_dk<4>(s);
      case 5: return ld::vonmises_kappa_from_s_dk<5>(s);
      default: return ld::vonmises_kappa_from_s_dk<6>(s);
    }
  }
};

// One route forced over the whole range, which is how the switch points are
// placed: each series is compared with the inverse where both are defined.
struct Small {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    if (!(s > 0.0)) return s * NAN;
    switch (n) {
      case 0: return dv::value_small(s);
      case 1: return dv::deriv_small<1>(s);
      case 2: return dv::deriv_small<2>(s);
      case 3: return dv::deriv_small<3>(s);
      case 4: return dv::deriv_small<4>(s);
      case 5: return dv::deriv_small<5>(s);
      default: return dv::deriv_small<6>(s);
    }
  }
};
struct Large {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    if (!(s > 0.0)) return s * NAN;
    switch (n) {
      case 0: return dv::value_large(s);
      case 1: return dv::deriv_large<1>(s);
      case 2: return dv::deriv_large<2>(s);
      case 3: return dv::deriv_large<3>(s);
      case 4: return dv::deriv_large<4>(s);
      case 5: return dv::deriv_large<5>(s);
      default: return dv::deriv_large<6>(s);
    }
  }
};
struct Middle {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    if (!(s > 0.0)) return s * NAN;
    const double kappa = dv::value_middle(s);
    switch (n) {
      case 0: return kappa;
      case 1: return dv::deriv_middle<1>(s, kappa);
      case 2: return dv::deriv_middle<2>(s, kappa);
      case 3: return dv::deriv_middle<3>(s, kappa);
      case 4: return dv::deriv_middle<4>(s, kappa);
      case 5: return dv::deriv_middle<5>(s, kappa);
      default: return dv::deriv_middle<6>(s, kappa);
    }
  }
};

// The joint routine, read at one order, so that its accuracy is compared with
// the per-order functions on the same references.
struct Joint {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    double out[7];
    ld::vonmises_kappa_from_s_upto<6>(s, out);
    return out[n];
  }
};

// All seven values from one call, summed, for the timing of the joint routine
// against the seven separate calls.
struct JointAll {
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    double out[7];
    ld::vonmises_kappa_from_s_upto<6>(s, out);
    double acc = 0.0;
    for (int n = 0; n <= 6; ++n) acc += out[n];
    return acc;
  }
};
struct SeparateAll {
  KOKKOS_INLINE_FUNCTION double operator()(double s) const {
    return ld::vonmises_kappa_from_s(s) + ld::vonmises_kappa_from_s_dk<1>(s) +
           ld::vonmises_kappa_from_s_dk<2>(s) + ld::vonmises_kappa_from_s_dk<3>(s) +
           ld::vonmises_kappa_from_s_dk<4>(s) + ld::vonmises_kappa_from_s_dk<5>(s) +
           ld::vonmises_kappa_from_s_dk<6>(s);
  }
};

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc < 3) {
    std::fprintf(stderr, "usage: distrib_vonmises_scale <reference dir> <output dir> [name filter]\n");
    return 2;
  }
  const std::string refdir = argv[1], outdir = argv[2];
  const std::string filter = argc > 3 ? argv[3] : "";
  bench::print_spaces();
  const std::size_t n = std::size_t(1) << 18;
  std::mt19937_64 gen(20261115);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  // s: half log-uniform over the whole working range (2^-27 to 2^2), half
  // uniform on the band where the inverse of the ratio does the work, which is
  // where the cost sits.
  const auto t_s = bench::make_time_args(n, [&] {
    return u01(gen) < 0.5 ? std::exp2(-27.0 + 29.0 * u01(gen)) : 0.05 + 3.95 * u01(gen);
  });

  auto run = [&](const std::string& name, const std::string& impl, auto f) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    bench::report(name.c_str(), impl.c_str(), bench::read_ref(refdir + "/" + name + ".csv"), f, outdir, t_s);
  };

  // The chosen map and the joint routine against the references, then each
  // single route over the whole grid: a route's error outside its own band is
  // what justifies the switch point, and inside it the three must agree.
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "vonmises_kappa" : "vonmises_kappa_d" + std::to_string(k);
    run(name, "own", Chosen{k});
    run(name, "joint", Joint{k});
    run(name, "series_v", Small{k});
    run(name, "series_rho", Large{k});
    run(name, "inverse", Middle{k});
  }

  // The joint routine against the seven separate calls, on the value's
  // reference only, since what is wanted here is the time and not the error.
  run("vonmises_kappa", "upto6_joint", JointAll{});
  run("vonmises_kappa", "upto6_separate", SeparateAll{});
  return 0;
}
