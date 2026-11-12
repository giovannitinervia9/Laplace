// Shared by the bench and the test of the quadrature rules: the integrands of
// tools/reference/quadrature.py (same ids) and the evaluation of one rule on
// one of them.
#pragma once

#include "measure.hpp"

#include <laplace/math/normal.hpp>
#include <laplace/math/quadrature.hpp>

namespace qbench {

namespace quad = laplace::math::quad;
constexpr double inf = __builtin_huge_val();

// the beta integrand from x and the distance to the nearer end (tanh-sinh's
// two-argument form): log(1 - x) from the distance on the right half
struct Beta {
  double p1, p2;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double xc) const {
    const double l1mx = xc > 0.0 ? Kokkos::log(xc) : Kokkos::log1p(-x);
    return Kokkos::exp((p1 - 1.0) * Kokkos::log(x) + (p2 - 1.0) * l1mx);
  }
};

struct Integrand {
  int id;
  double p1, p2;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (id) {
      case 0: return Kokkos::exp((p1 - 1.0) * Kokkos::log(x) + (p2 - 1.0) * Kokkos::log1p(-x));
      case 1: return Kokkos::exp((p1 - 1.0) * Kokkos::log(x) - x);
      case 2: return Kokkos::exp(-0.5 * x * x) * Kokkos::cos(p1 * x);
      case 3: return Kokkos::exp(-0.5 * (p1 + 1.0) * Kokkos::log1p(x * x / p1));
      case 4: return Kokkos::exp(-Kokkos::fabs(x));
      case 5: {
        const double l = Kokkos::log(x);
        return Kokkos::exp(p1 * l) * l * l;
      }
      case 6: {
        if (x == 0.0) return 0.0;
        const double ax = p1 * x;
        // 2 x^2 phi(x) phi(a x)^2 / Phi(a x), in logs
        return Kokkos::exp(Kokkos::log(2.0 * x * x) - 0.5 * x * x - ax * ax - 2.7568155996140185 -
                           laplace::math::log_Phi(ax));
      }
      case 7: return Kokkos::exp(-p1 * x * x / (1.0 + Kokkos::sqrt(1.0 + x * x)));
      default: {
        const double ax = Kokkos::fabs(x);  // sech^p = exp(-p log cosh x)
        return Kokkos::exp(-p1 * (ax + Kokkos::log1p(Kokkos::exp(-2.0 * ax)) - 0.69314718055994530942));
      }
    }
  }
};

// the methods
enum { TS = 0, ES, SS, GK, PAN, GL64, GH32, GLA32, NMETH };
inline const char* method_name(int m) {
  static const char* n[] = {"tanh_sinh_quad", "exp_sinh_quad", "sinh_sinh_quad", "gk15", "panels", "legendre64", "hermite32", "laguerre32"};
  return n[m];
}

// the domain of each integrand: 0 = [p1-dependent finite], 1 = [0, inf), 2 = R
KOKKOS_INLINE_FUNCTION int domain(int id) {
  switch (id) {
    case 0: case 4: case 5: return 0;
    case 1: case 3: return 1;
    default: return 2;
  }
}

// value, evaluations, converged (NaN when the rule does not apply)
KOKKOS_INLINE_FUNCTION quad::quad_result run(int id, double p1, double p2, int m, double stop) {
  const Integrand f{id, p1, p2};
  quad::de_quad_options o;
  o.stop = stop;
  quad::quad_result r;
  r.value = laplace::math::detail::quiet_nan();
  const int dom = domain(id);
  const double a = id == 4 ? p1 : 0.0, b = id == 4 ? p2 : 1.0;
  // centre and scale of each integrand
  double c = 0.0, s = 1.0;
  if (id == 1) c = p1 > 1.0 ? p1 - 1.0 : 0.0, s = Kokkos::sqrt(p1 > 1.0 ? p1 : 1.0);
  if (id == 3) s = Kokkos::sqrt(p1);
  if (id == 6) s = 1.0 / (1.0 + Kokkos::fabs(p1));
  if (id == 7) s = p1 < 1.0 ? 1.0 / p1 : 1.0 / Kokkos::sqrt(p1);
  if (id == 8) s = p1 < 1.0 ? 1.0 / p1 : 1.0 / Kokkos::sqrt(p1);
  const double kink = 0.0;
  switch (m) {
    case TS:
      if (id == 0) r = quad::tanh_sinh_quad(Beta{p1, p2}, a, b, o);
      else if (dom == 0) r = quad::tanh_sinh_quad(f, a, b, o);
      break;
    case ES: if (dom == 1) r = quad::exp_sinh_quad(f, 0.0, s, o); break;
    case SS: if (dom == 2) r = quad::sinh_sinh_quad(f, c, s, o); break;
    case GK: if (dom == 0) r = quad::gauss_kronrod_quad(f, a, b, 1e-14); break;
    case PAN:
      if (dom == 0) r = quad::panels_quad(f, a, b, id == 4 ? 0.0 : 0.5 * (a + b), id == 4 ? 1.0 : 0.25, id == 4 ? &kink : nullptr, id == 4 ? 1 : 0, o);
      if (dom == 1) r = quad::panels_quad(f, 0.0, inf, c, s, nullptr, 0, o);
      if (dom == 2) r = quad::panels_quad(f, -inf, inf, c, s, nullptr, 0, o);
      break;
    case GL64: if (dom == 0) r.value = quad::gauss_legendre_quad<64>(f, a, b), r.evals = 64, r.converged = true; break;
    case GH32: if (dom == 2) r.value = quad::gauss_hermite32_quad(f, c, s), r.evals = 32, r.converged = true; break;
    case GLA32: if (dom == 1) r.value = quad::gauss_laguerre32_quad(f, 0.0, s), r.evals = 32, r.converged = true; break;
  }
  return r;
}

struct Ref {
  std::vector<int> id;
  std::vector<double> p1, p2, hi, lo;
};
inline Ref read(const std::string& path) {
  Ref r;
  const auto b = bench::read_ref(path);  // id, p1, p2 as the three arguments
  for (std::size_t i = 0; i < b.x.size(); ++i) {
    r.id.push_back(int(b.x[i][0])), r.p1.push_back(b.x[i][1]), r.p2.push_back(b.x[i][2]);
    r.hi.push_back(b.hi[i]), r.lo.push_back(b.lo[i]);
  }
  return r;
}

struct Out {
  std::vector<double> value;
  std::vector<int> evals, conv;
};
inline Out eval(const Ref& r, int m, double stop) {
  const std::size_t n = r.id.size();
  Kokkos::View<double* [3], Kokkos::LayoutRight, bench::Exec::memory_space> in("in", n);
  auto h = Kokkos::create_mirror_view(in);
  for (std::size_t i = 0; i < n; ++i) h(i, 0) = r.id[i], h(i, 1) = r.p1[i], h(i, 2) = r.p2[i];
  Kokkos::deep_copy(in, h);
  Kokkos::View<double* [3], Kokkos::LayoutRight, bench::Exec::memory_space> out("out", n);
  Kokkos::parallel_for(
      "quad", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t i) {
        const auto q = run(int(in(i, 0)), in(i, 1), in(i, 2), m, stop);
        out(i, 0) = q.value, out(i, 1) = q.evals, out(i, 2) = q.converged ? 1.0 : 0.0;
      });
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  Out o;
  for (std::size_t i = 0; i < n; ++i)
    o.value.push_back(ho(i, 0)), o.evals.push_back(int(ho(i, 1))), o.conv.push_back(int(ho(i, 2)));
  return o;
}

}  // namespace qbench
