// Generic expectations of a distribution (laplace::distrib; note/distrib.md
// section 4): E[h(Y)] by quadrature over the support (continuous) or by a sum
// over the support (discrete), and the moments built on it. They read only
// logpdf / logpmf, the support, center() and scale(), and y_kinks() when the
// distribution has it.
//
//   expectation_by_integral(d, h)        E[h(Y)], as a quad_result
//   moment_by_integral(d, k, about)      E[(Y - about)^k]
//   detail_generic::expectation_vec_by_integral<M>(d, h)
//                                        M expectations on the same nodes or terms
//
// Continuous: panels_quad over (support_lower, support_upper), cut at center()
// and at the kinks in y, panels of width scale(). Discrete: the sum runs from
// floor(center()) upwards and downwards; a direction stops at the end of the
// support or, once it is more than 2 scale() from the centre and the mass
// decreases (ratio r of two successive masses below 1), when the geometric bound
// (|t_y| + |t_(y-1)|) / (1 - r) on what is left falls below 2^-60 of the sum of
// |t| (two terms, so that a zero of h does not stop it). The bound holds when the
// mass ratio does not rise again further out, which is true for log-concave
// masses (Poisson, binomial, negative binomial).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_generic {

// l(y) of either kind
template <class D>
LAPLACE_HD_FORCEINLINE double log_density(const D& d, double y) {
  if constexpr (D::kind == distrib_kind::continuous) return d.logpdf(y);
  else return d.logpmf(y);
}

// d^K l / dtheta^K of either kind
template <int K, class D>
LAPLACE_HD_FORCEINLINE sym_tensor<D::n_params, K> log_density_dk(const D& d, double y) {
  constexpr bool c = D::kind == distrib_kind::continuous;
  if constexpr (K == 1) { if constexpr (c) return d.logpdf_d1(y); else return d.logpmf_d1(y); }
  else if constexpr (K == 2) { if constexpr (c) return d.logpdf_d2(y); else return d.logpmf_d2(y); }
  else if constexpr (K == 3) { if constexpr (c) return d.logpdf_d3(y); else return d.logpmf_d3(y); }
  else if constexpr (K == 4) { if constexpr (c) return d.logpdf_d4(y); else return d.logpmf_d4(y); }
  else if constexpr (K == 5) { if constexpr (c) return d.logpdf_d5(y); else return d.logpmf_d5(y); }
  else { if constexpr (c) return d.logpdf_d6(y); else return d.logpmf_d6(y); }
}

// one direction of the discrete sum, from y0 by step +1 or -1, to the end e (inclusive)
template <class D, class H>
LAPLACE_HD void discrete_sweep(const D& d, const H& h, double y0, double e, int step, double c, double s,
                               math::quad::quad_result& r) {
  double prev_t = 0.0, prev_m = 0.0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double y = y0 + step * double(n);
    if (step > 0 ? y > e : y < e) return;
    const double m = math::exp(log_density(d, y));
    const double t = m == 0.0 ? 0.0 : h(y) * m;
    ++r.evals;
    if (!(Kokkos::fabs(t) < 1e308)) {  // a non-finite term: no bound can stop the sum
      r.value = t;
      r.converged = false;
      return;
    }
    r.value += t;
    r.error += Kokkos::fabs(t);  // the sum of |t| for now
    const bool far = Kokkos::fabs(y - c) > 2.0 * s + 2.0;
    if (far && m == 0.0) return;
    if (far && n > 0) {
      const double ratio = m / prev_m;
      if (ratio < 1.0 && (Kokkos::fabs(t) + Kokkos::fabs(prev_t)) <= 0x1p-60 * r.error * (1.0 - ratio)) return;
    }
    prev_t = t, prev_m = m;
  }
  r.converged = false;  // ponytail: cap of 2^26 terms a direction, enough for scale() up to ~1e6
}

// M expectations on one set of nodes (used by the expected information, whose
// components share every evaluation of the derivatives of l): value[m] and whether
// component m converged
template <int M>
struct vec_expectation {
  double value[M] = {};
  bool converged[M] = {};
  int evals = 0;
};

// the double-exponential rule of math::quad::detail::de_rule on M components:
// the same nodes, steps, trimming and stopping test, a level refined until every
// component passes its own test; h(x, out) writes the M values at x. The trimming
// of level 0 re-evaluates the outer nodes instead of storing M values per node.
template <int M, class H, class Node>
LAPLACE_HD void de_rule_vec(const H& h, const Node& node, vec_expectation<M>& out) {
  using math::quad::detail::t_max;
  const math::quad::de_quad_options o{};
  double s[M] = {}, l1[M] = {}, v[M], prev[M];
  bool conv[M] = {};
  auto eval = [&](double t) {  // v = f w at the node t; false for a node to skip
    double x, w, xc;
    if (node(t, x, w, xc) == 0) return false;
    h(x, v);
    ++out.evals;
    for (int m = 0; m < M; ++m) v[m] = Kokkos::fabs(v[m]) < 1e308 ? v[m] * w : 0.0;
    return true;
  };
  double step = 0.5;
  const int K = int(t_max / step);
  for (int k = -K; k <= K; ++k)
    if (eval(k * step))
      for (int m = 0; m < M; ++m) s[m] += v[m], l1[m] += Kokkos::fabs(v[m]);
  auto counts = [&](int k) {
    if (!eval(k * step)) return false;
    for (int m = 0; m < M; ++m)
      if (Kokkos::fabs(v[m]) > l1[m] * 0x1p-64) return true;
    return false;
  };
  int lo = -K, hi = K;
  while (lo < 0 && !counts(lo)) ++lo;
  while (hi > 0 && !counts(hi)) --hi;
  const double t_lo = Kokkos::fmax(-t_max, (lo - 1) * step), t_hi = Kokkos::fmin(t_max, (hi + 1) * step);
  for (int m = 0; m < M; ++m) prev[m] = s[m] * step;
  for (int level = 1; level <= o.max_level; ++level) {
    step *= 0.5;
    const int k0 = int(Kokkos::ceil(t_lo / step)), k1 = int(Kokkos::floor(t_hi / step));
    for (int k = k0; k <= k1; ++k)
      if (k % 2 != 0 && eval(k * step))
        for (int m = 0; m < M; ++m) s[m] += v[m], l1[m] += Kokkos::fabs(v[m]);
    bool all = true;
    for (int m = 0; m < M; ++m) {
      const double est = s[m] * step;
      conv[m] = Kokkos::fabs(est - prev[m]) <= o.stop * l1[m] * step;
      prev[m] = est;
      all = all && conv[m];
    }
    if (all) break;
  }
  for (int m = 0; m < M; ++m) {
    out.value[m] += prev[m];
    out.converged[m] = out.converged[m] && conv[m];
  }
}

// math::quad::panels_quad on M components, for a < b: the same cuts and panels
template <int M, class H>
LAPLACE_HD void panels_vec(const H& h, double a, double b, double c, double s, const double* kinks, int nk,
                           vec_expectation<M>& r) {
  namespace qd = math::quad::detail;
  for (int m = 0; m < M; ++m) r.converged[m] = true;
  if (!(a < b) || !qd::finite(c) || !qd::scale_ok(s)) {
    for (int m = 0; m < M; ++m) r.converged[m] = false;
    return;
  }
  double cut[10];
  int nc = 0;
  if (c > a && c < b) cut[nc++] = c;
  for (int i = 0; i < nk && i < 8; ++i)
    if (kinks[i] > a && kinks[i] < b) cut[nc++] = kinks[i];
  for (int i = 1; i < nc; ++i)
    for (int j = i; j > 0 && cut[j] < cut[j - 1]; --j) {
      const double t = cut[j];
      cut[j] = cut[j - 1], cut[j - 1] = t;
    }
  auto ts = [&](double u, double v) {
    if (u < v) de_rule_vec<M>(h, qd::tanh_sinh_node{u, v}, r);
  };
  auto finite = [&](double u, double v) {
    const bool from_u = Kokkos::fabs(u - c) <= Kokkos::fabs(v - c);
    double width = s, at = from_u ? u : v;
    for (int k = 0; k < 64; ++k) {
      const double next = from_u ? at + width : at - width;
      const bool last = !(width > 0.0) || next == at || (from_u ? !(next < v) : !(next > u));
      if (last) {
        from_u ? ts(at, v) : ts(u, at);
        return;
      }
      from_u ? ts(at, next) : ts(next, at);
      at = next;
      width *= 2.0;
    }
  };
  auto right_tail = [&](double u) { de_rule_vec<M>(h, qd::exp_sinh_node{u, s}, r); };
  auto left_tail = [&](double v) {
    de_rule_vec<M>([&](double x, double* out) { h(-x, out); }, qd::exp_sinh_node{-v, s}, r);
  };
  const bool inf_lo = !(a > -1e308), inf_hi = !(b < 1e308);
  double ends[12];
  int ne = 0;
  ends[ne++] = a;
  for (int i = 0; i < nc; ++i) ends[ne++] = cut[i];
  ends[ne++] = b;
  for (int i = 0; i + 1 < ne; ++i) {
    const double u = ends[i], v = ends[i + 1];
    if (i == 0 && inf_lo) {
      if (inf_hi && ne == 2) {  // the whole line without cuts: split at c
        finite(c, c + 63.0 * s);
        right_tail(c + 63.0 * s);
        finite(c - 63.0 * s, c);
        left_tail(c - 63.0 * s);
        return;
      }
      finite(v - 63.0 * s, v);
      left_tail(v - 63.0 * s);
    } else if (i + 2 == ne && inf_hi) {
      finite(u, u + 63.0 * s);
      right_tail(u + 63.0 * s);
    } else {
      finite(u, v);
    }
  }
}

// discrete_sweep on M components: a direction stops when every component passes its own bound
template <int M, class D, class H>
LAPLACE_HD void discrete_sweep_vec(const D& d, const H& h, double y0, double e, int step, double c, double s,
                                   double* l1, vec_expectation<M>& r) {
  double prev_t[M] = {}, t[M], prev_m = 0.0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double y = y0 + step * double(n);
    if (step > 0 ? y > e : y < e) return;
    const double mass = math::exp(log_density(d, y));
    if (mass == 0.0) {
      for (int m = 0; m < M; ++m) t[m] = 0.0;
    } else {
      h(y, t);
      for (int m = 0; m < M; ++m) t[m] *= mass;
    }
    ++r.evals;
    for (int m = 0; m < M; ++m) {
      if (!(Kokkos::fabs(t[m]) < 1e308)) {  // a non-finite term: no bound can stop this component
        r.converged[m] = false;
        t[m] = 0.0;
      }
      r.value[m] += t[m];
      l1[m] += Kokkos::fabs(t[m]);
    }
    const bool far = Kokkos::fabs(y - c) > 2.0 * s + 2.0;
    if (far && mass == 0.0) return;
    if (far && n > 0) {
      const double ratio = mass / prev_m;
      bool done = ratio < 1.0;
      for (int m = 0; m < M && done; ++m)
        done = (Kokkos::fabs(t[m]) + Kokkos::fabs(prev_t[m])) <= 0x1p-60 * l1[m] * (1.0 - ratio);
      if (done) return;
    }
    for (int m = 0; m < M; ++m) prev_t[m] = t[m];
    prev_m = mass;
  }
  for (int m = 0; m < M; ++m) r.converged[m] = false;  // the cap of 2^26 terms a direction
}

// E[h_m(Y)], m = 0..M-1, on one set of nodes or one sum; h(y, out) writes the M values
template <int M, class D, class H>
LAPLACE_HD vec_expectation<M> expectation_vec_by_integral(const D& d, const H& h) {
  const double lo = d.support_lower(), hi = d.support_upper(), c = d.center(), s = d.scale();
  vec_expectation<M> r;
  if constexpr (D::kind == distrib_kind::continuous) {
    auto g = [&](double y, double* out) {
      const double mass = math::exp(d.logpdf(y));
      if (mass == 0.0) {
        for (int m = 0; m < M; ++m) out[m] = 0.0;
        return;
      }
      h(y, out);
      for (int m = 0; m < M; ++m) out[m] *= mass;
    };
    if constexpr (requires { d.y_kinks(); }) {
      const distrib_kinks k = d.y_kinks();
      panels_vec<M>(g, lo, hi, c, s, k.point, k.count, r);
    } else {
      panels_vec<M>(g, lo, hi, c, s, nullptr, 0, r);
    }
  } else {
    for (int m = 0; m < M; ++m) r.converged[m] = true;
    double l1[M] = {};
    double y0 = Kokkos::floor(c);
    if (y0 < lo) y0 = lo;
    if (y0 > hi) y0 = hi;
    discrete_sweep_vec<M>(d, h, y0, hi, +1, c, s, l1, r);
    discrete_sweep_vec<M>(d, h, y0 - 1.0, lo, -1, c, s, l1, r);
  }
  return r;
}

}  // namespace detail_generic

// E[h(Y)] by quadrature (continuous) or by a sum over the support (discrete).
/**
 * \ingroup distrib_generic
 * \brief General expectation \f$\mathbb{E}[h(Y)]\f$ of a distribution by numerical quadrature or summation.
 *
 * Evaluates the expected value of an arbitrary integrand \f$h(y)\f$:
 * - For continuous distributions:
 *   \f[
 *   \mathbb{E}[h(Y)] = \int h(y) f(y) \, \mathrm{d}y
 *   \f]
 *   integrated by laplace::math::quad::panels_quad() over the support, from `support_lower()` to `support_upper()`, subdivided at `center()` and at `y_kinks()`, using panel scale `scale()`.
 * - For discrete distributions:
 *   \f[
 *   \mathbb{E}[h(Y)] = \sum_y h(y) p(y)
 *   \f]
 *   summed from \f$\lfloor \mathrm{center}() \rfloor\f$ upwards and downwards until the tail bound falls below \f$2^{-60}\f$ of the accumulated sum or \f$2^{26}\f$ terms are reached.
 *
 * \tparam D Distribution type providing `logpdf()` or `logpmf()`, support bounds, `center()`, and `scale()`.
 * \tparam H Integrand functor evaluating \f$y \mapsto h(y)\f$.
 * \param d Distribution object.
 * \param h Integrand functor.
 * \return A math::quad::quad_result containing the integral estimate, error estimate, evaluation count, and convergence status.
 *
 * \par Domain and special values
 * - Non-finite terms in the integrand or non-convergence result in `converged = false`.
 * - For discrete distributions, summation terminates if non-converged after \f$2^{26}\f$ terms per direction.
 *
 * \par Measured accuracy
 * Measured through the moments (see moment_by_integral()): 1.8 eps of the scale on gaussian1 and laplace, 3.0 on gamma1, 0.6 to 6.7 on the Poisson.
 *
 * \see moment_by_integral(), distrib_expected_info()
 */
template <class D, class H>
LAPLACE_HD math::quad::quad_result expectation_by_integral(const D& d, const H& h) {
  const double lo = d.support_lower(), hi = d.support_upper(), c = d.center(), s = d.scale();
  if constexpr (D::kind == distrib_kind::continuous) {
    auto g = [&](double y) {
      const double m = math::exp(d.logpdf(y));
      return m == 0.0 ? 0.0 : h(y) * m;
    };
    if constexpr (requires { d.y_kinks(); }) {
      const distrib_kinks k = d.y_kinks();
      return math::quad::panels_quad(g, lo, hi, c, s, k.point, k.count);
    } else {
      return math::quad::panels_quad(g, lo, hi, c, s);
    }
  } else {
    math::quad::quad_result r;
    r.converged = true;
    double y0 = Kokkos::floor(c);
    if (y0 < lo) y0 = lo;
    if (y0 > hi) y0 = hi;
    detail_generic::discrete_sweep(d, h, y0, hi, +1, c, s, r);
    detail_generic::discrete_sweep(d, h, y0 - 1.0, lo, -1, c, s, r);
    r.error = 0x1p-60 * r.error;  // the tail bound used
    return r;
  }
}

// E[(Y - about)^k], k >= 0, by expectation_by_integral.
/**
 * \ingroup distrib_generic
 * \brief Raw or central moment \f$\mathbb{E}[(Y - \mathrm{about})^k]\f$ of a distribution.
 *
 * Computes the moment of order \f$k\f$:
 * \f[
 * \mu_k = \mathbb{E}[(Y - \mathrm{about})^k]
 * \f]
 * via expectation_by_integral(). For raw moments, \f$\mathrm{about} = 0\f$; for central moments, \f$\mathrm{about} = \mathbb{E}[Y]\f$.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param k Non-negative moment order \f$k \ge 0\f$.
 * \param about Point about which the moment is taken (defaults to \f$0.0\f$).
 * \return The computed moment value, or quiet NaN if the integral does not converge.
 *
 * \par Domain and special values
 * - Returns quiet NaN if the underlying quadrature or summation fails to converge.
 *
 * \par Measured accuracy
 * Against closed forms in eps of the scale: 1.8 eps for gaussian1 and laplace, 3.0 eps for gamma1, and 0.6–6.7 eps for Poisson (\f$\mu \in [0.05, 1000]\f$).
 *
 * \see expectation_by_integral()
 */
template <class D>
LAPLACE_HD double moment_by_integral(const D& d, int k, double about = 0.0) {
  const auto r = expectation_by_integral(d, [&](double y) {
    double p = 1.0;
    for (int i = 0; i < k; ++i) p *= y - about;
    return p;
  });
  return r.converged ? r.value : math::detail::quiet_nan();
}

}  // namespace laplace::distrib
