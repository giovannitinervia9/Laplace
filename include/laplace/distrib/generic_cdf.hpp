// The cdf family of a distribution that has no closed form for it, and the
// dispatchers that take a distribution's own member when it has one
// (laplace::distrib; note/distrib.md section 4). A discrete distribution sums
// its mass, a continuous one integrates its density; the dispatchers choose by
// kind.
//
//   cdf_by_sum(d, y), ccdf_by_sum, log_cdf_by_sum, log_ccdf_by_sum
//   distrib_cdf(d, y), distrib_ccdf, distrib_log_cdf, distrib_log_ccdf
//   log_cdf_dk_by_sum<K>(d, y), log_ccdf_dk_by_sum<K>  the derivatives in the
//                               parameters of log F and log(1 - F), K = 1..6
//   cdf_by_integral(d, y), ccdf_by_integral, log_cdf_by_integral,
//   log_ccdf_by_integral, log_cdf_dk_by_integral<K>, log_ccdf_dk_by_integral<K>
//                               the same for a continuous distribution
//   distrib_log_cdf_dk<K>(d, y), distrib_log_ccdf_dk<K>
//
// With k = floor(y): below the centre the lower tail F(k) = sum_(i <= k) p(i) is
// summed downwards from k, and 1 - F(k) is its complement; from the centre up the
// upper tail 1 - F(k) = sum_(i > k) p(i) is summed upwards from k + 1, and F(k)
// is its complement. Each sum starts at its largest term, p(k) or p(k + 1), and
// adds e^(l(i) - l(start)), so that the logarithm of a tail far below the double
// range is still exact: log(tail) = l(start) + log(sum). A sum stops at the end of
// the support or when the geometric bound t r / (1 - r) on the rest (r the ratio
// of the last two terms, below 1) falls below 2^-60 of the sum: the terms
// decrease away from the mode, and the bound holds for log-concave masses.
//
// Continuous: the side of y that does not hold center() is integrated with
// panels_quad over it, cut at the kinks in y, in panels of width scale(), and
// the density at the end point y is divided out, so that a tail far below the
// double range still gives its logarithm exactly (log S = l(y) + log of the
// integral of f / f(y)). The other side is its complement on the log scale. The
// derivatives in the parameters come from the same identity as the discrete
// ones, d_T F = int f B_T(l), with the Bell polynomial of the derivatives of l
// built on their difference from l(y); it is valid where every parameter is
// smooth, since a parameter that moves a kink of l in y would leave a boundary
// term, and such a component is NaN.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_generic {

// log of the lower tail sum_(i <= k) (upper = false) or of the upper one sum_(i > k)
template <class D>
LAPLACE_HD double log_tail_by_sum(const D& d, double k, bool upper) {
  const double start = upper ? k + 1.0 : k, end = upper ? d.support_upper() : d.support_lower();
  const int step = upper ? 1 : -1;
  const double l0 = log_density(d, start);
  if (!(l0 > -distrib_infinity)) return l0;  // ponytail: a zero mass at the start means an empty tail (unimodal masses)
  double s = 0.0, prev = 0.0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double i = start + step * double(n);
    if (upper ? i > end : i < end) break;
    const double t = math::exp(log_density(d, i) - l0);
    s += t;
    if (n > 0) {
      const double r = t / prev;
      if (t == 0.0 || (r < 1.0 && t * r <= 0x1p-60 * s * (1.0 - r))) break;
    }
    prev = t;
  }
  return l0 + math::log(s);
}

// log F(k) or log(1 - F(k)) from the cheaper side
template <class D>
LAPLACE_HD double log_cdf_side_by_sum(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::discrete, "the cdf by summing the mass is for discrete distributions");
  const double k = Kokkos::floor(y);
  if (k < d.support_lower()) return upper ? 0.0 : -distrib_infinity;
  if (k >= d.support_upper()) return upper ? -distrib_infinity : 0.0;
  const bool lower_side = k < d.center();
  const double lt = log_tail_by_sum(d, k, !lower_side);
  if (lower_side != upper) return lt;  // the side asked for is the one summed
  return Kokkos::log1p(-math::exp(lt));
}

// The derivatives of a tail in the parameters. With S the tail summed,
// C = c[0..K-1] a sorted multi-index of parameters, T a sub-multiset of C (the
// bits of a mask over the positions of C) and B_T the complete Bell polynomial
// over T of the derivatives of a function (B_T = e^-g d_T e^g, B_empty = 1),
//
//   d_T S = sum_i p(i) B_T(l(i)),       l = log p,
//
// which summed as ratios to S itself would give d_T S / S directly. At high order
// in a tail those ratios are powers of the score (of size ((y - mu)/mu)^K on the
// Poisson) while d_T log S is of the size of d_T l alone, so the step from one to
// the other cancels: measured against the mpmath references of the Poisson's
// log F, up to 4e14 ulp at order 6. The sum is therefore centred on the first
// mass: with A = l(start) and m_i = l(i) - l(start),
//
//   S = e^A W,   W = sum_i e^(m_i),   d_T log S = d_T A + d_T log W,
//
// where d_T A is the mass's own derivative (exact and closed) and the derivatives
// of m_i, which vanish at the start, carry only the variation of l over the tail.
// This fills r[T] = d_T W / W (so r[empty] = 1) and l0[T] = d_T A, and returns
// log S. The walk and its stopping test are those of log_tail_by_sum, with the
// bound taken on the largest of the sums instead of the mass alone and asked for
// twice in a row: the mass decreases geometrically away from the mode and the
// |B_T| grow at most polynomially in i.
template <int K, class D>
LAPLACE_HD double tail_ratios_by_sum(const D& d, double k, bool upper, const int* c, double* r, double* l0) {
  constexpr int P = D::n_params;
  constexpr int full = (1 << K) - 1;
  const double start = upper ? k + 1.0 : k, end = upper ? d.support_upper() : d.support_lower();
  const int step = upper ? 1 : -1;
  const double la = log_density(d, start);
  for (int m = 0; m <= full; ++m) r[m] = l0[m] = 0.0;
  if (!(la > -distrib_infinity)) return la;
  log_density_partials<P, K> L0;
  L0.fill(d, start);
  for (int U = 1; U <= full; ++U) l0[U] = l_over(L0, nullptr, 0, c, U);  // d_T A, T not empty
  constexpr int n_partials = log_density_partials<P, K>::offset(K + 1);
  double A[1 << K], sabs[1 << K];
  for (int m = 0; m <= full; ++m) A[m] = sabs[m] = 0.0;
  double prev = 0.0;
  int quiet = 0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double i = start + step * double(n);
    if (upper ? i > end : i < end) break;
    const double w = math::exp(log_density(d, i) - la);
    log_density_partials<P, K> L;
    L.fill(d, i);
    for (int j = 0; j < n_partials; ++j) L.v[j] -= L0.v[j];  // the derivatives of m_i
    double B[1 << K], mc[1 << K];
    B[0] = 1.0;
    for (int U = 1; U <= full; ++U) mc[U] = l_over(L, nullptr, 0, c, U);
    for (int mask = 1; mask <= full; ++mask) {  // the block that holds the lowest index of T
      const int low = mask & -mask, rest = mask ^ low;
      double acc = 0.0;
      for (int sub = rest;; sub = (sub - 1) & rest) {
        const int U = sub | low;
        acc += mc[U] * B[mask ^ U];
        if (sub == 0) break;
      }
      B[mask] = acc;
    }
    double tmax = 0.0, amax = 0.0;
    for (int m = 0; m <= full; ++m) {
      const double t = w * B[m];
      A[m] += t;
      sabs[m] += Kokkos::fabs(t);
      if (Kokkos::fabs(t) > tmax) tmax = Kokkos::fabs(t);
      if (sabs[m] > amax) amax = sabs[m];
    }
    if (n > 0 && prev > 0.0) {
      const double ratio = w / prev;
      if (w == 0.0 || (ratio < 1.0 && tmax * ratio <= 0x1p-55 * amax * (1.0 - ratio))) {
        if (++quiet == 2) break;
      } else {
        quiet = 0;
      }
    }
    prev = w;
  }
  if (!(A[0] > 0.0)) return -distrib_infinity;
  for (int m = 0; m <= full; ++m) r[m] = A[m] / A[0];
  return la + math::log(A[0]);
}

// u[T] = d_T log G from r[T] = d_T G / G, the Bell relation solved for the
// derivatives of log G: r[T] = sum over the sub-multisets U of T that hold T's
// lowest index of u[U] r[T - U], and the term U = T leaves u[T] alone.
template <int K>
LAPLACE_HD void log_tail_dk_from_ratios(const double* r, double* u) {
  constexpr int full = (1 << K) - 1;
  u[0] = 0.0;
  for (int mask = 1; mask <= full; ++mask) {
    const int low = mask & -mask, rest = mask ^ low;
    double acc = r[mask];
    if (rest != 0) {
      for (int sub = (rest - 1) & rest;; sub = (sub - 1) & rest) {
        const int U = sub | low;
        acc -= u[U] * r[mask ^ U];
        if (sub == 0) break;
      }
    }
    u[mask] = acc;
  }
}

// the forward relation, r[T] = d_T G / G = B_T(u) from u[T] = d_T log G
template <int K>
LAPLACE_HD void ratios_from_log_tail_dk(const double* u, double* r) {
  constexpr int full = (1 << K) - 1;
  r[0] = 1.0;
  for (int mask = 1; mask <= full; ++mask) {
    const int low = mask & -mask, rest = mask ^ low;
    double acc = 0.0;
    for (int sub = rest;; sub = (sub - 1) & rest) {
      const int U = sub | low;
      acc += u[U] * r[mask ^ U];
      if (sub == 0) break;
    }
    r[mask] = acc;
  }
}

// d^K log F / dtheta^K (upper = false) or d^K log(1 - F) / dtheta^K (upper = true)
// by summing the mass: one walk per multi-index over the cheaper tail S. When the
// side asked for is the other one, O = 1 - S, its ratios follow from d_T O =
// -d_T S: d_T O / O = -(S/O) d_T S / S, with d_T S / S rebuilt from the
// derivatives of log S, and are small wherever S is (the side summed is the one
// of smaller probability, so S / O <= 1).
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_side_dk_by_sum(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::discrete,
                "the derivatives of the log cdf by summing the mass are for discrete distributions");
  constexpr int P = D::n_params;
  constexpr int full = (1 << K) - 1;
  sym_tensor<P, K> t;
  const double k = Kokkos::floor(y);
  const bool empty = upper ? k >= d.support_upper() : k < d.support_lower();
  if (empty) {  // the side asked for has probability 0: its logarithm is -inf
    for (int i = 0; i < sym_tensor<P, K>::size; ++i) t[i] = math::detail::quiet_nan();
    return t;
  }
  if (upper ? k < d.support_lower() : k >= d.support_upper()) return t;  // probability 1, every derivative 0
  const bool lower_side = k < d.center();
  const bool summed_is_asked = lower_side != upper;
  int c[K];
  for (int j = 0; j < K; ++j) c[j] = 0;
  for (int comp = 0; comp < sym_tensor<P, K>::size; ++comp) {
    double r[1 << K], u[1 << K], l0[1 << K];
    const double log_s = tail_ratios_by_sum<K>(d, k, !lower_side, c, r, l0);
    log_tail_dk_from_ratios<K>(r, u);                      // d_T log W
    for (int m = 1; m <= full; ++m) u[m] += l0[m];         // d_T log S = d_T A + d_T log W
    if (!summed_is_asked) {
      const double log_o = Kokkos::log1p(-math::exp(log_s));
      const double f = -math::exp(log_s - log_o);
      ratios_from_log_tail_dk<K>(u, r);
      for (int m = 1; m <= full; ++m) r[m] *= f;
      log_tail_dk_from_ratios<K>(r, u);
    }
    t[comp] = u[full];
    next_multi_index<P, K>(c);
  }
  return t;
}

// ---------------------------------------------------------------- continuous

// The position of a sorted multi-index of length k in the layout of
// log_density_partials<P, K>, which packs the derivatives order by order; the
// same arithmetic as that type's at(), kept here because the integrals below are
// indexed the same way.
template <int P, int K>
LAPLACE_HD_FORCEINLINE int mi_position(const int* s, int k) {
  int pos = 0, prev = 0;
  for (int j = 0; j < k; ++j) {
    for (int w = prev; w < s[j]; ++w) pos += binomial_coef(P - w + k - j - 2, k - j - 1);
    prev = s[j];
  }
  return log_density_partials<P, K>::offset(k) + pos;
}

// the next sorted multi-index of length k, with k at run time
template <int P>
LAPLACE_HD_FORCEINLINE void next_multi_index_k(int* c, int k) {
  int i = k - 1;
  while (i >= 0 && c[i] == P - 1) --i;
  if (i < 0) return;
  ++c[i];
  for (int j = i + 1; j < k; ++j) c[j] = c[i];
}

// The number of multi-indices of order 0 to K over P parameters, which is the
// length of the arrays below: B[0] and A[0] are the order-0 entry and
// B[1 + mi_position(U)] the one of U.
template <int P, int K>
LAPLACE_HD_FORCEINLINE constexpr int mi_count() {
  return 1 + log_density_partials<P, K>::offset(K + 1);
}

// The Bell recursion over the whole index space, as a table built once per call.
//
// B_U = d_U e^l / e^l, the complete Bell polynomial of the derivatives of l over
// the multiset U, obeys
//   B_U = sum_(V subset of U, V holds U's first slot) l_V B_(U - V),
// which fills B by increasing order. Where the sum is written out at every node
// of a quadrature, finding the positions of V and of U - V costs more than the
// arithmetic, so the positions are computed once into bell_table and the
// per-node loop is then a sequence of multiply-adds: for each multi-index U in
// the order of the layout, count[U] entries, each the position of l_V in the
// partials and the position of B_(U - V) (which is 0, where B is 1, when V is
// all of U).
template <int P, int K>
LAPLACE_HD_FORCEINLINE constexpr int bell_entries() {
  int e = 0;
  for (int k = 1; k <= K; ++k) e += sym_tensor_size(P, k) * (1 << (k - 1));
  return e;
}

template <int P, int K>
struct bell_table {
  static_assert(mi_count<P, K>() <= 256, "the table indexes the multi-indices in one byte");
  unsigned char lv[bell_entries<P, K>()];  // position of l_V in the partials
  unsigned char bw[bell_entries<P, K>()];  // position of B_(U - V), 0 for the empty complement
  unsigned char count[mi_count<P, K>()];   // entries of each multi-index, in layout order
  LAPLACE_HD bell_table() {
    for (int m = 0; m < mi_count<P, K>(); ++m) count[m] = 0;
    int e = 0, c[K];
    for (int k = 1; k <= K; ++k) {
      for (int j = 0; j < k; ++j) c[j] = 0;
      for (int comp = 0; comp < sym_tensor_size(P, k); ++comp) {
        const int rest = (1 << (k - 1)) - 1;  // the slots 1 .. k-1
        int n = 0;
        for (int sub = rest;; sub = (sub - 1) & rest) {
          const int mask = (sub << 1) | 1;
          int vi[K], wi[K];
          int nv = 0, nw = 0;
          for (int i = 0; i < k; ++i) {
            if (mask >> i & 1) vi[nv++] = c[i];
            else wi[nw++] = c[i];
          }
          lv[e] = static_cast<unsigned char>(mi_position<P, K>(vi, nv));
          bw[e] = static_cast<unsigned char>(nw == 0 ? 0 : 1 + mi_position<P, K>(wi, nw));
          ++e;
          ++n;
          if (sub == 0) break;
        }
        count[1 + mi_position<P, K>(c, k)] = static_cast<unsigned char>(n);
        next_multi_index_k<P>(c, k);
      }
    }
  }
};

// B over every multi-index of order 0 to K from the table and the partials L
template <int P, int K, class L>
LAPLACE_HD_FORCEINLINE void bell_all(const bell_table<P, K>& tb, const L& L_, double* B) {
  B[0] = 1.0;
  int e = 0;
  for (int u = 1; u < mi_count<P, K>(); ++u) {
    double acc = 0.0;
    const int n = tb.count[u];
    for (int i = 0; i < n; ++i, ++e) acc += L_.v[tb.lv[e]] * B[tb.bw[e]];
    B[u] = acc;
  }
}

// The derivatives of F in the parameters by quadrature of the density, every
// multi-index of order 0 to K in one pass: A[U] / A[0] = d_U W / W for the
// integral W of f / f(y) over the side of y that does not hold the centre. The
// density at the end point y is divided out, so a tail far below the double
// range still gives its logarithm (log S = l(y) + log W), and the Bell
// polynomial is built on the derivatives of l centred at y, which the caller
// recombines with those of l(y) as the discrete route does.
//
// One quadrature serves every component of every order: the l-partials, six
// derivative tensors per node, are evaluated once instead of once per component
// (decided with Giovanni on 2026-11-15, after the first version, which ran one
// adaptive quadrature per component, measured 94 ms per call at order 6).
template <int K, class D>
LAPLACE_HD double cdf_partials_by_integral(const D& d, double y, bool upper, double* A) {
  constexpr int P = D::n_params;
  constexpr int M = mi_count<P, K>();
  constexpr int n_partials = log_density_partials<P, K>::offset(K + 1);
  for (int m = 0; m < M; ++m) A[m] = 0.0;
  const double lf = d.logpdf(y);
  // where the density at y is zero or not finite nothing can be factored out and
  // the integrand is taken as it is
  const double la = (lf > -distrib_infinity && lf < distrib_infinity) ? lf : 0.0;
  log_density_partials<P, K> L0;
  L0.fill(d, y);
  const bell_table<P, K> tb;
  const auto g = [&](double t, double* out) {
    const double w = math::exp(d.logpdf(t) - la);
    if (!(w > 0.0)) {
      for (int m = 0; m < M; ++m) out[m] = 0.0;
      return;
    }
    log_density_partials<P, K> L;
    L.fill(d, t);
    for (int j = 0; j < n_partials; ++j) L.v[j] -= L0.v[j];
    double B[M];
    bell_all<P, K>(tb, L, B);
    for (int m = 0; m < M; ++m) out[m] = w * B[m];
  };
  const double lo = upper ? y : d.support_lower(), hi = upper ? d.support_upper() : y;
  vec_expectation<M> q;
  if constexpr (requires { d.y_kinks(); }) {
    const distrib_kinks k = d.y_kinks();
    panels_vec<M>(g, lo, hi, d.center(), d.scale(), k.point, k.count, q);
  } else {
    panels_vec<M>(g, lo, hi, d.center(), d.scale(), nullptr, 0, q);
  }
  if (!(q.value[0] > 0.0)) return -distrib_infinity;
  // The per-component stopping test of the rule (the change of the last
  // refinement against that component's own L1 norm) is conservative here: the
  // components it leaves unconverged are the high pure derivatives in the scale
  // in the far tail, where the Bell polynomial carries powers of z that cancel,
  // and they were measured against the Cauchy's closed forms at 1.9e-8 relative
  // at worst and 1e-13 elsewhere. A NaN would poison the whole tensor for a
  // number of that quality, so the value is returned and the flag is not read;
  // what is not usable (an empty or non-finite integral) is caught above.
  for (int m = 0; m < M; ++m) A[m] = q.value[m] / q.value[0];
  return la + math::log(q.value[0]);
}

// log F(y) (upper = false) or log(1 - F(y)) (upper = true) by quadrature of the
// density over the side that does not hold the centre, the other side by the
// complement on the log scale
template <class D>
LAPLACE_HD double log_cdf_side_by_integral(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::continuous,
                "the cdf by quadrature of the density is for continuous distributions");
  const double lo = d.support_lower(), hi = d.support_upper();
  if (y <= lo) return upper ? 0.0 : -distrib_infinity;
  if (y >= hi) return upper ? -distrib_infinity : 0.0;
  const bool lower_side = y < d.center();
  const bool integrated_is_asked = lower_side != upper;
  const double lf = d.logpdf(y);
  const double la = (lf > -distrib_infinity && lf < distrib_infinity) ? lf : 0.0;
  const auto f = [&](double t) { return math::exp(d.logpdf(t) - la); };
  math::quad::quad_result q;
  const double a = lower_side ? lo : y, b = lower_side ? y : hi;
  if constexpr (requires { d.y_kinks(); }) {
    const distrib_kinks k = d.y_kinks();
    q = math::quad::panels_quad(f, a, b, d.center(), d.scale(), k.point, k.count);
  } else {
    q = math::quad::panels_quad(f, a, b, d.center(), d.scale());
  }
  if (!q.converged || !(q.value > 0.0)) return q.value > 0.0 ? math::detail::quiet_nan() : -distrib_infinity;
  const double log_s = la + math::log(q.value);
  if (integrated_is_asked) return log_s;
  return Kokkos::log1p(-math::exp(log_s));
}

// d^K log F / dtheta^K (upper = false) or d^K log(1 - F) / dtheta^K (upper = true)
// by quadrature of the density: one pass for every component, from the
// derivatives of F over all multi-indices. The identity d_T F = int f B_T(l)
// holds where every parameter is smooth: a parameter of finite regularity moves
// a kink of l in y, and differentiating the integrand across it would miss a
// boundary term, so such a component is NaN.
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_side_dk_by_integral(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::continuous,
                "the derivatives of the log cdf by quadrature are for continuous distributions");
  constexpr int P = D::n_params;
  constexpr int full = (1 << K) - 1;
  sym_tensor<P, K> t;
  const double lo = d.support_lower(), hi = d.support_upper();
  if (upper ? y >= hi : y <= lo) {  // the side asked for has probability 0
    for (int i = 0; i < sym_tensor<P, K>::size; ++i) t[i] = math::detail::quiet_nan();
    return t;
  }
  if (upper ? y <= lo : y >= hi) return t;  // probability 1, every derivative 0
  const bool lower_side = y < d.center();
  const bool integrated_is_asked = lower_side != upper;
  double A[mi_count<P, K>()];
  const double log_s = cdf_partials_by_integral<K>(d, y, !lower_side, A);
  log_density_partials<P, K> L0;
  L0.fill(d, y);
  const double log_o = integrated_is_asked ? 0.0 : Kokkos::log1p(-math::exp(log_s));
  const double fac = integrated_is_asked ? 0.0 : -math::exp(log_s - log_o);
  // the descriptors as a value: a static constexpr array cannot be indexed on
  // the device (the same helper the expected information uses)
  constexpr regularity_orders<D> R = regularities_of<D>();
  int c[K];
  for (int j = 0; j < K; ++j) c[j] = 0;
  for (int comp = 0; comp < sym_tensor<P, K>::size; ++comp) {
    bool smooth = true;
    for (int j = 0; j < K; ++j)
      if (R.r[c[j]] != infinitely_smooth) smooth = false;
    if (!smooth) {
      t[comp] = math::detail::quiet_nan();
      next_multi_index<P, K>(c);
      continue;
    }
    double r[1 << K], u[1 << K];
    r[0] = 1.0;
    for (int mask = 1; mask <= full; ++mask) {  // the sub-multiset of c that the mask selects
      int idx[K];
      int n = 0;
      for (int i = 0; i < K; ++i)
        if (mask >> i & 1) idx[n++] = c[i];
      r[mask] = A[1 + mi_position<P, K>(idx, n)];
    }
    log_tail_dk_from_ratios<K>(r, u);  // d_T log W
    for (int m = 1; m <= full; ++m) u[m] += l_over(L0, nullptr, 0, c, m);  // + d_T l(y)
    if (!integrated_is_asked) {
      ratios_from_log_tail_dk<K>(u, r);
      for (int m = 1; m <= full; ++m) r[m] *= fac;
      log_tail_dk_from_ratios<K>(r, u);
    }
    t[comp] = u[full];
    next_multi_index<P, K>(c);
  }
  return t;
}

}  // namespace detail_generic

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of cumulative distribution function \f$\log F(y)\f$ for a discrete distribution by summation.
 *
 * For \f$k = \lfloor y \rfloor\f$, evaluates \f$\log \sum_{i \le k} p(i)\f$ by summing the lower tail downwards
 * when \f$k < \mathrm{center}()\f$, or from the upper tail complement \f$\log(1 - e^{\log(1 - F(k))})\f$
 * when \f$k \ge \mathrm{center}()\f$. Each tail factors out its dominant term to prevent underflow.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$-\infty\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * On the Poisson at \f$\mu\f$ = 0.05, 3.5, 40, 1000 (gcc), worst error 0.60, 2.64, 0.82, 5.91 ulp of \f$\max(|\log F|, 1)\f$ and \f$\max(|\log(1 - F)|, 1)\f$ over both functions.
 *
 * \see log_ccdf_by_sum(), cdf_by_sum(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double log_cdf_by_sum(const D& d, double y) { return detail_generic::log_cdf_side_by_sum(d, y, false); }

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of complementary cumulative distribution function \f$\log(1 - F(y))\f$ for a discrete distribution by summation.
 *
 * For \f$k = \lfloor y \rfloor\f$, evaluates \f$\log \sum_{i > k} p(i)\f$ by summing the upper tail upwards
 * when \f$k \ge \mathrm{center}()\f$, or from the lower tail complement \f$\log(1 - e^{\log F(k)})\f$
 * when \f$k < \mathrm{center}()\f$. Each tail factors out its dominant term to prevent underflow.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log(1 - F(y))\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$-\infty\f$.
 *
 * \par Measured accuracy
 * On the Poisson at \f$\mu\f$ = 0.05, 3.5, 40, 1000 (gcc), worst error 0.60, 2.64, 0.82, 5.91 ulp of \f$\max(|\log F|, 1)\f$ and \f$\max(|\log(1 - F)|, 1)\f$ over both functions.
 *
 * \see log_cdf_by_sum(), ccdf_by_sum(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double log_ccdf_by_sum(const D& d, double y) { return detail_generic::log_cdf_side_by_sum(d, y, true); }

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of cumulative distribution function \f$\log F(y)\f$ for a continuous distribution by quadrature.
 *
 * Integrates the density over the side of \f$y\f$ that does not hold `d.center()`, using
 * laplace::math::quad::panels_quad() split at `d.y_kinks()` when present with panel width `d.scale()`.
 * When \f$y < \mathrm{center}()\f$, it integrates the lower tail directly; when \f$y \ge \mathrm{center}()\f$,
 * it integrates the upper tail \f$S\f$ and returns the log-scale complement \f$\log(1 - e^{\log S})\f$.
 * Where \f$\ell(y) = \log f(y)\f$ is finite, the density at the endpoint is divided out, giving
 * \f$\log S = \ell(y) + \log W\f$ with \f$W = \int (f(t)/f(y))\,\mathrm{d}t\f$, so that tails far below
 * the smallest normal `double` keep their logarithm accurately.
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$y\f$ is at most `d.support_lower()`, returns \f$-\infty\f$.
 * - If \f$y\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 * - If quadrature fails to converge with a positive integral, returns quiet NaN.
 *
 * \par Measured accuracy
 * On Cauchy and logistic distributions (\f$\sigma \in \{0.25, 1, 7\}\f$, \f$y \in [-300, 300]\f$, gcc),
 * worst error relative to \f$\max(|\log F|, 1)\f$ is \f$5.6 \times 10^{-16}\f$ on Cauchy and
 * \f$4.4 \times 10^{-16}\f$ on logistic (test bound 8.0 ulp of \f$\max(|\log F|, 1)\f$).
 *
 * \see log_ccdf_by_integral(), cdf_by_integral(), distrib_log_cdf()
 */
// The same four quantities for a continuous distribution, by quadrature of its
// density over the side of y that does not hold the centre (note/distrib.md
// section 4). They are what a continuous family without a closed cdf inherits.
template <class D>
LAPLACE_HD double log_cdf_by_integral(const D& d, double y) { return detail_generic::log_cdf_side_by_integral(d, y, false); }

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of complementary cumulative distribution function \f$\log(1 - F(y))\f$ for a continuous distribution by quadrature.
 *
 * Evaluates \f$\log(1 - F(y))\f$ by integrating the upper tail directly when \f$y \ge \mathrm{center}()\f$,
 * or from the lower tail complement \f$\log(1 - e^{\log F(y)})\f$ when \f$y < \mathrm{center}()\f$,
 * using the endpoint-scaled quadrature of log_cdf_by_integral().
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log(1 - F(y))\f$.
 *
 * \par Domain and special values
 * - If \f$y\f$ is at most `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$y\f$ is at least `d.support_upper()`, returns \f$-\infty\f$.
 * - If quadrature fails to converge with a positive integral, returns quiet NaN.
 *
 * \par Measured accuracy
 * On Cauchy and logistic distributions (\f$\sigma \in \{0.25, 1, 7\}\f$, \f$y \in [-300, 300]\f$, gcc),
 * worst error relative to \f$\max(|\log(1 - F)|, 1)\f$ is \f$5.6 \times 10^{-16}\f$ on Cauchy and
 * \f$4.4 \times 10^{-16}\f$ on logistic (test bound 8.0 ulp of \f$\max(|\log(1 - F)|, 1)\f$).
 *
 * \see log_cdf_by_integral(), ccdf_by_integral(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double log_ccdf_by_integral(const D& d, double y) { return detail_generic::log_cdf_side_by_integral(d, y, true); }

/**
 * \ingroup distrib_generic
 * \brief Cumulative distribution function \f$F(y)\f$ for a continuous distribution by quadrature.
 *
 * Computes the exponential of log_cdf_by_integral().
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$y\f$ is at most `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$y\f$ is at least `d.support_upper()`, returns \f$1.0\f$.
 *
 * \par Measured accuracy
 * That of log_cdf_by_integral(), passed through the exponential.
 *
 * \see log_cdf_by_integral(), ccdf_by_integral(), distrib_cdf()
 */
template <class D>
LAPLACE_HD double cdf_by_integral(const D& d, double y) { return math::exp(log_cdf_by_integral(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Complementary cumulative distribution function \f$1 - F(y)\f$ for a continuous distribution by quadrature.
 *
 * Computes the exponential of log_ccdf_by_integral().
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$1 - F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$y\f$ is at most `d.support_lower()`, returns \f$1.0\f$.
 * - If \f$y\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * That of log_ccdf_by_integral(), passed through the exponential.
 *
 * \see log_ccdf_by_integral(), cdf_by_integral(), distrib_ccdf()
 */
template <class D>
LAPLACE_HD double ccdf_by_integral(const D& d, double y) { return math::exp(log_ccdf_by_integral(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ for a continuous distribution by quadrature.
 *
 * Integrates the side of \f$y\f$ that does not hold `d.center()`, dividing out \f$f(y)\f$ when
 * \f$\ell(y) = \log f(y)\f$ is finite so that \f$\log S = \ell(y) + \log W\f$ with \f$W = \int (f/f(y))\f$.
 * Uses the identity
 * \f[
 * \frac{\partial^{|T|} F}{\partial\theta_T} = \int f \, B_T(\ell),
 * \f]
 * where \f$B_T\f$ is the complete Bell polynomial of the derivatives of \f$\ell\f$ over the parameter
 * multiset \f$T\f$, built on the centred derivatives \f$\ell(t) - \ell(y)\f$ to obtain the ratios
 * \f$\partial_T W / W\f$ and compose them to \f$\partial_T \log S = \partial_T \ell(y) + \partial_T \log W\f$.
 * When \f$y \ge \mathrm{center}()\f$, the requested side \f$F = 1 - S\f$ is obtained from the upper
 * tail \f$S\f$ via \f$\partial_T F / F = -(S/F)\,\partial_T S / S\f$. One quadrature pass serves every
 * component, because the integrands are the Bell polynomials of every multi-index of order at most
 * \f$K\f$ evaluated on shared quadrature nodes.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor `sym_tensor<D::n_params, K>` of derivatives.
 *
 * \par Domain and special values
 * - Quiet NaN in every component when \f$y\f$ is at most `d.support_lower()` (\f$\log F = -\infty\f$).
 * - Zero in every component when \f$y\f$ is at least `d.support_upper()` (\f$F = 1\f$).
 * - Quiet NaN in any component whose parameter has finite regularity: that parameter moves a
 *   non-smooth point of \f$\ell\f$ in \f$y\f$, and differentiating the integrand across it would
 *   leave a boundary term.
 *
 * \par Measured accuracy
 * Against closed forms on Cauchy (orders 1 to 6, \f$\sigma = 0.25\f$, \f$y \in [-40, 25]\f$, gcc),
 * worst error over \f$\max(|f|, 1)\f$ is \f$1.8 \times 10^{-15}\f$, \f$1.4 \times 10^{-14}\f$,
 * \f$5.7 \times 10^{-14}\f$, \f$9.1 \times 10^{-13}\f$, \f$2.2 \times 10^{-11}\f$, and
 * \f$5.8 \times 10^{-10}\f$ (test bounds \f$6 \times 10^{-15}\f$, \f$5 \times 10^{-14}\f$,
 * \f$2 \times 10^{-13}\f$, \f$3 \times 10^{-12}\f$, \f$7 \times 10^{-11}\f$, and
 * \f$2 \times 10^{-9}\f$); on logistic at order 6, \f$1.5 \times 10^{-9}\f$ (test bound \f$5 \times 10^{-9}\f$).
 *
 * \see log_ccdf_dk_by_integral(), log_cdf_by_integral(), distrib_log_cdf_dk()
 */
// the derivatives in the parameters of log F and log(1 - F), K = 1..6, by
// quadrature of f B_T(l): one quadrature per component of the tensor
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_dk_by_integral(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_integral<K>(d, y, false);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ for a continuous distribution by quadrature.
 *
 * Applies the one-pass Bell-polynomial quadrature of log_cdf_dk_by_integral() to \f$\log(1 - F(y))\f$,
 * integrating the upper tail directly when \f$y \ge \mathrm{center}()\f$ and taking the complement of
 * the lower tail when \f$y < \mathrm{center}()\f$.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor `sym_tensor<D::n_params, K>` of derivatives.
 *
 * \par Domain and special values
 * - Zero in every component when \f$y\f$ is at most `d.support_lower()` (\f$1 - F = 1\f$).
 * - Quiet NaN in every component when \f$y\f$ is at least `d.support_upper()` (\f$\log(1 - F) = -\infty\f$).
 * - Quiet NaN in any component whose parameter has finite regularity.
 *
 * \par Measured accuracy
 * Same grid, worst errors, and test bounds as log_cdf_dk_by_integral().
 *
 * \see log_cdf_dk_by_integral(), log_ccdf_by_integral(), distrib_log_ccdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_ccdf_dk_by_integral(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_integral<K>(d, y, true);
}

/**
 * \ingroup distrib_generic
 * \brief Cumulative distribution function \f$F(y)\f$ for a discrete distribution by summation.
 *
 * Computes the exponential of log_cdf_by_sum().
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$1.0\f$.
 *
 * \par Measured accuracy
 * That of log_cdf_by_sum(), passed through the exponential.
 *
 * \see log_cdf_by_sum(), ccdf_by_sum(), distrib_cdf()
 */
template <class D>
LAPLACE_HD double cdf_by_sum(const D& d, double y) { return math::exp(log_cdf_by_sum(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Complementary cumulative distribution function \f$1 - F(y)\f$ for a discrete distribution by summation.
 *
 * Computes the exponential of log_ccdf_by_sum().
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$1 - F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$1.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * That of log_ccdf_by_sum(), passed through the exponential.
 *
 * \see log_ccdf_by_sum(), cdf_by_sum(), distrib_ccdf()
 */
template <class D>
LAPLACE_HD double ccdf_by_sum(const D& d, double y) { return math::exp(log_ccdf_by_sum(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Cumulative distribution function \f$F(y)\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls `d.cdf(y)` if provided by distribution `d`; otherwise calls cdf_by_sum() when `D::kind` is
 * `distrib_kind::discrete` and cdf_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$F(y)\f$.
 *
 * \see cdf_by_sum(), cdf_by_integral(), distrib_ccdf(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double distrib_cdf(const D& d, double y) {
  if constexpr (requires { d.cdf(y); }) return d.cdf(y);
  else if constexpr (D::kind == distrib_kind::discrete) return cdf_by_sum(d, y);
  else return cdf_by_integral(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Complementary cumulative distribution function \f$1 - F(y)\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls `d.ccdf(y)` if provided by distribution `d`; otherwise calls ccdf_by_sum() when `D::kind` is
 * `distrib_kind::discrete` and ccdf_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$1 - F(y)\f$.
 *
 * \see ccdf_by_sum(), ccdf_by_integral(), distrib_cdf(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double distrib_ccdf(const D& d, double y) {
  if constexpr (requires { d.ccdf(y); }) return d.ccdf(y);
  else if constexpr (D::kind == distrib_kind::discrete) return ccdf_by_sum(d, y);
  else return ccdf_by_integral(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of cumulative distribution function \f$\log F(y)\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls `d.log_cdf(y)` if provided by distribution `d`; otherwise calls log_cdf_by_sum() when `D::kind` is
 * `distrib_kind::discrete` and log_cdf_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log F(y)\f$.
 *
 * \see log_cdf_by_sum(), log_cdf_by_integral(), distrib_cdf(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double distrib_log_cdf(const D& d, double y) {
  if constexpr (requires { d.log_cdf(y); }) return d.log_cdf(y);
  else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_by_sum(d, y);
  else return log_cdf_by_integral(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of complementary cumulative distribution function \f$\log(1 - F(y))\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls `d.log_ccdf(y)` if provided by distribution `d`; otherwise calls log_ccdf_by_sum() when `D::kind` is
 * `distrib_kind::discrete` and log_ccdf_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log(1 - F(y))\f$.
 *
 * \see log_ccdf_by_sum(), log_ccdf_by_integral(), distrib_ccdf(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double distrib_log_ccdf(const D& d, double y) {
  if constexpr (requires { d.log_ccdf(y); }) return d.log_ccdf(y);
  else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_by_sum(d, y);
  else return log_ccdf_by_integral(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ for a discrete distribution by summation.
 *
 * With \f$k = \lfloor y \rfloor\f$ and \f$C\f$ a sorted multi-index of parameters,
 * \f$\partial_C F(k) = \sum_{i \le k} p(i) B_C(l(i))\f$, where \f$B_C\f$ is the complete Bell
 * polynomial of the derivatives of \f$l = \log p\f$ over the sub-multisets of \f$C\f$; the same
 * relation inverted gives the derivatives of \f$\log F\f$. The cheaper tail is summed and the
 * other side is taken from it by complement.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor \c sym_tensor<D::n_params, K> of derivatives.
 *
 * \par Domain and special values
 * - Quiet NaN when \f$\lfloor y \rfloor\f$ is below `d.support_lower()` (\f$\log F = -\infty\f$).
 * - Zero when \f$\lfloor y \rfloor\f$ is at least `d.support_upper()` (\f$F = 1\f$).
 *
 * \see log_ccdf_dk_by_sum(), log_cdf_by_sum(), distrib_log_cdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_dk_by_sum(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_sum<K>(d, y, false);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ for a discrete distribution by summation.
 *
 * The route of log_cdf_dk_by_sum() applied to the upper tail \f$\sum_{i > k} p(i)\f$.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor \c sym_tensor<D::n_params, K> of derivatives.
 *
 * \par Domain and special values
 * - Zero when \f$\lfloor y \rfloor\f$ is below `d.support_lower()` (\f$1 - F = 1\f$).
 * - Quiet NaN when \f$\lfloor y \rfloor\f$ is at least `d.support_upper()` (\f$\log(1 - F) = -\infty\f$).
 *
 * \see log_cdf_dk_by_sum(), log_ccdf_by_sum(), distrib_log_ccdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_ccdf_dk_by_sum(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_sum<K>(d, y, true);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls the distribution's own `log_cdf_dK(y)` when it has one; otherwise calls log_cdf_dk_by_sum()
 * when `D::kind` is `distrib_kind::discrete` and log_cdf_dk_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \see log_cdf_dk_by_sum(), log_cdf_dk_by_integral(), distrib_log_ccdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> distrib_log_cdf_dk(const D& d, double y) {
  if constexpr (K == 1) {
    if constexpr (requires { d.log_cdf_d1(y); }) return d.log_cdf_d1(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<1>(d, y);
    else return log_cdf_dk_by_integral<1>(d, y);
  } else if constexpr (K == 2) {
    if constexpr (requires { d.log_cdf_d2(y); }) return d.log_cdf_d2(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<2>(d, y);
    else return log_cdf_dk_by_integral<2>(d, y);
  } else if constexpr (K == 3) {
    if constexpr (requires { d.log_cdf_d3(y); }) return d.log_cdf_d3(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<3>(d, y);
    else return log_cdf_dk_by_integral<3>(d, y);
  } else if constexpr (K == 4) {
    if constexpr (requires { d.log_cdf_d4(y); }) return d.log_cdf_d4(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<4>(d, y);
    else return log_cdf_dk_by_integral<4>(d, y);
  } else if constexpr (K == 5) {
    if constexpr (requires { d.log_cdf_d5(y); }) return d.log_cdf_d5(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<5>(d, y);
    else return log_cdf_dk_by_integral<5>(d, y);
  } else {
    if constexpr (requires { d.log_cdf_d6(y); }) return d.log_cdf_d6(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_cdf_dk_by_sum<6>(d, y);
    else return log_cdf_dk_by_integral<6>(d, y);
  }
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$, dispatching to member or generic route by distribution kind.
 *
 * Calls the distribution's own `log_ccdf_dK(y)` when it has one; otherwise calls log_ccdf_dk_by_sum()
 * when `D::kind` is `distrib_kind::discrete` and log_ccdf_dk_by_integral() when it is `distrib_kind::continuous`.
 * The choice is made at compile time.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \see log_ccdf_dk_by_sum(), log_ccdf_dk_by_integral(), distrib_log_cdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> distrib_log_ccdf_dk(const D& d, double y) {
  if constexpr (K == 1) {
    if constexpr (requires { d.log_ccdf_d1(y); }) return d.log_ccdf_d1(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<1>(d, y);
    else return log_ccdf_dk_by_integral<1>(d, y);
  } else if constexpr (K == 2) {
    if constexpr (requires { d.log_ccdf_d2(y); }) return d.log_ccdf_d2(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<2>(d, y);
    else return log_ccdf_dk_by_integral<2>(d, y);
  } else if constexpr (K == 3) {
    if constexpr (requires { d.log_ccdf_d3(y); }) return d.log_ccdf_d3(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<3>(d, y);
    else return log_ccdf_dk_by_integral<3>(d, y);
  } else if constexpr (K == 4) {
    if constexpr (requires { d.log_ccdf_d4(y); }) return d.log_ccdf_d4(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<4>(d, y);
    else return log_ccdf_dk_by_integral<4>(d, y);
  } else if constexpr (K == 5) {
    if constexpr (requires { d.log_ccdf_d5(y); }) return d.log_ccdf_d5(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<5>(d, y);
    else return log_ccdf_dk_by_integral<5>(d, y);
  } else {
    if constexpr (requires { d.log_ccdf_d6(y); }) return d.log_ccdf_d6(y);
    else if constexpr (D::kind == distrib_kind::discrete) return log_ccdf_dk_by_sum<6>(d, y);
    else return log_ccdf_dk_by_integral<6>(d, y);
  }
}

}  // namespace laplace::distrib
