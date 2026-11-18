// Scalar bijector box_cox (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   box_cox: (0, inf) -> (-1/p1, inf) for p1 > 0, (-inf, -1/p1) for p1 < 0, R for p1 = 0; g(v) = (v^p1 - 1)/p1, log v at p1 = 0; p1 = exponent, p2 unused
//
// For each map NAME, with v in the domain of g and u in the domain of h = g^-1:
//   NAME_fwd<K>(v, p1, p2)      g^(K)(v), K = 0..6 (K = 0 the value)
//   NAME_inv<K>(u, p1, p2)      h^(K)(u)
//   NAME_fwd_lj<K>(v, p1, p2)   d^K/dv^K log|g'(v)|
//   NAME_inv_lj<K>(u, p1, p2)   d^K/du^K log|h'(u)|
//   NAME_fwd_partials<K>(v, p1, p2, out), NAME_inv_partials, NAME_fwd_lj_partials,
//   NAME_inv_lj_partials        the partial derivatives of total order K = 1..6 in
//                               (argument, p1, p2), out[bijector_partials<K>::index(n1, n2)]
//                               = d^K / (d arg^n0 d p1^n1 d p2^n2), n0 = K - n1 - n2;
//                               components in a parameter the map does not have are 0
//   NAME_dom_lo/hi(p1, p2), NAME_img_lo/hi(p1, p2)   the closed bounds of the domains of g and h
//   NAME_inv_to_upper/lower(u, p1, p2)   dom_hi - h(u) and h(u) - dom_lo, computed directly
// Invalid parameters: every function returns NaN. The arguments have been checked
// against the domain bounds by the dispatch; a value on a bound gives its limit. The forms
// below give NaN at an end of either domain, so the value and the argument derivatives return
// that limit instead (bc_fwd_at_end, bc_inv_at_end, bc_fwd_lj_at_end, bc_inv_lj_at_end, with
// the derivation and the ends in the comment above them). The partial derivatives in lambda, the
// NAME_*_partials below, return their limits there too (bc_fwd_partials_at_end,
// bc_fwd_lj_partials_at_end, bc_inv_partials_at_inf, bc_inv_partials_at_edge,
// bc_inv_lj_partials_at_inf, bc_inv_lj_partials_at_edge): the same exponent decides, a
// lambda-derivative only bringing a logarithm of the argument, except at the finite end of the
// image, where it lowers the power of w by one and the exponent becomes 1/lambda - K.
//
// Formulas (lambda = p1, L = log v, x = lambda u, w = 1 + x, t = 1/w, S = log h = u psi(x),
// phi and psi the functions exprel and log1prel of laplace::math; their derivatives are
// exprel_dK and log1prel_dK). Derived and checked with SymPy by
// tools/reference/bijector_box_cox_forms.py. No formula divides by lambda, so lambda = 0
// needs no branch: every term is a polynomial or exponential in lambda.
//
//   Forward g = L phi(lambda L). For lambda L in (-1, 1) the value is L exprel(lambda L), and
//     beyond it (v^lambda - 1)/lambda with pow, where no cancellation occurs.
//   g^(K) (pure in v, K >= 1) = (lambda-1)(lambda-2)...(lambda-K+1) v^(lambda-K).
//   d^n/dlambda^n g = L^(n+1) phi^(n)(lambda L)                                  (pure in lambda)
//   d^m/dv^m d^n/dlambda^n g = v^(lambda-m) sum_(p=0..min(m-1,n)) e_(m-1-p)(c) n!/(n-p)! L^(n-p),
//     m >= 1, c_r = lambda - 1 - r (r = 0..m-2), e_k the elementary symmetric polynomials of c
//     (from d^(m-1)_v (L^n v^(lambda-1)), with v d/dv acting as (d/dL + lambda - 1 - r)).
//   log|g'| = (lambda - 1) L: its derivatives are (lambda-1) d^m_v L and d^m_v L, both closed.
//
//   Inverse h = exp(S), S = u psi(lambda u) = log(1 + lambda u)/lambda (which equals log h
//     without cancellation at lambda = 0). d_u S = t.
//   h^(K) (pure in u, K >= 1) = P_K(lambda) exp((1 - K lambda) S), P_K = prod_(r=1..K-1)(1 - r lambda).
//   d^n_lambda S = u^(n+1) psi^(n)(x) (pure in lambda; psi^(0) = log1prel, psi^(n) = log1prel_dn),
//     and the Bell polynomials A_n of these: d^n_lambda h = h A_n, A_0 = 1,
//     A_(i+1) = sum_(j=0..i) C(i,j) x_(j+1) A_(i-j), x_j = d^j_lambda S.
//   d^K_u d^n_lambda h (mixed, K >= 1) = exp((1 - K lambda) S) sum_(j,i) C(n,j) C(n-j,i)
//     P_K^(j)(lambda) A_i (-1)^q K^(q) u^q t^q, q = n - j - i, K^(q) the rising factorial;
//     P_K^(j) from the Leibniz rule on the linear factors (1 - r lambda).
//   log|h'| = (1 - lambda) S (exact: d_u log h' = d_u[(1/lambda - 1) log w] = -t ...); its
//     derivatives are (1 - lambda) T_(K,n) - n T_(K,n-1), with T_(K,n) = d^K_u d^n_lambda S:
//     T_(0,n) = x_n, and for K >= 1 T_(K,n) = (-1)^(K-1)(K-1)! sum_j C(n,j) (K-1)!/(K-1-j)!
//     lambda^(K-1-j) (-1)^(n-j) K^(n-j) u^(n-j) t^(K+n-j).
//
//   Distances: the range of h is (0, inf), so the distance to the upper bound is +inf and
//   to the lower bound h(u) itself (no cancellation).
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/bijector/detail/scalar_yeo_johnson.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/log1prel.hpp>

namespace laplace::bijector::detail_scalar {

// Above this |lambda u| the inverse partials take the rescaled route rather than the
// forms; the value is measured in docs/numerics.md.
inline constexpr double bc_rescale_threshold = 1.0e3;

// lambda finite: NaN and the infinities fail x - x == 0
LAPLACE_HD_FORCEINLINE bool bc_finite(double x) { return x - x == 0.0; }

LAPLACE_HD_FORCEINLINE double bc_pow_int(double x, int k) {
  double r = 1.0;
  for (int i = 0; i < k; ++i) r *= x;
  return r;
}
LAPLACE_HD_FORCEINLINE double bc_fact(int k) {
  double f = 1.0;
  for (int i = 2; i <= k; ++i) f *= i;
  return f;
}
LAPLACE_HD_FORCEINLINE double bc_sgn(int k) { return (k & 1) ? -1.0 : 1.0; }  // (-1)^k
LAPLACE_HD_FORCEINLINE double bc_binom(int n, int k) {  // C(n, k), 0 <= k <= n <= 6, exact
  // integer arithmetic: the double division of the device is slow (FP64 at 1/64 rate)
  int c = 1;
  for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
  return double(c);
}
// a (a - 1) ... (a - k + 1), k factors
LAPLACE_HD_FORCEINLINE double bc_falling(double a, int k) {
  double r = 1.0;
  for (int i = 0; i < k; ++i) r *= a - i;
  return r;
}
// K (K + 1) ... (K + q - 1), q factors (the rising factorial K^(q))
LAPLACE_HD_FORCEINLINE double bc_rising(int K, int q) {
  double r = 1.0;
  for (int i = 0; i < q; ++i) r *= K + i;
  return r;
}

// exprel^(k)(y) and log1prel^(k)(x), k = 0..6
LAPLACE_HD_FORCEINLINE double bc_exprel_dk(int k, double y) {
  switch (k) {
    case 0: return m::exprel(y);
    case 1: return m::exprel_d1(y);
    case 2: return m::exprel_d2(y);
    case 3: return m::exprel_d3(y);
    case 4: return m::exprel_d4(y);
    case 5: return m::exprel_d5(y);
    default: return m::exprel_d6(y);
  }
}
LAPLACE_HD_FORCEINLINE double bc_log1prel_dk(int k, double x) {
  switch (k) {
    case 0: return m::log1prel(x);
    case 1: return m::log1prel_d1(x);
    case 2: return m::log1prel_d2(x);
    case 3: return m::log1prel_d3(x);
    case 4: return m::log1prel_d4(x);
    case 5: return m::log1prel_d5(x);
    default: return m::log1prel_d6(x);
  }
}

// P_K(lambda) = prod_(r=1..K-1) (1 - r lambda) and its derivatives P[j] = P_K^(j)(lambda), j = 0..6,
// by the Leibniz rule on one linear factor at a time (no expansion into coefficients, so the zero of
// P_K at lambda = 1/r stays exact)
LAPLACE_HD_FORCEINLINE void bc_PK_jets(int K, double lam, double (&P)[7]) {
  for (int j = 0; j < 7; ++j) P[j] = j == 0 ? 1.0 : 0.0;
  for (int k = 1; k <= K - 1; ++k) {
    const double f = 1.0 - k * lam;  // the factor, with f' = -k and no higher derivative
    for (int j = 6; j >= 0; --j) P[j] = f * P[j] - (j > 0 ? k * j * P[j - 1] : 0.0);
  }
}

// S = log h = u psi(lambda u), with psi = log1prel
LAPLACE_HD_FORCEINLINE double bc_S(double u, double lam) { return u * m::log1prel(lam * u); }

// d^K_u d^n_lambda S for K + n <= 6: for K = 0, x_n = u^(n+1) psi^(n)(lambda u); for K >= 1 the
// closed form in t = 1/(1 + lambda u) given above (T_(K,n))
LAPLACE_HD double bc_S_jet(int K, int n, double u, double lam) {
  const double x = lam * u;
  if (K == 0) return bc_pow_int(u, n + 1) * bc_log1prel_dk(n, x);
  const double t = 1.0 / (1.0 + x);
  double s = 0.0;
  for (int j = 0; j <= n && j <= K - 1; ++j) {
    const int q = n - j;
    // C(n, j) (K-1)!/(K-1-j)! lambda^(K-1-j) (-1)^q K^(q) u^q t^(K+q), the sum without its (K-1)! sign factor
    s += bc_binom(n, j) * bc_falling(K - 1, j) * bc_pow_int(lam, K - 1 - j) * bc_sgn(q) * bc_rising(K, q) *
         bc_pow_int(u, q) * bc_pow_int(t, K + q);
  }
  return bc_sgn(K - 1) * bc_fact(K - 1) * s;
}

// A_0..A_n: the Bell polynomials of x_j = d^j_lambda S, so that d^n_lambda h = h A_n
LAPLACE_HD_FORCEINLINE void bc_bell(int n, double u, double lam, double (&A)[7]) {
  double xs[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // x_j = u^(j+1) psi^(j)(lambda u), j = 1..n; the n derivatives of psi in one pass
  double psi[7];
  if (n >= 1) m::detail_log1prel::derivs_upto(n, lam * u, psi);
  for (int j = 1; j <= n; ++j) xs[j] = bc_pow_int(u, j + 1) * psi[j];
  A[0] = 1.0;
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j <= i; ++j) s += bc_binom(i, j) * xs[j + 1] * A[i - j];
    A[i + 1] = s;
  }
}

// ---------------------------------------------------------------- forward g

// g(v) = L phi(lambda L) for lambda L in (-1, 1) (no cancellation as lambda L -> 0), and
// (v^lambda - 1)/lambda beyond, where the exponential does not cancel; the limits at v = 0 and
// v = inf are those of the domain
LAPLACE_HD double bc_value(double v, double lam) {
  if (v == 0.0) return lam > 0.0 ? -1.0 / lam : -inf();
  if (v == inf()) return lam < 0.0 ? -1.0 / lam : inf();
  const double L = m::log(v), y = lam * L;
  if (Kokkos::fabs(y) < 1.0) return L * m::exprel(y);
  return (pow_pos(v, lam) - 1.0) / lam;
}

// v^(lambda - m) for integer m >= 0. The exponent lambda - m is not rounded (lambda - 1 rounds
// with an absolute error of eps, which log v magnifies): v^lambda is taken once and divided by v m
// times. At v = 0 and v = inf the division would form 0/0 or inf/inf, so the power goes directly.
// Where v^lambda itself underflows or overflows the division route carries no information left:
// v^lambda is 0 or inf and so is every quotient, although v^(lambda - m) may be an ordinary number
// (g^(6) at lambda = 6 is 120 for every v, and the route returned 0 below v = 1e-52 and inf above
// 1e52). The rounded exponent is then the only route; it costs |log v| eps relative, 1.5e-14 at
// the smallest double, which is what a result the other route cannot represent at all is worth.
LAPLACE_HD double bc_pow_shift(double v, double lam, int m) {
  if (v == 0.0 || v == inf()) return pow_pos(v, lam - m);
  const double p = pow_pos(v, lam);
  if (p == 0.0 || p == inf()) return pow_pos(v, lam - m);
  double r = p;
  for (int i = 0; i < m; ++i) r /= v;
  return r;
}

// d^m/dv^m d^n/dlambda^n g for m >= 1 (v > 0): see the formulas above
LAPLACE_HD double bc_fwd_mixed(int m, int n, double v, double lam) {
  const double L = m::log(v);
  const int j = m - 1;
  double e[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // e_k of c_r = lam - 1 - r, r < j
  for (int r = 0; r < j; ++r) {
    const double c = lam - 1.0 - r;
    for (int k = r + 1; k >= 1; --k) e[k] += c * e[k - 1];
  }
  double s = 0.0;
  for (int p = 0; p <= j && p <= n; ++p) s += e[j - p] * bc_falling(n, p) * bc_pow_int(L, n - p);
  return bc_pow_shift(v, lam, m) * s;
}

// every d^m_v d^n_lambda g, m + n <= 6
LAPLACE_HD double bc_fwd_any(int m, int n, double v, double lam) {
  if (m == 0 && n == 0) return bc_value(v, lam);
  if (n == 0) {
    const double f = bc_falling(lam - 1.0, m - 1);
    // f vanishes exactly at an integer lambda below m, where g is a polynomial of degree lambda
    // and g^(m) is identically 0. The power is then not formed: it overflows for a small enough v
    // and the product would be 0 times inf (at lambda = 1 and m = 3, v below 1e-103).
    if (f == 0.0) return 0.0;
    return f * bc_pow_shift(v, lam, m);
  }
  if (m == 0) {
    const double L = m::log(v);
    return bc_pow_int(L, n + 1) * bc_exprel_dk(n, lam * L);
  }
  return bc_fwd_mixed(m, n, v, lam);
}

// ---------------------------------------------------------------- inverse h

// d^K_u d^n_lambda h for K >= 1, n >= 0 (K + n <= 6): exp((1 - K lambda) S) times the sum
// over j (lambda derivatives of P_K) and i (Bell polynomials), with t^q
LAPLACE_HD double bc_inv_mixed(int K, int n, double u, double lam) {
  double P[7], A[7];
  bc_PK_jets(K, lam, P);
  // P_K vanishes exactly at lambda = 1/r with r below K, where h is a polynomial of degree r and
  // h^(K) is identically 0. The exponential is then not formed: it overflows for a w small enough
  // and the product would be 0 times inf. Only the pure u-derivative is P_K alone; a derivative in
  // lambda carries the other P_K^(j) as well.
  if (n == 0 && P[0] == 0.0) return 0.0;
  bc_bell(n, u, lam, A);
  const double t = 1.0 / (1.0 + lam * u);
  double s = 0.0;
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n - j; ++i) {
      const int q = n - j - i;
      s += bc_binom(n, j) * bc_binom(n - j, i) * P[j] * A[i] * bc_sgn(q) * bc_rising(K, q) * bc_pow_int(u, q) *
           bc_pow_int(t, q);
    }
  return m::exp((1.0 - K * lam) * bc_S(u, lam)) * s;
}

// every d^K_u d^n_lambda h, K + n <= 6
LAPLACE_HD double bc_inv_any(int K, int n, double u, double lam) {
  if (K == 0) {
    double A[7];
    bc_bell(n, u, lam, A);
    return m::exp(bc_S(u, lam)) * A[n];
  }
  return bc_inv_mixed(K, n, u, lam);
}

// ---------------------------------------------------------------- log-Jacobians

// d^m_v d^n_lambda [(lambda - 1) log v], the forward log|g'|
LAPLACE_HD double bc_fwd_lj_any(int m, int n, double v, double lam) {
  if (n >= 2) return 0.0;
  // at lambda = 1 the map is the identity and log|g'| is identically 0, so no v-derivative of it
  // is formed: v^(-m) overflows for a small enough v and (lambda - 1) v^(-m) would be 0 times inf
  // (at m = 6, v below 1e-52). The derivative in lambda is log v and does not vanish there.
  if (n == 0 && lam == 1.0) return 0.0;
  if (m == 0) return n == 0 ? (lam - 1.0) * m::log(v) : m::log(v);
  const double d = bc_sgn(m - 1) * bc_fact(m - 1) * pow_pos(v, -m);  // d^m_v log v
  return n == 0 ? (lam - 1.0) * d : d;
}

// d^K_u d^n_lambda [(1 - lambda) S], the inverse log|h'|
LAPLACE_HD double bc_inv_lj_any(int K, int n, double u, double lam) {
  if (n == 0) return (1.0 - lam) * bc_S_jet(K, 0, u, lam);
  return (1.0 - lam) * bc_S_jet(K, n, u, lam) - n * bc_S_jet(K, n - 1, u, lam);
}

// ------------------------------------------- the limits at the ends of the two domains
//
// The forms above give NaN at an end of either domain although the limit exists. At an infinite
// argument the cause is the one the forms were written for: each of them passes through exprel or
// log1prel so that lambda = 0 needs no branch, and at an infinite argument those two primitives
// are themselves evaluated at an infinite argument, where each is inf/inf. At a finite end the
// cause is different and the point is finite: a prefactor that vanishes meets a power that is
// infinite, which is 0 times inf.
//
// The ends. g has domain (0, inf), so they are v = 0 and v = +inf whatever lambda is. The domain
// of h is the image of g: (-1/lambda, inf) for lambda > 0, so u = -1/lambda and u = +inf;
// (-inf, -1/lambda) for lambda < 0, so u = -inf and u = -1/lambda; the whole line for lambda = 0,
// which has no finite end.
//
// The rule. The argument enters every closed form only through a power of v or of w = 1 + lambda u,
// so the sign of that power's exponent decides each limit: where the base runs to +inf the power
// runs to +inf for a positive exponent and dies for a negative one, where the base goes to 0 from
// above the two cases exchange, and on the knife edge where the exponent vanishes the power is 1
// and the prefactor is the whole limit. The exponent is never formed: it vanishes where lambda is
// a small integer (forward) or where a product of lambda with one equals 1 (inverse), so the two
// are compared and never subtracted. A subtraction is what nvcc contracts into an FMA, and the
// exact product of 3 with the double nearest 1/3 is 1 - 2^-54, not 1, so the edge would be taken
// on one platform and not on the other. Each limit is compared with SymPy's own limit of the
// definition by tools/reference/bijector_box_cox_forms.py (check_limits).

// pref w^e, with the base running to +inf (grow) or to 0 from above, and e_sign the sign of the
// exponent. A prefactor vanishes only where the quantity vanishes identically, so it gives 0.
LAPLACE_HD_FORCEINLINE double bc_limit_power(double pref, bool grow, int e_sign) {
  if (e_sign == 0) return pref;
  if (grow ? (e_sign < 0) : (e_sign > 0)) return 0.0;
  return pref > 0.0 ? inf() : (pref < 0.0 ? -inf() : 0.0);
}

// g and its v-derivatives at v = 0 and at v = +inf. The value is the end of the image of g,
// -1/lambda where that end is finite. For m >= 1, g^(m) = F_m(lambda) v^(lambda - m) with
// F_m = prod_(k=1..m-1)(lambda - k), and the exponent vanishes exactly at lambda = m, where
// F_m = (m-1)!; where F_m itself vanishes, that is at an integer lambda below m, g is a
// polynomial of degree lambda and g^(m) is identically 0.
template <int K>
LAPLACE_HD double bc_fwd_at_end(bool at_zero, double lam) {
  if constexpr (K == 0) {
    return at_zero ? (lam > 0.0 ? -1.0 / lam : -inf()) : (lam < 0.0 ? -1.0 / lam : inf());
  } else {
    double pref = 1.0;
    for (int k = 1; k <= K - 1; ++k) pref *= lam - k;
    const int es = lam == double(K) ? 0 : (lam > double(K) ? 1 : -1);
    return bc_limit_power(pref, !at_zero, es);
  }
}

// log|g'| = (lambda - 1) log v, whose m-th v-derivative is (lambda - 1)(-1)^(m-1)(m-1)! v^(-m).
// At lambda = 1 the map is the identity and every order is 0. Otherwise the value runs to an
// infinity of the sign of (lambda - 1) log v, each derivative to one of the sign of its own
// coefficient at v = 0, and each derivative dies at v = +inf.
template <int K>
LAPLACE_HD double bc_fwd_lj_at_end(bool at_zero, double lam) {
  if (lam == 1.0) return 0.0;
  if constexpr (K == 0) {
    const double s = at_zero ? 1.0 - lam : lam - 1.0;
    return s > 0.0 ? inf() : -inf();
  } else {
    if (!at_zero) return 0.0;
    const double s = (lam - 1.0) * bc_sgn(K - 1);
    return s > 0.0 ? inf() : -inf();
  }
}

// h and its u-derivatives at an end of the image of g: h^(K) = P_K(lambda) w^(1/lambda - K) with
// P_K = prod_(r=1..K-1)(1 - r lambda). At u = +inf, an end only for lambda >= 0, w runs to +inf
// and 1 - K lambda carries the sign of the exponent, zero included; lambda = 0, where h = e^u,
// falls under its positive case with P_K = 1. At u = -inf, an end only for lambda <= 0, the
// exponent is negative and h dies with every order of it. At the finite end u = -1/lambda, w goes
// to 0 from above, so the same exponent decides the other way round; below 0 it is negative
// whatever K is, and P_K is then a product of factors above 1.
// end: 1 for u = +inf, -1 for u = -inf, 0 for u = -1/lambda.
template <int K>
LAPLACE_HD double bc_inv_at_end(int end, double lam) {
  if (end < 0) return 0.0;
  double pref = 1.0;
  for (int r = 1; r <= K - 1; ++r) pref *= 1.0 - r * lam;
  if (end == 0 && lam < 0.0) return inf();
  const double kl = double(K) * lam;  // compared with 1, never subtracted from it
  const int es = kl == 1.0 ? 0 : (kl < 1.0 ? 1 : -1);
  return bc_limit_power(pref, end > 0, es);
}

// log|h'| = (1 - lambda) S, S = log(1 + lambda u)/lambda (u itself at lambda = 0), whose K-th
// u-derivative is (1 - lambda)(-1)^(K-1)(K-1)! lambda^(K-1) w^(-K). At lambda = 1 the map is the
// identity and every order is 0. S runs to +inf at u = +inf and to -inf at u = -inf, and at the
// finite end to -inf for lambda > 0 and to +inf for lambda < 0. Every u-derivative carries w^(-K)
// and dies at an infinite u, except at lambda = 0, where log|h'| = u and the first derivative is
// its slope 1; at the finite end each grows to an infinity of the sign of its own coefficient.
// end as in bc_inv_at_end.
template <int K>
LAPLACE_HD double bc_inv_lj_at_end(int end, double lam) {
  if (lam == 1.0) return 0.0;
  if constexpr (K == 0) {
    double s;
    if (end > 0) s = 1.0 - lam;                           // S runs to +inf
    else if (end < 0) s = lam - 1.0;                      // S runs to -inf, and 1 - lambda > 0
    else s = lam > 0.0 ? lam - 1.0 : 1.0 - lam;           // -inf above 0, +inf below
    return s > 0.0 ? inf() : -inf();
  } else if constexpr (K == 1) {
    if (end != 0) return lam == 0.0 ? 1.0 : 0.0;
    return (1.0 - lam) > 0.0 ? inf() : -inf();
  } else {
    if (end != 0) return 0.0;
    const double s = (1.0 - lam) * bc_sgn(K - 1) * bc_pow_int(lam, K - 1);
    return s > 0.0 ? inf() : -inf();
  }
}

// ------------------------------------- the partials at the ends of the two domains
//
// out[n1] is d^(K-n1)_x d^n1_lambda, n1 = 0..K, which are the entries of bijector_partials<K>
// with n2 = 0, the only ones this map writes: index(n1, 0) == n1. The caller has already zeroed
// the array, so the entries with n2 >= 1 stay zero.
//
// The rule of the pure limits carries over. The argument enters each closed form through one
// power, and a lambda-derivative multiplies the form by a logarithm of the argument, which loses
// to a negative power of it and wins against a vanishing exponent: the three-way verdict of the
// exponent's sign is therefore unchanged, and only the value on the knife edge moves, from the
// prefactor itself at n1 = 0 to an infinity above it. The exception is the finite end of the image,
// where the argument is at a finite point and a lambda-derivative brings u/w rather than log w,
// which raises the power of 1/w by one: there the exponent is 1/lambda - K and depends on the
// total order alone, not on how it splits between the argument and lambda.
//
// Derived with SymPy and checked, each rule against SymPy's own limit of the definition, by
// tools/reference/bijector_box_cox_forms.py (check_partials_limits).

// g and its mixed derivatives at v = 0 and v = +inf.
//   d^m_v d^n_lambda g = v^(lambda - m) sum_(p=0..min(m-1,n)) e_(m-1-p)(c) n!/(n-p)! L^(n-p),
// L = log v and e_k the elementary symmetric polynomials of c_r = lambda - 1 - r, r < m - 1. The
// dominant term is the one with the smallest p whose coefficient does not vanish, since a smaller p
// carries a higher power of L; if every coefficient in range vanishes the sum is identically zero.
// At m = 0 the value tends to an end of the image, whose lambda-derivatives are those of -1/lambda
// where that end is finite and infinite otherwise.
template <int K>
LAPLACE_HD void bc_fwd_partials_at_end(bool at_zero, double lam, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    if (n0 == 0) {  // the lambda-derivatives of the end of the image
      const bool finite_end = at_zero ? (lam > 0.0) : (lam < 0.0);
      const double s = (n1 % 2) ? 1.0 : -1.0;  // (-1)^(n1+1)
      if (finite_end) out[n1] = s * bc_fact(n1) / bc_pow_int(lam, n1 + 1);
      else out[n1] = at_zero ? (s > 0.0 ? inf() : -inf()) : inf();
      continue;
    }
    double e[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int j = n0 - 1;
    for (int r = 0; r < j; ++r) {
      const double c = lam - 1.0 - r;
      for (int k = r + 1; k >= 1; --k) e[k] += c * e[k - 1];
    }
    int p0 = -1;
    for (int p = 0; p <= j && p <= n1; ++p)
      if (e[j - p] != 0.0) {
        p0 = p;
        break;
      }
    if (p0 < 0) {  // the sum vanishes identically: so does this derivative of g
      out[n1] = 0.0;
      continue;
    }
    double coef = e[j - p0];
    for (int i = 0; i < p0; ++i) coef *= double(n1 - i);  // n1!/(n1 - p0)!
    // at v = 0 the dominant L^(n1-p0) carries (-1)^(n1-p0); at v = +inf it is positive
    const double sign = (at_zero && ((n1 - p0) % 2)) ? -coef : coef;
    const int es = lam == double(n0) ? 0 : (lam > double(n0) ? 1 : -1);
    if (es == 0) {  // the power is 1, and only p0 == n1 leaves no logarithm
      out[n1] = (p0 == n1) ? coef : (sign > 0.0 ? inf() : -inf());
      continue;
    }
    out[n1] = bc_limit_power(sign, !at_zero, es);
  }
}

// log|g'| = (lambda - 1) log v is linear in lambda, so only n1 = 0 and n1 = 1 are not identically
// zero. At n1 = 0 the quantity is (lambda - 1) d^n0_v log v, which is 0 at lambda = 1 and otherwise
// runs to an infinity of its own sign at v = 0 while every argument derivative dies at v = +inf; at
// n1 = 1 it is d^n0_v log v, free of lambda and so not zero at lambda = 1.
template <int K>
LAPLACE_HD void bc_fwd_lj_partials_at_end(bool at_zero, double lam, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    if (n1 >= 2) {
      out[n1] = 0.0;
      continue;
    }
    if (n1 == 1) {
      if (n0 == 0) out[n1] = at_zero ? -inf() : inf();          // log v itself
      else if (!at_zero) out[n1] = 0.0;                         // v^(-n0) dies
      else out[n1] = ((n0 - 1) % 2) ? -inf() : inf();           // (-1)^(n0-1) v^(-n0)
      continue;
    }
    if (lam == 1.0) {  // the identity: log|g'| is 0 and so is every v-derivative of it
      out[n1] = 0.0;
      continue;
    }
    if (n0 == 0) {
      const double s = at_zero ? 1.0 - lam : lam - 1.0;
      out[n1] = s > 0.0 ? inf() : -inf();
      continue;
    }
    if (!at_zero) {
      out[n1] = 0.0;
      continue;
    }
    const double s = (lam - 1.0) * bc_sgn(n0 - 1);
    out[n1] = s > 0.0 ? inf() : -inf();
  }
}

// h and its mixed derivatives at an INFINITE end of the image.
//   d^n0_u h = P_n0(lambda) w^(1/lambda - n0),   P_n0 = prod_(r=1..n0-1)(1 - r lambda),
// and at an infinite u a lambda-derivative multiplies this by -log w/lambda^2 at leading order, so
// it carries the sign (-1)^n1 and never changes the verdict of the exponent, whose sign is that of
// 1 - n0 lambda. At u = -inf, an end only for lambda <= 0, the exponent is negative whatever n0 is
// and h dies with every derivative of it, pure or mixed. At lambda = 0, where h = e^u, the first
// case covers it with n0 lambda = 0 and P_n0 = 1.
template <int K>
LAPLACE_HD void bc_inv_partials_at_inf(bool upper, double lam, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!upper) {  // u = -inf: everything dies with h
      out[n1] = 0.0;
      continue;
    }
    const int n0 = K - n1;
    double pref = 1.0;
    for (int r = 1; r <= n0 - 1; ++r) pref *= 1.0 - r * lam;
    if (n1 % 2) pref = -pref;                 // (-1)^n1 from the lambda-derivatives
    const double na = double(n0) * lam;       // compared with 1, never subtracted from it
    if (na == 1.0 && n1 == 0) {
      out[n1] = pref;  // the exponent vanishes and no logarithm is left
      continue;
    }
    out[n1] = bc_limit_power(pref, true, na <= 1.0 ? 1 : -1);
  }
}

// log|h'| = (1 - lambda) S, S = log w / lambda, and its mixed derivatives at an INFINITE end.
// Away from lambda = 0 an argument derivative carries w^(-n0) and dies, so only n0 = 0 survives,
// and there the quantity is c_n1 log w with c_n1 = d^n1_lambda[(1 - lambda)/lambda]: the logarithm
// runs to +inf at both infinite ends, so the limit is an infinity of the sign of c_n1, which is
// (1 - lambda)/lambda at n1 = 0 and (-1)^n1 n1!/lambda^(n1+1) above it, and 0 only where c_0
// vanishes, at lambda = 1.
// At lambda = 0 the log-Jacobian is the series (1 - lambda) sum_j (-1)^j u^(j+1) lambda^j/(j+1), so
// d^n1_lambda is a polynomial in u of degree n1 + 1 whose leading coefficient has the sign
// (-1)^n1: its n0-th argument derivative runs to an infinity below n1 + 1, is the constant
// (-1)^n1 (n1!)^2 at n0 = n1 + 1, and vanishes above. The degree n1 + 1 - n0 decides the sign at
// u = -inf, both ends agreeing wherever it is even.
template <int K>
LAPLACE_HD void bc_inv_lj_partials_at_inf(bool upper, double lam, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    if (lam == 0.0) {
      const int deg = n1 + 1 - n0;
      if (deg < 0) {
        out[n1] = 0.0;
      } else if (deg == 0) {
        const double f = bc_fact(n1);
        out[n1] = ((n1 % 2) ? -1.0 : 1.0) * f * f;
      } else {
        double s = (n1 % 2) ? -1.0 : 1.0;
        if (!upper && (deg % 2)) s = -s;
        out[n1] = s > 0.0 ? inf() : -inf();
      }
      continue;
    }
    if (n0 >= 1) {  // w^(-n0) dies and a logarithm cannot save it
      out[n1] = 0.0;
      continue;
    }
    const double c = (n1 == 0) ? (1.0 - lam) / lam
                               : ((n1 % 2) ? -1.0 : 1.0) * bc_fact(n1) / bc_pow_int(lam, n1 + 1);
    out[n1] = c > 0.0 ? inf() : (c < 0.0 ? -inf() : 0.0);
  }
}

// log|h'| and its mixed derivatives at the FINITE end of the image, u = -1/lambda, where w goes to
// 0 from above. There a lambda-derivative brings u/w rather than log w, so the most singular term
// is C u^n1 w^(-n0-n1) and every entry with n1 >= 1 is an infinity; its sign, with u at the end
// itself, is that of -(1 - lambda)(-1)^n0/lambda^(n1+1-n0). At lambda = 1 the factor 1 - lambda
// vanishes and the next term takes over, carrying -1/lambda^2 in its place, which leaves (-1)^n0.
// The column n1 = 0 is the pure log-Jacobian, bc_inv_lj_at_end.
template <int K>
LAPLACE_HD void bc_inv_lj_partials_at_edge(double lam, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    if (n1 == 0) {
      out[n1] = bc_inv_lj_at_end<K>(0, lam);
      continue;
    }
    if (lam == 1.0) {
      out[n1] = (n0 % 2) ? -inf() : inf();
      continue;
    }
    // sign((lambda - 1) (-1)^n0) times sign(lambda) to the parity of n1 + 1 - n0
    double s = (lam - 1.0) * bc_sgn(n0);
    if (lam < 0.0 && ((n1 + 1 - n0) % 2 != 0)) s = -s;
    out[n1] = s > 0.0 ? inf() : -inf();
  }
}

// h and its mixed derivatives at the FINITE end of the image, u = -1/lambda, where w goes to 0 from
// above. A lambda-derivative brings u/w rather than log w and so lowers the power of w by one: the
// exponent is 1/lambda - K and depends on the total order alone. Writing P_n for
// prod_(r=1..n-1)(1 - r lambda), the most singular term is
//   P_n0 u^n1 prod_(i=0..n1-1)(1/lambda - n0 - i) w^(1/lambda - K),
// so above the edge the sign is that of P_n0 (-1)^n1 prod_i (1 - (n0 + i) lambda), each factor
// compared with 1 and never subtracted from it, and the sign of lambda cancels between u^n1 and the
// product. Below 0 the exponent is negative whatever K is, every factor of the product is negative
// and P_n0 is a product of factors above 1, which leaves (-1)^n1.
//
// Two degeneracies, both at lambda = 1/M with M an integer below K, and both measured with SymPy:
// there either P_n0 vanishes (M below n0) or one factor of the product does (M from n0 up), the
// leading term goes with it and the next takes over. The limit is then 0 at n1 = 0, where h^(K) is
// identically 0, and an infinity of sign (-1)^(n0+M) above it.
//
// On the edge itself, K lambda = 1, the power is 1 and no logarithm is left, so every n1 is finite:
// the values are P_K(lambda) (-1/lambda^2)^n1, the ratio between consecutive n1 being -1/lambda^2,
// which is -K^2 there. Measured with SymPy for K = 1 to 6: at K = 6 and lambda = 1/6 the column is
// 5/324, -5/9, 20, -720, 25920, -933120, 33592320, each 36 times the one before it with the sign
// reversed.
template <int K>
LAPLACE_HD void bc_inv_partials_at_edge(double lam, double* out) {
  // the exponent 1/lambda - K: negative for every K below 0, and of the sign of 1 - K lambda above
  int es;
  if (lam < 0.0) {
    es = -1;
  } else {
    const double kl = double(K) * lam;
    es = kl == 1.0 ? 0 : (kl < 1.0 ? 1 : -1);
  }
  if (es == 0) {  // on the edge: finite for every n1
    double pk = 1.0;
    for (int r = 1; r <= K - 1; ++r) pk *= 1.0 - r * lam;
    const double q = -1.0 / (lam * lam);
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = pk * bc_pow_int(q, n1);
    return;
  }
  if (es > 0) {  // the power dies
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = 0.0;
    return;
  }
  int M = 0;  // lambda is 1/M with M an integer below K, if any: the degenerate case
  for (int r = 1; r <= K - 1; ++r)
    if (r * lam == 1.0) {
      M = r;
      break;
    }
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    if (M > 0) {
      out[n1] = (n1 == 0) ? 0.0 : (((n0 + M) % 2) ? -inf() : inf());
      continue;
    }
    double s = 1.0;
    for (int r = 1; r <= n0 - 1; ++r) s *= 1.0 - r * lam;  // P_n0, never 0 here
    if (n1 % 2) s = -s;
    for (int i = 0; i < n1; ++i)
      if (double(n0 + i) * lam > 1.0) s = -s;  // the factor 1 - (n0 + i) lambda is negative
    out[n1] = s > 0.0 ? inf() : -inf();
  }
}

// which end of the image of g the argument is, if any: the finite end is compared with -1/lambda,
// the expression the dispatch itself compared it with, and not with 1 + lambda u, whose rounding
// is not the bound's
LAPLACE_HD_FORCEINLINE int bc_inv_end_of(double u, double lam) {
  if (u == inf()) return 1;
  if (u == -inf()) return -1;
  if (lam != 0.0 && u == -1.0 / lam) return 0;
  return 2;  // inside
}

// ---------------------------------------------------------------- the maps

template <int K>
LAPLACE_HD double box_cox_fwd(double v, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  if (v == 0.0 || v == inf()) return bc_fwd_at_end<K>(v == 0.0, lam);
  return bc_fwd_any(K, 0, v, lam);
}
template <int K>
LAPLACE_HD double box_cox_inv(double u, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  const int end = bc_inv_end_of(u, lam);
  if (end != 2) return bc_inv_at_end<K>(end, lam);
  return bc_inv_any(K, 0, u, lam);
}
template <int K>
LAPLACE_HD double box_cox_fwd_lj(double v, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  if (v == 0.0 || v == inf()) return bc_fwd_lj_at_end<K>(v == 0.0, lam);
  return bc_fwd_lj_any(K, 0, v, lam);
}
template <int K>
LAPLACE_HD double box_cox_inv_lj(double u, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  const int end = bc_inv_end_of(u, lam);
  if (end != 2) return bc_inv_lj_at_end<K>(end, lam);
  return bc_inv_lj_any(K, 0, u, lam);
}

// the components n1 = 0..K (n2 = 0) fill the first K + 1 entries; the others are 0
template <int K>
LAPLACE_HD void box_cox_fwd_partials(double v, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
  if (!bc_finite(lam)) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = nan();
    return;
  }
  if (v == 0.0 || v == inf()) {
    bc_fwd_partials_at_end<K>(v == 0.0, lam, out);
    return;
  }
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = bc_fwd_any(K - n1, n1, v, lam);
}
// The Bell polynomials A_0..A_K of the lambda-derivatives of S do not depend on the component:
// they are formed once (K calls of log1prel_dk) instead of once per component, which cost 1190
// ns per call of the order-6 partials on one gcc thread (measured 2026-11-12).
template <int K>
LAPLACE_HD void box_cox_inv_partials(double u, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
  if (!bc_finite(lam)) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = nan();
    return;
  }
  {
    const int end = bc_inv_end_of(u, lam);
    if (end != 2) {
      if (end == 0) bc_inv_partials_at_edge<K>(lam, out);
      else bc_inv_partials_at_inf<K>(end > 0, lam, out);
      return;
    }
  }
  // At a large lambda u the forms below compute inf times 0, psi^(j)(x) underflowing while
  // u^(j+1) overflows, and the column comes out NaN (measured: 3 of the 7 entries at u = 1e60,
  // 10 at 1e100, 21 at 1e300 over orders 1 to 6). The rescaled route of the yeo_johnson session
  // takes the same structure with q_j = x^(j+1) psi^(j)(x) kept bounded, and is exactly this
  // column: its only Yeo-Johnson-specific line, the -1 of h = exp(S) - 1, is guarded by
  // n0 == 0 && n1 == 0, which needs K = 0 and so never fires for a partial. It takes lambda and u
  // directly and performs no reflection, which is what makes it right here.
  if (Kokkos::fabs(lam * u) > bc_rescale_threshold) {
    yj_inv_partials_rescaled_pos<K>(lam, u, out);
    return;
  }
  double A[7];
  bc_bell(K, u, lam, A);
  const double S = bc_S(u, lam);
  const double t = 1.0 / (1.0 + lam * u);
  double C[K + 1][K + 1];  // the binomial coefficients, by Pascal's rule (exact; bc_binom divides)
  for (int n = 0; n <= K; ++n) {
    C[n][0] = C[n][n] = 1.0;
    for (int k = 1; k < n; ++k) C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
  }
  for (int n1 = 0; n1 <= K; ++n1) {
    const int Kd = K - n1;
    if (Kd == 0) {
      out[n1] = m::exp(S) * A[n1];
      continue;
    }
    double P[7];
    bc_PK_jets(Kd, lam, P);
    double s = 0.0;
    for (int j = 0; j <= n1; ++j)
      for (int i = 0; i <= n1 - j; ++i) {
        const int q = n1 - j - i;
        s += C[n1][j] * C[n1 - j][i] * P[j] * A[i] * bc_sgn(q) * bc_rising(Kd, q) * bc_pow_int(u, q) *
             bc_pow_int(t, q);
      }
    out[n1] = m::exp((1.0 - Kd * lam) * S) * s;
  }
}
template <int K>
LAPLACE_HD void box_cox_fwd_lj_partials(double v, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
  if (!bc_finite(lam)) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = nan();
    return;
  }
  if (v == 0.0 || v == inf()) {
    bc_fwd_lj_partials_at_end<K>(v == 0.0, lam, out);
    return;
  }
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = bc_fwd_lj_any(K - n1, n1, v, lam);
}
// log|h'| = (1 - lambda) log(1 + lambda u)/lambda is the positive branch of Yeo-Johnson's: its
// partials are generated there (scalar_yeo_johnson.hpp), where the mixed ones are written in
// e = 1/(1 + lambda u) for |lambda u| > 1/2 (the sums T_(K,n) above cancel there: up to 49 ulpC
// at orders 4-6, measured 2026-11-12)
template <int K>
LAPLACE_HD void box_cox_inv_lj_partials(double u, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
  if (!bc_finite(lam)) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = nan();
    return;
  }
  // the guard sits here, before the shared branch is called, so that
  // yeo_johnson_inv_lj_pos_partials stays untouched and shared with yeo_johnson
  const int end = bc_inv_end_of(u, lam);
  if (end != 2) {
    if (end == 0) bc_inv_lj_partials_at_edge<K>(lam, out);
    else bc_inv_lj_partials_at_inf<K>(end > 0, lam, out);
    return;
  }
  yeo_johnson_inv_lj_pos_partials<K>(u, lam, out);
}

LAPLACE_HD_FORCEINLINE double box_cox_dom_lo(double p1, double p2) { return bc_finite(p1) ? 0.0 : nan(); }
LAPLACE_HD_FORCEINLINE double box_cox_dom_hi(double p1, double p2) { return bc_finite(p1) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double box_cox_img_lo(double p1, double p2) {
  if (!bc_finite(p1)) return nan();
  return p1 > 0.0 ? -1.0 / p1 : -inf();
}
LAPLACE_HD_FORCEINLINE double box_cox_img_hi(double p1, double p2) {
  if (!bc_finite(p1)) return nan();
  return p1 < 0.0 ? -1.0 / p1 : inf();
}
// the range of h is (0, inf): the distance to the upper bound is inf, to the lower one h(u)
LAPLACE_HD_FORCEINLINE double box_cox_inv_to_upper(double u, double p1, double p2) {
  return bc_finite(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double box_cox_inv_to_lower(double u, double p1, double p2) {
  return box_cox_inv<0>(u, p1, p2);
}

}  // namespace laplace::bijector::detail_scalar
