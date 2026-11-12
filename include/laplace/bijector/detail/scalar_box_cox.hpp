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
// against the domain bounds by the dispatch; a value on a bound gives its limit.
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
LAPLACE_HD double bc_pow_shift(double v, double lam, int m) {
  if (v == 0.0 || v == inf()) return pow_pos(v, lam - m);
  double r = pow_pos(v, lam);
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
  if (n == 0) return bc_falling(lam - 1.0, m - 1) * bc_pow_shift(v, lam, m);
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
  if (m == 0) return n == 0 ? (lam == 1.0 ? 0.0 : (lam - 1.0) * m::log(v)) : m::log(v);
  const double d = bc_sgn(m - 1) * bc_fact(m - 1) * pow_pos(v, -m);  // d^m_v log v
  return n == 0 ? (lam - 1.0) * d : d;
}

// d^K_u d^n_lambda [(1 - lambda) S], the inverse log|h'|
LAPLACE_HD double bc_inv_lj_any(int K, int n, double u, double lam) {
  if (n == 0) return (1.0 - lam) * bc_S_jet(K, 0, u, lam);
  return (1.0 - lam) * bc_S_jet(K, n, u, lam) - n * bc_S_jet(K, n - 1, u, lam);
}

// ---------------------------------------------------------------- the maps

template <int K>
LAPLACE_HD double box_cox_fwd(double v, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  return bc_fwd_any(K, 0, v, lam);
}
template <int K>
LAPLACE_HD double box_cox_inv(double u, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  return bc_inv_any(K, 0, u, lam);
}
template <int K>
LAPLACE_HD double box_cox_fwd_lj(double v, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  return bc_fwd_lj_any(K, 0, v, lam);
}
template <int K>
LAPLACE_HD double box_cox_inv_lj(double u, double lam, double p2) {
  if (!bc_finite(lam)) return nan();
  return bc_inv_lj_any(K, 0, u, lam);
}

// the components n1 = 0..K (n2 = 0) fill the first K + 1 entries; the others are 0
template <int K>
LAPLACE_HD void box_cox_fwd_partials(double v, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = bc_fwd_any(K - n1, n1, v, lam);
}
// The Bell polynomials A_0..A_K of the lambda-derivatives of S do not depend on the component:
// they are formed once (K calls of log1prel_dk) instead of once per component, which cost 1190
// ns per call of the order-6 partials on one gcc thread (measured 2026-11-12).
template <int K>
LAPLACE_HD void box_cox_inv_partials(double u, double lam, double p2, double* out) {
  constexpr int size = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < size; ++i) out[i] = 0.0;
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
