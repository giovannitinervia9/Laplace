// skewnormal_distrib (laplace::distrib): the skew normal in its centred parametrization,
// the parameters being moments of the law: mu the mean, sigma > 0 the standard deviation
// and gamma in (-g, g) the skewness, with g = ((4 - pi)/2) (b / sqrt(1 - b^2))^3 = 0.99527...,
// b = sqrt(2/pi), the supremum of |gamma| over the family (the half normal, alpha -> +-inf).
//
// The density is the classic skew normal (2/omega) phi(t) Phi(alpha t), t = (y - xi)/omega,
// at the triple that the three moments imply. With
//
//   cc    = sign(gamma) (2 |gamma| / (4 - pi))^(1/3),
//   xi    = mu - sigma cc,   omega = sigma sqrt(1 + cc^2),
//   alpha = cc / (b sqrt(1 - q)),   1 - q = -expm1((2/3) log(|gamma| / g)) = 1 - kappa cc^2,
//
// (kappa = pi/2 - 1; 1 - q = (1 + cc^2)(1 - delta^2), delta = alpha / sqrt(1 + alpha^2); the
// expm1 form keeps 1 - q accurate as |gamma| -> g, where alpha -> infinity), and z = (y - mu)/sigma,
//
//   t = (z + cc)/s,   s = sqrt(1 + cc^2),   u = alpha t,
//   l(mu, sigma, gamma; y) = log 2 - log sigma - log(2 pi)/2 + W(z, cc),
//   W(z, cc) = -log s - t^2/2 + log Phi(u),
//
// with log Phi from math::log_Phi, never log(Phi): Phi underflows in the lower tail.
//
// The route for the derivatives. y and mu enter l only through z, and gamma only through cc,
// so the tensor in (mu, sigma, gamma) is built in two steps.
//  1. The table of mixed derivatives of W in (z, cc), order j in z and k in cc, j + k <= K. Both t
//     and u are linear in z with coefficients that depend on cc alone,
//       t = A(cc) z + B(cc),  u = C(cc) z + D(cc),  A = 1/s, B = cc/s, C = alpha A, D = alpha B,
//     so the derivatives in z are those of the three parts of W by Leibniz (-t^2/2 has z-derivatives
//     0, 1, 2 only) and, for log Phi(u), f_j(u) C^j with f_j = log_Phi_dj, and the derivative in cc of
//     f_j(u(cc)) C(cc)^j is Leibniz over Faa di Bruno's formula for one inner function (the partial
//     Bell polynomials B_(r,m) of math/faa_di_bruno.hpp, with u^(m)(cc) = C^(m) z + D^(m)). The cc
//     derivatives of A, B, C, D, log s and alpha are exact recurrences from the differential equations
//       (1 + x^2) A' = -x A,    (1 - kappa x^2) E' = kappa x E   (E = (1 - kappa x^2)^(-1/2),
//       alpha = x E / b),
//     which are written in terms of 1 - kappa x^2 = 1 - q so that the factor stays accurate at |gamma| -> g.
//  2. The only place where gamma differs from cc: the cc index of the table is replaced by
//     gamma with Faa di Bruno in the single variable gamma, cc(gamma) = cc_0 sign |gamma/g_0|^(1/3):
//       d^c/dgamma^c = sum_(k=1..c) d^k/dcc^k B_(c,k)(cc', cc'', ...),
//       cc^(n) = cc (1/3)(1/3 - 1) ... (1/3 - n + 1) / gamma^n.
//     The blow-up of cc^(n) at gamma -> 0 (like gamma^(1/3 - n)) therefore enters once, in the
//     last step, and not in every term of the composition. The table so obtained, D[c (K + 1) + j]
//     = d^c/dgamma^c d^j/dz^j W, is handed to detail_location_scale::mixed_shape_dk, which writes the
//     tensor in (mu, sigma, gamma); -log sigma adds its own derivatives to the all-sigma component.
// The density is smooth in cc and so in gamma away from 0, but it is not C^inf in gamma at gamma = 0:
// the expansion of l in cc has cc^3 He_3(z)/6 (= gamma He_3(z)/6) and then the kurtosis term of order
// cc^4 = gamma^(4/3), so l_gamma is continuous at 0 and every derivative of order two and above in gamma
// diverges like gamma^(-2/3) (the expectation E[l_gammagamma] stays finite). At gamma = 0 exactly the
// components with at most one gamma index are the limits (l_gamma = He_3(z)/6) and the others are NaN.
//
// The cdf: F(y) = F_SN(t; alpha), math::log_sn_cdf, and 1 - F(y) = F_SN(-t; -alpha). The derivatives of
// log F in (mu, sigma, gamma) come from math::log_sn_cdf_d1 .. _d6 (the table in (t, alpha)) composed
// the same way, now with two inner scalars t(z, cc) and alpha(cc): the z derivatives are
// S_(t^j) A^j, and the cc derivatives of S_(t^j)(t(cc), alpha(cc)) are the bivariate Faa di Bruno
// formula, written as the truncated series of the Taylor polynomial of S_(t^j) composed with the series
// of t and of alpha. At gamma = 0 exactly, d/dgamma log F = -(z^2 - 1) phi(z) / (6 Phi(z)) (from the
// Gram-Charlier term gamma He_3/6 of the density), and its z-derivatives follow by Leibniz on
// log_Phi_dk; higher gamma orders are NaN there.
// The expected information has no closed form: the generic route of generic_info.hpp. The quantile and
// the draw are generic (quantile_by_root, rng_by_inversion).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/owen.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_skewnormal {

inline constexpr double pi_value = 3.14159265358979323846;

// square root for the constants below (Newton; usable in constant expressions on every compiler)
LAPLACE_HD constexpr double csqrt(double x) {
  double r = x > 1.0 ? x : 1.0;
  for (int i = 0; i < 80; ++i) r = 0.5 * (r + x / r);
  return r;
}

// b = sqrt(2/pi), kappa = pi/2 - 1 = (1 - b^2)/b^2 and g = ((4 - pi)/2) (b/sqrt(1 - b^2))^3 = ((4 - pi)/2) kappa^(-3/2)
inline constexpr double b_value = csqrt(2.0 / pi_value);
inline constexpr double kappa = pi_value / 2.0 - 1.0;
inline constexpr double g_max = ((4.0 - pi_value) / 2.0) / (kappa * csqrt(kappa));
// g_max as a double-double, g_max + g_max_lo: with a rounding of g_max alone, |gamma| = g - 1e-10 would
// carry a relative error 1e-6 in 1 - q = -expm1((2/3) log1p((|gamma| - g)/g)), the distance to g being 1e-10
// and the error of the constant 1e-16 (g_max_lo = true g - g_max, from mpmath at 300 bits, checked in the test)
inline constexpr double g_max_lo = -1.9209167999475743e-16;

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// the quantities of the map gamma -> (cc, alpha) that the density reads
struct implied_params {
  double cc;     // sign(gamma) (2 |gamma| / (4 - pi))^(1/3)
  double s2;     // 1 + cc^2
  double omq;    // 1 - q = 1 - kappa cc^2 = -expm1((2/3) log(|gamma| / g))
  double alpha;  // cc / (b sqrt(1 - q))
};

LAPLACE_HD implied_params imply(double g) {
  const double ag = Kokkos::fabs(g);
  const double lx = ag > 0.5 * g_max ? Kokkos::log1p(((ag - g_max) - g_max_lo) / g_max) : math::log(ag / g_max);
  implied_params m;
  m.omq = -Kokkos::expm1((2.0 / 3.0) * lx);
  m.cc = (g < 0.0 ? -1.0 : 1.0) * Kokkos::cbrt(2.0 * ag / (4.0 - pi_value));
  m.s2 = 1.0 + m.cc * m.cc;
  m.alpha = m.cc / (b_value * Kokkos::sqrt(m.omq));
  return m;
}

// out[m] = d^m/dx^m (a b) = sum_r C(m, r) a^(r) b^(m - r), m = 0..N
LAPLACE_HD_FORCEINLINE void leibniz(const double* a, const double* b, int N, double* out) {
  for (int m = 0; m <= N; ++m) {
    double s = 0.0;
    for (int r = 0; r <= m; ++r) s += binomial_coef(m, r) * a[r] * b[m - r];
    out[m] = s;
  }
}

// derivatives in x = cc, order 0..K, of the pieces of the density
struct cc_series {
  double A[7];   // 1/s
  double B[7];   // cc/s
  double ls[7];  // log s
  double al[7];  // alpha
  double C[7];   // alpha A
  double D[7];   // alpha B
};

template <int K>
LAPLACE_HD void fill_cc_series(const implied_params& m, cc_series& S) {
  const double x = m.cc;
  double E[7];
  S.A[0] = 1.0 / Kokkos::sqrt(m.s2);
  E[0] = 1.0 / Kokkos::sqrt(m.omq);
  for (int n = 0; n < K; ++n) {
    const double an1 = n > 0 ? double(n) * n * S.A[n - 1] : 0.0;
    S.A[n + 1] = -((2.0 * n + 1.0) * x * S.A[n] + an1) / m.s2;
    const double en1 = n > 0 ? double(n) * n * E[n - 1] : 0.0;
    E[n + 1] = kappa * ((2.0 * n + 1.0) * x * E[n] + en1) / m.omq;
  }
  for (int n = 0; n <= K; ++n) {
    S.B[n] = x * S.A[n] + (n > 0 ? double(n) * S.A[n - 1] : 0.0);
    S.al[n] = (x * E[n] + (n > 0 ? double(n) * E[n - 1] : 0.0)) / b_value;
  }
  leibniz(S.al, S.A, K, S.C);
  leibniz(S.al, S.B, K, S.D);
  double A2[7];
  leibniz(S.A, S.A, K, A2);
  S.ls[0] = 0.5 * Kokkos::log1p(x * x);
  for (int n = 0; n < K; ++n) S.ls[n + 1] = x * A2[n] + (n > 0 ? double(n) * A2[n - 1] : 0.0);
}

// d^n/dgamma^n cc, n = 1..K, for gamma != 0: cc (1/3)(1/3 - 1) ... (1/3 - n + 1) / gamma^n
template <int K>
LAPLACE_HD void cc_gamma_derivs(double cc, double g, double* cp) {
  cp[1] = cc * (1.0 / 3.0) / g;
  for (int n = 1; n < K; ++n) cp[n + 1] = cp[n] * (1.0 / 3.0 - n) / g;
}

// The table in (z, cc), W[j][k] = d^j/dz^j d^k/dcc^k, to the table in (z, gamma) in the layout of
// detail_location_scale::mixed_shape_dk: D[c (K + 1) + j] = d^c/dgamma^c d^j/dz^j, c + j <= K.
// At gamma = 0 the components with c >= 2 are NaN and those with c = 1 are filled by the caller.
template <int K>
LAPLACE_HD void to_gamma_table(const double (&W)[7][7], double cc, double g, double* D) {
  for (int j = 0; j <= K; ++j) D[j] = W[j][0];
  if (g == 0.0) {
    for (int c = 1; c <= K; ++c)
      for (int j = 0; j + c <= K; ++j) D[c * (K + 1) + j] = c == 1 ? 0.0 : math::detail::quiet_nan();
    return;
  }
  double cp[8] = {};
  if constexpr (K >= 1) cc_gamma_derivs<K>(cc, g, cp);
  for (int c = 1; c <= K; ++c)
    for (int j = 0; j + c <= K; ++j) {
      double s = 0.0;
      for (int k = 1; k <= c; ++k) s += W[j][k] * math::bell_partial(c, k, cp);
      D[c * (K + 1) + j] = s;
    }
}

// W(z, cc) = -log s - t^2/2 + log Phi(u): the table W[j][k], j + k <= K
template <int K>
LAPLACE_HD void density_table_cc(double z, const implied_params& m, double (&W)[7][7]) {
  cc_series S;
  fill_cc_series<K>(m, S);
  double tz[7], uz[7];
  for (int r = 0; r <= K; ++r) {
    tz[r] = S.A[r] * z + S.B[r];
    uz[r] = S.C[r] * z + S.D[r];
  }
  const double u0 = uz[0];
  double f[7];
  f[0] = math::log_Phi(u0);
  if constexpr (K >= 1) f[1] = math::log_Phi_d1(u0);
  if constexpr (K >= 2) f[2] = math::log_Phi_d2(u0);
  if constexpr (K >= 3) f[3] = math::log_Phi_d3(u0);
  if constexpr (K >= 4) f[4] = math::log_Phi_d4(u0);
  if constexpr (K >= 5) f[5] = math::log_Phi_d5(u0);
  if constexpr (K >= 6) f[6] = math::log_Phi_d6(u0);
  double TT[7], TA[7], A2[7];
  leibniz(tz, tz, K, TT);
  leibniz(tz, S.A, K, TA);
  leibniz(S.A, S.A, K, A2);
  double Pw[7][7] = {};  // Pw[j][n] = d^n/dcc^n C^j
  Pw[0][0] = 1.0;
  for (int j = 1; j <= K; ++j) leibniz(Pw[j - 1], S.C, K, Pw[j]);
  double G[7][7] = {};  // G[j][r] = d^r/dcc^r f_j(u(cc))
  for (int j = 0; j <= K; ++j)
    for (int r = 0; j + r <= K; ++r) {
      if (r == 0) {
        G[j][0] = f[j];
      } else {
        double s = 0.0;
        for (int mm = 1; mm <= r; ++mm) s += f[j + mm] * math::bell_partial(r, mm, uz);
        G[j][r] = s;
      }
    }
  for (int j = 0; j <= K; ++j)
    for (int k = 0; j + k <= K; ++k) {
      double w = 0.0;
      if (j == 0) w -= S.ls[k] + 0.5 * TT[k];
      else if (j == 1) w -= TA[k];
      else if (j == 2) w -= A2[k];
      for (int r = 0; r <= k; ++r) w += binomial_coef(k, r) * Pw[j][k - r] * G[j][r];
      W[j][k] = w;
    }
}

// the table of the log density in (z, gamma) at gamma = g
template <int K>
LAPLACE_HD void density_table(double z, double g, double* D) {
  const implied_params m = imply(g);
  double W[7][7] = {};
  density_table_cc<K>(z, m, W);
  to_gamma_table<K>(W, m.cc, g, D);
  if constexpr (K >= 1) {
    if (g == 0.0) {  // l_gamma = He_3(z)/6 and its z-derivatives
      const double lim[4] = {(z * z * z - 3.0 * z) / 6.0, (z * z - 1.0) / 2.0, z, 1.0};
      for (int j = 0; j + 1 <= K; ++j) D[1 * (K + 1) + j] = j < 4 ? lim[j] : 0.0;
    }
  }
}

// H[p][q] = d^(p+q) H / da^p db^q at (a(0), b(0)) with a(x), b(x) given by their derivatives a[1..R], b[1..R];
// G[r] = d^r/dx^r H(a(x), b(x)), r = 0..R, from the Taylor polynomial of H composed with the series of a and b
// (this is the bivariate Faa di Bruno formula)
LAPLACE_HD void compose2(const double (&H)[7][7], const double* a, const double* b, int R, double* G) {
  double da[7] = {}, db[7] = {};
  double mf = 1.0;
  for (int mm = 1; mm <= R; ++mm) {
    mf *= mm;
    da[mm] = a[mm] / mf;
    db[mm] = b[mm] / mf;
  }
  double pa[7][7] = {}, pb[7][7] = {};
  pa[0][0] = 1.0;
  pb[0][0] = 1.0;
  for (int p = 1; p <= R; ++p)
    for (int n = 0; n <= R; ++n) {
      double sa = 0.0, sb = 0.0;
      for (int i = 0; i <= n; ++i) {
        sa += pa[p - 1][i] * da[n - i];
        sb += pb[p - 1][i] * db[n - i];
      }
      pa[p][n] = sa;
      pb[p][n] = sb;
    }
  double acc[7] = {};
  for (int p = 0; p <= R; ++p)
    for (int q = 0; p + q <= R; ++q) {
      const double coef = H[p][q] / (fact(p) * fact(q));
      for (int n = p + q; n <= R; ++n) {
        double s = 0.0;
        for (int i = 0; i <= n; ++i) s += pa[p][i] * pb[q][n - i];
        acc[n] += coef * s;
      }
    }
  G[0] = H[0][0];
  for (int r = 1; r <= R; ++r) G[r] = acc[r] * fact(r);
}

// the table of log F (sgn = +1) or log(1 - F) (sgn = -1) in (z, cc): W[j][k], j + k <= K
template <int K>
LAPLACE_HD void cdf_table_cc(double z, const implied_params& m, double sgn, double (&W)[7][7]) {
  cc_series S;
  fill_cc_series<K>(m, S);
  double ta[7], aa[7], Aq[7];
  for (int r = 0; r <= K; ++r) {
    ta[r] = sgn * (S.A[r] * z + S.B[r]);
    aa[r] = sgn * S.al[r];
    Aq[r] = sgn * S.A[r];
  }
  double M[7][7] = {};  // M[n][q] = d^n / dt^(n - q) dalpha^q of log F_SN
  M[0][0] = math::log_sn_cdf(ta[0], aa[0]);
  if constexpr (K >= 1) {
    const auto r = math::log_sn_cdf_d1(ta[0], aa[0]);
    for (int q = 0; q <= 1; ++q) M[1][q] = r.c[q];
  }
  if constexpr (K >= 2) {
    const auto r = math::log_sn_cdf_d2(ta[0], aa[0]);
    for (int q = 0; q <= 2; ++q) M[2][q] = r.c[q];
  }
  if constexpr (K >= 3) {
    const auto r = math::log_sn_cdf_d3(ta[0], aa[0]);
    for (int q = 0; q <= 3; ++q) M[3][q] = r.c[q];
  }
  if constexpr (K >= 4) {
    const auto r = math::log_sn_cdf_d4(ta[0], aa[0]);
    for (int q = 0; q <= 4; ++q) M[4][q] = r.c[q];
  }
  if constexpr (K >= 5) {
    const auto r = math::log_sn_cdf_d5(ta[0], aa[0]);
    for (int q = 0; q <= 5; ++q) M[5][q] = r.c[q];
  }
  if constexpr (K >= 6) {
    const auto r = math::log_sn_cdf_d6(ta[0], aa[0]);
    for (int q = 0; q <= 6; ++q) M[6][q] = r.c[q];
  }
  double Pw[7][7] = {};  // Pw[j][n] = d^n/dcc^n (sgn A)^j
  Pw[0][0] = 1.0;
  for (int j = 1; j <= K; ++j) leibniz(Pw[j - 1], Aq, K, Pw[j]);
  for (int j = 0; j <= K; ++j) {
    const int R = K - j;
    double H[7][7] = {};
    for (int p = 0; p <= R; ++p)
      for (int q = 0; p + q <= R; ++q) H[p][q] = M[j + p + q][q];
    double G[7] = {};
    compose2(H, ta, aa, R, G);
    for (int k = 0; k <= R; ++k) {
      double w = 0.0;
      for (int r = 0; r <= k; ++r) w += binomial_coef(k, r) * Pw[j][k - r] * G[r];
      W[j][k] = w;
    }
  }
}

template <int K>
LAPLACE_HD void cdf_table(double z, double g, double sgn, double* D) {
  const implied_params m = imply(g);
  double W[7][7] = {};
  cdf_table_cc<K>(z, m, sgn, W);
  to_gamma_table<K>(W, m.cc, g, D);
  if constexpr (K >= 1) {
    if (g == 0.0) {
      // d/dgamma log F = -(z^2 - 1) R(z) / 6 and d/dgamma log(1 - F) = (z^2 - 1) R(-z) / 6, R = phi/Phi = log_Phi_d1
      double Rd[7];  // d^i/dz^i R(sgn z)
      Rd[0] = math::log_Phi_d1(sgn * z);
      if constexpr (K >= 2) Rd[1] = sgn * math::log_Phi_d2(sgn * z);
      if constexpr (K >= 3) Rd[2] = math::log_Phi_d3(sgn * z);
      if constexpr (K >= 4) Rd[3] = sgn * math::log_Phi_d4(sgn * z);
      if constexpr (K >= 5) Rd[4] = math::log_Phi_d5(sgn * z);
      if constexpr (K >= 6) Rd[5] = sgn * math::log_Phi_d6(sgn * z);
      for (int j = 0; j + 1 <= K; ++j) {
        double s = (z * z - 1.0) * Rd[j];
        if (j >= 1) s += 2.0 * z * j * Rd[j - 1];
        if (j >= 2) s += double(j) * (j - 1) * Rd[j - 2];
        D[1 * (K + 1) + j] = -sgn / 6.0 * s;
      }
    }
  }
}

}  // namespace detail_skewnormal

struct skewnormal_distrib {
  /// Mean.
  double mu;
  /// Standard deviation.
  double sigma;
  /// Skewness, in (-g, g) with g = 0.99527...
  double gamma;

  static constexpr distrib_kind kind = distrib_kind::continuous;
  static constexpr int n_params = 3;
  static constexpr int y_regularity = infinitely_smooth;
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      // gamma carries a non-smooth point of its own, at gamma = 0: the map to
      // the classic triple runs through a cube root, cc is proportional to
      // |gamma|^(1/3), and the expansion of l in cc gives gamma He_3(z)/6 and
      // then a term of order cc^4, that is |gamma|^(4/3). So at zero skewness
      // the first derivative in gamma is continuous and every one above it
      // diverges like gamma^(-2/3); the expected information stays finite
      // there (1/6 in its gamma entry), while E[l_gamma gamma] carries a
      // term in gamma^(2/3) and E[l_mu gamma] one in gamma^(4/3), so the
      // derivatives of the information in gamma are infinite at zero.
      //
      // The descriptor says infinitely_smooth all the same, because
      // `regularity` and `param_kinks` describe a kink in y whose position
      // depends on the parameter (the Laplace's and the enet's mu = y), which
      // this family does not have: for every gamma other than zero the law is
      // C^inf in all three parameters, and nothing jumps in y. Declaring a
      // finite regularity here would make the generic information drop the
      // derivatives in gamma at every gamma, not only at the one point where
      // they do not exist. A descriptor for a non-smooth point on a parameter's
      // own axis is what the interface lacks (note/piano_distrib_reali.md).
      {"gamma", -detail_skewnormal::g_max, detail_skewnormal::g_max, false, infinitely_smooth, false,
       infinitely_smooth}};

  /// The log density log f(y; mu, sigma, gamma).
  LAPLACE_HD double logpdf(double y) const {
    const auto m = detail_skewnormal::imply(gamma);
    const double z = (y - mu) / sigma, s = Kokkos::sqrt(m.s2), t = (z + m.cc) / s;
    return 0.69314718055994530942 - math::log(sigma) - 0.5 * Kokkos::log1p(m.cc * m.cc) -
           0.91893853320467274178 - 0.5 * t * t + math::log_Phi(m.alpha * t);
  }

  /// The derivatives of order K of the log density in (mu, sigma, gamma), a sym_tensor<3, K>.
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double D[(K + 1) * (K + 1)] = {};
    detail_skewnormal::density_table<K>(z, gamma, D);
    sym_tensor<3, K> t = detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
    int idx[K];
    for (int i = 0; i < K; ++i) idx[i] = 1;
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[sym_tensor_position<3, K>(idx)] += detail_skewnormal::sign_of_power(K) * detail_skewnormal::fact(K - 1) * isk;
    return t;
  }
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  // l depends on y and mu only through y - mu, so d/dy = -d/dmu and the component with a derivatives in
  // mu, b in sigma and c in gamma of d^J/dy^J (order K) is (-1)^J times the component (a + J, b, c) of
  // order J + K.
  /// J derivatives in y and the tensor of order K in (mu, sigma, gamma), J + K <= 6.
  template <int J, int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<3, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<3, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<3, K> r;
      const double s = detail_skewnormal::sign_of_power(J);
      if constexpr (K == 0) {
        r[0] = s * t[0];
      } else {
        for (int b = 0; b <= K; ++b)
          for (int c = 0; b + c <= K; ++c) {
            const int a = K - b - c;
            int ik[K], ij[J + K];
            int p = 0, q = 0;
            for (int i = 0; i < a; ++i) ik[p++] = 0;
            for (int i = 0; i < a + J; ++i) ij[q++] = 0;
            for (int i = 0; i < b; ++i) ik[p++] = 1, ij[q++] = 1;
            for (int i = 0; i < c; ++i) ik[p++] = 2, ij[q++] = 2;
            r[sym_tensor_position<3, K>(ik)] = s * t[sym_tensor_position<3, J + K>(ij)];
          }
      }
      return r;
    }
  }

  /// log F(y) = log_sn_cdf(t, alpha).
  LAPLACE_HD double log_cdf(double y) const {
    const auto m = detail_skewnormal::imply(gamma);
    const double z = (y - mu) / sigma, t = (z + m.cc) / Kokkos::sqrt(m.s2);
    return math::log_sn_cdf(t, m.alpha);
  }
  /// log(1 - F(y)) = log F_SN(-t; -alpha).
  LAPLACE_HD double log_ccdf(double y) const {
    const auto m = detail_skewnormal::imply(gamma);
    const double z = (y - mu) / sigma, t = (z + m.cc) / Kokkos::sqrt(m.s2);
    return math::log_sn_cdf(-t, -m.alpha);
  }
  LAPLACE_HD double cdf(double y) const { return math::exp(log_cdf(y)); }
  LAPLACE_HD double ccdf(double y) const { return math::exp(log_ccdf(y)); }

  /// The derivatives of order K of log F(y) in (mu, sigma, gamma).
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double D[(K + 1) * (K + 1)] = {};
    detail_skewnormal::cdf_table<K>(z, gamma, 1.0, D);
    return detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
  }
  /// The derivatives of order K of log(1 - F(y)) in (mu, sigma, gamma).
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_ccdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double D[(K + 1) * (K + 1)] = {};
    detail_skewnormal::cdf_table<K>(z, gamma, -1.0, D);
    return detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
  }
  LAPLACE_HD sym_tensor<3, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  LAPLACE_HD sym_tensor<3, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  LAPLACE_HD sym_tensor<3, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  LAPLACE_HD sym_tensor<3, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  LAPLACE_HD sym_tensor<3, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  LAPLACE_HD sym_tensor<3, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  LAPLACE_HD sym_tensor<3, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  LAPLACE_HD sym_tensor<3, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  LAPLACE_HD sym_tensor<3, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  LAPLACE_HD sym_tensor<3, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  LAPLACE_HD sym_tensor<3, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  LAPLACE_HD sym_tensor<3, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }

  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  LAPLACE_HD double center() const { return mu; }
  LAPLACE_HD double scale() const { return sigma; }
  // no kink in y for any parameter; the non-smooth point of gamma at zero is
  // on its own axis, which this member does not describe (see params above)
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }

  // Starting value (generic_start.hpp): by construction the parameters are the mean, the standard deviation
  // and the skewness, so the sample's own are the exact moment estimates. The summary carries no skewness
  // field (distrib_sample_summary of core.hpp): when it gets one, named skewness, it is used; until then
  // gamma is estimated from the mean and the median, gamma ~ 6 (mean - median)/sd (Pearson, exact to first
  // order in the skewness). Kept strictly inside (-g, g).
  template <class Summary>
  skewnormal_distrib start_value(const Summary& s) const {
    const double sd = Kokkos::sqrt(s.variance);
    double sk;
    if constexpr (requires { s.skewness; }) sk = s.skewness;
    else sk = sd > 0.0 ? 6.0 * (s.mean - s.median) / sd : 0.0;
    return {start_inside(s.mean, params[0]), start_inside(sd, params[1]), start_inside(sk, params[2])};
  }
};

}  // namespace laplace::distrib
