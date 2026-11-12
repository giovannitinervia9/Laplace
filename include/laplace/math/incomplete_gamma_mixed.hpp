// Mixed partial derivatives of log P(a, x) and log Q(a, x) in the shape a and
// the argument x, total orders K = 1 to 6, Laplace's own (a > 0, x >= 0).
// P and Q = 1 - P are the regularized lower and upper incomplete gamma
// functions.
//
// One function per order: own::log_gamma_p_mixed_d1 .. _d6 and
// own::log_gamma_q_mixed_d1 .. _d6 (wrapped in laplace::math). Order K returns
// a gamma_inc_mixed_partials<K>, whose v[j] = d^K log S / da^(K-j) dx^j for
// j = 0..K (j derivatives in x; v[0] is the pure derivative in a, the value
// of log_gamma_p_daK or log_gamma_q_daK up to their measured errors).
//
// Mathematics. The routes of include/laplace/math/incomplete_gamma_deriv.hpp
// (read its top comment first) carry log S(a + h) as a Taylor polynomial in
// the increment h of the shape. Here they carry log S(a + h, x (1 + e)) as a
// polynomial in two increments, h and the relative increment e of x, of total
// degree K; d^(i+j) log S / da^i dx^j = i! j! B_ij / x^j, with B_ij the
// coefficient of h^i e^j. In the variable d = log(t/x) every side is an
// integral whose x-dependence is exp-linear like its a-dependence:
//
//   upper  Gamma(a) Q = x^a e^-x I_U,  I_U = int_0^inf exp(a d + y) dd,
//   lower  Gamma(a + 1) P = x^a e^-x I_L,  I_L = 1 + x int_-inf^0 exp((a + 1) d + y) dd,
//
// with y = x - t = -x (e^d - 1). Raising x to x (1 + e) multiplies the
// integrand by exp(e y), and the prefactor x^a e^-x by exp(a log(1 + e) - x e).
// So the coefficients B_ij with j >= 1 are the joint cumulants of (d, y)
// divided by i! j!, plus the explicit terms of a log(1 + e) - x e and
// h log(1 + e):
//
//   B_01 = a - x + kappa_01,   B_0j = a (-1)^(j-1) / j + kappa_0j / j!   (j >= 2),
//   B_1j = (-1)^(j-1) / j + kappa_1j / j!,   B_ij = kappa_ij / (i! j!)   (i >= 2).
//
// For i >= 2 nothing is added: the mixed partials are cumulants alone. B_01
// = x d log S / dx = +-x f / S (f the gamma density) is taken in closed form
// from the integral of the side (a / I_L for P, -1 / I_U for Q), where the
// sum a - x + kappa_01 would cancel at small x. The point mass of the lower
// side (its integral was taken by parts once, which removes the pole 1/a of
// the cumulants in a) does not move with x while the density part carries a
// factor x (1 + e): with K(h, e) the joint cumulant generating function of
// (d, y) under the normalized measure nu of the side and pi the mass of its
// density part,
//
//   log I_L(h, e) = log I_L + K(h, e) + log(1 + e (pi + (1 - pi) (1 - exp(-K(h, e))))),
//
// whose terms in h alone are those of K: the derivatives in a are unchanged.
//
// The routes, each the bivariate version of the route of the same name in
// incomplete_gamma_deriv.hpp:
// 1. quad: the same rules and nodes; the nodes and weights are stored, and
//    the joint central moments of (d, y) up to total order K are summed in a
//    second pass about the weighted means (Pebay's one-pass update in two
//    variables would cost about 200 multiply-adds per node at K = 6).
// 2. lower series: I_L = sum_n tau_n, tau_n = prod_{l<=n} x/(a + l), and
//    tau_n(x (1 + e)) = tau_n (1 + e)^n, so the coefficient of e^m of term n is
//    C(n, m) tau_n. The series runs until the tail of every sum
//    sum_n C(n, m) tau_n is below 1e-22 of it.
// 3. upper asymptotic series: x I_U = sum_n prod_{l<=n} (a - l)/x, term n times
//    (1 + e)^-n, coefficient (-1)^m C(n + m - 1, m).
// 4. the closed form of the upper side at small x (upper_small_x): every
//    series of it in (h, e); L = a log x - log Gamma(1 + a) gains
//    (a + h) log(1 + e), its quotient c = -L/a gains -log(1 + e), and the
//    series K = sum (-x)^n / (n! (a + n)) gains the factors C(n, m).
// 5. Temme's expansion: lambda = x/a becomes lambda (1 + e)/(1 + h/a), the
//    rest is composed as in temme_side; -a phi gains -x e + (a + h) log(1 + e).
// 6. the complement log(1 - S) from the series of log S as in complement_dk:
//    S(h, e)/S = exp(B(h, e)) (times 1 + h/a for the upper side, whose series
//    are those of log(Q/a) in h), log(1 - S) = log(1 - S) + log(1 - rho (S(h, e)/S - 1)),
//    rho = S / (1 - S).
// 7. the ratio S(a + h, x (1 + e))/S through the density (density_ratio):
//    S(h, e)/S = exp(hR(h)) +- r int_0^e Psi(h, u) du, with hR the terms in h
//    alone of log S (log R for the upper side), r = x f/S, and Psi = f(a + h,
//    x (1 + u))/f(a, x) in closed form, its factor (1 + u)^(a - 1 + h) as a
//    product of linear factors. Used for the complement of the lower side
//    (log Q where P is small, the Poisson cdf far from its mean among them),
//    where exp of the series of log P cancels in e (to C(a, m) and less, for
//    example to 0 at a = 1), and for the terms in e of the closed form 4.
// The fraction in Taylor form (route 7 of incomplete_gamma_deriv.hpp, host
// only) has no bivariate version: its region takes the quadrature. The
// dispatch is that of log_gamma_da_impl otherwise, with three changes. For
// log Q at a < 2, 1/4 <= x < 3/2 the component d^K / da^(K-1) dx comes from the
// quadrature of the upper side (measured on 2026-11-12, order 6, a >= 1: 44
// ulp1 against 1.1e3 for the closed form and the complement). Temme's
// expansion is used from a = 50, not 20, and for the components with three or
// more derivatives in x only within |x/a - 1| < 1/4 (mixed_log_series;
// 2026-11-12, evening and night, measured on the Poisson cdf and at fixed
// ratio); its log G is formed without the exponential and logarithm of the
// Gaussian part (temme_log_g), and on the side wanted also beyond x = a while
// zeta >= -1 (temme_zeta_cross).
//
// Accuracy (docs/numerics.md, "Mixed partial derivatives of log P(a, x) and
// log Q(a, x)"): log P within the errors of the derivatives in a of the same
// order (651 ulp at order 6, at a < 0.1); log Q for a >= 1 within 235 ulp at
// order 6. For a < 1 and x below about 3/2, the components of log Q with one
// to three derivatives in a and at least one in x lose up to 1e5 ulp1 at order
// 6 (1e3 to 3e4 at orders 4 and 5): there log Q = log a + log R with the pole of
// log a, and every route forms these components from series whose terms in h
// carry powers of 1/a (the complement of P) or from the cumulants of the closed
// form, which cancel between the terms of order one of its factor F.
//
// Bivariate series. A truncated series is an array c[i][j], i + j <= K. The
// logarithm L of F follows from the Euler operator theta = h d/dh + e d/de,
// theta F = F theta L: (i + j) L_ij F_00 = (i + j) F_ij - sum (p + q) L_pq F_(i-p, j-q)
// over (p, q) != (0, 0), (i, j); the exponential E = exp(L) from
// (i + j) E_ij = sum (p + q) L_pq E_(i-p, j-q). These are the recursions of
// cumulants_from_moments and moments_from_cumulants in two variables.
// Derived from the formulas above; no code copied.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/incomplete_gamma_deriv.hpp>

#if defined(__CUDACC__)
#define LAPLACE_GM_NOINLINE __noinline__
#else
#define LAPLACE_GM_NOINLINE
#endif

namespace laplace::math {

/**
 * \ingroup incomplete_gamma
 * \brief Return struct for mixed partial derivatives of total order \f$K\f$ of \f$\log P(a, x)\f$ or \f$\log Q(a, x)\f$.
 *
 * Holds an array `v` of \f$K + 1\f$ elements. The component \f$v_j\f$ contains the mixed partial derivative
 * with \f$K - j\f$ derivatives in the shape \f$a\f$ and \f$j\f$ derivatives in the argument \f$x\f$:
 * \f[
 * v_j = \frac{\partial^K}{\partial a^{K-j} \partial x^j} \log S(a, x), \quad j = 0, \dots, K
 * \f]
 * where \f$S(a, x) = P(a, x)\f$ or \f$S(a, x) = Q(a, x)\f$.
 *
 * In particular, \f$v_0\f$ is the pure derivative with respect to \f$a\f$ alone (the value of
 * `log_gamma_p_daK()` or `log_gamma_q_daK()` up to their measured errors) and \f$v_K\f$ is the
 * pure derivative with respect to \f$x\f$ alone.
 *
 * \tparam K Total derivative order (\f$K \in \{1, \dots, 6\}\f$).
 *
 * \see incomplete_gamma
 */
template <int K>
struct gamma_inc_mixed_partials {
  double v[K + 1];
};

}  // namespace laplace::math

namespace laplace::math::own {

namespace detail_gamma_mixed {

namespace gd = detail_gamma_deriv;

// ---------------------------------------------------------------- bivariate series

// c[i][j]: the coefficient of h^i e^j, i + j <= k <= 6
struct bser {
  double c[7][7];
};

LAPLACE_HD_FORCEINLINE void bzero(bser& p) {
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 7; ++j) p.c[i][j] = 0.0;
}

// r = p q (r may not alias p or q)
LAPLACE_HD void bmul(const bser& p, const bser& q, bser& r, int k) {
  for (int n = 0; n <= k; ++n) {
    for (int i = 0; i <= n; ++i) {
      const int j = n - i;
      double acc = 0.0;
      for (int u = 0; u <= i; ++u)
        for (int w = 0; w <= j; ++w) acc += p.c[u][w] * q.c[i - u][j - w];
      r.c[i][j] = acc;
    }
  }
}

// L = log(F / F_00) (L_00 = 0); F_00 > 0
LAPLACE_HD void blog(const bser& F, bser& L, int k) {
  const double i0 = 1.0 / F.c[0][0];
  L.c[0][0] = 0.0;
  for (int n = 1; n <= k; ++n) {
    for (int i = 0; i <= n; ++i) {
      const int j = n - i;
      double acc = n * F.c[i][j];
      for (int u = 0; u <= i; ++u)
        for (int w = 0; w <= j; ++w) {
          const int s = u + w;
          if (s == 0 || s == n) continue;
          acc -= s * L.c[u][w] * F.c[i - u][j - w];
        }
      L.c[i][j] = acc * i0 / n;
    }
  }
}

// E = exp(L - L_00) (E_00 = 1)
LAPLACE_HD void bexp(const bser& L, bser& E, int k) {
  E.c[0][0] = 1.0;
  for (int n = 1; n <= k; ++n) {
    for (int i = 0; i <= n; ++i) {
      const int j = n - i;
      double acc = 0.0;
      for (int u = 0; u <= i; ++u)
        for (int w = 0; w <= j; ++w) {
          const int s = u + w;
          if (s == 0) continue;
          acc += s * L.c[u][w] * E.c[i - u][j - w];
        }
      E.c[i][j] = acc / n;
    }
  }
}

// r = sum_{m=0}^{k} cm[m] d^m, d without constant term (Horner)
LAPLACE_HD void bcompose(const double* cm, const bser& d, bser& r, int k) {
  bser t;
  bzero(r);
  r.c[0][0] = cm[k];
  for (int m = k - 1; m >= 0; --m) {
    bmul(r, d, t, k);
    r = t;
    r.c[0][0] += cm[m];
  }
}

LAPLACE_HD_FORCEINLINE double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^(m-1) / m, the coefficients of log(1 + e)
LAPLACE_HD_FORCEINLINE double log1p_coef(int m) { return ((m % 2 == 1) ? 1.0 : -1.0) / m; }

// log S as a series in (h, e) on the side computed; with direct false and the
// upper side, the terms in h alone are those of log(Q/a) (see complement_dk)
struct mixed_side {
  bool lower, direct;
  double log_s;
  bser B;
  double hR[7];  // the terms in h alone of log P (lower) or log(Q/a) (upper)
};

// the terms in h alone from the cumulants kap[1..k] of d (as from_cumulants)
LAPLACE_HD void pure_a_terms(double a, double x, int k, bool lower, bool direct, const double* kap, mixed_side& s) {
  const bool q = !lower && direct;
  const double lr = gd::log_ratio(x, a);
  s.hR[1] = (lr + kap[1]) - gd::psi1p_minus_log(a);
  for (int j = 2; j <= k; ++j) s.hR[j] = (kap[j] - math::polygamma(j - 1, a + 1.0)) / fact(j);
  if (q) {
    s.B.c[1][0] = (lr + kap[1]) - gd::psi_minus_log(a);
    for (int j = 2; j <= k; ++j) s.B.c[j][0] = (kap[j] - math::polygamma(j - 1, a)) / fact(j);
  } else {
    for (int j = 1; j <= k; ++j) s.B.c[j][0] = s.hR[j];
  }
}

// the terms with e from the cumulant series K of the side (its coefficients
// with j >= 1) and the explicit terms of (a + h) log(1 + e) - x e; B_01 is
// set by the caller
LAPLACE_HD void mixed_terms(double a, int k, const bser& Kc, double extra_a, bser& B) {
  for (int m = 2; m <= k; ++m) B.c[0][m] = (a + extra_a) * log1p_coef(m) + Kc.c[0][m];
  for (int m = 1; m + 1 <= k; ++m) B.c[1][m] = log1p_coef(m) + Kc.c[1][m];
  for (int i = 2; i <= k; ++i)
    for (int m = 1; i + m <= k; ++m) B.c[i][m] = Kc.c[i][m];
}

// ---------------------------------------------------------------- 1. the quadrature

// the nodes of the rules of incomplete_gamma_deriv.hpp, kept for two passes
// (at most 2 x 57 tanh-sinh nodes and 158 nodes of the exp-exp rule)
struct node_store {
  static constexpr int cap = 288;
  double W = 0;
  bool six = false;
  int n = 0;
  double wv[cap], dv[cap];
  LAPLACE_HD void add(double w, double d) {
    W += w;
    if (n < cap) {
      wv[n] = w;
      dv[n] = d;
      ++n;
    }
  }
};

LAPLACE_HD void quad_mixed(double a, double x, int k, bool lower, bool direct, mixed_side& s) {
  node_store c;
  c.six = k >= 6;
  if (lower) {
    // the point mass at d = 0 (y = 0) is added below, not here: the rule stops
    // when a node adds less than 1e-20 of the running weight, and the terms in
    // e come from the density part alone, which is x/(a + 1) of the point mass
    // at small x (with the point mass in the running weight the rule stopped at
    // the first node past t = 0 and lost a factor 61 at x = 3e-24)
    gd::lower_rule<8, 0>(c, (a - x) + 1.0, x);
  } else if (x < 0.25) {
    const double L = -math::log(x);
    const gd::log_integrand g{a - x, x};
    if (L > 4.0) {
      gd::tanh_sinh_quad<8, 0>(c, 0.0, L - 3.0, g);
      gd::tanh_sinh_quad<8, 0>(c, L - 3.0, 3.0, g);
    } else {
      gd::tanh_sinh_quad<8, 0>(c, 0.0, L, g);
    }
    gd::upper_rule<8, 0>(c, a, x, 1.0, L);
  } else {
    gd::upper_rule<8, 0>(c, a, x, x, 0.0);
  }
  // first pass: the weighted means of d and y = -x expm1(d)
  double yv[node_store::cap];
  double Wd = 0.0, sd = 0.0, sy = 0.0;  // Wd: the weight of the nodes (the density part)
  for (int i = 0; i < c.n; ++i) {
    yv[i] = -x * Kokkos::expm1(c.dv[i]);
    Wd += c.wv[i];
    sd += c.wv[i] * c.dv[i];
    sy += c.wv[i] * yv[i];
  }
  const double W = (lower ? 1.0 : 0.0) + Wd;
  const double iW = 1.0 / W;
  const double md = sd * iW, my = sy * iW;
  // second pass: sum w (d - md)^p (y - my)^q, p + q <= k
  bser M;
  bzero(M);
  auto acc = [&](double w, double dd, double yy) {
    double dp[7], yp[7];
    dp[0] = yp[0] = 1.0;
    for (int p = 1; p <= k; ++p) dp[p] = dp[p - 1] * dd, yp[p] = yp[p - 1] * yy;
    for (int p = 0; p <= k; ++p) {
      const double wd = w * dp[p];
      for (int q = 0; p + q <= k; ++q) M.c[p][q] += wd * yp[q];
    }
  };
  if (lower) acc(1.0, -md, -my);
  for (int i = 0; i < c.n; ++i) acc(c.wv[i], c.dv[i] - md, yv[i] - my);
  for (int p = 0; p <= k; ++p)
    for (int q = 0; p + q <= k; ++q) M.c[p][q] *= iW / (fact(p) * fact(q));
  bser Kc;
  blog(M, Kc, k);
  Kc.c[1][0] += md;
  Kc.c[0][1] += my;
  double kap[7];
  for (int j = 1; j <= k; ++j) kap[j] = fact(j) * Kc.c[j][0];
  s.lower = lower;
  s.direct = direct;
  bzero(s.B);
  pure_a_terms(a, x, k, lower, direct, kap, s);
  if (lower) {
    // the factor 1 + e of the density part: Lam = log(1 + e D),
    // D = pi + (1 - pi) (1 - exp(-K)), 1 - pi = 1/W
    bser nK, Ei, Y, Lam;
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j) nK.c[i][j] = -Kc.c[i][j];
    bexp(nK, Ei, k);
    bzero(Y);
    Y.c[0][0] = 1.0;
    for (int i = 0; i <= k; ++i)
      for (int j = 1; i + j <= k; ++j) {
        // coefficient (i, j - 1) of D
        // pi = Wd / W, not (W - 1)/W: at x = 1e-24 the density part is 1e-24 of W
        Y.c[i][j] = (i == 0 && j == 1) ? Wd * iW : -iW * Ei.c[i][j - 1];
      }
    blog(Y, Lam, k);
    for (int i = 0; i <= k; ++i)
      for (int j = 1; i + j <= k; ++j) Kc.c[i][j] += Lam.c[i][j];
    mixed_terms(a, k, Kc, 0.0, s.B);
    s.B.c[0][1] = a * iW;
    s.log_s = detail_igamma::log_prefactor(a, x) + math::log(W);
  } else {
    mixed_terms(a, k, Kc, 0.0, s.B);
    s.B.c[0][1] = -iW;
    s.log_s = detail_igamma::log_prefactor(a, x) + math::log(W) + math::log(a);
  }
}

// ---------------------------------------------------------------- 2, 3. the series

// The lower series (lower true) or the upper asymptotic series in (h, e); as
// gd::series_moments, false when the first pass has not converged within nmax
// terms.
LAPLACE_HD bool series_mixed(double a, double x, bool lower, bool direct, int k, int nmax, mixed_side& s) {
  constexpr double tol = 1e-22;
  const double ix = 1.0 / x;
  // first pass: S0, S1 (the first Taylor coefficient in h) and the sums of
  // the coefficients of e^m, em[m]; cb[m] = C(n, m) (lower) or C(n + m - 1, m)
  double t0 = 1.0, t1 = 0.0, s0 = 1.0, s1 = 0.0;
  double cb[7] = {1, 0, 0, 0, 0, 0, 0}, em[7] = {1, 0, 0, 0, 0, 0, 0};
  int n_end = -1;
  if (lower) {
    for (int n = 1; n <= nmax; ++n) {
      const double z = -1.0 / (a + n), q = -x * z;
      t1 = q * (t1 + z * t0);
      t0 = q * t0;
      s0 += t0;
      s1 += t1;
      for (int m = k; m >= 1; --m) cb[m] += cb[m - 1];
      bool done = n >= k && t0 <= tol * s0;
      for (int m = 1; m <= k; ++m) {
        em[m] += cb[m] * t0;
        if (!(cb[m] * t0 <= tol * em[m])) done = false;
      }
      if (done) {
        n_end = n;
        break;
      }
    }
  } else {
    for (int n = 1; n <= nmax; ++n) {
      const double q = (a - n) * ix;
      if (q <= -1.0) break;  // past the smallest term: the series diverges
      t1 = q * t1 + t0 * ix;
      t0 = q * t0;
      s0 += t0;
      s1 += t1;
      for (int m = 1; m <= k; ++m) cb[m] += cb[m - 1];
      bool done = Kokkos::fabs(t0) <= tol * s0 && Kokkos::fabs(t1) <= tol * (Kokkos::fabs(s1) + s0);
      for (int m = 1; m <= k; ++m) {
        const double tm = ((m % 2 == 1) ? -cb[m] : cb[m]) * t0;
        em[m] += tm;
        if (!(Kokkos::fabs(tm) <= tol * (Kokkos::fabs(em[m]) + s0))) done = false;
      }
      if (done) {
        n_end = n;
        break;
      }
    }
  }
  if (n_end < 0) return false;
  const double mu = s1 / s0;
  // second pass: T[i] the Taylor coefficients in h of tau_n e^(-mu h)
  double T[7];
  T[0] = 1.0;
  for (int i = 1; i <= k; ++i) T[i] = T[i - 1] * (-mu) / i;
  bser S;
  bzero(S);
  for (int i = 0; i <= k; ++i) S.c[i][0] = T[i];
  for (int m = 0; m <= k; ++m) cb[m] = m == 0 ? 1.0 : 0.0;
  for (int n = 1; n <= n_end; ++n) {
    if (lower) {
      const double z = -1.0 / (a + n), q = -x * z;
      double U = T[0];
      T[0] = q * U;
      for (int i = 1; i <= k; ++i) {
        U = T[i] + z * U;
        T[i] = q * U;
      }
      for (int m = k; m >= 1; --m) cb[m] += cb[m - 1];
    } else {
      const double q = (a - n) * ix;
      for (int i = k; i >= 1; --i) T[i] = q * T[i] + T[i - 1] * ix;
      T[0] = q * T[0];
      for (int m = 1; m <= k; ++m) cb[m] += cb[m - 1];
    }
    for (int m = 0; m <= k; ++m) {
      const double c = lower ? cb[m] : ((m % 2 == 1) ? -cb[m] : cb[m]);
      for (int i = 0; i + m <= k; ++i) S.c[i][m] += T[i] * c;
    }
  }
  bser Kc;
  blog(S, Kc, k);
  Kc.c[1][0] += mu;
  double kap[7];
  for (int j = 1; j <= k; ++j) kap[j] = fact(j) * Kc.c[j][0];
  s.lower = lower;
  s.direct = direct;
  bzero(s.B);
  pure_a_terms(a, x, k, lower, direct, kap, s);
  const double S0 = S.c[0][0];
  if (lower) {
    mixed_terms(a, k, Kc, 0.0, s.B);
    s.B.c[0][1] = a / S0;
    s.log_s = detail_igamma::log_prefactor(a, x) + math::log(S0);
  } else {
    // log I_U = log(x I_U) - log x - log(1 + e)
    mixed_terms(a, k, Kc, -1.0, s.B);
    s.B.c[0][1] = -x / S0;
    s.log_s = detail_igamma::log_prefactor(a, x) + (math::log(S0) - math::log(x)) + math::log(a);
  }
  return true;
}

// ---------------------------------------------------------------- 4. the upper side at small x

// gd::upper_small_x in (h, e), for log Q (direct) only
LAPLACE_HD bool small_x_mixed(double a, double x, int k, mixed_side& s) {
  const double lx = math::log(x);
  const double lg1p = Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
  // L(h, e) = (a + h)(log x + log(1 + e)) - log Gamma(1 + a + h); dz = (L - L_00)/2
  const double L0 = a * lx - lg1p;
  if (!(Kokkos::fabs(L0) <= 2.0)) return false;
  bser dz;
  bzero(dz);
  dz.c[1][0] = 0.5 * (lx - math::digamma(a + 1.0));
  for (int j = 2; j <= k; ++j) dz.c[j][0] = -0.5 * math::polygamma(j - 1, a + 1.0) / fact(j);
  for (int m = 1; m <= k; ++m) dz.c[0][m] = 0.5 * a * log1p_coef(m);
  for (int m = 1; m + 1 <= k; ++m) dz.c[1][m] = 0.5 * log1p_coef(m);
  // lambda(a + h) = log Gamma(1 + a + h)/(a + h), as in gd::upper_small_x
  double lam[7];
  if (a < 0.5) {
    const double* cz = nullptr;
    KOKKOS_IF_ON_HOST((cz = gd::tab::lgam_c;))
    KOKKOS_IF_ON_DEVICE((cz = gd::tab::d_lgam_c;))
    for (int j = 0; j <= k; ++j) lam[j] = 0.0;
    int M = a < 1e-3 ? 12 + (k > 5 ? 4 : 0) : static_cast<int>(-53.4 / math::log(a)) + (k > 5 ? 12 : 8);
    if (M > 90) M = 90;
    for (int m = M; m >= 0; --m) {
      double bin = 1.0;
      for (int j = 0; j <= k && j <= m; ++j) {
        lam[j] = lam[j] * a + cz[m] * bin;
        bin = bin * (m - j) / (j + 1);
      }
    }
  } else {
    const double ia = 1.0 / a;
    lam[0] = lg1p * ia;
    double f = 1.0;
    for (int j = 1; j <= k; ++j) {
      f *= j;
      const double G = (j == 1 ? math::digamma(a + 1.0) : math::polygamma(j - 1, a + 1.0)) / f;
      lam[j] = (G - lam[j - 1]) * ia;
    }
  }
  // c(h, e) = -L/(a + h) = lambda(a + h) - log x - log(1 + e)
  bser cc;
  bzero(cc);
  cc.c[0][0] = lam[0] - lx;
  for (int j = 1; j <= k; ++j) cc.c[j][0] = lam[j];
  for (int m = 1; m <= k; ++m) cc.c[0][m] = -log1p_coef(m);
  // sinhc(z + dz), z = L_00/2
  const double z = 0.5 * L0;
  double sig[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    double zpow[25];
    zpow[0] = 1.0;
    for (int i = 1; i < 25; ++i) zpow[i] = zpow[i - 1] * z;
    double coef = 1.0;
    for (int n = 0; n <= 12; ++n) {
      if (n > 0) coef /= (2.0 * n) * (2.0 * n + 1.0);
      double bin = coef;
      for (int m = 0; m <= k && m <= 2 * n; ++m) {
        sig[m] += bin * zpow[2 * n - m];
        bin = bin * (2 * n - m) / (m + 1);
      }
    }
  }
  bser Sh, E, Kc, t1, t2, F;
  bcompose(sig, dz, Sh, k);
  bexp(dz, E, k);
  const double eh = math::exp(z);
  // K(h, e) = sum_n (-x (1 + e))^n / (n! (a + n + h)): coefficient of h^j e^m
  // sum_n (-x)^n/n! C(n, m) (-1)^j / (a + n)^(j+1)
  bzero(Kc);
  {
    double term = 1.0, cb[7] = {1, 0, 0, 0, 0, 0, 0};
    for (int n = 1; n < 60; ++n) {
      term *= -x / n;
      for (int m = k; m >= 1; --m) cb[m] += cb[m - 1];
      const double u = 1.0 / (a + n);
      double p = term * u;
      for (int j = 0; j <= k; ++j) {
        for (int m = 0; j + m <= k; ++m) Kc.c[j][m] += p * cb[m];
        p *= -u;
      }
      if (n > k && Kokkos::fabs(term) * cb[k] <= 1e-20 * Kokkos::fabs(Kc.c[0][0])) break;
    }
  }
  // F = c S - e^(L/2) E K
  bmul(cc, Sh, t1, k);
  bmul(E, Kc, t2, k);
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) F.c[i][j] = t1.c[i][j] - eh * t2.c[i][j];
  if (!(F.c[0][0] > 0.0)) return false;
  bser lF;
  blog(F, lF, k);
  s.lower = false;
  s.direct = true;
  bzero(s.B);
  // log Q = log(a + h) + L/2 + log F
  double ia_pow = 1.0;
  for (int j = 1; j <= k; ++j) {
    ia_pow /= a;
    s.hR[j] = dz.c[j][0] + lF.c[j][0];
    s.B.c[j][0] = s.hR[j] + log1p_coef(j) * ia_pow;
  }
  for (int i = 0; i <= k; ++i)
    for (int m = 1; i + m <= k; ++m) s.B.c[i][m] = dz.c[i][m] + lF.c[i][m];
  // x d log Q / dx = -x f / Q = -e^(L/2 - x) / F
  s.B.c[0][1] = -math::exp(z - x) / F.c[0][0];
  s.log_s = math::log(a) + z + math::log(F.c[0][0]);
  return true;
}

// ---------------------------------------------------------------- 5. Temme's expansion

// log G for Temme's expansion, G = A0 exp(Lz) + sgn Si (Lz without constant
// term, A0 = sqrt(pi a/2) erfcx(zeta) > 0), as Lz + log(1 + sgn Si exp(-Lz)/A0),
// lG_00 = 0. Formed as blog of A0 exp(Lz) + sgn Si, the coefficients of
// exp(Lz) (the moments of the Gaussian part, of size dz^m with dz ~ sqrt(a/2)
// per derivative in x or c) are taken back to its cumulants, which are small
// differences of them near zeta = 0: the pure derivatives of order 5 and 6 in
// x and c at a >= 300 lost 1e4-3e5 ulp of their value there (measured
// 2026-11-12, night). Here the Gaussian part Lz enters as it is, and only the
// correction, of relative size 1/sqrt(a), goes through exp and log.
LAPLACE_HD void temme_log_g(double A0, double sgn, const bser& Lz, const bser& Si, int k, bser& lG) {
  bser nL, Ei, T, Y, lY;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) nL.c[i][j] = (i + j == 0) ? 0.0 : -Lz.c[i][j];
  bexp(nL, Ei, k);
  bmul(Si, Ei, T, k);
  const double r = sgn / A0;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) Y.c[i][j] = r * T.c[i][j];
  Y.c[0][0] += 1.0;
  blog(Y, lY, k);
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) lG.c[i][j] = (i + j == 0) ? 0.0 : Lz.c[i][j] + lY.c[i][j];
}

// Temme's expansion of the side S wanted (want_p) also on the far side of
// x = a, where zeta = sgn eta sqrt(a/2) < 0, for zeta >= -temme_zeta_cross:
// there log S is taken from Temme's G directly (erfcx and the derivatives of
// log Phi hold for negative arguments), not as the complement log(1 - S') of
// the other side, whose exponential and logarithm of the series in x (or c)
// cancel like moments and cumulants: near x = a the pure derivatives of order
// 5 and 6 of the complement lost up to 3e4 ulp of their value at a = 7827
// (2.9e4 at a zero of the derivative, 900-3.7e3 next to it), against 1-10
// directly (measured 2026-11-12, night). Beyond it the complement is kept,
// since -a phi and log G cancel in the value and the low orders as zeta^2 grows.
constexpr double temme_zeta_cross = 1.0;
// Temme's expansion serves every component from a = temme_all_from within
// |x/a - 1| < temme_band, and outside that band only the components with at
// most two derivatives in x (or c). Measured 2026-11-12 (night), after the
// two repairs above, on the references of incomplete_ratio.py, worst error in
// ulp of the order's scale at order 6 (P, Q): every component from a = 300
// (before) 4.3e3, 5.8e3; from a = 100 4.0e3, 2.3e3; from a = 50 2.6e3, 1.9e3
// (order 5: 9.0e3, 5.2e3 from a = 300; 2.1e3, 900 from a = 50); from a = 20
// the P side lost 8.8e3 at order 6. The band: Temme's series loses on its
// direct side from |x/a - 1| = 0.25-0.3 on (temme_c_band in
// incomplete_gamma_ratio.hpp).
constexpr double temme_all_from = 50.0;
constexpr double temme_band = 0.25;

LAPLACE_HD bool temme_mixed(double a, double x, int k, bool want_p, mixed_side& s, double zeta_min = 0.0) {
  constexpr int off[11] = {0, 15, 28, 39, 48, 55, 64, 71, 76, 79, 80};
  const double sigma = (x - a) / a;
  const bool lower = zeta_min < 0.0 ? want_p : x < a;
  const bool direct = lower == want_p;
  const double lam = x / a, ia = 1.0 / a;
  const double* dd = nullptr;
  const double* cc = nullptr;
  KOKKOS_IF_ON_HOST((dd = gd::tab::eta_d; cc = gd::tab::temme_c;))
  KOKKOS_IF_ON_DEVICE((dd = gd::tab::d_eta_d; cc = gd::tab::d_temme_c;))
  double em[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    const double as = Kokkos::fabs(sigma);
    int M = as < 1e-3 ? 12 : static_cast<int>(-53.4 / math::log(as)) + 8;
    if (M > 90) M = 90;
    gd::taylor_shift(dd, M + 1, sigma, k, em);
  }
  const double eta = em[0];
  // sigma(h, e) - sigma = lambda ((1 + e)/(1 + h/a) - 1)
  bser ds, de;
  bzero(ds);
  {
    double ph = 1.0;  // (-1/a)^i
    for (int i = 0; i <= k; ++i) {
      if (i >= 1) ds.c[i][0] = lam * ph;
      if (i + 1 <= k) ds.c[i][1] = lam * ph;
      ph *= -ia;
    }
  }
  {
    double c0[7];
    c0[0] = 0.0;
    for (int m = 1; m <= k; ++m) c0[m] = em[m];
    bcompose(c0, ds, de, k);
  }
  // powers of de
  bser P[7];
  bzero(P[0]);
  P[0].c[0][0] = 1.0;
  P[1] = de;
  for (int m = 2; m <= k; ++m) bmul(P[m - 1], de, P[m], k);
  // S(h, e) = sum_i C_i(eta(h, e)) (a + h)^-i by Horner in w = 1/(a + h)
  bser w, S, T;
  bzero(w);
  bzero(S);
  {
    double f = ia;
    for (int j = 0; j <= k; ++j) {
      w.c[j][0] = f;
      f *= -ia;
    }
  }
  for (int i = 9; i >= 0; --i) {
    const int b = off[i], e = off[i + 1];
    double cm[7] = {0, 0, 0, 0, 0, 0, 0};
    gd::taylor_shift(cc + b, e - b, eta, k, cm);
    bmul(S, w, T, k);
    for (int p = 0; p <= k; ++p)
      for (int q = 0; p + q <= k; ++q) {
        double acc = 0.0;
        for (int m = 0; m <= p + q; ++m) acc += cm[m] * P[m].c[p][q];
        S.c[p][q] = T.c[p][q] + acc;
      }
  }
  // sqrt((a + h)/a) and its inverse
  bser sq, isq;
  bzero(sq);
  bzero(isq);
  {
    double c = 1.0, c2 = 1.0, f = 1.0;
    for (int j = 0; j <= k; ++j) {
      sq.c[j][0] = c * f;
      isq.c[j][0] = c2 * f;
      c = c * (0.5 - j) / (j + 1);
      c2 = c2 * (-0.5 - j) / (j + 1);
      f *= ia;
    }
  }
  const double sgn = lower ? -1.0 : 1.0;
  const double rt = Kokkos::sqrt(0.5 * a);
  // zeta(h, e) = sgn eta(h, e) sqrt((a + h)/2)
  bser ez, zh, dz;
  for (int p = 0; p <= k; ++p)
    for (int q = 0; p + q <= k; ++q) ez.c[p][q] = sgn * de.c[p][q];
  ez.c[0][0] = sgn * eta;
  bmul(ez, sq, zh, k);
  const double zeta = rt * zh.c[0][0];
  if (!(zeta >= zeta_min)) return false;
  for (int p = 0; p <= k; ++p)
    for (int q = 0; p + q <= k; ++q) dz.c[p][q] = rt * zh.c[p][q];
  dz.c[0][0] = 0.0;
  double Lm[7];
  gd::log_erfcx_taylor(zeta, k, Lm);
  bser Lz, Si, lG;
  Lm[0] = 0.0;
  bcompose(Lm, dz, Lz, k);
  constexpr double sqrt_pi_half = 1.25331413731550025121;
  const double A0 = sqrt_pi_half * Kokkos::sqrt(a) * math::erfcx(zeta);
  // G = sqrt((a + h)/a) (A0 exp(Lz) + sgn S / sqrt((a + h)/a))
  const double G00 = A0 + sgn * S.c[0][0];
  if (!(G00 > 0.0)) return false;
  bmul(S, isq, Si, k);
  temme_log_g(A0, sgn, Lz, Si, k, lG);
  {
    double f = ia;
    for (int i = 1; i <= k; ++i, f *= ia) lG.c[i][0] += 0.5 * log1p_coef(i) * f;
  }
  s.lower = lower;
  s.direct = direct;
  bzero(s.B);
  const double aphi = -a * gd::log1pmx(sigma);
  constexpr double half_log_2pi = 0.91893853320467274178;
  s.log_s = -aphi - 0.5 * math::log(a) - half_log_2pi + math::log(G00);
  // the terms in h alone, as in gd::temme_side
  double f1 = 1.0, f2 = 1.0, ap = 1.0;
  for (int j = 1; j <= k; ++j) {
    if (j >= 2) f1 *= (j - 1);
    if (j >= 3) f2 *= (j - 2);
    const double aj = ap * a;
    double v = fact(j) * lG.c[j][0];
    if (j == 1) {
      v += Kokkos::log1p(sigma);
    } else {
      v -= ((j % 2 == 0) ? 1.0 : -1.0) * f2 / ap;
    }
    v -= 0.5 * ((j % 2 == 1) ? 1.0 : -1.0) * f1 / aj;
    const double vR = lower ? v : v - ((j % 2 == 1) ? 1.0 : -1.0) * f1 / aj;
    s.hR[j] = vR / fact(j);
    s.B.c[j][0] = (direct || lower ? v : vR) / fact(j);
    ap = aj;
  }
  // -a phi gains -x e + (a + h) log(1 + e)
  mixed_terms(a, k, lG, 0.0, s.B);
  s.B.c[0][1] = (a - x) + lG.c[0][1];
  return true;
}

// ---------------------------------------------------------------- 7. the ratio S(h, e)/S through the density

// E(h, e) = S(a + h, x (1 + e)) / S(a, x) for the lower side, R(...)/R with
// R = Q/a for the upper side, formed without the exponential of the series of
// log S in e (whose terms in e^m cancel to C(a, m) and less, for example to
// 0 at a = 1, m >= 2): S(a + h, x (1 + e)) = S(a + h, x) +- x int_0^e f(a + h, x (1 + u)) du,
// so with r = x f / S (= +-B_01) and Psi(h, u) = f(a + h, x (1 + u)) / f(a, x),
//   E(h, e) = exp(hR(h)) +- r int_0^e Psi(h, u) du    (lower: +, upper: - and Psi/(1 + h/a)),
//   Psi(h, u) (a/(a + h)) = x^h Gamma(a + 1)/Gamma(a + 1 + h) (1 + u)^(a - 1 + h) e^(-x u),
// For a < 2 or x < a/2 the binomial series of (1 + u)^(a - 1 + h) is formed as
// products of the linear factors (a - 1 - l + h)/(l + 1), which cancel only
// where a - 1 - l is exactly small. For a >= 2 and x >= a/2 the factor
// (1 + u)^(a - 1 + h) e^(-x u) is the exponential of its logarithm,
// ((a - 1) - x + h) u + (a - 1 + h) sum_(m >= 2) (-1)^(m-1) u^m / m: there the
// product of the binomial coefficients C(a - 1, m) ~ a^m / m! with those of
// e^(-x u), (-x)^m / m!, sums terms of alternating sign that grow like
// (a x)^(m/2) while the result stays of the size of the Gaussian's (measured
// 2026-11-12 on the Poisson cdf's log F, order 6, k = 40 at mu = 40: 3.3e5 ulp
// of the value with the products, 787 with the exponential). Below x = a/2 the
// products are exact for integer a and the exponential lost up to 4.7e4 ulp
// (k = 2 at mu = 0.5). Method 9 keeps the products everywhere.
LAPLACE_HD void density_ratio(double a, double x, int k, const mixed_side& s, bser& E, bool products = false) {
  // exp of the terms in h alone
  bser Bh;
  bzero(Bh);
  for (int i = 1; i <= k; ++i) Bh.c[i][0] = s.hR[i];
  bexp(Bh, E, k);
  if (k < 1) return;
  // A(h) = x^h Gamma(a + 1)/Gamma(a + 1 + h)
  bser lA, A;
  bzero(lA);
  lA.c[1][0] = gd::log_ratio(x, a) - gd::psi1p_minus_log(a);
  for (int i = 2; i <= k; ++i) lA.c[i][0] = -math::polygamma(i - 1, a + 1.0) / fact(i);
  bexp(lA, A, k);
  // (1 + u)^(a - 1 + h) e^(-x u), degree k - 1
  bser Bu, Cu, T, Psi;
  if (products || a < 2.0 || x < 0.5 * a) {
    bzero(Bu);
    Bu.c[0][0] = 1.0;
    for (int m = 1; m <= k - 1; ++m) {
      // column m = column m-1 times (a - m + h)/m
      const double c0 = (a - m) / m, c1 = 1.0 / m;
      for (int i = 0; i + m <= k; ++i) Bu.c[i][m] = c0 * Bu.c[i][m - 1] + (i > 0 ? c1 * Bu.c[i - 1][m - 1] : 0.0);
    }
    bzero(Cu);
    {
      double t = 1.0;
      for (int m = 0; m <= k; ++m) {
        Cu.c[0][m] = t;
        t *= -x / (m + 1);
      }
    }
    bmul(A, Bu, T, k - 1);
  } else {
    bser Lu;
    bzero(Lu);
    Lu.c[0][1] = (a - 1.0) - x;
    if (k >= 3) Lu.c[1][1] = 1.0;
    for (int m = 2; m <= k - 1; ++m) {
      Lu.c[0][m] = (a - 1.0) * log1p_coef(m);
      if (m + 1 <= k - 1) Lu.c[1][m] = log1p_coef(m);
    }
    bexp(Lu, T, k - 1);
    Cu = A;
  }
  bmul(T, Cu, Psi, k - 1);
  if (s.lower) {
    // times (a + h)/a
    const double ia = 1.0 / a;
    for (int i = k - 1; i >= 1; --i)
      for (int m = 0; i + m <= k - 1; ++m) Psi.c[i][m] += Psi.c[i - 1][m] * ia;
  }
  // r = x f / S; the integral in u
  const double r = s.lower ? s.B.c[0][1] : -s.B.c[0][1];
  const double sg = s.lower ? 1.0 : -1.0;
  for (int i = 0; i <= k; ++i)
    for (int m = 1; i + m <= k; ++m) E.c[i][m] = sg * r * Psi.c[i][m - 1] / m;
}

// the terms with e of log S from density_ratio (the direct side)
LAPLACE_HD void density_log(double a, double x, int k, mixed_side& s, bool products = false) {
  bser E, L;
  density_ratio(a, x, k, s, E, products);
  blog(E, L, k);
  for (int i = 0; i <= k; ++i)
    for (int m = 1; i + m <= k; ++m) s.B.c[i][m] = L.c[i][m];
}

// ---------------------------------------------------------------- 6. the complement

// C = log(1 - S) as a series in (h, e), from the series of log S (ratio false)
// or from density_ratio (ratio true)
LAPLACE_HD void complement_mixed(int k, const mixed_side& s, double a, double x, bool ratio, bser& C,
                                 bool products = false) {
  bser B = s.B, E, M;
  B.c[0][0] = 0.0;
  if (ratio) {
    density_ratio(a, x, k, s, E, products);
  } else {
    bexp(B, E, k);
  }
  if (!s.lower) {
    // S(h, e)/S = (1 + h/a) R(h, e)/R
    const double ia = 1.0 / a;
    for (int i = k; i >= 1; --i)
      for (int j = 0; i + j <= k; ++j) E.c[i][j] += E.c[i - 1][j] * ia;
  }
  const double ls = s.log_s;
  const double l1ms = ls < gd::ln_half ? Kokkos::log1p(-math::exp(ls)) : math::log(-Kokkos::expm1(ls));
  const double lrho = ls - l1ms;
  M.c[0][0] = 1.0;
  if (lrho > -700.0) {
    const double rho = math::exp(lrho);
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j)
        if (i + j > 0) M.c[i][j] = -rho * E.c[i][j];
  } else {
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j) {
        if (i + j == 0) continue;
        const double b = E.c[i][j];
        M.c[i][j] = b == 0.0 ? 0.0 : -Kokkos::copysign(math::exp(lrho + math::log(Kokkos::fabs(b))), b);
      }
  }
  blog(M, C, k);
}

// ---------------------------------------------------------------- limits and dispatch

// x = 0 for P: the components with at most one derivative in a are infinite
// (a log x); the others are the cumulants of I_L(a + h, eps) = sum_m eps^m
// prod_{l<=m} 1/(a + l + h), in the unscaled increment eps of x
LAPLACE_HD void p_at_zero(double a, int k, double* v) {
  const double inf = math::detail::pos_inf();
  bser F, L;
  bzero(F);
  double T[7] = {1, 0, 0, 0, 0, 0, 0};
  F.c[0][0] = 1.0;
  for (int m = 1; m <= k; ++m) {
    const double z = -1.0 / (a + m), q = -z;
    double U = T[0];
    T[0] = q * U;
    for (int i = 1; i <= k; ++i) {
      U = T[i] + z * U;
      T[i] = q * U;
    }
    for (int i = 0; i + m <= k; ++i) F.c[i][m] = T[i];
  }
  blog(F, L, k);
  for (int j = 1; j <= k; ++j) {
    const int i = k - j;
    v[j] = i <= 1 ? ((j % 2 == 1) ? inf : -inf) : fact(i) * fact(j) * L.c[i][j];
  }
}

// x = 0 for Q: log Q = -P (1 + O(P)), P ~ x^a / Gamma(a + 1); the leading
// term of d^(i+j)/da^i dx^j is -(a)_j (log x)^i x^(a-j) / Gamma(a + 1), with
// (a)_j = a (a - 1) ... (a - j + 1): 0 for a > j, infinite for a < j; NaN for
// an integer a <= j (where P is a polynomial times e^-x near 0)
LAPLACE_HD void q_at_zero(double a, int k, double* v) {
  const double inf = math::detail::pos_inf();
  for (int j = 1; j <= k; ++j) {
    const int i = k - j;
    if (a > j) {
      v[j] = 0.0;
    } else if (a == Kokkos::floor(a)) {
      v[j] = math::detail::quiet_nan();
    } else {
      // the sign of (a)_j: one minus sign per factor a - l < 0, l = 0..j-1
      const int neg = j - 1 - static_cast<int>(Kokkos::floor(a));
      const bool pos = ((neg % 2 == 0) ? 1 : -1) * ((i % 2 == 0) ? -1 : 1) > 0;
      v[j] = pos ? inf : -inf;
    }
  }
}

// The bivariate route on the side (a, x, lower), as gd::chosen_side without
// the fraction; nl, nu the limits on the series' terms.
// Returns 1 for the closed form at small x, 0 otherwise.
LAPLACE_HD int chosen_mixed(double a, double x, int k, bool lower, bool direct, int nl, int nu, bool smallx,
                            mixed_side& s) {
  if (smallx && direct && !lower && x < 1.5 && a < 2.0 && small_x_mixed(a, x, k, s)) return 1;
  if ((lower || x >= 36.0) && series_mixed(a, x, lower, direct, k, lower ? nl : nu, s)) return 0;
  quad_mixed(a, x, k, lower, direct, s);
  return 0;
}

// method: 0 the chosen route; 1 the chosen route with the limits of the device
// (for measuring them on the host); 2 the quadrature alone (no Temme, no
// series, no closed form); 3 the chosen route without Temme; 4 the chosen
// route without density_ratio (every series in e from the cumulants); 5 the
// chosen route with density_ratio for every complement; 6 log Q for a <= 1,
// x < 3/2 by the quadrature of the upper side (no closed form, no complement);
// 8 the chosen route without the quadrature for the component j = 1 of log Q
// at small x (below); 9 the chosen route with the binomial products in
// density_ratio for every a (before 2026-11-12, evening); 10 the chosen route
// with Temme for every component (no route without Temme for the components
// with three or more derivatives in x at a < 300, below).
//
// The series C(h, e) of log S(a + h, x (1 + e)) - log S(a, x), i + j <= k, for
// 0 < x < inf, and log S in log_s.
// One route (Temme's expansion allowed when temme is true); used_temme says
// whether it was taken.
template <bool Fixed>
LAPLACE_HD void mixed_series_route(int k, double a, double x, bool want_p, int method, bool temme, bser& C,
                                   double& log_s, bool& used_temme) {
  mixed_side s;
  bool done = false, smallx = false;
  if (temme && (Fixed || method == 0 || method == 1 || method >= 4)) {
    const double sg = Kokkos::fabs(x - a) / a;
    const double a_temme = (Fixed || method <= 1) ? 50.0 : 20.0;
    if ((a >= a_temme && sg < 0.3) || (a >= 300.0 && sg < 0.4) || (a >= 1000.0 && sg < 0.5))
      done = temme_mixed(a, x, k, want_p, s, -temme_zeta_cross) || temme_mixed(a, x, k, want_p, s);
  }
  if (!done) {
    const bool lower = gd::side_lower(a, x, want_p);
    const bool direct = lower == want_p;
    int nl = gd::nmax_lower_host, nu = gd::nmax_upper_host;
    KOKKOS_IF_ON_DEVICE((nl = a < 2.0 ? gd::nmax_lower_device : 0; nu = gd::nmax_upper_device;))
    if (!Fixed && method == 1) nl = a < 2.0 ? gd::nmax_lower_device : 0, nu = gd::nmax_upper_device;
    if (!Fixed && method == 2) {
      quad_mixed(a, x, k, lower, direct, s);
    } else if (!Fixed && method == 6 && !want_p && a <= 1.0 && x < 1.5) {
      quad_mixed(a, x, k, false, true, s);
    } else {
      smallx = chosen_mixed(a, x, k, lower, direct, nl, nu, true, s) == 1;
    }
  }
  // density_ratio: for the closed form at small x (whose series in e cancel
  // at small a) and for the complement of the lower side
  const bool use_ratio = Fixed || method != 4;
  const bool products = !Fixed && method == 9;
  if (s.direct) {
    if (smallx && use_ratio) density_log(a, x, k, s, products);
    C = s.B;
    log_s = s.log_s;
  } else {
    complement_mixed(k, s, a, x, use_ratio && (s.lower || (!Fixed && method == 5)), C, products);
    log_s = s.log_s < gd::ln_half ? Kokkos::log1p(-math::exp(s.log_s)) : math::log(-Kokkos::expm1(s.log_s));
  }
  C.c[0][0] = 0.0;
  // d^K log Q / da^(K-1) dx for a < 2, 1/4 <= x < 3/2 from the quadrature of
  // the upper side: the closed form and the complement of the lower side lose
  // there up to 1.1e3 ulp1 at order 6 for a >= 1 (the quadrature 44); the
  // other components of the quadrature are worse (1.3e4 at j = 6)
  if ((Fixed || method <= 1 || method >= 9) && !want_p && a < 2.0 && x >= 0.25 && x < 1.5 && k >= 2) {
    mixed_side u;
    quad_mixed(a, x, k, false, true, u);
    for (int i = 1; i <= k - 1; ++i) C.c[i][1] = u.B.c[i][1];
  }
  used_temme = done;
}

template <bool Fixed>
LAPLACE_HD void mixed_log_series(int k, double a, double x, bool want_p, int method, bser& C, double& log_s) {
  if (!Fixed) method = method < 0 ? 0 : method;
  bool used_temme = false;
  mixed_series_route<Fixed>(k, a, x, want_p, method, true, C, log_s, used_temme);
  // Where |x - a| >= a/4 the components with three or more derivatives in x
  // come from the route without Temme, and below a = 50 Temme is not used
  // (above). Until the night of 2026-11-12 this held for every x at
  // 50 <= a < 300: measured 2026-11-12 (evening) on the references at integer a near the
  // mean (incomplete_ratio.py), order 6, in ulp of the component's largest
  // value over the x of the same a: log Q j = 0 3.1e5 with Temme from a = 20,
  // 1.5e3 now; j = 6 827 and 339; log P j = 4 1.0e5 and 4.1e3. Temme's series
  // in (h, e) has 10 terms in 1/a, differentiated up to six times in a (the
  // likely cause below a = 50). Most of the loss at 50 <= a < 300 came from
  // log_erfcx_taylor and from blog of G (repaired, see temme_log_g and
  // temme_all_from). Where |x - a| >= a/4 Temme's series still loses on its
  // direct side (log P below a, log Q above) on the derivatives of order
  // three and more in x, at every a (temme_band; measured 2026-11-12, night).
  if (used_temme && (a < temme_all_from || Kokkos::fabs(x - a) >= temme_band * a) && k >= 3 && (Fixed || method <= 1)) {
    bser C2;
    double ls2;
    bool u2;
    mixed_series_route<Fixed>(k, a, x, want_p, method, false, C2, ls2, u2);
    for (int j = 3; j <= k; ++j)
      for (int i = 0; i + j <= k; ++i) C.c[i][j] = C2.c[i][j];
  }
}

template <bool Fixed>
LAPLACE_GM_NOINLINE LAPLACE_HD void log_gamma_mixed_impl(int k, double a, double x, bool want_p, int method,
                                                          double* v) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(x >= 0.0) || a == inf) {
    for (int j = 0; j <= k; ++j) v[j] = nan;
    return;
  }
  if (x == 0.0) {
    v[0] = want_p ? (k == 1 ? -inf : -math::polygamma(k - 1, a + 1.0)) : 0.0;
    if (want_p) p_at_zero(a, k, v);
    else q_at_zero(a, k, v);
    return;
  }
  if (x == inf) {
    v[0] = want_p ? 0.0 : (k == 1 ? inf : -math::polygamma(k - 1, a));
    for (int j = 1; j <= k; ++j) v[j] = (!want_p && k == 1) ? -1.0 : 0.0;
    return;
  }
  bser C;
  double log_s;
  mixed_log_series<Fixed>(k, a, x, want_p, method, C, log_s);
  for (int j = 0; j <= k; ++j) {
    double r = fact(k - j) * fact(j) * C.c[k - j][j];
    for (int l = 0; l < j; ++l) r /= x;
    v[j] = r;
  }
}

LAPLACE_HD void log_gamma_mixed(int k, double a, double x, bool want_p, int method, double* v) {
  log_gamma_mixed_impl<false>(k, a, x, want_p, method, v);
}

}  // namespace detail_gamma_mixed

/**
 * \ingroup incomplete_gamma
 * \brief First mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the first mixed partial derivatives of total order \f$K = 1\f$:
 * \f[
 * v_0 = \frac{\partial}{\partial a}\log P(a, x) = \log(x/a) - (\psi(a + 1) - \log a) + \kappa_{10}
 * \f]
 * \f[
 * v_1 = \frac{\partial}{\partial x}\log P(a, x) = \frac{a}{x I_L}
 * \f]
 * where \f$I_L\f$ is the lower incomplete gamma integral, and \f$\kappa_{10}\f$ is the first cumulant
 * of \f$d = \log(t/x)\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<1>` struct containing \f$v_0\f$ and \f$v_1\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\infty\f$, \f$v_1 = +\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = 0.0\f$, \f$v_1 = 0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.37 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 8.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d1(), log_gamma_p_mixed_d2(), log_gamma_p_da1()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<1> log_gamma_p_mixed_d1(double a, double x) {
  math::gamma_inc_mixed_partials<1> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(1, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Second mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the second mixed partial derivatives of total order \f$K = 2\f$:
 * \f[
 * v_0 = \frac{\partial^2}{\partial a^2}\log P(a, x) = \kappa_{20} - \psi'(a + 1)
 * \f]
 * \f[
 * v_1 = \frac{\partial^2}{\partial a \partial x}\log P(a, x) = \frac{1 + \kappa_{11}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^2}{\partial x^2}\log P(a, x) = \frac{-a + \kappa_{02}}{x^2}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<2>` struct containing \f$v_j = \frac{\partial^2 \log P}{\partial a^{2-j} \partial x^j}\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\psi'(a + 1)\f$, \f$v_1 = +\infty\f$, \f$v_2 = -\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Measured accuracy
 * Maximum error 8.03 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 12.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d2(), log_gamma_p_mixed_d1(), log_gamma_p_mixed_d3(), log_gamma_p_da2()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<2> log_gamma_p_mixed_d2(double a, double x) {
  math::gamma_inc_mixed_partials<2> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(2, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Third mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the third mixed partial derivatives of total order \f$K = 3\f$:
 * \f[
 * v_0 = \frac{\partial^3}{\partial a^3}\log P(a, x) = \kappa_{30} - \psi''(a + 1)
 * \f]
 * \f[
 * v_1 = \frac{\partial^3}{\partial a^2 \partial x}\log P(a, x) = \frac{\kappa_{21}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^3}{\partial a \partial x^2}\log P(a, x) = \frac{-1 + \kappa_{12}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^3}{\partial x^3}\log P(a, x) = \frac{2a + \kappa_{03}}{x^3}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<3>` struct containing \f$v_j = \frac{\partial^3 \log P}{\partial a^{3-j} \partial x^j}\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\psi''(a + 1)\f$, \f$v_1\f$ is finite, \f$v_2 = -\infty\f$, \f$v_3 = +\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Measured accuracy
 * Maximum error 12.9 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 20.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d3(), log_gamma_p_mixed_d2(), log_gamma_p_mixed_d4(), log_gamma_p_da3()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<3> log_gamma_p_mixed_d3(double a, double x) {
  math::gamma_inc_mixed_partials<3> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(3, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fourth mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the fourth mixed partial derivatives of total order \f$K = 4\f$:
 * \f[
 * v_0 = \frac{\partial^4}{\partial a^4}\log P(a, x) = \kappa_{40} - \psi'''(a + 1)
 * \f]
 * \f[
 * v_1 = \frac{\partial^4}{\partial a^3 \partial x}\log P(a, x) = \frac{\kappa_{31}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^4}{\partial a^2 \partial x^2}\log P(a, x) = \frac{\kappa_{22}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^4}{\partial a \partial x^3}\log P(a, x) = \frac{2 + \kappa_{13}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^4}{\partial x^4}\log P(a, x) = \frac{-6a + \kappa_{04}}{x^4}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<4>` struct containing \f$v_j = \frac{\partial^4 \log P}{\partial a^{4-j} \partial x^j}\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\psi'''(a + 1)\f$, \f$v_1\f$ and \f$v_2\f$ are finite, \f$v_3 = +\infty\f$, \f$v_4 = -\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Measured accuracy
 * Maximum error 33.4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 50.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d4(), log_gamma_p_mixed_d3(), log_gamma_p_mixed_d5(), log_gamma_p_da4()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<4> log_gamma_p_mixed_d4(double a, double x) {
  math::gamma_inc_mixed_partials<4> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(4, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fifth mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the fifth mixed partial derivatives of total order \f$K = 5\f$:
 * \f[
 * v_0 = \frac{\partial^5}{\partial a^5}\log P(a, x) = \kappa_{50} - \psi^{(4)}(a + 1)
 * \f]
 * \f[
 * v_1 = \frac{\partial^5}{\partial a^4 \partial x}\log P(a, x) = \frac{\kappa_{41}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^5}{\partial a^3 \partial x^2}\log P(a, x) = \frac{\kappa_{32}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^5}{\partial a^2 \partial x^3}\log P(a, x) = \frac{\kappa_{23}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^5}{\partial a \partial x^4}\log P(a, x) = \frac{-6 + \kappa_{14}}{x^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^5}{\partial x^5}\log P(a, x) = \frac{24a + \kappa_{05}}{x^5}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<5>` struct containing \f$v_j = \frac{\partial^5 \log P}{\partial a^{5-j} \partial x^j}\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\psi^{(4)}(a + 1)\f$, \f$v_1, v_2, v_3\f$ are finite, \f$v_4 = -\infty\f$, \f$v_5 = +\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Measured accuracy
 * Maximum error 176.0 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 265.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d5(), log_gamma_p_mixed_d4(), log_gamma_p_mixed_d6(), log_gamma_p_da5()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<5> log_gamma_p_mixed_d5(double a, double x) {
  math::gamma_inc_mixed_partials<5> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(5, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Sixth mixed partial derivatives of \f$\log P(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the sixth mixed partial derivatives of total order \f$K = 6\f$:
 * \f[
 * v_0 = \frac{\partial^6}{\partial a^6}\log P(a, x) = \kappa_{60} - \psi^{(5)}(a + 1)
 * \f]
 * \f[
 * v_1 = \frac{\partial^6}{\partial a^5 \partial x}\log P(a, x) = \frac{\kappa_{51}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^6}{\partial a^4 \partial x^2}\log P(a, x) = \frac{\kappa_{42}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^6}{\partial a^3 \partial x^3}\log P(a, x) = \frac{\kappa_{33}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^6}{\partial a^2 \partial x^4}\log P(a, x) = \frac{\kappa_{24}}{x^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^6}{\partial a \partial x^5}\log P(a, x) = \frac{24 + \kappa_{15}}{x^5}
 * \f]
 * \f[
 * v_6 = \frac{\partial^6}{\partial x^6}\log P(a, x) = \frac{-120a + \kappa_{06}}{x^6}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized lower measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<6>` struct containing \f$v_j = \frac{\partial^6 \log P}{\partial a^{6-j} \partial x^j}\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = -\psi^{(5)}(a + 1)\f$, \f$v_1, v_2, v_3, v_4\f$ are finite, \f$v_5 = +\infty\f$, \f$v_6 = -\infty\f$.
 * - \f$x = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Measured accuracy
 * Maximum error 651.0 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 980.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_mixed_d6(), log_gamma_p_mixed_d5(), log_gamma_p_da6()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<6> log_gamma_p_mixed_d6(double a, double x) {
  math::gamma_inc_mixed_partials<6> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(6, a, x, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief First mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the first mixed partial derivatives of total order \f$K = 1\f$:
 * \f[
 * v_0 = \frac{\partial}{\partial a}\log Q(a, x) = \log(x/a) + (\log a - \psi(a)) + \kappa_{10}
 * \f]
 * \f[
 * v_1 = \frac{\partial}{\partial x}\log Q(a, x) = -\frac{1}{x I_U}
 * \f]
 * where \f$I_U\f$ is the upper incomplete gamma integral, and \f$\kappa_{10}\f$ is the first cumulant
 * of \f$d = \log(t/x)\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<1>` struct containing \f$v_0\f$ and \f$v_1\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; \f$v_1 = 0.0\f$ for \f$a > 1\f$, \f$-\infty\f$ for non-integer \f$a < 1\f$, quiet NaN for integer \f$a \le 1\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = +\infty\f$, \f$v_1 = -1.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 31.0 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 47.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d1(), log_gamma_q_mixed_d2(), log_gamma_q_da1()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<1> log_gamma_q_mixed_d1(double a, double x) {
  math::gamma_inc_mixed_partials<1> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(1, a, x, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Second mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the second mixed partial derivatives of total order \f$K = 2\f$:
 * \f[
 * v_0 = \frac{\partial^2}{\partial a^2}\log Q(a, x) = \kappa_{20} - \psi'(a)
 * \f]
 * \f[
 * v_1 = \frac{\partial^2}{\partial a \partial x}\log Q(a, x) = \frac{1 + \kappa_{11}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^2}{\partial x^2}\log Q(a, x) = \frac{-a + \kappa_{02}}{x^2}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<2>` struct containing \f$v_j = \frac{\partial^2 \log Q}{\partial a^{2-j} \partial x^j}\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \in \{1, 2\}\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = -\psi'(a)\f$, \f$v_1 = 0.0\f$, \f$v_2 = 0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 40.4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 61.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d2(), log_gamma_q_mixed_d1(), log_gamma_q_mixed_d3(), log_gamma_q_da2()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<2> log_gamma_q_mixed_d2(double a, double x) {
  math::gamma_inc_mixed_partials<2> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(2, a, x, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Third mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the third mixed partial derivatives of total order \f$K = 3\f$:
 * \f[
 * v_0 = \frac{\partial^3}{\partial a^3}\log Q(a, x) = \kappa_{30} - \psi''(a)
 * \f]
 * \f[
 * v_1 = \frac{\partial^3}{\partial a^2 \partial x}\log Q(a, x) = \frac{\kappa_{21}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^3}{\partial a \partial x^2}\log Q(a, x) = \frac{-1 + \kappa_{12}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^3}{\partial x^3}\log Q(a, x) = \frac{2a + \kappa_{03}}{x^3}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<3>` struct containing \f$v_j = \frac{\partial^3 \log Q}{\partial a^{3-j} \partial x^j}\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \ge 1\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = -\psi''(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, 2, 3\f$.
 *
 * \par Measured accuracy
 * Maximum error 113.0 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 170.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d3(), log_gamma_q_mixed_d2(), log_gamma_q_mixed_d4(), log_gamma_q_da3()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<3> log_gamma_q_mixed_d3(double a, double x) {
  math::gamma_inc_mixed_partials<3> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(3, a, x, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fourth mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the fourth mixed partial derivatives of total order \f$K = 4\f$:
 * \f[
 * v_0 = \frac{\partial^4}{\partial a^4}\log Q(a, x) = \kappa_{40} - \psi'''(a)
 * \f]
 * \f[
 * v_1 = \frac{\partial^4}{\partial a^3 \partial x}\log Q(a, x) = \frac{\kappa_{31}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^4}{\partial a^2 \partial x^2}\log Q(a, x) = \frac{\kappa_{22}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^4}{\partial a \partial x^3}\log Q(a, x) = \frac{2 + \kappa_{13}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^4}{\partial x^4}\log Q(a, x) = \frac{-6a + \kappa_{04}}{x^4}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<4>` struct containing \f$v_j = \frac{\partial^4 \log Q}{\partial a^{4-j} \partial x^j}\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \ge 1\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = -\psi'''(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 4\f$.
 *
 * \par Measured accuracy
 * Maximum error 8150.0 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 12300.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d4(), log_gamma_q_mixed_d3(), log_gamma_q_mixed_d5(), log_gamma_q_da4()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<4> log_gamma_q_mixed_d4(double a, double x) {
  math::gamma_inc_mixed_partials<4> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(4, a, x, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fifth mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the fifth mixed partial derivatives of total order \f$K = 5\f$:
 * \f[
 * v_0 = \frac{\partial^5}{\partial a^5}\log Q(a, x) = \kappa_{50} - \psi^{(4)}(a)
 * \f]
 * \f[
 * v_1 = \frac{\partial^5}{\partial a^4 \partial x}\log Q(a, x) = \frac{\kappa_{41}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^5}{\partial a^3 \partial x^2}\log Q(a, x) = \frac{\kappa_{32}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^5}{\partial a^2 \partial x^3}\log Q(a, x) = \frac{\kappa_{23}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^5}{\partial a \partial x^4}\log Q(a, x) = \frac{-6 + \kappa_{14}}{x^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^5}{\partial x^5}\log Q(a, x) = \frac{24a + \kappa_{05}}{x^5}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<5>` struct containing \f$v_j = \frac{\partial^5 \log Q}{\partial a^{5-j} \partial x^j}\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \ge 1\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = -\psi^{(4)}(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 5\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.6e4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 39000.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d5(), log_gamma_q_mixed_d4(), log_gamma_q_mixed_d6(), log_gamma_q_da5()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<5> log_gamma_q_mixed_d5(double a, double x) {
  math::gamma_inc_mixed_partials<5> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(5, a, x, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Sixth mixed partial derivatives of \f$\log Q(a, x)\f$ in \f$a\f$ and \f$x\f$.
 *
 * Computes the sixth mixed partial derivatives of total order \f$K = 6\f$:
 * \f[
 * v_0 = \frac{\partial^6}{\partial a^6}\log Q(a, x) = \kappa_{60} - \psi^{(5)}(a)
 * \f]
 * \f[
 * v_1 = \frac{\partial^6}{\partial a^5 \partial x}\log Q(a, x) = \frac{\kappa_{51}}{x}
 * \f]
 * \f[
 * v_2 = \frac{\partial^6}{\partial a^4 \partial x^2}\log Q(a, x) = \frac{\kappa_{42}}{x^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^6}{\partial a^3 \partial x^3}\log Q(a, x) = \frac{\kappa_{33}}{x^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^6}{\partial a^2 \partial x^4}\log Q(a, x) = \frac{\kappa_{24}}{x^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^6}{\partial a \partial x^5}\log Q(a, x) = \frac{24 + \kappa_{15}}{x^5}
 * \f]
 * \f[
 * v_6 = \frac{\partial^6}{\partial x^6}\log Q(a, x) = \frac{-120a + \kappa_{06}}{x^6}
 * \f]
 * where \f$\kappa_{ij}\f$ is the joint cumulant of \f$i\f$ copies of \f$d = \log(t/x)\f$ and \f$j\f$ copies of
 * \f$y = x - t\f$ under the normalized upper measure (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<6>` struct containing \f$v_j = \frac{\partial^6 \log Q}{\partial a^{6-j} \partial x^j}\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN for all components.
 * - \f$x = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \ge 1\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$x = +\infty\f$: returns \f$v_0 = -\psi^{(5)}(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 6\f$.
 *
 * \par Measured accuracy
 * Maximum error 1.05e5 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\min(\mathrm{ulpL}, \mathrm{ulp1})\f$; test bound 158000.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_mixed_d6(), log_gamma_q_mixed_d5(), log_gamma_q_da6()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<6> log_gamma_q_mixed_d6(double a, double x) {
  math::gamma_inc_mixed_partials<6> r;
  detail_gamma_mixed::log_gamma_mixed_impl<true>(6, a, x, false, 0, r.v);
  return r;
}

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls.
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<1> log_gamma_p_mixed_d1(double a, double x) { return own::log_gamma_p_mixed_d1(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<2> log_gamma_p_mixed_d2(double a, double x) { return own::log_gamma_p_mixed_d2(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<3> log_gamma_p_mixed_d3(double a, double x) { return own::log_gamma_p_mixed_d3(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<4> log_gamma_p_mixed_d4(double a, double x) { return own::log_gamma_p_mixed_d4(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<5> log_gamma_p_mixed_d5(double a, double x) { return own::log_gamma_p_mixed_d5(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<6> log_gamma_p_mixed_d6(double a, double x) { return own::log_gamma_p_mixed_d6(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<1> log_gamma_q_mixed_d1(double a, double x) { return own::log_gamma_q_mixed_d1(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<2> log_gamma_q_mixed_d2(double a, double x) { return own::log_gamma_q_mixed_d2(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<3> log_gamma_q_mixed_d3(double a, double x) { return own::log_gamma_q_mixed_d3(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<4> log_gamma_q_mixed_d4(double a, double x) { return own::log_gamma_q_mixed_d4(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<5> log_gamma_q_mixed_d5(double a, double x) { return own::log_gamma_q_mixed_d5(a, x); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<6> log_gamma_q_mixed_d6(double a, double x) { return own::log_gamma_q_mixed_d6(a, x); }

}  // namespace laplace::math
