// Partial derivatives of log P(a, a c) and log Q(a, a c) in the shape a and
// the ratio c = x/a, total orders K = 1 to 6, Laplace's own (a > 0, c >= 0).
// P and Q = 1 - P are the regularized incomplete gamma functions.
//
// One function per order: own::log_gamma_p_ratio_mixed_d1 .. _d6 and
// own::log_gamma_q_ratio_mixed_d1 .. _d6 (wrapped in laplace::math). Order K
// returns a gamma_inc_mixed_partials<K>, whose v[j] = d^K log S(a, a c) /
// da^(K-j) dc^j for j = 0..K: the derivatives in a are taken at fixed c.
//
// Why. The gamma distribution with mean mu and dispersion phi has
// F(y) = P(a, a c) with a = 1/phi and c = y/mu: a depends on phi alone and c
// on mu alone. Composed from the partials in (a, x) of
// incomplete_gamma_mixed.hpp, a derivative at fixed c is
// d/da + c d/dx, and at large a the two terms are large and of opposite sign:
// at a = x = 50 the sixth derivative in a at fixed c is about a^(-3) times
// the largest term of the sum (1.7e7 ulp lost on the gamma distribution's
// log F at phi = 0.02). The routes below carry the increment of a with c fixed
// from the start, so the cancellation is never formed.
//
// Mathematics. With x = a c and the increments h of a and g (relative) of c,
// the point (a + h, c (1 + g)) has x' = x (1 + h/a)(1 + g). Write
//   log Q(a, x) = A(a, c) + log I_U,   log P(a, x) = A(a, c) - log a + log I_L,
//   A(a, c) = a log x - x - log Gamma(a) = (1/2) log a - a phi_c - log sqrt(2 pi) - delta(a),
// with phi_c = c - 1 - log c >= 0 and delta Stirling's error (stirling.hpp).
// A is a series in (h, g) with no cancellation: in h,
// h ((1/2)/a - phi_c) + (1/2) log(1 + h/a) - (delta(a + h) - delta(a)) (each
// term of the size of the result at c = 1); in g, (a + h)(log(1 + g) - c g).
// In the variable d = log(t/x) the integrals are those of
// incomplete_gamma_mixed.hpp, I_U = int_0^inf exp(a d + y) dd and
// I_L = 1 + x int_-inf^0 exp((a + 1) d + y) dd, y = -x (e^d - 1). Moving
// (a, c) to (a + h, c (1 + g)) multiplies the integrand by
//   exp(h w + g (1 + h/a) y),   w = d + y/a = (1 - c) d - c (e^d - 1 - d),
// and the density part of I_L by (1 + h/a)(1 + g). w is computed at each
// node as written, without the cancellation of d against y/a; it vanishes at
// d = 0 and stays small where the measure lies, so its cumulants are of the
// size of the derivatives at fixed c. The joint cumulants of (w, y) in
// (h, t), then t = g (1 + h/a) (a rescaling of the coefficients with positive
// binomial weights), give log I(h, g).
//
// The routes, each the counterpart of the route of the same name in
// incomplete_gamma_mixed.hpp (the same rules, nodes and cut-offs; the
// dispatch differs only in Temme's region, see temme_region below):
// 1. quad: the cumulants of (w, y) by the quadrature. On the lower side the
//    point mass at d = 0 (w = y = 0) is in the measure; with Kf its joint
//    cumulant series, 1 - pi = 1/I_L the point mass's share and
//    e = (1 + h/a)(1 + g) - 1,
//      log I_L(h, g)/I_L = Kf + log(1 + e (1 - (1 - pi) exp(-Kf)))       (pi < 1/2),
//                        = Kf + log(1 + e) + log(1 - e/(1 + e) (1 - pi) exp(-Kf))   (pi >= 1/2);
//    in the second form log(1 + e) = log(1 + h/a) + log(1 + g) is written
//    out, and its first term cancels the -log(a + h) of log P exactly.
// 2. lower series: I_L = sum_n tau_n, and tau_n(h, g) = (1 + g)^n
//    prod_(l<=n) c (a + h)/(a + h + l); the factor c (a + h)/(a + h + l) has
//    the Taylor coefficients x/(a + l) and c l (-1)^(m+1)/(a + l)^(m+1),
//    m >= 1: positive increments in h that no longer carry the log x of the
//    route in (a, x).
// 3. upper asymptotic series: x I_U = sum_n prod_(l<=n) (a - l)/x; the factor
//    (a + h - l)/(c (a + h)) has the Taylor coefficients (a - l)/x and
//    (-1/a)^(m-1) l/(a x), m >= 1, and the term n carries (1 + g)^-n.
// 4. Temme's expansion: at fixed c, sigma = c - 1 and eta do not move with
//    h; -(a + h) phi(sigma) is linear in h, and the factor sqrt((a + h)/a) of
//    the erfcx term is divided out of G (log S = -(a + h) phi - (1/2) log a -
//    log sqrt(2 pi) + log(G / sqrt((a + h)/a))), so that the terms of size 1/a
//    of (1/2) log(a + h) and log G are never formed apart. log G is
//    formed as Lz + log(1 + sgn S exp(-Lz)/A0), Lz the series of
//    log erfcx (gm::temme_log_g), and on the side wanted also beyond c = 1
//    while zeta >= -1 (gm::temme_zeta_cross), not as a complement. Used for
//    a >= 300 (|c - 1| < 0.4, 0.5 from a = 1000) and for 50 <= a < 300,
//    |c - 1| < 0.3, with the components with three or more derivatives in c
//    only within |c - 1| < 0.25 (temme_c_band; outside it, and for
//    50 <= a < 300 those with more than two derivatives in c (a >= 100) or
//    with any (a < 100), from routes 1-3).
// 5. the complement log(1 - S) from exp of the series of log S, as in
//    complement_mixed.
// Below a = 2 (gamma distributions with phi > 1/2) the two directions are not
// large together and the series of incomplete_gamma_mixed.hpp in (h, e) is
// composed with e = (1 + h/a)(1 + g) - 1 = h/a + g + h g/a (the fallback,
// which keeps its closed form at small x and its density ratio).
//
// Accuracy: docs/numerics.md, "Partial derivatives of log P(a, a c) and
// log Q(a, a c) at fixed ratio". Derived from the formulas above; no code
// copied.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/incomplete_gamma_mixed.hpp>
#include <laplace/math/stirling.hpp>

namespace laplace::math::own {

namespace detail_gamma_ratio {

namespace gd = detail_gamma_deriv;
namespace gm = detail_gamma_mixed;
using gm::bser;

// below this shape the fallback (the series in (h, e), composed)
constexpr double a_fallback = 2.0;
constexpr double half_log_2pi = 0.91893853320467274178;

// phi_c = c - 1 - log c >= 0
LAPLACE_HD_FORCEINLINE double phi_of(double c) {
  if (c > 0.5 && c < 2.0) return -gd::log1pmx(c - 1.0);
  return (c - 1.0) - math::log(c);
}

// delta^(j)(a), j = 1..k
LAPLACE_HD void stirling_derivs(double a, int k, double* dl) {
  dl[1] = math::stirling_error_d1(a);
  if (k >= 2) dl[2] = math::stirling_error_d2(a);
  if (k >= 3) dl[3] = math::stirling_error_d3(a);
  if (k >= 4) dl[4] = math::stirling_error_d4(a);
  if (k >= 5) dl[5] = math::stirling_error_d5(a);
  if (k >= 6) dl[6] = math::stirling_error_d6(a);
}

// A(a + h, c (1 + g)) - A(a, c) as a series in (h, g), A = a log x - x -
// log Gamma(a); returns A(a, c)
LAPLACE_HD double prefactor_ratio(double a, double c, int k, bser& P) {
  double dl[7];
  stirling_derivs(a, k, dl);
  const double ph = phi_of(c);
  gm::bzero(P);
  P.c[1][0] = (0.5 / a - ph) - dl[1];
  const double ia = 1.0 / a;
  double ap = ia;
  for (int i = 2; i <= k; ++i) {
    ap *= ia;
    P.c[i][0] = 0.5 * gm::log1p_coef(i) * ap - dl[i] / gm::fact(i);
  }
  const double omc = 1.0 - c;
  P.c[0][1] = a * omc;
  if (k >= 2) P.c[1][1] = omc;
  for (int m = 2; m <= k; ++m) {
    P.c[0][m] = a * gm::log1p_coef(m);
    if (m + 1 <= k) P.c[1][m] = gm::log1p_coef(m);
  }
  return (0.5 * math::log(a) - half_log_2pi) - a * ph - math::stirling_error(a);
}

// the series in (h, t) at t = g (1 + h/a): r_ij = sum_l s_(i-l, j) C(j, l) a^-l
LAPLACE_HD void scale_g(const bser& s, double a, int k, bser& r) {
  const double ia = 1.0 / a;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) {
      double acc = 0.0, b = 1.0, al = 1.0;
      for (int l = 0; l <= i && l <= j; ++l) {
        acc += s.c[i - l][j] * b * al;
        b = b * (j - l) / (l + 1);
        al *= ia;
      }
      r.c[i][j] = acc;
    }
}

// r += sign * log(1 + h/a) (in h alone)
LAPLACE_HD_FORCEINLINE void add_log1p_h(bser& r, double a, int k, double sign) {
  const double ia = 1.0 / a;
  double ap = 1.0;
  for (int i = 1; i <= k; ++i) {
    ap *= ia;
    r.c[i][0] += sign * gm::log1p_coef(i) * ap;
  }
}

struct ratio_side {
  bool lower, direct;
  double log_s;
  bser R;  // log S(a + h, c (1 + g)) - log S(a, c)
};

// ---------------------------------------------------------------- 1. the quadrature

LAPLACE_HD void quad_ratio(double a, double c, int k, bool lower, ratio_side& s) {
  const double x = a * c;
  gm::node_store st;
  st.six = k >= 6;
  if (lower) {
    gd::lower_rule<8, 0>(st, (a - x) + 1.0, x);
  } else if (x < 0.25) {
    const double L = -math::log(x);
    const gd::log_integrand g{a - x, x};
    if (L > 4.0) {
      gd::tanh_sinh_quad<8, 0>(st, 0.0, L - 3.0, g);
      gd::tanh_sinh_quad<8, 0>(st, L - 3.0, 3.0, g);
    } else {
      gd::tanh_sinh_quad<8, 0>(st, 0.0, L, g);
    }
    gd::upper_rule<8, 0>(st, a, x, 1.0, L);
  } else {
    gd::upper_rule<8, 0>(st, a, x, x, 0.0);
  }
  // first pass: the weighted means of w and y
  const double omc = 1.0 - c;
  double wv[gm::node_store::cap], yv[gm::node_store::cap];
  double Wd = 0.0, sw = 0.0, sy = 0.0;
  for (int i = 0; i < st.n; ++i) {
    const double d = st.dv[i];
    yv[i] = -x * Kokkos::expm1(d);
    wv[i] = omc * d - c * gd::expm1mx(d);
    Wd += st.wv[i];
    sw += st.wv[i] * wv[i];
    sy += st.wv[i] * yv[i];
  }
  const double W = (lower ? 1.0 : 0.0) + Wd;
  const double iW = 1.0 / W;
  const double mw = sw * iW, my = sy * iW;
  // second pass: sum q (w - mw)^p (y - my)^r, p + r <= k
  bser M;
  gm::bzero(M);
  auto acc = [&](double q, double ww, double yy) {
    double wp[7], yp[7];
    wp[0] = yp[0] = 1.0;
    for (int p = 1; p <= k; ++p) wp[p] = wp[p - 1] * ww, yp[p] = yp[p - 1] * yy;
    for (int p = 0; p <= k; ++p) {
      const double qw = q * wp[p];
      for (int r = 0; p + r <= k; ++r) M.c[p][r] += qw * yp[r];
    }
  };
  if (lower) acc(1.0, -mw, -my);
  for (int i = 0; i < st.n; ++i) acc(st.wv[i], wv[i] - mw, yv[i] - my);
  for (int p = 0; p <= k; ++p)
    for (int r = 0; p + r <= k; ++r) M.c[p][r] *= iW / (gm::fact(p) * gm::fact(r));
  bser Kt, Kc;
  gm::blog(M, Kt, k);
  Kt.c[1][0] += mw;
  Kt.c[0][1] += my;
  scale_g(Kt, a, k, Kc);
  bser P;
  const double A = prefactor_ratio(a, c, k, P);
  s.lower = lower;
  gm::bzero(s.R);
  if (lower) {
    const double ia = 1.0 / a;
    bser nK, Ei, Y, Lam, e;
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j) nK.c[i][j] = (i + j == 0) ? 0.0 : -Kc.c[i][j];
    gm::bexp(nK, Ei, k);
    gm::bzero(e);
    gm::bzero(Y);
    const double pi = Wd * iW;
    if (pi < 0.5) {
      // Y = 1 + e D, D = 1 - (1 - pi) exp(-Kf), D_00 = pi
      bser D;
      for (int i = 0; i <= k; ++i)
        for (int j = 0; i + j <= k; ++j) D.c[i][j] = -iW * Ei.c[i][j];
      D.c[0][0] = pi;
      e.c[1][0] = ia;
      e.c[0][1] = 1.0;
      if (k >= 2) e.c[1][1] = ia;
      gm::bmul(e, D, Y, k);
      Y.c[0][0] = 1.0;
      gm::blog(Y, Lam, k);
      add_log1p_h(s.R, a, k, -1.0);
    } else {
      // Y = 1 - e/(1 + e) (1 - pi) exp(-Kf), e/(1 + e) = 1 - 1/((1 + h/a)(1 + g))
      double ph = 1.0;
      for (int i = 0; i <= k; ++i, ph *= -ia) {
        double pg = 1.0;
        for (int j = 0; i + j <= k; ++j, pg = -pg)
          if (i + j > 0) e.c[i][j] = -ph * pg;
      }
      bser t;
      gm::bmul(e, Ei, t, k);
      for (int i = 0; i <= k; ++i)
        for (int j = 0; i + j <= k; ++j) Y.c[i][j] = -iW * t.c[i][j];
      Y.c[0][0] = 1.0;
      gm::blog(Y, Lam, k);
      for (int j = 1; j <= k; ++j) s.R.c[0][j] += gm::log1p_coef(j);
    }
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j)
        if (i + j > 0) s.R.c[i][j] += P.c[i][j] + Kc.c[i][j] + Lam.c[i][j];
    s.R.c[0][1] = a * iW;
    s.log_s = (A - math::log(a)) + math::log(W);
  } else {
    for (int i = 0; i <= k; ++i)
      for (int j = 0; i + j <= k; ++j)
        if (i + j > 0) s.R.c[i][j] = P.c[i][j] + Kc.c[i][j];
    s.R.c[0][1] = -iW;
    s.log_s = A + math::log(W);
  }
}

// ---------------------------------------------------------------- 2, 3. the series

// The Taylor coefficients F[0..k] in h of the factor of term n:
// lower c (a + h)/(a + h + n), upper (a + h - n)/(c (a + h))
LAPLACE_HD_FORCEINLINE void series_factor(double a, double c, double x, bool lower, int n, int k, double* F) {
  if (lower) {
    const double z = 1.0 / (a + n);
    F[0] = x * z;
    double t = c * n * z * z;  // c n / (a + n)^2
    for (int m = 1; m <= k; ++m) {
      F[m] = t;
      t *= -z;
    }
  } else {
    F[0] = (a - n) / x;
    double t = n / (a * x);
    for (int m = 1; m <= k; ++m) {
      F[m] = t;
      t *= -1.0 / a;
    }
  }
}

// as gm::series_mixed; false when the first pass has not converged within nmax terms
LAPLACE_HD bool series_ratio(double a, double c, bool lower, int k, int nmax, ratio_side& s) {
  constexpr double tol = 1e-22;
  const double x = a * c;
  double F[7];
  double t0 = 1.0, t1 = 0.0, s0 = 1.0, s1 = 0.0;
  double cb[7] = {1, 0, 0, 0, 0, 0, 0}, em[7] = {1, 0, 0, 0, 0, 0, 0};
  int n_end = -1;
  for (int n = 1; n <= nmax; ++n) {
    series_factor(a, c, x, lower, n, 1, F);
    if (!lower && F[0] <= -1.0) break;  // past the smallest term: the series diverges
    t1 = F[0] * t1 + F[1] * t0;
    t0 = F[0] * t0;
    s0 += t0;
    s1 += t1;
    bool done;
    if (lower) {
      for (int m = k; m >= 1; --m) cb[m] += cb[m - 1];
      done = n >= k && t0 <= tol * s0 && Kokkos::fabs(t1) <= tol * (Kokkos::fabs(s1) + s0);
      for (int m = 1; m <= k; ++m) {
        em[m] += cb[m] * t0;
        if (!(cb[m] * t0 <= tol * em[m])) done = false;
      }
    } else {
      for (int m = 1; m <= k; ++m) cb[m] += cb[m - 1];
      done = Kokkos::fabs(t0) <= tol * s0 && Kokkos::fabs(t1) <= tol * (Kokkos::fabs(s1) + s0);
      for (int m = 1; m <= k; ++m) {
        const double tm = ((m % 2 == 1) ? -cb[m] : cb[m]) * t0;
        em[m] += tm;
        if (!(Kokkos::fabs(tm) <= tol * (Kokkos::fabs(em[m]) + s0))) done = false;
      }
    }
    if (done) {
      n_end = n;
      break;
    }
  }
  if (n_end < 0) return false;
  const double mu = s1 / s0;
  // second pass: T the Taylor coefficients in h of tau_n e^(-mu h)
  double T[7], U[7];
  T[0] = 1.0;
  for (int i = 1; i <= k; ++i) T[i] = T[i - 1] * (-mu) / i;
  bser S;
  gm::bzero(S);
  for (int i = 0; i <= k; ++i) S.c[i][0] = T[i];
  for (int m = 0; m <= k; ++m) cb[m] = m == 0 ? 1.0 : 0.0;
  for (int n = 1; n <= n_end; ++n) {
    series_factor(a, c, x, lower, n, k, F);
    for (int i = 0; i <= k; ++i) {
      double acc = 0.0;
      for (int m = 0; m <= i; ++m) acc += F[m] * T[i - m];
      U[i] = acc;
    }
    for (int i = 0; i <= k; ++i) T[i] = U[i];
    if (lower) {
      for (int m = k; m >= 1; --m) cb[m] += cb[m - 1];
    } else {
      for (int m = 1; m <= k; ++m) cb[m] += cb[m - 1];
    }
    for (int m = 0; m <= k; ++m) {
      const double w = lower ? cb[m] : ((m % 2 == 1) ? -cb[m] : cb[m]);
      for (int i = 0; i + m <= k; ++i) S.c[i][m] += T[i] * w;
    }
  }
  bser Kc, P;
  gm::blog(S, Kc, k);
  Kc.c[1][0] += mu;
  const double A = prefactor_ratio(a, c, k, P);
  const double S0 = S.c[0][0];
  s.lower = lower;
  gm::bzero(s.R);
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j)
      if (i + j > 0) s.R.c[i][j] = P.c[i][j] + Kc.c[i][j];
  add_log1p_h(s.R, a, k, -1.0);
  if (lower) {
    s.R.c[0][1] = a / S0;
    s.log_s = (A - math::log(a)) + math::log(S0);
  } else {
    // log I_U = log(x I_U) - log x - log(1 + h/a) - log(1 + g)
    for (int j = 2; j <= k; ++j) s.R.c[0][j] -= gm::log1p_coef(j);
    s.R.c[0][1] = -x / S0;
    s.log_s = A + (math::log(S0) - math::log(x));
  }
  return true;
}

// ---------------------------------------------------------------- 4. Temme's expansion

// zeta_min < 0: the side wanted also beyond c = 1, as gm::temme_mixed
LAPLACE_HD bool temme_ratio(double a, double c, int k, bool want_p, ratio_side& s, double zeta_min = 0.0) {
  constexpr int off[11] = {0, 15, 28, 39, 48, 55, 64, 71, 76, 79, 80};
  const double sigma = c - 1.0;
  const bool lower = zeta_min < 0.0 ? want_p : c < 1.0;
  const double ia = 1.0 / a;
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
  // eta(sigma + c g) - eta: a series in g alone
  bser de;
  gm::bzero(de);
  {
    double cp = 1.0;
    for (int m = 1; m <= k; ++m) {
      cp *= c;
      de.c[0][m] = em[m] * cp;
    }
  }
  bser P[7];
  gm::bzero(P[0]);
  P[0].c[0][0] = 1.0;
  P[1] = de;
  for (int m = 2; m <= k; ++m) gm::bmul(P[m - 1], de, P[m], k);
  // S(h, g) = sum_i C_i(eta(g)) (a + h)^-i by Horner in w = 1/(a + h)
  bser w, S, T;
  gm::bzero(w);
  gm::bzero(S);
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
    gm::bmul(S, w, T, k);
    for (int p = 0; p <= k; ++p)
      for (int q = 0; p + q <= k; ++q) {
        double acc = 0.0;
        for (int m = 0; m <= p + q; ++m) acc += cm[m] * P[m].c[p][q];
        S.c[p][q] = T.c[p][q] + acc;
      }
  }
  // sqrt((a + h)/a) and its inverse
  bser sq, isq;
  gm::bzero(sq);
  gm::bzero(isq);
  {
    double b1 = 1.0, b2 = 1.0, f = 1.0;
    for (int j = 0; j <= k; ++j) {
      sq.c[j][0] = b1 * f;
      isq.c[j][0] = b2 * f;
      b1 = b1 * (0.5 - j) / (j + 1);
      b2 = b2 * (-0.5 - j) / (j + 1);
      f *= ia;
    }
  }
  const double sgn = lower ? -1.0 : 1.0;
  const double rt = Kokkos::sqrt(0.5 * a);
  // zeta(h, g) = sgn eta(g) sqrt((a + h)/2)
  bser ez, zh, dz;
  for (int p = 0; p <= k; ++p)
    for (int q = 0; p + q <= k; ++q) ez.c[p][q] = sgn * de.c[p][q];
  ez.c[0][0] = sgn * eta;
  gm::bmul(ez, sq, zh, k);
  const double zeta = rt * zh.c[0][0];
  if (!(zeta >= zeta_min)) return false;
  for (int p = 0; p <= k; ++p)
    for (int q = 0; p + q <= k; ++q) dz.c[p][q] = rt * zh.c[p][q];
  dz.c[0][0] = 0.0;
  double Lm[7];
  gd::log_erfcx_taylor(zeta, k, Lm);
  bser Lz, Si, lG;
  Lm[0] = 0.0;
  gm::bcompose(Lm, dz, Lz, k);
  constexpr double sqrt_pi_half = 1.25331413731550025121;
  const double A0 = sqrt_pi_half * Kokkos::sqrt(a) * math::erfcx(zeta);
  // G / sqrt((a + h)/a) = A0 exp(Lz) + sgn S / sqrt((a + h)/a)
  const double G00 = A0 + sgn * S.c[0][0];
  if (!(G00 > 0.0)) return false;
  gm::bmul(S, isq, Si, k);
  gm::temme_log_g(A0, sgn, Lz, Si, k, lG);
  s.lower = lower;
  s.direct = lower == want_p;
  s.R = lG;
  s.R.c[0][0] = 0.0;
  // -(a + h) phi(sigma (g)), phi(sigma + c g) = phi(sigma) + c g - log(1 + g)
  const double lpm = gd::log1pmx(sigma);  // -phi(sigma)
  s.R.c[1][0] += lpm;
  const double omc = 1.0 - c;
  s.R.c[0][1] += a * omc;
  if (k >= 2) s.R.c[1][1] += omc;
  for (int m = 2; m <= k; ++m) {
    s.R.c[0][m] += a * gm::log1p_coef(m);
    if (m + 1 <= k) s.R.c[1][m] += gm::log1p_coef(m);
  }
  s.log_s = a * lpm - 0.5 * math::log(a) - half_log_2pi + math::log(G00);
  return true;
}

// ---------------------------------------------------------------- 5. the complement

// C = log(1 - S(a + h, c (1 + g))) - log(1 - S) from the series R of log S
LAPLACE_HD void complement_ratio(int k, const ratio_side& s, bser& C, double& log_c) {
  bser B = s.R, E, M;
  B.c[0][0] = 0.0;
  gm::bexp(B, E, k);
  const double ls = s.log_s;
  const double l1ms = ls < gd::ln_half ? Kokkos::log1p(-math::exp(ls)) : math::log(-Kokkos::expm1(ls));
  log_c = l1ms;
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
  gm::blog(M, C, k);
}

// ---------------------------------------------------------------- the fallback

// R(h, g) = C(h, e(h, g)), e = h/a + g + h g/a, by Horner in e
LAPLACE_HD void compose_ratio(const bser& C, double a, int k, bser& R) {
  const double ia = 1.0 / a;
  bser e, t;
  gm::bzero(e);
  e.c[1][0] = ia;
  e.c[0][1] = 1.0;
  if (k >= 2) e.c[1][1] = ia;
  gm::bzero(R);
  for (int i = 0; i + k <= k; ++i) R.c[i][0] = C.c[i][k];
  for (int j = k - 1; j >= 0; --j) {
    gm::bmul(R, e, t, k);
    R = t;
    for (int i = 0; i + j <= k; ++i) R.c[i][0] += C.c[i][j];
  }
}

// ---------------------------------------------------------------- limits and dispatch

// c = 0 for P: log P(a + h, c (a + h)) = (a + h) log c + log((a + h)^(a + h) e^... )
// + log sum_m c^m (a + h)^m / ((a + h + 1) ... (a + h + m)): the components
// with at most one derivative in a are infinite through (a + h) log c, the
// others are the cumulants of the last sum in the unscaled increment of c
LAPLACE_HD void p_ratio_at_zero(double a, int k, double* v) {
  const double inf = math::detail::pos_inf();
  bser F, L;
  gm::bzero(F);
  double T[7] = {1, 0, 0, 0, 0, 0, 0}, G[7], U[7];
  F.c[0][0] = 1.0;
  for (int m = 1; m <= k; ++m) {
    // (a + h)/(a + m + h): a/(a + m), then m (-1)^(n+1)/(a + m)^(n+1)
    const double z = 1.0 / (a + m);
    G[0] = a * z;
    double t = m * z * z;
    for (int n = 1; n <= k; ++n) {
      G[n] = t;
      t *= -z;
    }
    for (int i = 0; i <= k; ++i) {
      double acc = 0.0;
      for (int l = 0; l <= i; ++l) acc += G[l] * T[i - l];
      U[i] = acc;
    }
    for (int i = 0; i <= k; ++i) T[i] = U[i];
    for (int i = 0; i + m <= k; ++i) F.c[i][m] = T[i];
  }
  gm::blog(F, L, k);
  for (int j = 1; j <= k; ++j) {
    const int i = k - j;
    v[j] = i <= 1 ? ((j % 2 == 1) ? inf : -inf) : gm::fact(i) * gm::fact(j) * L.c[i][j];
  }
}

// Temme's expansion from a = 300 (wide = true: from a = 20 as in
// incomplete_gamma_mixed.hpp); the dispatch also uses it within the band for
// 50 <= a < 300 (gm::temme_all_from). Measured 2026-11-12 against the references of
// incomplete_ratio.py, by component, for 20 <= a < 300 in the band
// |c - 1| < 0.3: the derivatives with three or more in c lose up to 1e5-3e6
// ulp of their value in Temme's series in (h, g) (1e3-4e4 by the quadrature
// and the series), and for a < 50 every component does (likely Temme's 10 terms in
// 1/a, differentiated six times in a); the components with no derivative in c
// (a >= 50) and with at most two (a >= 100) are better from Temme (order 6,
// 100 <= a < 300: j = 0 12 against 284 ulp, j = 2 1.7e3 against 3.2e4). The
// dispatch below combines the two there. Most of that loss came from the
// Taylor coefficients of log erfcx (incomplete_gamma_deriv.hpp,
// log_erfcx_taylor) and from blog of Temme's G, both repaired in the night of
// 2026-11-12; since then Temme serves every component within |c - 1| < 0.25
// from a = 50 (gm::temme_all_from).
// Temme's series in (h, g) loses on the derivatives of order 3 and more in c
// on its direct side far from c = 1 (P for c < 1, Q for c > 1), from
// |c - 1| = 0.25-0.3 on, whatever a: measured 2026-11-12 (night), order 6,
// against references from the closed form, log P at a = 300, c = 0.51: 1.1e5
// ulp of the value (the series 1.0), c = 0.74: 41 (8.5); log Q at a = 300,
// c = 1.49: 3.3e4 (52), c = 1.29: 19 (14); at a = 1000 log P from c = 0.70
// (36 against 0.6) and log Q from c = 1.32. Likely the truncation of the
// power series in eta of Temme's coefficients C_k, differentiated in c. From
// this distance those components come from the route without Temme.
constexpr double temme_c_band = gm::temme_band;

LAPLACE_HD_FORCEINLINE bool temme_region(double a, double c, bool wide) {
  const double sg = Kokkos::fabs(c - 1.0);
  return (wide && a >= 20.0 && sg < 0.3) || (a >= 300.0 && sg < 0.4) || (a >= 1000.0 && sg < 0.5);
}

// method: 0 the chosen route; 1 the chosen route with the series limits of the
// device; 2 the quadrature alone; 3 the chosen route without Temme; 4 the
// chosen route with Temme from a = 20 (as in incomplete_gamma_mixed.hpp); 7 the
// fallback for every a (the series in (h, e) of incomplete_gamma_mixed.hpp,
// composed: the route of gamma1 before 2026-11-12, evening, without its
// composition over (mu, phi)).
template <bool Fixed>
LAPLACE_GM_NOINLINE LAPLACE_HD void log_gamma_ratio_impl(int k, double a, double c, bool want_p, int method,
                                                          double* v) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(c >= 0.0) || a == inf) {
    for (int j = 0; j <= k; ++j) v[j] = nan;
    return;
  }
  if (c == 0.0) {
    if (want_p) {
      if (k == 1) {
        v[0] = -inf;
      } else {
        // d^K/da^K of a log a - log Gamma(a + 1)
        double ak = 1.0;
        for (int i = 1; i < k; ++i) ak *= a;
        v[0] = ((k % 2 == 0) ? 1.0 : -1.0) * gm::fact(k - 2) / ak - math::polygamma(k - 1, a + 1.0);
      }
      p_ratio_at_zero(a, k, v);
    } else {
      v[0] = 0.0;
      gm::q_at_zero(a, k, v);
    }
    return;
  }
  if (c == inf || a * c == inf) {
    for (int j = 0; j <= k; ++j) v[j] = 0.0;
    if (!want_p) {
      // log Q = -a phi_c + (1/2) log a - delta(a) - log sqrt(2 pi) - log(a (c - 1)) + O(1/c)
      if (k == 1) {
        v[0] = -inf;
        v[1] = -a;
      } else {
        double ak = 1.0;
        for (int i = 0; i < k; ++i) ak *= a;
        double dl[7];
        stirling_derivs(a, k, dl);
        v[0] = -1.5 * gm::log1p_coef(k) * gm::fact(k) / ak - dl[k];
        if (k == 2) v[1] = -1.0;
      }
    }
    return;
  }
  if (!Fixed) method = method < 0 ? 0 : method;
  const double x = a * c;
  bser R;
  if (a < a_fallback || (!Fixed && method == 7)) {
    bser C;
    double ls;
    gm::mixed_log_series<Fixed>(k, a, x, want_p, 0, C, ls);
    compose_ratio(C, a, k, R);
  } else {
    ratio_side s;
    bool done = false;
    if (Fixed || method <= 1 || method == 4) {
      if (temme_region(a, c, !Fixed && method == 4)) done = temme_ratio(a, c, k, want_p, s, -gm::temme_zeta_cross) || temme_ratio(a, c, k, want_p, s);
    }
    // the route without Temme: the series or the quadrature on the side
    auto other_route = [&](ratio_side& o) {
      const bool lower = gd::side_lower(a, x, want_p);
      o.direct = lower == want_p;
      int nl = gd::nmax_lower_host, nu = gd::nmax_upper_host;
      KOKKOS_IF_ON_DEVICE((nl = a < 2.0 ? gd::nmax_lower_device : 0; nu = gd::nmax_upper_device;))
      if (!Fixed && method == 1) nl = a < 2.0 ? gd::nmax_lower_device : 0, nu = gd::nmax_upper_device;
      if ((Fixed || method != 2) && (lower || x >= 36.0) && series_ratio(a, c, lower, k, lower ? nl : nu, o)) {
      } else {
        quad_ratio(a, c, k, lower, o);
      }
    };
    if (!done) other_route(s);
    if (s.direct) {
      R = s.R;
    } else {
      double lc;
      complement_ratio(k, s, R, lc);
    }
    // 50 <= a < 300 in the band of Temme's expansion: the components with at
    // most jt derivatives in c from Temme, the others from the route above
    // 50 <= a < 300 in the band of Temme's expansion: the components with at
    // most jt derivatives in c from Temme, the others from the route above
    // (all of them within |c - 1| < temme_c_band, gm::temme_all_from)
    const int jt = Kokkos::fabs(c - 1.0) < temme_c_band ? k : (a >= 100.0 ? 2 : 0);
    if (!done && (Fixed || method <= 1) && a >= gm::temme_all_from && a < 300.0 && temme_region(a, c, true)) {
      ratio_side t;
      if (temme_ratio(a, c, k, want_p, t, -gm::temme_zeta_cross) || temme_ratio(a, c, k, want_p, t)) {
        bser Rt;
        if (t.direct) {
          Rt = t.R;
        } else {
          double lc;
          complement_ratio(k, t, Rt, lc);
        }
        for (int i = 0; i <= k; ++i)
          for (int j = 0; j <= jt && i + j <= k; ++j) R.c[i][j] = Rt.c[i][j];
      }
    }
    // a >= 300 and |c - 1| >= temme_c_band: the components with three or
    // more derivatives in c from the route without Temme (temme_c_band below)
    if (done && k >= 3 && (Fixed || method <= 1) && Kokkos::fabs(c - 1.0) >= temme_c_band) {
      ratio_side o;
      other_route(o);
      bser Ro;
      if (o.direct) {
        Ro = o.R;
      } else {
        double lc;
        complement_ratio(k, o, Ro, lc);
      }
      for (int j = 3; j <= k; ++j)
        for (int i = 0; i + j <= k; ++i) R.c[i][j] = Ro.c[i][j];
    }
  }
  double ic = 1.0 / c;
  for (int j = 0; j <= k; ++j) {
    double r = gm::fact(k - j) * gm::fact(j) * R.c[k - j][j];
    for (int l = 0; l < j; ++l) r *= ic;
    v[j] = r;
  }
}

LAPLACE_HD void log_gamma_ratio(int k, double a, double c, bool want_p, int method, double* v) {
  log_gamma_ratio_impl<false>(k, a, c, want_p, method, v);
}

}  // namespace detail_gamma_ratio

/**
 * \ingroup incomplete_gamma
 * \brief First partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the first partial derivatives of total order \f$K = 1\f$:
 * \f[
 * v_0 = \frac{\partial}{\partial a}\log P(a, ac) = R_{10}
 * \f]
 * \f[
 * v_1 = \frac{\partial}{\partial c}\log P(a, ac) = \frac{R_{01}}{c} = a \frac{\partial}{\partial x}\log P(a, x)
 * \f]
 * where the derivative in \f$a\f$ is taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<1>` struct containing \f$v_0\f$ and \f$v_1\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = -\infty\f$, \f$v_1 = +\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = 0.0\f$, \f$v_1 = 0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 1496 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{1-j} c^j v_j| / (a^{1-j} c^j)\f$; test bound 2250 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d1(), log_gamma_p_ratio_mixed_d2(), log_gamma_p_mixed_d1()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<1> log_gamma_p_ratio_mixed_d1(double a, double c) {
  math::gamma_inc_mixed_partials<1> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(1, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Second partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the second partial derivatives of total order \f$K = 2\f$:
 * \f[
 * v_0 = \frac{\partial^2}{\partial a^2}\log P(a, ac) = 2 R_{20}
 * \f]
 * \f[
 * v_1 = \frac{\partial^2}{\partial a \partial c}\log P(a, ac) = \frac{R_{11}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^2}{\partial c^2}\log P(a, ac) = \frac{2 R_{02}}{c^2} = a^2 \frac{\partial^2}{\partial x^2}\log P(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<2>` struct containing \f$v_j = \frac{\partial^2 \log P}{\partial a^{2-j} \partial c^j}\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 1/a - \psi'(a + 1)\f$, \f$v_1 = +\infty\f$, \f$v_2 = -\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Measured accuracy
 * Maximum error 2113 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{2-j} c^j v_j| / (a^{2-j} c^j)\f$; test bound 3200 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d2(), log_gamma_p_ratio_mixed_d1(), log_gamma_p_ratio_mixed_d3(), log_gamma_p_mixed_d2()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<2> log_gamma_p_ratio_mixed_d2(double a, double c) {
  math::gamma_inc_mixed_partials<2> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(2, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Third partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the third partial derivatives of total order \f$K = 3\f$:
 * \f[
 * v_0 = \frac{\partial^3}{\partial a^3}\log P(a, ac) = 6 R_{30}
 * \f]
 * \f[
 * v_1 = \frac{\partial^3}{\partial a^2 \partial c}\log P(a, ac) = \frac{2 R_{21}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^3}{\partial a \partial c^2}\log P(a, ac) = \frac{2 R_{12}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^3}{\partial c^3}\log P(a, ac) = \frac{6 R_{03}}{c^3} = a^3 \frac{\partial^3}{\partial x^3}\log P(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<3>` struct containing \f$v_j = \frac{\partial^3 \log P}{\partial a^{3-j} \partial c^j}\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = -1/a^2 - \psi''(a + 1)\f$, \f$v_1\f$ is finite, \f$v_2 = -\infty\f$, \f$v_3 = +\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Measured accuracy
 * Maximum error 2082 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{3-j} c^j v_j| / (a^{3-j} c^j)\f$; test bound 3150 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d3(), log_gamma_p_ratio_mixed_d2(), log_gamma_p_ratio_mixed_d4(), log_gamma_p_mixed_d3()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<3> log_gamma_p_ratio_mixed_d3(double a, double c) {
  math::gamma_inc_mixed_partials<3> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(3, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fourth partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the fourth partial derivatives of total order \f$K = 4\f$:
 * \f[
 * v_0 = \frac{\partial^4}{\partial a^4}\log P(a, ac) = 24 R_{40}
 * \f]
 * \f[
 * v_1 = \frac{\partial^4}{\partial a^3 \partial c}\log P(a, ac) = \frac{6 R_{31}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^4}{\partial a^2 \partial c^2}\log P(a, ac) = \frac{4 R_{22}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^4}{\partial a \partial c^3}\log P(a, ac) = \frac{6 R_{13}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^4}{\partial c^4}\log P(a, ac) = \frac{24 R_{04}}{c^4} = a^4 \frac{\partial^4}{\partial x^4}\log P(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<4>` struct containing \f$v_j = \frac{\partial^4 \log P}{\partial a^{4-j} \partial c^j}\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 2/a^3 - \psi'''(a + 1)\f$, \f$v_1, v_2\f$ are finite, \f$v_3 = +\infty\f$, \f$v_4 = -\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Measured accuracy
 * Maximum error 1736 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{4-j} c^j v_j| / (a^{4-j} c^j)\f$; test bound 2650 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d4(), log_gamma_p_ratio_mixed_d3(), log_gamma_p_ratio_mixed_d5(), log_gamma_p_mixed_d4()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<4> log_gamma_p_ratio_mixed_d4(double a, double c) {
  math::gamma_inc_mixed_partials<4> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(4, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fifth partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the fifth partial derivatives of total order \f$K = 5\f$:
 * \f[
 * v_0 = \frac{\partial^5}{\partial a^5}\log P(a, ac) = 120 R_{50}
 * \f]
 * \f[
 * v_1 = \frac{\partial^5}{\partial a^4 \partial c}\log P(a, ac) = \frac{24 R_{41}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^5}{\partial a^3 \partial c^2}\log P(a, ac) = \frac{12 R_{32}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^5}{\partial a^2 \partial c^3}\log P(a, ac) = \frac{12 R_{23}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^5}{\partial a \partial c^4}\log P(a, ac) = \frac{24 R_{14}}{c^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^5}{\partial c^5}\log P(a, ac) = \frac{120 R_{05}}{c^5} = a^5 \frac{\partial^5}{\partial x^5}\log P(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<5>` struct containing \f$v_j = \frac{\partial^5 \log P}{\partial a^{5-j} \partial c^j}\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = -6/a^4 - \psi^{(4)}(a + 1)\f$, \f$v_1, v_2, v_3\f$ are finite, \f$v_4 = -\infty\f$, \f$v_5 = +\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Measured accuracy
 * Maximum error 9.0e3 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{5-j} c^j v_j| / (a^{5-j} c^j)\f$; test bound 13500 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d5(), log_gamma_p_ratio_mixed_d4(), log_gamma_p_ratio_mixed_d6(), log_gamma_p_mixed_d5()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<5> log_gamma_p_ratio_mixed_d5(double a, double c) {
  math::gamma_inc_mixed_partials<5> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(5, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Sixth partial derivatives of \f$\log P(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the sixth partial derivatives of total order \f$K = 6\f$:
 * \f[
 * v_0 = \frac{\partial^6}{\partial a^6}\log P(a, ac) = 720 R_{60}
 * \f]
 * \f[
 * v_1 = \frac{\partial^6}{\partial a^5 \partial c}\log P(a, ac) = \frac{120 R_{51}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^6}{\partial a^4 \partial c^2}\log P(a, ac) = \frac{48 R_{42}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^6}{\partial a^3 \partial c^3}\log P(a, ac) = \frac{36 R_{33}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^6}{\partial a^2 \partial c^4}\log P(a, ac) = \frac{48 R_{24}}{c^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^6}{\partial a \partial c^5}\log P(a, ac) = \frac{120 R_{15}}{c^5}
 * \f]
 * \f[
 * v_6 = \frac{\partial^6}{\partial c^6}\log P(a, ac) = \frac{720 R_{06}}{c^6} = a^6 \frac{\partial^6}{\partial x^6}\log P(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log P(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<6>` struct containing \f$v_j = \frac{\partial^6 \log P}{\partial a^{6-j} \partial c^j}\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 24/a^5 - \psi^{(5)}(a + 1)\f$, \f$v_1, v_2, v_3, v_4\f$ are finite, \f$v_5 = +\infty\f$, \f$v_6 = -\infty\f$.
 * - \f$c = +\infty\f$: returns \f$v_j = 0.0\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Measured accuracy
 * Maximum error 7.3e4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{6-j} c^j v_j| / (a^{6-j} c^j)\f$; test bound 110000 ulp.
 *
 * \see incomplete_gamma, log_gamma_q_ratio_mixed_d6(), log_gamma_p_ratio_mixed_d5(), log_gamma_p_mixed_d6()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<6> log_gamma_p_ratio_mixed_d6(double a, double c) {
  math::gamma_inc_mixed_partials<6> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(6, a, c, true, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief First partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the first partial derivatives of total order \f$K = 1\f$:
 * \f[
 * v_0 = \frac{\partial}{\partial a}\log Q(a, ac) = R_{10}
 * \f]
 * \f[
 * v_1 = \frac{\partial}{\partial c}\log Q(a, ac) = \frac{R_{01}}{c} = a \frac{\partial}{\partial x}\log Q(a, x)
 * \f]
 * where the derivative in \f$a\f$ is taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<1>` struct containing \f$v_0\f$ and \f$v_1\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; \f$v_1 = 0.0\f$ for \f$a > 1\f$, \f$-\infty\f$ for non-integer \f$a < 1\f$, quiet NaN for integer \f$a \le 1\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = -\infty\f$, \f$v_1 = -a\f$.
 *
 * \par Measured accuracy
 * Maximum error 1704 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{1-j} c^j v_j| / (a^{1-j} c^j)\f$; test bound 2600 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d1(), log_gamma_q_ratio_mixed_d2(), log_gamma_q_mixed_d1()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<1> log_gamma_q_ratio_mixed_d1(double a, double c) {
  math::gamma_inc_mixed_partials<1> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(1, a, c, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Second partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the second partial derivatives of total order \f$K = 2\f$:
 * \f[
 * v_0 = \frac{\partial^2}{\partial a^2}\log Q(a, ac) = 2 R_{20}
 * \f]
 * \f[
 * v_1 = \frac{\partial^2}{\partial a \partial c}\log Q(a, ac) = \frac{R_{11}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^2}{\partial c^2}\log Q(a, ac) = \frac{2 R_{02}}{c^2} = a^2 \frac{\partial^2}{\partial x^2}\log Q(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<2>` struct containing \f$v_j = \frac{\partial^2 \log Q}{\partial a^{2-j} \partial c^j}\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \in \{1, 2\}\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = \frac{3}{2a^2} - \delta''(a)\f$, \f$v_1 = -1.0\f$, \f$v_2 = 0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 1432 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{2-j} c^j v_j| / (a^{2-j} c^j)\f$; test bound 2150 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d2(), log_gamma_q_ratio_mixed_d1(), log_gamma_q_ratio_mixed_d3(), log_gamma_q_mixed_d2()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<2> log_gamma_q_ratio_mixed_d2(double a, double c) {
  math::gamma_inc_mixed_partials<2> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(2, a, c, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Third partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the third partial derivatives of total order \f$K = 3\f$:
 * \f[
 * v_0 = \frac{\partial^3}{\partial a^3}\log Q(a, ac) = 6 R_{30}
 * \f]
 * \f[
 * v_1 = \frac{\partial^3}{\partial a^2 \partial c}\log Q(a, ac) = \frac{2 R_{21}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^3}{\partial a \partial c^2}\log Q(a, ac) = \frac{2 R_{12}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^3}{\partial c^3}\log Q(a, ac) = \frac{6 R_{03}}{c^3} = a^3 \frac{\partial^3}{\partial x^3}\log Q(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<3>` struct containing \f$v_j = \frac{\partial^3 \log Q}{\partial a^{3-j} \partial c^j}\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j \in \{1, 2, 3\}\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = -\frac{3}{a^3} - \delta'''(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, 2, 3\f$.
 *
 * \par Measured accuracy
 * Maximum error 1204 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{3-j} c^j v_j| / (a^{3-j} c^j)\f$; test bound 1850 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d3(), log_gamma_q_ratio_mixed_d2(), log_gamma_q_ratio_mixed_d4(), log_gamma_q_mixed_d3()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<3> log_gamma_q_ratio_mixed_d3(double a, double c) {
  math::gamma_inc_mixed_partials<3> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(3, a, c, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fourth partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the fourth partial derivatives of total order \f$K = 4\f$:
 * \f[
 * v_0 = \frac{\partial^4}{\partial a^4}\log Q(a, ac) = 24 R_{40}
 * \f]
 * \f[
 * v_1 = \frac{\partial^4}{\partial a^3 \partial c}\log Q(a, ac) = \frac{6 R_{31}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^4}{\partial a^2 \partial c^2}\log Q(a, ac) = \frac{4 R_{22}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^4}{\partial a \partial c^3}\log Q(a, ac) = \frac{6 R_{13}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^4}{\partial c^4}\log Q(a, ac) = \frac{24 R_{04}}{c^4} = a^4 \frac{\partial^4}{\partial x^4}\log Q(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<4>` struct containing \f$v_j = \frac{\partial^4 \log Q}{\partial a^{4-j} \partial c^j}\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j = 1, \dots, 4\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = \frac{9}{a^4} - \delta^{(4)}(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 4\f$.
 *
 * \par Measured accuracy
 * Maximum error 1179 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{4-j} c^j v_j| / (a^{4-j} c^j)\f$; test bound 1800 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d4(), log_gamma_q_ratio_mixed_d3(), log_gamma_q_ratio_mixed_d5(), log_gamma_q_mixed_d4()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<4> log_gamma_q_ratio_mixed_d4(double a, double c) {
  math::gamma_inc_mixed_partials<4> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(4, a, c, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Fifth partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the fifth partial derivatives of total order \f$K = 5\f$:
 * \f[
 * v_0 = \frac{\partial^5}{\partial a^5}\log Q(a, ac) = 120 R_{50}
 * \f]
 * \f[
 * v_1 = \frac{\partial^5}{\partial a^4 \partial c}\log Q(a, ac) = \frac{24 R_{41}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^5}{\partial a^3 \partial c^2}\log Q(a, ac) = \frac{12 R_{32}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^5}{\partial a^2 \partial c^3}\log Q(a, ac) = \frac{12 R_{23}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^5}{\partial a \partial c^4}\log Q(a, ac) = \frac{24 R_{14}}{c^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^5}{\partial c^5}\log Q(a, ac) = \frac{120 R_{05}}{c^5} = a^5 \frac{\partial^5}{\partial x^5}\log Q(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<5>` struct containing \f$v_j = \frac{\partial^5 \log Q}{\partial a^{5-j} \partial c^j}\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j = 1, \dots, 5\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = -\frac{36}{a^5} - \delta^{(5)}(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 5\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.8e4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{5-j} c^j v_j| / (a^{5-j} c^j)\f$; test bound 87000 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d5(), log_gamma_q_ratio_mixed_d4(), log_gamma_q_ratio_mixed_d6(), log_gamma_q_mixed_d5()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<5> log_gamma_q_ratio_mixed_d5(double a, double c) {
  math::gamma_inc_mixed_partials<5> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(5, a, c, false, 0, r.v);
  return r;
}

/**
 * \ingroup incomplete_gamma
 * \brief Sixth partial derivatives of \f$\log Q(a, a c)\f$ in the shape \f$a\f$ (at fixed ratio \f$c\f$) and \f$c\f$.
 *
 * Computes the sixth partial derivatives of total order \f$K = 6\f$:
 * \f[
 * v_0 = \frac{\partial^6}{\partial a^6}\log Q(a, ac) = 720 R_{60}
 * \f]
 * \f[
 * v_1 = \frac{\partial^6}{\partial a^5 \partial c}\log Q(a, ac) = \frac{120 R_{51}}{c}
 * \f]
 * \f[
 * v_2 = \frac{\partial^6}{\partial a^4 \partial c^2}\log Q(a, ac) = \frac{48 R_{42}}{c^2}
 * \f]
 * \f[
 * v_3 = \frac{\partial^6}{\partial a^3 \partial c^3}\log Q(a, ac) = \frac{36 R_{33}}{c^3}
 * \f]
 * \f[
 * v_4 = \frac{\partial^6}{\partial a^2 \partial c^4}\log Q(a, ac) = \frac{48 R_{24}}{c^4}
 * \f]
 * \f[
 * v_5 = \frac{\partial^6}{\partial a \partial c^5}\log Q(a, ac) = \frac{120 R_{15}}{c^5}
 * \f]
 * \f[
 * v_6 = \frac{\partial^6}{\partial c^6}\log Q(a, ac) = \frac{720 R_{06}}{c^6} = a^6 \frac{\partial^6}{\partial x^6}\log Q(a, x)
 * \f]
 * where the derivatives in \f$a\f$ are taken at fixed ratio \f$c = x/a\f$, and \f$R_{ij}\f$ are the series coefficients of \f$\log Q(a + h, c(1 + g))\f$ in \f$(h, g)\f$ (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param c Non-negative ratio \f$c = x/a \in [0, \infty)\f$.
 * \return A `gamma_inc_mixed_partials<6>` struct containing \f$v_j = \frac{\partial^6 \log Q}{\partial a^{6-j} \partial c^j}\f$ for \f$j = 0, \dots, 6\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, c \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$c < 0\f$, or \f$a\f$ is NaN, or \f$c\f$ is NaN: returns quiet NaN for all components.
 * - \f$c = 0\f$: returns \f$v_0 = 0.0\f$; for \f$j = 1, \dots, 6\f$, \f$v_j = 0.0\f$ for \f$a > j\f$, \f$\pm\infty\f$ for non-integer \f$a < j\f$, quiet NaN for integer \f$a \le j\f$.
 * - \f$c = +\infty\f$: returns \f$v_0 = \frac{180}{a^6} - \delta^{(6)}(a)\f$, \f$v_j = 0.0\f$ for \f$j = 1, \dots, 6\f$.
 *
 * \par Measured accuracy
 * Maximum error 6.2e4 ulp (gcc/OpenMP; CUDA not yet measured) in \f$\max_j |a^{6-j} c^j v_j| / (a^{6-j} c^j)\f$; test bound 94000 ulp.
 *
 * \see incomplete_gamma, log_gamma_p_ratio_mixed_d6(), log_gamma_q_ratio_mixed_d5(), log_gamma_q_mixed_d6()
 */
LAPLACE_HD math::gamma_inc_mixed_partials<6> log_gamma_q_ratio_mixed_d6(double a, double c) {
  math::gamma_inc_mixed_partials<6> r;
  detail_gamma_ratio::log_gamma_ratio_impl<true>(6, a, c, false, 0, r.v);
  return r;
}

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls.
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<1> log_gamma_p_ratio_mixed_d1(double a, double c) { return own::log_gamma_p_ratio_mixed_d1(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<2> log_gamma_p_ratio_mixed_d2(double a, double c) { return own::log_gamma_p_ratio_mixed_d2(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<3> log_gamma_p_ratio_mixed_d3(double a, double c) { return own::log_gamma_p_ratio_mixed_d3(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<4> log_gamma_p_ratio_mixed_d4(double a, double c) { return own::log_gamma_p_ratio_mixed_d4(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<5> log_gamma_p_ratio_mixed_d5(double a, double c) { return own::log_gamma_p_ratio_mixed_d5(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<6> log_gamma_p_ratio_mixed_d6(double a, double c) { return own::log_gamma_p_ratio_mixed_d6(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<1> log_gamma_q_ratio_mixed_d1(double a, double c) { return own::log_gamma_q_ratio_mixed_d1(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<2> log_gamma_q_ratio_mixed_d2(double a, double c) { return own::log_gamma_q_ratio_mixed_d2(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<3> log_gamma_q_ratio_mixed_d3(double a, double c) { return own::log_gamma_q_ratio_mixed_d3(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<4> log_gamma_q_ratio_mixed_d4(double a, double c) { return own::log_gamma_q_ratio_mixed_d4(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<5> log_gamma_q_ratio_mixed_d5(double a, double c) { return own::log_gamma_q_ratio_mixed_d5(a, c); }
LAPLACE_HD_FORCEINLINE gamma_inc_mixed_partials<6> log_gamma_q_ratio_mixed_d6(double a, double c) { return own::log_gamma_q_ratio_mixed_d6(a, c); }

}  // namespace laplace::math
