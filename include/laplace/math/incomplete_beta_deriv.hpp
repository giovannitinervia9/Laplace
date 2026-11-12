// Derivatives of log I_x(a, b) and log(1 - I_x(a, b)) with respect to the
// shapes a and b, all mixed components of total order 1 to 5, Laplace's own
// (a, b > 0, 0 < x < 1). I_x(a, b) is the regularized incomplete beta function.
//
// One function per order: own::log_ibeta_d1 .. log_ibeta_d5 and
// own::log_ibetac_d1 .. log_ibetac_d5. Order K returns mixed_partials<K>, whose
// component c[j] is d^K / da^(K-j) db^j. They share the private helpers below,
// which compute every component up to the order asked for.
//
// Notation: s = a + b, y = 1 - x, lambda = a - s x (positive when x lies below
// the mean a / s), B(a, b) the beta function, psi^(n) the polygamma functions,
// l_ij = d^(i+j) log S / da^i db^j for the side S computed. Every side is
// handled as the lower side of a frame: 1 - I_x(a, b) = I_y(b, a) is the
// lower side of (b, a, y), and in a frame "a" is the side's own shape (the
// exponent of t) and "b" the other one.
//
// Mathematics. B(a, b) I_x(a, b) = J(a, b) = int_0^x t^(a-1) (1 - t)^(b-1) dt,
// and the derivatives of log J are the joint cumulants kappa_ij of
// (log T, log(1 - T)), T a Beta(a, b) variable truncated to T < x. Hence
//     l_ij = kappa_ij - d^(i+j) log B(a, b) / da^i db^j,
// with the derivatives of log B: psi^(i-1)(a) - psi^(i-1)(s) (j = 0),
// psi^(j-1)(b) - psi^(j-1)(s) (i = 0), -psi^(i+j-1)(s) (i, j > 0). The
// differences of polygamma functions are computed without forming the two
// terms (dpsi below), since b / a can be 2^-20; the first-order terms are
// written as l_10 = (c1 - log(a/s)) + E[X] + D(a) - D(s), with
// D(z) = log z - psi(z), X = log T - c1 and c1 the logarithm of t at the anchor
// of the quadrature (below), so that nothing of size log(a/s) cancels.
//
// In u = logit t the integrand t^a (1 - t)^b du is log-concave, with its
// maximum at the mean t_c = a / s. The side that does not contain t_c (lower
// when lambda >= 0) is monotone away from the end point; the other contains
// the maximum.
//
// The other side, log(1 - S), is composed from the l_ij (Faa di Bruno's
// formula as a composition of truncated bivariate Taylor polynomials,
// complement below). As b -> 0, S is proportional to b, and log S has a pole
// log b whose derivatives (n-1)!/b^n would cancel in the composition: it
// starts from S / b, whose logarithm (log Gamma(b + 1) in place of
// log Gamma(b)) has regular derivatives. S itself comes from the side's own
// computation: log S = log(x^a y^b / B) (detail_ibx::log_power, without
// cancellation) plus the logarithm of the quadrature's zeroth moment or of the
// series' sum; own::beta_inc is not called (it cost 0.3-0.45 us per call).
//
// Which side is computed (derivs below; each condition on a probability is
// checked on the S of the side computed, and the next choice taken if it fails):
//   - the side asked for, when its logarithm has a pole in a small shape (log I
//     in b, log(1 - I) in a, below pole_min = 1), its probability is at most
//     p_direct = 3/4, and its own shape is at least own_min = 1 or the series
//     applies to it: composed from the other side, the pole would cancel;
//   - otherwise the series where it applies, on the side asked for or on the
//     other side if its probability is at most p_mono = 3/4;
//   - otherwise the quadrature on the monotone side, or on the side with the
//     maximum where the monotone one has probability above p_mono (there
//     l_ij = kappa_ij - kappa_ij(full beta) would cancel).
//
// Candidate 1, quad: double-exponential rules on rays from an anchor (the end
// point x on the monotone side, t_c on the other). A ray towards t = 0 runs in
// v = log t, one towards t = 1 in w = -log(1 - t): the singularity of the
// integrand that is closest (t = 1, or t = 0) then lies behind the ray. (In
// u = logit t the integrand has poles of order s at u = +-i pi; where a ray
// runs towards Re u = 0 they narrow its strip of analyticity, and the first
// version in u converged like e^(-2.9/h).) In its own coordinate d <= 0,
//     X = d, Y = log1p(-r expm1(d)), r = t_c / (1 - t_c)       (towards 0)
// and the exponent a X + (b - 1) Y, its first-order terms written exactly near
// the anchor (ray_node). Nodes d = d0 - sigma exp(alpha (t - e^-t)), the map
// of Takahasi and Mori for exponentially decaying integrands, step h = 1/5,
// alpha = 1/2 where the start is Gaussian-like, 3/4 otherwise, sigma the
// distance where the exponent drops by one (a few evaluations), capped at 1.
// The nodes of the map are tabulated (constant memory on the device); a ray
// starts at d = 1e-6 sigma, one node at the weighted mean of the omitted ones
// standing for them. On the side with the maximum, towards the end point tanh-sinh when the
// exponent drops by less than 4.5 there, otherwise the ray from t_c minus the
// ray from x. The 21 moments of (X, Y) about their means (orders 3 and 5: the
// nodes stored and summed twice; order 4: the means from a first pass at 4h;
// orders 1-2: about the anchor) come from one evaluation per node; the cumulants follow from the
// moments by the bivariate recursion
//     mu_ij = sum_{k<i, l<=j} C(i-1, k) C(j, l) kappa_{k+1,l} mu_{i-1-k, j-l}.
// Its weakness: for a small own shape a the cumulants kappa_n0 ~ (n-1)!/a^n of
// the tail t^(a-1) cancel against psi^(n-1)(a).
//
// Candidate 2, series: for x <= 1/2 and (b <= 1 or b x <= 0.7) the power series
// (DiDonato and Morris' BPSER)
//     I_x(a, b) = x^a / (a B(a, b)) Sigma,  Sigma = 1 + sum_{n>=1} a / (a + n) (1 - b)_n x^n / n!,
// differentiated term by term: d^i/da^i of a / (a + n) in closed form, the
// b-derivatives of g_n = (1 - b)_n x^n / n! by the recursion
//     g_n^(j) = (x / n) ((n - b) g_(n-1)^(j) - j g_(n-1)^(j-1)),
// and log Sigma's derivatives by the cumulant recursion; log(a B(a, b)) =
// log Gamma(a + 1) + log Gamma(b) - log Gamma(s) has no pole at a = 0. The
// same on the upper side with y and the shapes exchanged, and up to x = 0.8 (quad_par::x_wide)
// for b <= 1 on a side computed directly; elsewhere the quadrature. This is
// the version chosen (chosen_method below; docs/numerics.md).
//
// Derived from the formulas above; no code copied (R's toms708.c is GPL).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/ibeta_expansions.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/special.hpp>

// On the device the side computations are subroutines, not inlined into the
// kernel (see LAPLACE_IBX_NOINLINE in detail/ibeta_expansions.hpp).
#if defined(__CUDACC__)
#define LAPLACE_IBD_NOINLINE __noinline__
#else
#define LAPLACE_IBD_NOINLINE
#endif

namespace laplace::math::own {

// The components of order K: c[j] = d^K / da^(K-j) db^j, j = 0..K.
/**
 * \ingroup special
 * \brief Container holding mixed partial derivatives of total order \f$K\f$ in two variables.
 *
 * Holds the \f$K + 1\f$ mixed partial derivative components of total order \f$K\f$:
 * \f[
 * c[j] = \frac{\partial^K}{\partial u^{K-j} \partial v^j} f(u, v), \quad j = 0, \dots, K
 * \f]
 * where \f$j\f$ is the number of derivatives with respect to the second variable \f$v\f$,
 * and \f$K - j\f$ is the number of derivatives with respect to the first variable \f$u\f$.
 *
 * The array `c[j]` is filled by each module as follows:
 * - Incomplete beta: \f$u = a\f$ and \f$v = b\f$, where `c[j]` holds \f$\frac{\partial^K}{\partial a^{K-j} \partial b^j} I_x(a, b)\f$
 *   or the corresponding derivatives of \f$1 - I_x\f$, \f$\log I_x\f$, or \f$\log(1 - I_x)\f$.
 * - Student \f$t\f$: \f$u = z\f$ and \f$v = \nu\f$, where `c[j]` holds \f$\frac{\partial^K}{\partial z^{K-j} \partial \nu^j} \log T_\nu(z)\f$.
 * - Skew normal: \f$u = z\f$ and \f$v = \alpha\f$, where `c[j]` holds \f$\frac{\partial^K}{\partial z^{K-j} \partial \alpha^j} \log F(z; \alpha)\f$.
 *
 * \tparam K Total order of differentiation, \f$K \in \{1, \dots, 5\}\f$.
 */
template <int K>
struct mixed_partials {
  /// Array of \f$K + 1\f$ mixed partial derivative components of total order \f$K\f$, where `c[j]` holds the derivative with \f$j\f$ differentiations in the second variable.
  double c[K + 1];
};

namespace detail_ibd {

namespace ibx = detail_ibx;

// derivatives up to order 5 in one array: (i, j) at (i + j)(i + j + 1)/2 + j
constexpr int NC = 21;
LAPLACE_HD_FORCEINLINE constexpr int ix(int i, int j) { return (i + j) * (i + j + 1) / 2 + j; }

LAPLACE_HD_FORCEINLINE double binom(int n, int k) {
  double c = 1.0;
  for (int m = 1; m <= k; ++m) c = c * (n - k + m) / m;
  return c;
}
LAPLACE_HD_FORCEINLINE double factorial(int n) {
  double f = 1.0;
  for (int m = 2; m <= n; ++m) f *= m;
  return f;
}

// ---------------------------------------------------------------- polygamma differences
//
// For w > 0, h >= 0 and p >= 1, w^-p - (w + h)^-p = w^-p E_p with
// E_p = 1 - rho^p, rho = w / (w + h): E_1 = h / (w + h) and
// E_(p+1) = E_p + E_1 rho^p, a sum of positive terms.
//
// dpsi(n, z, h) = psi^(n)(z) - psi^(n)(z + h), n = 1..4: with
// |psi^(n)(w)| = n! sum_k (w + k)^-(n+1), the terms below z + M >= 15 are
// differenced one by one, and at Z = z + M the asymptotic expansion
// (polygamma.hpp) term by term.
// dD(z, h) = D(z) - D(z + h), D(w) = log w - psi(w) = 1/(2w) + sum_k B_2k / (2k w^2k) + ...:
// the same, with D(z) = D(z + M) - log((z + M)/z) + sum_{k<M} 1/(z + k), so
// that dD = dD(Z, h) + h sum_k 1 / ((z + k)(z + k + h)) - log1p(M h / (z (z + h + M))).
// B_2k, k = 1..10 (a switch: device code cannot index a namespace-scope array
// at run time)
LAPLACE_HD_FORCEINLINE double bernoulli(int k) {
  switch (k) {
    case 1: return 1.0 / 6.0;
    case 2: return -1.0 / 30.0;
    case 3: return 1.0 / 42.0;
    case 4: return -1.0 / 30.0;
    case 5: return 5.0 / 66.0;
    case 6: return -691.0 / 2730.0;
    case 7: return 7.0 / 6.0;
    case 8: return -3617.0 / 510.0;
    case 9: return 43867.0 / 798.0;
    default: return -174611.0 / 330.0;
  }
}

LAPLACE_HD double dpsi(int n, double z, double h) {
  if (!(h > 0.0)) return 0.0;
  const int m = n + 1;
  const int M = z < 15.0 ? static_cast<int>(15.0 - z) + 1 : 0;
  // asymptotic part at Z: (n-1)! Z^-n + n!/2 Z^-(n+1) + sum_k B_2k (2k+1)...(2k+n-1) Z^-(2k+n)
  const double Z = z + M, Zh = Z + h;
  const double E1 = h / Zh, rho = Z / Zh, iZ = 1.0 / Z;
  double Ep = E1, rp = rho, zp = iZ;  // E_p, rho^p, Z^-p for p = 1
  double asym = 0.0;
  for (int p = 1; p <= 20 + n; ++p) {
    double c = 0.0;
    if (p == n) c += factorial(n - 1);
    if (p == n + 1) c += 0.5 * factorial(n);
    if (p > n && ((p - n) % 2 == 0)) {
      const int k = (p - n) / 2;
      double ck = bernoulli(k);
      for (int i = 1; i <= n - 1; ++i) ck *= static_cast<double>(2 * k + i);
      c += ck;
    }
    asym += c * zp * Ep;
    Ep += E1 * rp;
    rp *= rho;
    zp *= iZ;
  }
  double rec = 0.0;
  for (int k = M - 1; k >= 0; --k) {
    const double w = z + k, wh = w + h, e1 = h / wh, r = w / wh, iw = 1.0 / w;
    double e = e1, rr = r, wp = iw;
    for (int p = 1; p < m; ++p) {
      e += e1 * rr;
      rr *= r;
      wp *= iw;
    }
    rec += wp * e;
  }
  const double mag = asym + factorial(n) * rec;
  return (n % 2 == 1) ? mag : -mag;
}

LAPLACE_HD double dD(double z, double h) {
  if (!(h > 0.0)) return 0.0;
  const int M = z < 15.0 ? static_cast<int>(15.0 - z) + 1 : 0;
  const double Z = z + M, Zh = Z + h;
  const double E1 = h / Zh, rho = Z / Zh, iZ = 1.0 / Z;
  double Ep = E1, rp = rho, zp = iZ;
  double asym = 0.0;
  for (int p = 1; p <= 20; ++p) {
    double c = 0.0;
    if (p == 1) c = 0.5;
    if (p % 2 == 0) c = bernoulli(p / 2) / p;
    asym += c * zp * Ep;
    Ep += E1 * rp;
    rp *= rho;
    zp *= iZ;
  }
  if (M == 0) return asym;
  double sum = 0.0;
  for (int k = M - 1; k >= 0; --k) sum += h / ((z + k) * (z + k + h));
  return asym + sum - Kokkos::log1p(M * h / (z * (z + h + M)));
}

// psi(z) - psi(z + h) = -log1p(h / z) - dD(z, h), both terms negative
LAPLACE_HD_FORCEINLINE double dpsi0(double z, double h) {
  if (!(h > 0.0)) return 0.0;
  return -Kokkos::log1p(h / z) - dD(z, h);
}

// psi^(n)(z) - psi^(n)(w), any order of z and w (d = w - z exact)
LAPLACE_HD_FORCEINLINE double dpsi_any(int n, double z, double w, double d) {
  if (d >= 0.0) return n == 0 ? dpsi0(z, d) : dpsi(n, z, d);
  return n == 0 ? -dpsi0(w, -d) : -dpsi(n, w, -d);
}

// All the differences at once, for one evaluation per call (the separate
// functions above repeat the shift and the asymptotic powers for each order;
// measured 2026-11-09, the log-beta terms of order 5 cost 0.5-1 us that way):
//   out[n] = psi^(n)(z) - psi^(n)(z + h), n = 0..N (N <= 4), and dDv = D(z) - D(z + h).
LAPLACE_HD void dpsi_all(double z, double h, int N, double* out, double& dDv) {
  for (int n = 0; n <= N; ++n) out[n] = 0.0;
  dDv = 0.0;
  if (!(h > 0.0)) return;
  const int M = z < 15.0 ? static_cast<int>(15.0 - z) + 1 : 0;
  const double Z = z + M, Zh = Z + h;
  const double E1 = h / Zh, rho = Z / Zh, iZ = 1.0 / Z;
  // zE[p] = Z^-p E_p, p = 1..20 + N, then the asymptotic sums with constant
  // coefficients: D: 1/2 at p = 1, B_2k / (2k) at p = 2k; psi^(n): (n-1)! at
  // p = n, n!/2 at p = n + 1, B_2k (2k+1)...(2k+n-1) at p = 2k + n
  double zE[25];
  double Ep = E1, rp = rho, zp = iZ;
  for (int p = 1; p <= 20 + N; ++p) {
    zE[p] = zp * Ep;
    Ep += E1 * rp;
    rp *= rho;
    zp *= iZ;
  }
  const double fact[5] = {1.0, 1.0, 2.0, 6.0, 24.0};
  constexpr double cD[10] = {0.08333333333333333,  -0.008333333333333333, 0.003968253968253968, -0.004166666666666667,
                             0.007575757575757576, -0.021092796092796094, 0.08333333333333333,  -0.4432598039215686,
                             3.0539543302701198,   -26.456212121212122};
  constexpr double cP[4][10] = {
      {0.16666666666666666, -0.03333333333333333, 0.023809523809523808, -0.03333333333333333, 0.07575757575757576,
       -0.2531135531135531, 1.1666666666666667, -7.092156862745098, 54.971177944862156, -529.1242424242424},
      {0.5, -0.16666666666666666, 0.16666666666666666, -0.3, 0.8333333333333334, -3.2904761904761903, 17.5,
       -120.56666666666666, 1044.452380952381, -11111.60909090909},
      {2.0, -1.0, 1.3333333333333333, -3.0, 10.0, -46.06666666666667, 280.0, -2170.2, 20889.04761904762, -244455.4},
      {10.0, -7.0, 12.0, -33.0, 130.0, -691.0, 4760.0, -41233.8, 438670.0, -5622474.2}};
  double asym[5] = {0.0, 0.0, 0.0, 0.0, 0.0}, asymD = 0.0;
  for (int k = 10; k >= 1; --k) asymD += cD[k - 1] * zE[2 * k];
  asymD += 0.5 * zE[1];
  for (int n = 1; n <= N; ++n) {
    double t = 0.0;
    for (int k = 10; k >= 1; --k) t += cP[n - 1][k - 1] * zE[2 * k + n];
    asym[n] = fact[n - 1] * zE[n] + 0.5 * fact[n] * zE[n + 1] + t;
  }
  double rec[5] = {0.0, 0.0, 0.0, 0.0, 0.0}, sumD = 0.0;
  for (int k = M - 1; k >= 0; --k) {
    // the same operations as dpsi and dD
    const double w = z + k, wh = w + h, iw = 1.0 / w;
    const double e1 = h / wh, r = w / wh;
    sumD += h / (w * wh);
    double e = e1, rr = r, wp = iw;
    for (int p = 1; p <= N; ++p) {  // E_(p+1) and w^-(p+1) for order n = p
      e += e1 * rr;
      rr *= r;
      wp *= iw;
      rec[p] += wp * e;
    }
  }
  dDv = M == 0 ? asymD : asymD + sumD - Kokkos::log1p(M * h / (z * (z + h + M)));
  out[0] = -Kokkos::log1p(h / z) - dDv;
  for (int n = 1; n <= N; ++n) {
    const double mag = asym[n] + fact[n] * rec[n];
    out[n] = (n % 2 == 1) ? mag : -mag;
  }
}

// psi^(n)(z) - psi^(n)(w), n = 0..N, any order of z and w (d = w - z exact)
LAPLACE_HD_FORCEINLINE void dpsi_any_all(double z, double w, double d, int N, double* out) {
  double dDv;
  if (d >= 0.0) {
    dpsi_all(z, d, N, out, dDv);
  } else {
    dpsi_all(w, -d, N, out, dDv);
    for (int n = 0; n <= N; ++n) out[n] = -out[n];
  }
}

// psi^(n)(s), n = 1..N (N <= 4), with one shift to Z = s + M >= 15 and the
// asymptotic expansion there (as own::polygamma, polygamma.hpp)
LAPLACE_HD void psi_all(double s, int N, double* out) {
  const int M = s < 15.0 ? static_cast<int>(15.0 - s) + 1 : 0;
  const double Z = s + M, iZ = 1.0 / Z, w2 = iZ * iZ;
  const double fact[6] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0};
  double rec[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int k = M - 1; k >= 0; --k) {
    const double iw = 1.0 / (s + k);
    double wp = iw;
    for (int n = 1; n <= N; ++n) {
      wp *= iw;
      rec[n] += wp;
    }
  }
  double zn = 1.0;
  for (int n = 1; n <= N; ++n) {
    zn *= iZ;  // Z^-n
    // sum_k B_2k (2k+1)...(2k+n-1) Z^-2k, Horner in w2 from k = 10
    double sb = 0.0;
    for (int k = 10; k >= 1; --k) {
      double ck = bernoulli(k);
      for (int i = 1; i <= n - 1; ++i) ck *= static_cast<double>(2 * k + i);
      sb = (sb + ck) * w2;
    }
    const double mag = (fact[n - 1] + 0.5 * fact[n] * iZ + sb) * zn + fact[n] * rec[n];
    out[n] = (n % 2 == 1) ? mag : -mag;
  }
}

// ---------------------------------------------------------------- moments and cumulants

// kappa from the moments mu (mu_00 = 1), orders 1..K
template <int K>
LAPLACE_HD void cumulants(const double* mu, double* kap) {
  kap[0] = 0.0;
  for (int n = 1; n <= K; ++n) {
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = mu[ix(i, j)];
      if (i >= 1) {
        for (int k = 0; k <= i - 1; ++k)
          for (int l = 0; l <= j; ++l) {
            if (k == i - 1 && l == j) continue;
            v -= binom(i - 1, k) * binom(j, l) * kap[ix(k + 1, l)] * mu[ix(i - 1 - k, j - l)];
          }
      } else {
        for (int l = 0; l <= j - 2; ++l) v -= binom(j - 1, l) * kap[ix(0, l + 1)] * mu[ix(0, j - 1 - l)];
      }
      kap[ix(i, j)] = v;
    }
  }
}

// truncated product of bivariate Taylor coefficient arrays, total order <= K
template <int K>
LAPLACE_HD void tmul(const double* p, const double* q, double* r) {
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      for (int k = 0; k <= i; ++k)
        for (int l = 0; l <= j; ++l) v += p[ix(k, l)] * q[ix(i - k, j - l)];
      r[ix(i, j)] = v;
    }
}

// moments mu (mu_00 = 1, mu_ij = S_ij / S) from the cumulants kappa (the
// derivatives of log S), orders 1..K, by the recursion of cumulants above
// solved the other way
template <int K>
LAPLACE_HD void moments(const double* kap, double* mu) {
  mu[0] = 1.0;
  for (int n = 1; n <= K; ++n) {
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      if (i >= 1) {
        for (int k = 0; k <= i - 1; ++k)
          for (int l = 0; l <= j; ++l) v += binom(i - 1, k) * binom(j, l) * kap[ix(k + 1, l)] * mu[ix(i - 1 - k, j - l)];
      } else {
        for (int l = 0; l <= j - 1; ++l) v += binom(j - 1, l) * kap[ix(0, l + 1)] * mu[ix(0, j - 1 - l)];
      }
      mu[ix(i, j)] = v;
    }
  }
}

// Derivatives of log(1 - S) for the lower side S of a frame with shapes a, b.
// As b -> 0, S is proportional to b (1 - S stays finite): log S has a pole
// log b whose derivatives, of size 1/b^n, would cancel in the composition.
// The composition therefore starts from S~ = S / b, whose logarithm lt has
// regular derivatives (log Gamma(b + 1) in place of log Gamma(b)):
//   M~ = S~_ij / S~ from lt (moments from cumulants),
//   M_ij = S_ij / S = M~_ij + (j / b) M~_(i,j-1)   (Leibniz, S = b S~),
//   (1 - S)_ij / (1 - S) = -r M_ij,  r = S / (1 - S),
//   and the derivatives of log(1 - S) from these (cumulants from moments).
// This is Faa di Bruno's formula; a first version composed truncated bivariate
// Taylor polynomials, and cost 0.95 us at order 5 against about 0.2 us
// (measured 2026-11-09). Where r is below 2^-900 (S subnormal or nearly) the
// result is -r M to first order, formed as exp(log r + log|M_ij|) so that the
// digits of a subnormal r are not lost.
template <int K>
LAPLACE_HD void complement(const double* lt, double b, double r, double log_r, double* out) {
  double Mt[NC], M[NC];
  moments<K>(lt, Mt);
  const double ib = 1.0 / b;
  M[0] = 1.0;
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      M[ix(i, j)] = Mt[ix(i, j)] + (j > 0 ? j * ib * Mt[ix(i, j - 1)] : 0.0);
    }
  if (r < 7.4e-272) {  // 2^-900
    for (int n = 1; n <= K; ++n)
      for (int j = 0; j <= n; ++j) {
        const double t = M[ix(n - j, j)];
        out[ix(n - j, j)] = t == 0.0 ? 0.0 : (t > 0.0 ? -1.0 : 1.0) * math::exp(log_r + math::log(Kokkos::fabs(t)));
      }
    out[0] = 0.0;
    return;
  }
  for (int c = 1; c < NC; ++c) M[c] *= -r;
  cumulants<K>(M, out);
  out[0] = 0.0;
}

// l_ij from the cumulants kappa of (X, Y) = (log T - c1, log(1 - T) - c2) on
// the lower side: l = kappa - d log B, with the first order written as above
// (c1off = c1 - log(a/s), c2off = c2 - log(b/s)); and, when want_lt, lt, the
// derivatives of log(S / b), which differ in the pure b-derivatives:
//   lt_01 = l_01 - 1/b = c2off + kappa_01 + (D(b) - 1/b) - D(s),
//   lt_0j = kappa_0j - (psi^(j-1)(b + 1) - psi^(j-1)(s)),  j >= 2.
template <int K>
LAPLACE_HD void assemble(double a, double b, double c1off, double c2off, const double* kap, double* l, double* lt,
                         bool want_lt) {
  const double s = a + b;
  double Pa[5], Pb[5], Ps[5], Pt[5], Da, Db;
  dpsi_all(a, b, K - 1, Pa, Da);  // psi^(n)(a) - psi^(n)(s)
  dpsi_all(b, a, K - 1, Pb, Db);  // psi^(n)(b) - psi^(n)(s)
  if (K >= 2) psi_all(s, K - 1, Ps);
  if (want_lt) dpsi_any_all(b + 1.0, s, a - 1.0, K - 1, Pt);  // psi^(n)(b + 1) - psi^(n)(s)
  l[0] = lt[0] = 0.0;
  l[ix(1, 0)] = lt[ix(1, 0)] = c1off + kap[ix(1, 0)] + Da;
  l[ix(0, 1)] = c2off + kap[ix(0, 1)] + Db;
  // (D(b) - 1/b) - D(s): for b >= 1, D(b) < 1/(2b) and the difference with
  // 1/b loses at most a factor 2; below, log(b/s) - (psi(b + 1) - psi(s))
  if (want_lt) lt[ix(0, 1)] = c2off + kap[ix(0, 1)] + (b >= 1.0 ? Db - 1.0 / b : -Kokkos::log1p(a / b) - Pt[0]);
  for (int n = 2; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      if (j == 0) {
        l[ix(i, j)] = lt[ix(i, j)] = kap[ix(i, j)] - Pa[i - 1];
      } else if (i == 0) {
        l[ix(i, j)] = kap[ix(i, j)] - Pb[j - 1];
        if (want_lt) lt[ix(i, j)] = kap[ix(i, j)] - Pt[j - 1];
      } else {
        l[ix(i, j)] = lt[ix(i, j)] = kap[ix(i, j)] + Ps[n - 1];
      }
    }
}

// ---------------------------------------------------------------- the quadrature

struct quad_par {
  double h = 0.2;         // step of the rays
  double hmean = 4.0;     // the first pass (the means) at this multiple of the steps
  double hlow = 1.0;      // step multiple for orders 1 to 3
  double alpha_g = 0.5;   // power of the ray map where the start is Gaussian-like
  double alpha_e = 0.75;  // and where it is exponential-like
  double g_switch = 0.5;  // Gaussian-like when (slope at the start) x (scale) is below this
  double cap = 1.0;       // largest scale of a ray (the scale of the structure of t and 1 - t)
  double thi = 30.0;      // last t of a ray (the loop stops earlier, once phi < -cut)
  double hts = 0.125;     // tanh-sinh: step
  double tts = 3.25;      // tanh-sinh: t in [-tts, tts]
  double cut = 80.0;      // nodes where phi is below -cut (relative to the anchor) end a ray
  double kx_fin = 4.5;    // tanh-sinh to the end point when phi drops less than this there
  double p_mono = 0.75;   // integrate the monotone side when its probability is at most this
  double pole_min = 1.0;  // the requested side directly when its pole shape is below this
  double p_direct = 0.75; // and its probability at most this
  double own_min = 1.0;   // and its own shape at least this, unless the series applies
  double drop_lo = 0.5;   // the scale search stops when the drop is in [drop_lo, drop_hi]
  double drop_hi = 2.0;
  bool own_elem = false;  // own::exp and own::log in the nodes (a variant for measurement)
  double e_start = 1e-6;  // rays start at the first table node with d / sigma >= e_start (one node stands for those before)
  double gl_c16 = 1e-4;   // Gauss-Laguerre with 16, 20, 24, 32 nodes on a ray where kappa / (2 g^2) is at most these
  double gl_c20 = 1e-3;   // (the errors on the references are those of the double-exponential rule up to
  double gl_c24 = 3e-3;   // 1e-2 for 32 nodes; at 3e-2 they exceed the test bounds; measured 2026-11-09)
  double gl_c32 = 1e-2;
  bool gauss_device = false;  // Gauss-Laguerre also on the device (on CUDA it was 10 per cent slower:
                              // divergence between the two rules; host only, decided 2026-11-09)
  double series_below = 1e300;  // the series only where min(a, b) is below this
  double x_wide = 0.8;    // the series on a side computed directly, for b <= 1, up to x = x_wide (0.95 and 0.8 give the same errors)
  bool pure_check = true;   // the series stops on its pure components only (the same errors as all 21)
  bool store = true;      // orders 3-5: store the nodes and centre the moments in a second pass
  int* count = nullptr;   // measurement only (host): incremented once per node evaluated
  int* sides = nullptr;   // measurement only (host): incremented once per side computed
};

// e^d - 1 - d for |d| <= 1: sum_{k=2..19} d^k / k!, by Horner with the
// coefficients as constants (multiply-adds only; truncation below 2^-60
// relative). A first version divided by k at each step, and its 17 divisions
// were the largest cost of a node (measured 2026-11-09).
LAPLACE_HD_FORCEINLINE double expm1mx(double d) {
  double s = 8.2206352466243297e-18;  // 1/19!
  s = s * d + 1.5619206968586226e-16;  // 1/18!
  s = s * d + 2.8114572543455208e-15;
  s = s * d + 4.7794773323873853e-14;
  s = s * d + 7.6471637318198165e-13;
  s = s * d + 1.1470745597729725e-11;
  s = s * d + 1.6059043836821613e-10;
  s = s * d + 2.0876756987868100e-09;
  s = s * d + 2.5052108385441720e-08;
  s = s * d + 2.7557319223985888e-07;
  s = s * d + 2.7557319223985893e-06;
  s = s * d + 2.4801587301587302e-05;
  s = s * d + 1.9841269841269841e-04;
  s = s * d + 1.3888888888888889e-03;
  s = s * d + 8.3333333333333333e-03;
  s = s * d + 4.1666666666666667e-02;
  s = s * d + 1.6666666666666667e-01;
  s = s * d + 0.5;  // 1/2!
  return d * d * s;
}

constexpr double halfpi = 1.57079632679489661923;

// The elementary functions of the nodes. With own = false (the default): the
// platform's exp, expm1 and log1p (Kokkos), except where the argument is small
// (|d| <= 1/2 for expm1, 0 <= q <= 1 for log1p), where the Taylor series of
// expm1mx above and the atanh series of detail_ibx::log1pmx are used, as
// accurate and cheaper. With own = true (a variant for measurement only; the
// choice of elementary functions is open), own::exp and own::log
// (elementary.hpp) everywhere.
LAPLACE_HD_FORCEINLINE double node_exp(double z, bool own) { return own ? math::own::exp(z) : math::exp(z); }
LAPLACE_HD_FORCEINLINE double node_expm1(double d, bool own) {
  if (d >= -0.5 && d <= 0.5) return d + expm1mx(d);
  return own ? math::own::exp(d) - 1.0 : Kokkos::expm1(d);
}
// log(1 + q), q >= 0
LAPLACE_HD_FORCEINLINE double node_log1p(double q, bool own) {
  if (q <= 1.0) return ibx::log1p_poly(q);
  return own ? math::own::log(1.0 + q) : Kokkos::log1p(q);
}

// The nodes of the ray map d / sigma = exp(alpha (t - e^-t)), t = tlo + k h,
// tlo = -log(42 / alpha), h = 1/5, for alpha = 1/2 (185 nodes) and 3/4 (129
// nodes), until d / sigma >= 1e7: pairs (e_k, alpha (1 + e^-t_k) e_k),
// correctly rounded (mpmath). Written by a script of the session that wrote
// this file (2026-11-09); beyond the table the nodes are computed.
// clang-format off
#define LAPLACE_IBD_MAP_DATA \
  6.273243091895032e-20, 2.6661283140553885e-18, \
  1.4039103625218656e-16, 4.897778789244434e-15, \
  7.903778758695924e-14, 2.2647046571749946e-12, \
  1.4375480332468866e-11, 3.385438371640234e-10, \
  1.0366982928203647e-09, 2.0082729087042977e-08, \
  3.5055092954248434e-08, 5.591615627384432e-07, \
  6.375932476835945e-07, 8.38445124561907e-06, \
  6.979865679714051e-06, 7.578090762404052e-05, \
  5.042304954460773e-05, 0.00045278152622089225, \
  0.00025918675230459406, 0.0019290112200562486, \
  0.0010082992328914139, 0.006235405031811696, \
  0.003122438379084916, 0.01609221262570621, \
  0.00802204070419088, 0.034576231135414706, \
  0.017687472038826178, 0.06401963321077628, \
  0.03440920959389559, 0.10508650478908754, \
  0.06041817728609764, 0.15654693341764486, \
  0.09754699993490108, 0.21577518973806334, \
  0.1470348364310238, 0.27961281638647956, \
  0.20950561413303584, 0.345180683638785, \
  0.28508131677839726, 0.41039520050960016, \
  0.3735642630881043, 0.47414899361128016, \
  0.4746286235613903, 0.5362421711848169, \
  0.5879833338071881, 0.5971842592561826, \
  0.713490968123932, 0.6579650327701431, \
  0.8512428887194805, 0.7198530177201543, \
  1.0015991038218623, 0.7842458622980261, \
  1.1652036689913214, 0.852574520834153, \
  1.3429856077471245, 0.92625233836402, \
  1.5361531311049252, 1.0066569324643937, \
  1.7461865593372712, 1.0951336466380226, \
  1.9748333383727292, 1.1930118115002417, \
  2.2241070603002493, 1.3016277124262734, \
  2.49629141152628, 1.422350414096031, \
  2.7939493792319436, 1.5566082658453964, \
  3.1199377333694516, 1.7059150493152317, \
  3.477426671459493, 1.871895447795918, \
  3.8699244954298657, 2.056309937441576, \
  4.301307234883695, 2.261079426240997, \
  4.775853208474054, 2.4883100727685035, \
  5.298282606095866, 2.740318754961915, \
  5.873802269391954, 3.019659662788915, \
  6.508155941896231, 3.329152478009958, \
  7.207680351438011, 3.6719125910739074, \
  7.9793675764064975, 4.051383795829275, \
  8.830934235237423, 4.471373900294928, \
  9.770898126565651, 4.936093697471941, \
  10.808663037564678, 5.450199754397257, \
  11.954612531754126, 6.018841500209863, \
  13.220213626643329, 6.647713124654677, \
  14.618131377584838, 7.343110836947484, \
  16.162355498688882, 8.111996081130785, \
  17.868340276138433, 8.962065357970877, \
  19.7531591653005, 9.901827365237981, \
  21.835675612254896, 10.940688238181501, \
  24.13673180443613, 12.08904575061479, \
  26.67935723579017, 13.358393424851872, \
  29.48899917110203, 14.761435596521762, \
  32.59377731402727, 16.31221458888868, \
  36.02476522612227, 18.026251271728228, \
  39.81630131228988, 19.920700413189756, \
  44.006332484256816, 22.014522380698732, \
  48.63679394096594, 24.328672910260245, \
  53.754028866396546, 26.886312844121765, \
  59.409252244950544, 29.71303993642726, \
  65.65906343617371, 32.837145047227345, \
  72.56601263864592, 36.28989528920017, \
  80.1992269122633, 40.10584696111715, \
  88.63510202425168, 44.32319140015021, \
  97.95806704306263, 48.98413721454009, \
  108.26142933240907, 54.13533272222301, \
  119.64830840238272, 59.82833282338577, \
  132.23266796391886, 66.12011497961299, \
  146.14045651573306, 73.07364946375348, \
  161.5108678790909, 80.75852958778339, \
  178.49773429625324, 89.25166821621727, \
  197.27106603518578, 98.63806753602135, \
  218.0187529094069, 109.01166978715523, \
  240.94844474235185, 120.47629746814701, \
  266.2896295965681, 133.14669242660074, \
  294.29593056735615, 147.14966423420768, \
  325.2476441279361, 162.62535933958733, \
  359.45454543074993, 179.72866370106885, \
  397.25898864129675, 198.63075293743506, \
  439.0393333336837, 219.52080551105945, \
  485.2137312404093, 242.60789608955054, \
  536.2443112554441, 268.12308803530334, \
  592.6418045755262, 296.3217459652974, \
  654.970656269638, 327.4860915260178, \
  723.8546744349721, 361.9280279625533, \
  799.9832734780362, 399.9922617511093, \
  884.1183740057518, 442.05975253728286, \
  977.1020283829791, 488.55152590824156, \
  1079.8648482756073, 539.9328871583123, \
  1193.4353185248956, 596.7180782207619, \
  1318.95009056945, 659.4754243739083, \
  1457.6653584348624, 728.8330222315269, \
  1610.9694311457408, 805.4850259448723, \
  1780.3966273890526, 890.198594530737, \
  1967.6426314912478, 983.8215698567429, \
  2174.581464396936, 1087.2909621277222, \
  2403.284239500381, 1201.642327798787, \
  2656.039891044474, 1328.0201337723952, \
  2935.378082543978, 1467.6892116077781, \
  3244.094524508227, 1622.0474163803108, \
  3585.27895485158, 1792.6396168849415, \
  3962.346062028959, 1981.1731572023389, \
  4379.069660385627, 2189.534944372311, \
  4839.620459759576, 2419.8103331936704, \
  5348.607807347411, 2674.303997155972, \
  5911.125819600344, 2955.562994386425, \
  6532.804365853837, 3266.402259463726, \
  7219.865413952171, 3609.9327762294524, \
  7979.185301793906, 3989.592713559991, \
  8818.36355803279, 4409.1818357162565, \
  9745.798960714816, 4872.899531661564, \
  10770.773595071825, 5385.386843957834, \
  11903.54575175034, 5951.772917879461, \
  13155.452595232264, 6577.726335623186, \
  14539.023629987589, 7269.511849383999, \
  16068.106099966428, 8034.053081100758, \
  17758.0035764707, 8879.001816391667, \
  19625.629121439408, 9812.814586196595, \
  21689.67455905723, 10844.83730258106, \
  23970.797549813477, 11985.398795765452, \
  26491.82833931154, 13245.914188529516, \
  29277.99825104044, 14638.99914259789, \
  32357.192209940797, 16178.596120422913, \
  35760.22782410612, 17880.113926035076, \
  39521.16381776275, 19760.581921532823, \
  43677.64090242943, 21838.82046266222, \
  48271.25849780971, 24135.629259262987, \
  53347.991072766345, 26673.9955457556, \
  58958.648273258244, 29479.324145109644, \
  65159.38344235495, 32579.691728850965, \
  72012.25562176782, 36006.127817827175, \
  79585.85065959833, 39792.92533608169, \
  87955.96764055827, 43977.9838259638, \
  97206.37750868656, 48603.188759486984, \
  107429.66147511358, 53714.830742210994, \
  118728.1376019397, 59364.06880518115, \
  131214.8848357912, 65607.44242170613, \
  145014.8747399251, 72507.43737341047, \
  160266.2222516388, 80133.1111289392, \
  177121.56798298444, 88560.78399431513, \
  195749.60589931867, 97874.80295221362, \
  216336.77166520793, 108168.38583491517, \
  239089.1085552226, 119544.55427970257, \
  264234.3296042825, 132117.1648040335, \
  292024.09663624724, 146012.0483198358, \
  322736.53898003604, 161368.26949156725, \
  356679.0370814338, 178339.51854211872, \
  394191.298869907, 197095.64943622192, \
  435648.7596697412, 217824.3798360183, \
  481466.339682955, 240733.16984251598, \
  532102.5966501411, 266051.2983260102, \
  588064.3152504606, 294032.15762608056, \
  649911.579173049, 324955.7895872938, \
  718263.3766228267, 359131.68831210944, \
  793803.7953625034, 396901.89768188156, \
  877288.8692928376, 438644.43464698875, \
  969554.1450940263, 484777.0725475289, \
  1071523.0446577182, 535761.5223293257, \
  1184216.107003677, 592108.0535022607, \
  1308761.2021773076, 654380.6010890359, \
  1446404.8193521597, 723202.4096764256, \
  1598524.542112537, 799262.2710565813, \
  1766642.8357730294, 883321.4178867977, \
  1952442.2847231012, 976221.1423618067, \
  2157782.4322971962, 1078891.2161488298, \
  2384718.3917094427, 1192359.195854931, \
  2635521.4143174486, 1317760.7071589138, \
  2912701.621069282, 1456350.8105348125, \
  3219033.1246375996, 1609516.562318955, \
  3557581.7936716876, 1778790.8968359844, \
  3931735.9370413767, 1965867.9685208155, \
  4345240.2151710605, 2172620.1075856453, \
  4802233.117859849, 2401116.558930028, \
  5307288.383678469, 2653644.191839329, \
  5865460.775482173, 2932730.3877411718, \
  6482336.67017635, 3241168.335088252, \
  7164089.969054248, 3582044.984527194, \
  7917543.888276236, 3958771.944138181, \
  8750239.247910494, 4375119.623955304, \
  9670509.942994816, 4835254.971497459, \
  10687566.35195927, 5343783.175979681, \
  2.80860548598085e-20, 1.2006788452568135e-18, \
  6.60773665895938e-17, 2.3217400535331314e-15, \
  3.910775141935455e-14, 1.1303486222989432e-12, \
  7.47765045185026e-12, 1.779687441408931e-10, \
  5.669044975890721e-10, 1.1123696020705656e-08, \
  2.0152241838432272e-08, 3.264851909455416e-07, \
  3.8532809592533023e-07, 5.163456943498087e-06, \
  4.4345418376253895e-06, 4.925477763069809e-05, \
  3.367794035460626e-05, 0.00031083572868067583, \
  0.00018198849751386236, 0.0013999562180780501, \
  0.0007442782032028779, 0.004788746912768702, \
  0.002423005858526357, 0.013093275871811434, \
  0.006544254392964964, 0.029842808096794904, \
  0.015168959867721288, 0.0586961428417331, \
  0.03102269026173052, 0.10249967067627794, \
  0.057264714971662675, 0.1626923119618843, \
  0.09719594027143887, 0.23929762641854577, \
  0.15401718276677578, 0.33139528187327044, \
  0.23070622966808402, 0.4377872976761696, \
  0.3300252256728212, 0.5576015185274784, \
  0.4546303529211899, 0.6907002703133087, \
  0.6072419337018979, 0.8378810590249977, \
  0.7908380729587403, 1.000922843579143, \
  1.0088480452495714, 1.1825485773071998, \
  1.2653349324755487, 1.3863634292722908, \
  1.5651668911000427, 1.6168076537171614, \
  1.9141822851123604, 1.8791445291681292, \
  2.3193566295237877, 2.1794910741280793, \
  2.7889800509428335, 2.524892299934926, \
  3.3328537955531736, 2.923437034288454, \
  3.9625138740256, 3.3844131809122873, \
  4.6914896058818405, 3.918501419325044, \
  5.535604788881155, 4.538008016392908, \
  6.5133295323165905, 5.257139212416214, \
  7.646191461792031, 6.0923213934279214, \
  8.959256014070071, 7.062572937881239, \
  10.481686882356557, 8.189935266084465, \
  12.247399343729887, 9.499972287748895, \
  14.295821213726594, 11.022349207476992, \
  16.672778554905616, 12.791503580149826, \
  19.431526056906957, 14.847423676078874, \
  22.63394525896896, 17.236551686679775, \
  26.351937569493884, 20.012832144023978, \
  30.669043432597956, 23.23892921490019, \
  35.68232409544425, 26.987640341956496, \
  41.50454835625363, 31.34353813080698, \
  48.26673355407922, 36.404877524810665, \
  56.12109805247347, 42.28581128592003, \
  65.24449175007234, 49.11896374610151, \
  75.84238193106688, 57.05842086637875, \
  88.15348429058452, 66.28320402173304, \
  102.4551435165143, 77.00130583146986, \
  119.06958470828314, 89.45437902197736, \
  138.3711765455495, 103.92318402757961, \
  160.79486992836405, 120.73391813717234, \
  186.84600230927825, 140.26556886498446, \
  217.1116887245642, 162.95845731101352, \
  252.2740563001881, 189.32416410082757, \
  293.12562056479914, 219.9570616601446, \
  340.58715018356094, 255.5477127894054, \
  395.7284228215683, 296.8984375736669, \
  459.79234001837597, 344.94139954188995, \
  534.2229446749797, 400.75961877889904, \
  620.6979727283353, 465.6113856730848, \
  721.1666727993113, 540.9586256406291, \
  837.8937463519145, 628.4998542306777, \
  973.5103988716437, 730.2084654931532, \
  1131.0736528690657, 848.3772167143921, \
  1314.1352597546581, 985.6699123056067, \
  1526.8217640108846, 1145.1814519139975, \
  1773.9275244850523, 1330.5075963744423, \
  2061.022789708796, 1545.8260241813448, \
  2394.5792635012517, 1795.9905056735756, \
  2782.1159913828096, 2086.640317827785, \
  3232.3688564025474, 2424.3273661124763, \
  3755.487505191038, 2816.6638790116617, \
  4363.264143388404, 3272.494004580535, \
  5069.399358003957, 3802.09317720043, \
  5889.810958932992, 4417.399748702238, \
  6842.992801603569, 5132.284105337926, \
  7950.431679411752, 5962.861337093976, \
  9237.091683625658, 6927.854507607014, \
  10731.97694930517, 8049.016713591751, \
  12468.785472778916, 9351.621447937408, \
  14486.668739175138, 10865.032320345092, \
  16831.114283696486, 12623.364978273612, \
  19554.971081528238, 14666.256149360861, \
  22719.640880943734, 17039.75714124357, \
  26396.46233489882, 19797.371940244233, \
  30668.31913251281, 23001.263309973423, \
  35631.50838128815, 26723.654077986448, \
  41397.911357549965, 31048.455198605378, \
  48097.51555863761, 36073.15729205539, \
  55881.34490950113, 41911.02829940522, \
  64924.864177061936, 48693.666793331, \
  75431.93433539198, 56573.96850199445, \
  87639.4080444202, 65729.57291806664, \
  101822.46883445402, 76366.86768711352, \
  118300.83435358363, 88725.64104314364, \
  137445.96351293105, 103084.48716753986, \
  159689.42999479198, 119767.08632016076, \
  185532.65088112542, 139149.50131070334, \
  215558.189707267, 161668.65478898352, \
  250442.88873677954, 187832.17845106966, \
  290973.12648805825, 218229.8561842328, \
  338062.54445119924, 253546.91910459404, \
  392772.6425946681, 294579.49218712223, \
  456336.70793019223, 342252.54068930005, \
  530187.6155388323, 397640.7209206739, \
  615990.1287550534, 461992.6053809048, \
  715678.4266266254, 536758.82835469, \
  831499.704602577, 623624.7864277262, \
  966064.8313054667, 724548.6310659094, \
  1122407.203304032, 841805.4096948204, \
  1304051.1246034023, 978038.3503173807, \
  1515091.252278326, 1136318.445738772, \
  1760284.8991302904, 1320213.680559272, \
  2045159.2740752562, 1533869.4614650554, \
  2376136.077698333, 1782102.0638941964, \
  2760676.2616356886, 2070507.2015731009, \
  3207448.214981412, 2405586.166321649, \
  3726523.169014162, 2794892.381598185, \
  4329602.225099784, 3247201.673426471, \
  5030280.123482624, 3772710.0969891762, \
  5844351.69889943, 4383263.778338302, \
  6790167.931205476, 5092625.952364769, \
  7889049.617184121, 5916787.216655589, \
  9165767.988620419, 6874325.995049069, \
  10649103.1108367, 7986827.336536497
// clang-format on
constexpr int map_n_g = 185, map_n_e = 129;
#if defined(__CUDACC__)
static __constant__ double map_dev[2 * (map_n_g + map_n_e)] = {LAPLACE_IBD_MAP_DATA};
#endif
inline constexpr double map_host[2 * (map_n_g + map_n_e)] = {LAPLACE_IBD_MAP_DATA};
LAPLACE_HD_FORCEINLINE double map_tab(int i) {
  KOKKOS_IF_ON_DEVICE((return map_dev[i];))
  KOKKOS_IF_ON_HOST((return map_host[i];))
}

// Gauss-Laguerre rules, n = 12, 16, 20, 24, 32: pairs (z_i, w_i e^(z_i)), so
// that int_0^inf g(z) dz ~ sum_i w_i e^(z_i) g(z_i) for g ~ e^-z times a smooth
// function (Golub-Welsch and Newton at 80 digits, mpmath; written by a script
// of the session that wrote this file, 2026-11-09)
// clang-format off
#define LAPLACE_IBD_GL_DATA   0.11572211735802068, 0.2972096360444108, \
  0.6117574845151307, 0.6964629804305972, \
  1.5126102697764188, 1.1077813946157522, \
  2.8337513377435073, 1.5384642390428291, \
  4.5992276394183484, 1.9983276062742428, \
  6.844525453115177, 2.5007457691008668, \
  9.621316842456867, 3.0653215182823863, \
  13.006054993306348, 3.7232891107827717, \
  17.116855187462257, 4.529814029981735, \
  22.151090379397004, 5.597258461835321, \
  28.487967250984, 7.212995460925877, \
  37.09912104446692, 10.543837461910082, \
  0.08764941047892784, 0.22503631486424724, \
  0.46269632891508083, 0.5258360527623425, \
  1.141057774831227, 0.831961391687087, \
  2.1292836450983805, 1.1460992409637516, \
  3.4370866338932067, 1.4717513169668086, \
  5.078018614549768, 1.813134687381348, \
  7.070338535048234, 2.1755175196946075, \
  9.438314336391938, 2.565762750165029, \
  12.21422336886616, 2.993215086371375, \
  15.441527368781617, 3.4712344831020903, \
  19.180156856753136, 4.020044086444669, \
  23.515905693991908, 4.672516607732854, \
  28.57872974288214, 5.487420657986153, \
  34.58339870228662, 6.5853612332892135, \
  41.94045264768833, 8.276357984364234, \
  51.70116033954332, 11.824277551658435, \
  0.07053988969198875, 0.18108006241898925, \
  0.37212681800161146, 0.422556767878564, \
  0.9165821024832735, 0.6669095467018481, \
  1.707306531028344, 0.9153523727830737, \
  2.749199255309432, 1.169539707195546, \
  4.048925313850887, 1.431354985928206, \
  5.6151749708616165, 1.7029811379850228, \
  7.459017453671064, 1.9870158907927473, \
  9.594392869581096, 2.286635781253431, \
  12.038802546964316, 2.6058347275538334, \
  14.81429344263074, 2.949783734213951, \
  17.948895520519375, 3.3253957820093194, \
  21.47878824028501, 3.742255470589811, \
  25.451702793186904, 4.21423671025188, \
  29.93255463170061, 4.76251846149021, \
  35.013434240479, 5.421726044245574, \
  40.83305705672857, 6.254012356932421, \
  47.6199940473465, 7.387314389054435, \
  55.810795750063896, 9.15132873098748, \
  66.52441652561575, 12.893388645939996, \
  0.05901985218150798, 0.15149441285950946, \
  0.31123914619848375, 0.35325658252992387, \
  0.7660969055459367, 0.5567845632881526, \
  1.4255975908036131, 0.7626853176973091, \
  2.2925620586321904, 0.9718726322465476, \
  3.3707742642089977, 1.185357893037801, \
  4.665083703467171, 1.4042656272844185, \
  6.1815351187367655, 1.6298686157570415, \
  7.927539247172152, 1.8636350553320729, \
  9.912098015077706, 2.1072911510814802, \
  12.146102711729766, 2.362905891041935, \
  14.642732289596674, 2.633008753163857, \
  17.417992646508978, 2.9207575797277245, \
  20.491460082616424, 3.2301851334923537, \
  23.887329848169735, 3.5665733773687567, \
  27.635937174332717, 3.9370437554551603, \
  31.776041352374722, 4.351531188863512, \
  36.35840580165162, 4.8244818548980355, \
  41.45172048487077, 5.378022079789182, \
  47.153106445156325, 6.048417812619965, \
  53.60857454469507, 6.900898352180496, \
  61.05853144721876, 8.069965156146957, \
  69.96224003510503, 9.902793319484225, \
  81.49827923394889, 13.820532094792005, \
  0.04448936583326702, 0.11418710576810485, \
  0.23452610951961853, 0.2660652168976152, \
  0.5768846293018864, 0.418793137324853, \
  1.0724487538178176, 0.5725328464998047, \
  1.7224087764446454, 0.7276487883809714, \
  2.5283367064257947, 0.8845367193402497, \
  3.4922132730219944, 1.043618875892077, \
  4.616456769749767, 1.2053492741523526, \
  5.903958504174244, 1.3702213385217812, \
  7.358126733186241, 1.5387772564686448, \
  8.982940924212595, 1.7116193526864572, \
  10.783018632539973, 1.889424063449484, \
  12.763697986742725, 2.0729593402465336, \
  14.931139755522556, 2.2631066339969634, \
  17.292454336715316, 2.460889072488236, \
  19.855860940336054, 2.667508126397117, \
  22.630889013196775, 2.8843920929220417, \
  25.628636022459247, 3.113261327039586, \
  28.862101816323474, 3.3562176925958025, \
  32.346629153964734, 3.615869856484269, \
  36.10049480575197, 3.8955130449485496, \
  40.14571977153944, 4.199394104711586, \
  44.509207995754934, 4.533114978534361, \
  49.22439498730864, 4.9042702876112445, \
  54.33372133339691, 5.323500972023666, \
  59.89250916213402, 5.8063332142336215, \
  65.97537728793505, 6.3766146741596526, \
  72.68762809066271, 7.0735265807072425, \
  80.18744697791352, 7.9676935092959, \
  88.7353404178924, 9.20504033127819, \
  98.82954286828397, 11.163013090767873, \
  111.7513980979377, 15.390180415260643
// clang-format on
#if defined(__CUDACC__)
static __constant__ double gl_dev[208] = {LAPLACE_IBD_GL_DATA};
#endif
inline constexpr double gl_host[208] = {LAPLACE_IBD_GL_DATA};
LAPLACE_HD_FORCEINLINE double gl_tab(int i) {
  KOKKOS_IF_ON_DEVICE((return gl_dev[i];))
  KOKKOS_IF_ON_HOST((return gl_host[i];))
}
// offset (in pairs) of the rule with n nodes
LAPLACE_HD_FORCEINLINE int gl_offset(int n) { return n == 12 ? 0 : n == 16 ? 12 : n == 20 ? 28 : n == 24 ? 48 : 72; }

// A ray goes from the anchor t_c towards t = 0 in v = log t ("left"), or
// towards t = 1 in w = -log(1 - t) ("right"), which is the left ray of the
// exchanged problem (b, a, 1 - t). In its own coordinate d <= 0 (d = v - v_c,
// or d = -(w - w_c)), with A the exponent of its own variable (a left, b
// right), B the other one, rr = t_c / (1 - t_c) (left) or (1 - t_c) / t_c (right):
//     P = d,  Q = log1p(-rr expm1(d)) >= 0,
//     u-measure: t^a (1 - t)^b du = (ref) e^(A P + (B - 1) Q) dd / (1 - t_c)   (left; t_c right),
// so that X = P, Y = Q on a left ray and X = Q, Y = P on a right one. Near the
// anchor the exponent is written as
//     lam_v d - (B - 1) rr (e^d - 1 - d) + (B - 1) L(q),  q = -rr expm1(d),
// lam_v = A - (B - 1) rr given exactly by the caller, so that its first-order
// terms do not cancel. The nearest singularity of the integrand (t = 1 for a
// left ray, t = 0 for a right one) lies behind the ray.
struct ray {
  double A, B, rr, lamv;
  double fac;  // weight factor: t_c / (1 - t_c) on left rays, 1 on right rays
  bool left;
};

struct rnode {
  double E, P, Q;
};

LAPLACE_HD rnode ray_node(const ray& R, double d, bool own = false) {
  rnode v;
  v.P = d;
  const double em = node_expm1(d, own);
  const double q = -R.rr * em;
  const double Bm1 = R.B - 1.0;
  if (d >= -1.0 && q <= 1.0) {
    const double Lq = ibx::log1pmx(q);
    v.Q = q + Lq;
    v.E = R.lamv * d - Bm1 * R.rr * expm1mx(d) + Bm1 * Lq;
  } else {
    v.Q = node_log1p(q, own);
    v.E = R.A * d + Bm1 * v.Q;
  }
  return v;
}

template <int K>
LAPLACE_HD_FORCEINLINE void add_node(double* m, double W, double X, double Y) {
  double xp[K + 1], yp[K + 1];
  xp[0] = yp[0] = 1.0;
  for (int i = 1; i <= K; ++i) {
    xp[i] = xp[i - 1] * X;
    yp[i] = yp[i - 1] * Y;
  }
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) m[ix(n - j, j)] += W * xp[n - j] * yp[j];
}

// distance from d0 towards -inf at which phi drops by about one below
// phi(d0) = E0; the first guess from the local model g delta + kappa delta^2/2 = 1
// (g >= 0 the rate of decrease, kappa the curvature)
LAPLACE_HD double drop_scale(const ray& R, double d0, double E0, double g, double kappa, const quad_par& qp) {
  double s = 2.0 / (g + Kokkos::sqrt(g * g + 2.0 * kappa));
  if (!(s > 0.0) || !(s < 1e300)) s = 1.0;
  for (int it = 0; it < 16; ++it) {
    const double k = E0 - ray_node(R, d0 - s, qp.own_elem).E;
    if (qp.count) ++*qp.count;
    if (k >= qp.drop_lo && k <= qp.drop_hi) break;
    double f = k > 0.0 ? 1.0 / Kokkos::sqrt(k) : 8.0;
    f = f < 0.125 ? 0.125 : (f > 8.0 ? 8.0 : f);
    s *= f;
  }
  return s;
}

// One piece of the integral: a ray from d0 towards -inf (weights times sign,
// stopped below dstop), or tanh-sinh on [-L, 0] (finite).
struct piece {
  ray R;
  double d0, sign, dstop;
  double sig, alpha;  // the ray's scale and map power
  double L;
  bool finite;
  int gauss_n = 0;  // > 0: Gauss-Laguerre with this many nodes in z = (d0 - d) / sig
};

// a ray piece, with its scale from drop_scale. The map is
// d = d0 - sigma exp(alpha (t - e^-t)), the double-exponential map for
// integrands that decay exponentially (Takahasi and Mori); alpha = 1/2 suits a
// Gaussian-like start, where the slope times the scale is small. sigma is
// capped at qp.cap, the scale of the structure of t and 1 - t.
LAPLACE_HD piece ray_piece(const ray& R, double d0, double E0, double g, double kappa, double sign, double dstop,
                           const quad_par& qp) {
  // Gauss-Laguerre in z = (d0 - d) g where the ray is dominated by its slope g:
  // c = kappa / (2 g^2) small (measured 2026-11-09: 16-24 nodes against about
  // 45 of the double-exponential rule)
  bool gauss = true;
  KOKKOS_IF_ON_DEVICE((gauss = qp.gauss_device;))
  if (gauss && g > 0.0) {
    const double cc = kappa / (2.0 * g * g);
    const int n = cc <= qp.gl_c16 ? 16 : cc <= qp.gl_c20 ? 20 : cc <= qp.gl_c24 ? 24 : cc <= qp.gl_c32 ? 32 : 0;
    if (n > 0) {
      piece c{R, d0, sign, dstop, 1.0 / g, 0.0, 0.0, false};
      c.gauss_n = n;
      return c;
    }
  }
  const double sd = drop_scale(R, d0, E0, g, kappa, qp);
  piece c{R, d0, sign, dstop, sd, g * sd < qp.g_switch ? qp.alpha_g : qp.alpha_e, 0.0, false};
  if (sd > qp.cap) {
    c.sig = qp.cap;
    c.alpha = qp.alpha_e;
  }
  return c;
}

// sum over the pieces of w e^phi (X - cx)^i (Y - cy)^j, i + j <= K, with the
// step h of the rays and hts of the finite pieces multiplied by hmul (an
// integer for the rays). On a ray, t = tlo, tlo + h, ..., tlo = -log(42 / alpha)
// (the first node within e^-42 sigma of d0), from the table where it applies;
// the ray stops where phi < -cut (phi decreases along it).
// Where the weighted nodes go: the moments about (cx, cy) up to order K, or
// stored (orders 3 to 5: the moments are then taken about the means in a
// second pass over the stored nodes, without evaluating them again).
template <int K>
struct moment_sink {
  double* m;
  double cx, cy;
  LAPLACE_HD_FORCEINLINE void add(double W, double X, double Y) { add_node<K>(m, W, X - cx, Y - cy); }
};
constexpr int store_cap = 160;
struct store_sink {
  double W[store_cap], X[store_cap], Y[store_cap];
  int n = 0;
  bool overflow = false;
  LAPLACE_HD_FORCEINLINE void add(double w, double x, double y) {
    if (n < store_cap) {
      W[n] = w;
      X[n] = x;
      Y[n] = y;
      ++n;
    } else {
      overflow = true;
    }
  }
};

template <class Sink>
LAPLACE_HD void integrate(const piece* pc, int np, double hmul, const quad_par& qp, Sink& sink) {
  const bool own = qp.own_elem;
  for (int k = 0; k < np; ++k) {
    const piece& c = pc[k];
    const ray& R = c.R;
    if (c.gauss_n > 0) {
      const int off = gl_offset(c.gauss_n);
      for (int i = 0; i < c.gauss_n; ++i) {
        const double d = c.d0 - c.sig * gl_tab(2 * (off + i));
        if (d < c.dstop) continue;
        const rnode v = ray_node(R, d, own);
        if (qp.count) ++*qp.count;
        if (v.E < -qp.cut) continue;
        const double W = c.sign * R.fac * c.sig * gl_tab(2 * (off + i) + 1) * node_exp(v.E, own);
        if (R.left) sink.add(W, v.P, v.Q);
        else sink.add(W, v.Q, v.P);
      }
    } else if (c.finite) {
      const double h = qp.hts * hmul;
      const int n = static_cast<int>(qp.tts / h + 0.5);
      for (int i = -n; i <= n; ++i) {
        const double t = i * h;
        const double u = halfpi * Kokkos::sinh(t);
        const double e = math::exp(-2.0 * Kokkos::fabs(u));
        const double near = c.L * e / (1.0 + e);  // distance to the nearer end
        const double d = u < 0.0 ? -near : -(c.L - near);
        const double w = h * c.L * halfpi * Kokkos::cosh(t) * 2.0 * e / ((1.0 + e) * (1.0 + e));
        const rnode v = ray_node(R, d, own);
        if (qp.count) ++*qp.count;
        if (v.E < -qp.cut) continue;
        const double W = R.fac * w * node_exp(v.E, own);
        if (R.left) sink.add(W, v.P, v.Q);
        else sink.add(W, v.Q, v.P);
      }
    } else {
      const double h = qp.h * hmul;
      const double tlo = -math::log(42.0 / c.alpha);
      const int n = static_cast<int>((qp.thi - tlo) / h) + 1;
      // the table: h = 1/5 and the two default powers of the map
      const int stride = static_cast<int>(hmul + 0.5);
      const bool tab = qp.h == 0.2 && hmul == stride && (c.alpha == 0.5 || c.alpha == 0.75);
      const int base = c.alpha == 0.5 ? 0 : 2 * map_n_g, ntab = c.alpha == 0.5 ? map_n_g : map_n_e;
      const double hs = h * c.sig;
      // A later start: the table nodes with e below qp.e_start are not
      // evaluated; one node at the weighted mean of their positions, with
      // their total weight, stands for them (exact where the integrand is
      // linear over [0, e_start sigma]; the rest is of order e_start^2).
      int i0 = 0;
      if (tab && qp.e_start > 0.0) {
        double sw = 0.0, swe = 0.0;
        while (i0 * stride < ntab && map_tab(base + 2 * i0 * stride) < qp.e_start) {
          const double wk = map_tab(base + 2 * i0 * stride + 1);
          sw += wk;
          swe += wk * map_tab(base + 2 * i0 * stride);
          ++i0;
        }
        if (sw > 0.0) {
          const double d = c.d0 - c.sig * (swe / sw);
          const rnode v = ray_node(R, d, own);
          if (qp.count) ++*qp.count;
          const double W = c.sign * R.fac * hs * sw * node_exp(v.E, own);
          if (R.left) sink.add(W, v.P, v.Q);
          else sink.add(W, v.Q, v.P);
        }
      }
      for (int i = i0; i <= n; ++i) {
        double e, wf;
        if (tab && i * stride < ntab) {
          e = map_tab(base + 2 * i * stride);
          wf = map_tab(base + 2 * i * stride + 1);
        } else {
          const double t = tlo + i * h, ex = math::exp(-t);
          e = math::exp(c.alpha * (t - ex));
          wf = c.alpha * (1.0 + ex) * e;
        }
        const double d = c.d0 - c.sig * e;
        if (d < c.dstop) break;
        const rnode v = ray_node(R, d, own);
        if (qp.count) ++*qp.count;
        if (v.E < -qp.cut) break;
        const double W = c.sign * R.fac * hs * wf * node_exp(v.E, own);
        if (R.left) sink.add(W, v.P, v.Q);
        else sink.add(W, v.Q, v.P);
      }
    }
  }
}

// l_ij and lt_ij of the lower side of p (the frame) by the quadrature. The
// moments are taken about the means of X and Y, estimated by a first pass at
// hmean times the step, so that the cumulants are not formed from raw moments that
// cancel.
template <int K>
LAPLACE_IBD_NOINLINE LAPLACE_HD void quad_lower(const ibx::args& p, const quad_par& qp, double* l, double* lt,
                                                 bool want_lt, double& logS) {
  const double a = p.a, b = p.b, lam = p.lambda;
  const double inf = math::detail::pos_inf();
  piece pc[3];
  int np = 0;
  double c1off = 0.0, c2off = 0.0;
  if (lam >= 0.0) {
    // monotone: anchor at the end point x, one left ray
    const double r = p.x / p.y;
    const ray R{a, b, r, (lam + p.x) / p.y, 1.0, true};
    pc[np++] = ray_piece(R, 0.0, 0.0, R.lamv, Kokkos::fabs(b - 1.0) * r * (1.0 + r), 1.0, -inf, qp);
    // log(x s / a): from lambda near the mean, from log x in the tail
    c1off = lam <= 0.5 * a ? Kokkos::log1p(-lam / a) : p.lx + Kokkos::log1p(b / a);
    c2off = Kokkos::log1p(lam / b);
  } else {
    // the maximum t_c = a / s inside: a left ray from it, and towards the end
    // point x, at e_x = -log((1 - x) / (1 - t_c)) in w
    const double r = a / b, ri = b / a;
    const ray Lr{a, b, r, r, r, true};
    const ray Rr{b, a, ri, ri, 1.0, false};
    pc[np++] = ray_piece(Lr, 0.0, 0.0, r, Kokkos::fabs(b - 1.0) * r * (1.0 + r), 1.0, -inf, qp);
    // -log(y s / b): from lambda near the mean, from log y when y s / b is small
    const double ex = lam >= -0.5 * b ? -Kokkos::log1p(lam / b) : -(p.ly + Kokkos::log1p(a / b));
    const double EL = ray_node(Rr, -ex, qp.own_elem).E;
    const double kr = Kokkos::fabs(a - 1.0) * ri * (1.0 + ri);
    if (-EL <= qp.kx_fin) {
      pc[np++] = piece{Rr, 0.0, 1.0, -inf, 0.0, 0.0, ex, true};
    } else {
      pc[np++] = ray_piece(Rr, 0.0, 0.0, ri, kr, 1.0, -EL < qp.cut ? -inf : -ex, qp);
      if (-EL < qp.cut) {
        // minus the ray beyond the end point; its local rate of decrease
        const double em = Kokkos::expm1(-ex), q = -ri * em;
        const double gx = b - (a - 1.0) * ri * (1.0 + em) / (1.0 + q);
        pc[np++] = ray_piece(Rr, -ex, EL, gx > 0.0 ? gx : 0.0, kr, -1.0, -inf, qp);
      }
    }
  }
  double m[NC], mu[NC], kap[NC];
  double cx = 0.0, cy = 0.0;
  for (int i = 0; i < NC; ++i) m[i] = 0.0;
  bool done = false;
  // stored nodes for orders 3 and 5; order 4 keeps the first pass (with the
  // stored nodes the worst row of log(1 - I) at order 4, a = 1.4999,
  // b = 19.04, x = 0.0571, moved from 65 to 78 ulpL by rounding, above the bound
  // of the test; measured 2026-11-09)
  if ((K == 3 || K == 5) && qp.store) {
    store_sink st;
    integrate(pc, np, 1.0, qp, st);
    if (!st.overflow) {
      double s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int i = 0; i < st.n; ++i) {
        s0 += st.W[i];
        s1 += st.W[i] * st.X[i];
        s2 += st.W[i] * st.Y[i];
      }
      cx = s1 / s0;
      cy = s2 / s0;
      for (int i = 0; i < st.n; ++i) add_node<K>(m, st.W[i], st.X[i] - cx, st.Y[i] - cy);
      done = true;
    }
  }
  if (!done) {
    if (qp.hmean > 0.0 && K >= 3) {
      double m1[NC];
      for (int i = 0; i < NC; ++i) m1[i] = 0.0;
      moment_sink<1> s1{m1, 0.0, 0.0};
      integrate(pc, np, qp.hmean, qp, s1);
      cx = m1[ix(1, 0)] / m1[0];
      cy = m1[ix(0, 1)] / m1[0];
    }
    moment_sink<K> sk{m, cx, cy};
    integrate(pc, np, K >= 4 ? 1.0 : qp.hlow, qp, sk);
  }
  const double i0 = 1.0 / m[0];
  for (int i = 0; i < NC; ++i) mu[i] = m[i] * i0;
  cumulants<K>(mu, kap);
  kap[ix(1, 0)] += cx;
  kap[ix(0, 1)] += cy;
  assemble<K>(a, b, c1off, c2off, kap, l, lt, want_lt);
  // log S = log J - log B(a, b), J = t_c^a (1 - t_c)^b m_0 / (1 - t_c) on the
  // monotone side (anchor x), / t_c on the other (anchor a / s); the factor
  // t_c^a (1 - t_c)^b / B(a, b) from detail_ibx::log_power, without cancellation
  if (lam >= 0.0) {
    logS = ibx::log_power(p) - p.ly + math::log(m[0]);
  } else {
    const double s = a + b, la = -Kokkos::log1p(b / a), lb = -Kokkos::log1p(a / b);
    const ibx::args pm{a, b, a / s, b / s, la, lb, 0.0, true, true};
    logS = ibx::log_power(pm) - la + math::log(m[0]);
  }
}

// ---------------------------------------------------------------- the series

// whether the series converges fast on the lower side of p; with wide, also up
// to x = x_wide when b <= 1 (0.8 in quad_par; the default 0.95 is unused;
// terms below x^n: at most about 800)
LAPLACE_HD_FORCEINLINE bool series_ok(const ibx::args& p, bool wide = false, double x_wide = 0.95) {
  return (p.x <= 0.5 && (p.b <= 1.0 || p.b * p.x <= 0.7)) || (wide && p.x <= x_wide && p.b <= 1.0);
}

template <int K>
LAPLACE_IBD_NOINLINE LAPLACE_HD void series_lower(const ibx::args& p, double* l, double* lt, bool want_lt, double& logS,
                                                   bool pure_check = false) {
  const double a = p.a, b = p.b, x = p.x, s = a + b;
  double S[NC], g[K + 1], f[K + 1];
  for (int i = 0; i < NC; ++i) S[i] = 0.0;
  S[0] = 1.0;
  for (int j = 0; j <= K; ++j) g[j] = j == 0 ? 1.0 : 0.0;
  for (int n = 1; n < 100000; ++n) {
    for (int j = K; j >= 0; --j) g[j] = (x / n) * ((n - b) * g[j] - (j > 0 ? j * g[j - 1] : 0.0));
    const double ia = 1.0 / (a + n);
    f[0] = a * ia;
    double pw = n * ia;  // (-1)^(i+1) i! n ia^(i+1)
    for (int i = 1; i <= K; ++i) {
      pw *= ia * (i > 1 ? -static_cast<double>(i) : 1.0);
      f[i] = pw;
    }
    bool done = true;
    for (int nn = 0; nn <= K; ++nn)
      for (int j = 0; j <= nn; ++j) {
        const double t = g[j] * f[nn - j];
        double& Sv = S[ix(nn - j, j)];
        Sv += t;
        if ((!pure_check || j == 0 || j == nn) && Kokkos::fabs(t) > 1e-17 * Kokkos::fabs(Sv)) done = false;
      }
    if (done) break;
  }
  double mu[NC], kap[NC];
  const double i0 = 1.0 / S[0];
  for (int i = 0; i < NC; ++i) mu[i] = S[i] * i0;
  cumulants<K>(mu, kap);
  // log I = a log x - log(a B(a, b)) + log Sigma, log(a B(a, b)) = log Gamma(a + 1)
  // + log Gamma(b) - log Gamma(s); for lt, log(a B(a, b) / b) with log Gamma(b + 1)
  // shared polygamma differences: psi^(n)(a + 1) - psi^(n)(s), psi^(n)(b) - psi^(n)(s),
  // psi^(n)(b + 1) - psi^(n)(s) (for lt), psi^(n)(s)
  double Pa[5], Pb[5], Pt[5], Ps[5], Db;
  dpsi_any_all(a + 1.0, s, b - 1.0, K - 1, Pa);
  dpsi_all(b, a, K - 1, Pb, Db);
  if (want_lt) dpsi_any_all(b + 1.0, s, a - 1.0, K - 1, Pt);
  if (K >= 2) psi_all(s, K - 1, Ps);
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double lb, lbt;  // d^(i+j) log(a B(a, b)) / da^i db^j, and of log(a B(a, b) / b)
      if (j == 0) lb = lbt = Pa[i - 1];
      else if (i == 0) {
        lb = Pb[j - 1];
        lbt = want_lt ? Pt[j - 1] : 0.0;
      } else lb = lbt = -Ps[n - 1];
      l[ix(i, j)] = kap[ix(i, j)] - lb + (i == 1 && j == 0 ? p.lx : 0.0);
      lt[ix(i, j)] = kap[ix(i, j)] - lbt + (i == 1 && j == 0 ? p.lx : 0.0);
    }
  l[0] = lt[0] = 0.0;
  // log I = log(x^a y^b / B(a, b)) - log a - b log y + log Sigma
  logS = ibx::log_power(p) - math::log(a) - b * p.ly + math::log(S[0]);
}

// ---------------------------------------------------------------- the driver

// one side in its frame; returns log S
template <int K>
LAPLACE_HD double side(const ibx::args& f, bool series, bool want_lt, const quad_par& qp, double* l, double* lt) {
  double logS;
  if (qp.sides) ++*qp.sides;
  if (series) series_lower<K>(f, l, lt, want_lt, logS, qp.pure_check);
  else quad_lower<K>(f, qp, l, lt, want_lt, logS);
  return logS;
}

// method: 0 quad everywhere, 1 series where it converges and quad elsewhere
template <int K>
LAPLACE_HD void derivs(double a, double b, double x, bool upper, int method, const quad_par& qp, double* out) {
  const double nan = math::detail::quiet_nan();
  for (int i = 0; i < NC; ++i) out[i] = 0.0;
  const bool shapes_ok = a > 0.0 && b > 0.0 && a < math::detail::pos_inf() && b < math::detail::pos_inf();
  // log I = 0 at x = 1 and log(1 - I) = 0 at x = 0 for all shapes: derivatives 0
  if (shapes_ok && ((x == 1.0 && !upper) || (x == 0.0 && upper))) return;
  if (!shapes_ok || !(x > 0.0) || !(x < 1.0)) {
    for (int i = 0; i < NC; ++i) out[i] = nan;
    return;
  }
  const double y = 1.0 - x;
  const double lx = x < 0.5 ? math::log(x) : Kokkos::log1p(-y);
  const double ly = x < 0.5 ? Kokkos::log1p(-x) : math::log(y);
  const ibx::args p{a, b, x, y, lx, ly, ibx::lambda_of(a, b, x), true, x >= 0.5};
  const ibx::args ps = ibx::swapped(p);
  const ibx::args& fr = upper ? ps : p;  // the requested side's frame
  const ibx::args& fo = upper ? p : ps;  // the other side's
  // The side computed, chosen in this order; each choice that depends on the
  // side's probability S is checked on the S that its own computation gives
  // (log S from log_power and m_0, or the series' sum), and abandoned for the
  // next one when the check fails (rare):
  //  1. the requested side, when its logarithm has a pole in a small shape
  //     (log I in b, log(1 - I) in a: I is proportional to b as b -> 0), its
  //     own shape is at least own_min or the series applies to it, and
  //     S <= p_direct: composed from the other side, the pole's derivatives
  //     (n-1)!/shape^n would cancel;
  //  2. the series on the requested side, or on the other side if there
  //     S <= p_mono (composing log(1 - S) from S near 1 is ill-conditioned);
  //  3. the quadrature on the monotone side if there S <= p_mono, otherwise
  //     on the other side.
  const double lp_direct = math::log(qp.p_direct), lp_mono = math::log(qp.p_mono);
  double l[NC], lt[NC], o[NC], logS = 0.0;
  bool lower = !upper, done = false;
  if ((upper ? a : b) < qp.pole_min) {
    const bool ser = method == 1 && series_ok(fr, true, qp.x_wide);
    if ((upper ? b : a) >= qp.own_min || ser) {
      logS = side<K>(fr, ser, false, qp, l, lt);
      done = logS <= lp_direct;
    }
  }
  if (!done && method == 1 && Kokkos::fmin(a, b) < qp.series_below) {
    if (series_ok(fr)) {
      logS = side<K>(fr, true, false, qp, l, lt);
      done = true;
    } else if (series_ok(fo)) {
      logS = side<K>(fo, true, true, qp, l, lt);
      if (logS <= lp_mono) {
        lower = upper;
        done = true;
      }
    }
  }
  if (!done) {
    const bool mono_lower = p.lambda >= 0.0;
    logS = side<K>(mono_lower ? p : ps, false, mono_lower == upper, qp, l, lt);
    lower = mono_lower;
    if (!(logS <= lp_mono)) {
      logS = side<K>(mono_lower ? ps : p, false, mono_lower != upper, qp, l, lt);
      lower = !mono_lower;
    }
  }
  if (lower != upper) {
    for (int c = 0; c < NC; ++c) o[c] = l[c];
  } else {
    // r = S / (1 - S)
    const double S = math::exp(logS), l1mS = Kokkos::log1p(-S);
    complement<K>(lt, (lower ? p : ps).b, S / -Kokkos::expm1(logS), logS - l1mS, o);
  }
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) out[ix(n - j, j)] = lower ? o[ix(n - j, j)] : o[ix(j, n - j)];
}

template <int K>
LAPLACE_HD_FORCEINLINE mixed_partials<K> pick(const double* d) {
  mixed_partials<K> r;
  for (int j = 0; j <= K; ++j) r.c[j] = d[ix(K - j, j)];
  return r;
}

// the version chosen for the public functions
constexpr int chosen_method = 1;

template <int K>
LAPLACE_HD mixed_partials<K> order(double a, double b, double x, bool upper) {
  double d[NC];
  derivs<K>(a, b, x, upper, chosen_method, quad_par{}, d);
  return pick<K>(d);
}

}  // namespace detail_ibd

// d^K log I_x(a, b) / da^(K-j) db^j in c[j]
/**
 * \ingroup incomplete_beta
 * \brief First shape derivatives of \f$\log I_x(a, b)\f$.
 *
 * Computes all mixed partial derivatives of total order 1 with respect to the shape parameters:
 * \f[
 * c[0] = \frac{\partial}{\partial a} \log I_x(a, b) = \kappa_{1,0} - (\psi(a) - \psi(a+b)), \quad
 * c[1] = \frac{\partial}{\partial b} \log I_x(a, b) = \kappa_{0,1} - (\psi(b) - \psi(a+b))
 * \f]
 * where \f$\kappa_{i,j}\f$ are the joint cumulants of \f$(\log T, \log(1 - T))\f$ under \f$T \sim \mathrm{Beta}(a, b)\f$ truncated to \f$T < x\f$, and \f$\psi\f$ is the digamma function.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in (0, 1]\f$.
 * \return A mixed_partials of order 1 containing \f$c[0] = \frac{\partial}{\partial a} \log I_x(a, b)\f$ and \f$c[1] = \frac{\partial}{\partial b} \log I_x(a, b)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in (0, 1]\f$: standard domain.
 * - At \f$x = 1\f$: returns all zeros in all components.
 * - At \f$x = 0\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1580 ulpN (gcc, CUDA within 1.9 times); test bound 1600.0 ulpN / 55.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibeta(), log_ibetac_d1(), log_ibeta_d2()
 */
LAPLACE_HD mixed_partials<1> log_ibeta_d1(double a, double b, double x) { return detail_ibd::order<1>(a, b, x, false); }

/**
 * \ingroup incomplete_beta
 * \brief Second shape derivatives of \f$\log I_x(a, b)\f$.
 *
 * Computes all mixed partial derivatives of total order 2 with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^2}{\partial a^{2-j} \partial b^j} \log I_x(a, b) = \kappa_{2-j, j} - \frac{\partial^2}{\partial a^{2-j} \partial b^j} \log B(a, b), \quad j = 0, 1, 2
 * \f]
 * where \f$\kappa_{i,j}\f$ are the joint cumulants of \f$(\log T, \log(1 - T))\f$ under \f$T \sim \mathrm{Beta}(a, b)\f$ truncated to \f$T < x\f$, and \f$\frac{\partial^2}{\partial a^2} \log B(a, b) = \psi^{(1)}(a) - \psi^{(1)}(a+b)\f$, \f$\frac{\partial^2}{\partial a \partial b} \log B(a, b) = -\psi^{(1)}(a+b)\f$, \f$\frac{\partial^2}{\partial b^2} \log B(a, b) = \psi^{(1)}(b) - \psi^{(1)}(a+b)\f$.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in (0, 1]\f$.
 * \return A mixed_partials of order 2 containing \f$c[j] = \frac{\partial^2}{\partial a^{2-j} \partial b^j} \log I_x(a, b)\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in (0, 1]\f$: standard domain.
 * - At \f$x = 1\f$: returns all zeros in all components.
 * - At \f$x = 0\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1340 ulpN (gcc, CUDA within 1.9 times); test bound 1800.0 ulpN / 245.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibeta(), log_ibetac_d2(), log_ibeta_d1(), log_ibeta_d3()
 */
LAPLACE_HD mixed_partials<2> log_ibeta_d2(double a, double b, double x) { return detail_ibd::order<2>(a, b, x, false); }

/**
 * \ingroup incomplete_beta
 * \brief Third shape derivatives of \f$\log I_x(a, b)\f$.
 *
 * Computes all mixed partial derivatives of total order 3 with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^3}{\partial a^{3-j} \partial b^j} \log I_x(a, b) = \kappa_{3-j, j} - \frac{\partial^3}{\partial a^{3-j} \partial b^j} \log B(a, b), \quad j = 0, \dots, 3
 * \f]
 * where \f$\kappa_{i,j}\f$ are the joint cumulants of \f$(\log T, \log(1 - T))\f$ under \f$T \sim \mathrm{Beta}(a, b)\f$ truncated to \f$T < x\f$, and \f$\frac{\partial^3}{\partial a^{3-j} \partial b^j} \log B(a, b)\f$ are polygamma differences of order 2.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in (0, 1]\f$.
 * \return A mixed_partials of order 3 containing \f$c[j] = \frac{\partial^3}{\partial a^{3-j} \partial b^j} \log I_x(a, b)\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in (0, 1]\f$: standard domain.
 * - At \f$x = 1\f$: returns all zeros in all components.
 * - At \f$x = 0\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1620 ulpN (gcc, CUDA within 1.9 times); test bound 3100.0 ulpN / 315.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibeta(), log_ibetac_d3(), log_ibeta_d2(), log_ibeta_d4()
 */
LAPLACE_HD mixed_partials<3> log_ibeta_d3(double a, double b, double x) { return detail_ibd::order<3>(a, b, x, false); }

/**
 * \ingroup incomplete_beta
 * \brief Fourth shape derivatives of \f$\log I_x(a, b)\f$.
 *
 * Computes all mixed partial derivatives of total order 4 with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^4}{\partial a^{4-j} \partial b^j} \log I_x(a, b) = \kappa_{4-j, j} - \frac{\partial^4}{\partial a^{4-j} \partial b^j} \log B(a, b), \quad j = 0, \dots, 4
 * \f]
 * where \f$\kappa_{i,j}\f$ are the joint cumulants of \f$(\log T, \log(1 - T))\f$ under \f$T \sim \mathrm{Beta}(a, b)\f$ truncated to \f$T < x\f$, and \f$\frac{\partial^4}{\partial a^{4-j} \partial b^j} \log B(a, b)\f$ are polygamma differences of order 3.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in (0, 1]\f$.
 * \return A mixed_partials of order 4 containing \f$c[j] = \frac{\partial^4}{\partial a^{4-j} \partial b^j} \log I_x(a, b)\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in (0, 1]\f$: standard domain.
 * - At \f$x = 1\f$: returns all zeros in all components.
 * - At \f$x = 0\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1370 ulpN (gcc, CUDA within 1.9 times); test bound 1800.0 ulpN / 115.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibeta(), log_ibetac_d4(), log_ibeta_d3(), log_ibeta_d5()
 */
LAPLACE_HD mixed_partials<4> log_ibeta_d4(double a, double b, double x) { return detail_ibd::order<4>(a, b, x, false); }

/**
 * \ingroup incomplete_beta
 * \brief Fifth shape derivatives of \f$\log I_x(a, b)\f$.
 *
 * Computes all mixed partial derivatives of total order 5 with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^5}{\partial a^{5-j} \partial b^j} \log I_x(a, b) = \kappa_{5-j, j} - \frac{\partial^5}{\partial a^{5-j} \partial b^j} \log B(a, b), \quad j = 0, \dots, 5
 * \f]
 * where \f$\kappa_{i,j}\f$ are the joint cumulants of \f$(\log T, \log(1 - T))\f$ under \f$T \sim \mathrm{Beta}(a, b)\f$ truncated to \f$T < x\f$, and \f$\frac{\partial^5}{\partial a^{5-j} \partial b^j} \log B(a, b)\f$ are polygamma differences of order 4.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in (0, 1]\f$.
 * \return A mixed_partials of order 5 containing \f$c[j] = \frac{\partial^5}{\partial a^{5-j} \partial b^j} \log I_x(a, b)\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in (0, 1]\f$: standard domain.
 * - At \f$x = 1\f$: returns all zeros in all components.
 * - At \f$x = 0\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1950 ulpN (gcc, CUDA within 1.9 times); test bound 3300.0 ulpN / 420.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibeta(), log_ibetac_d5(), log_ibeta_d4()
 */
LAPLACE_HD mixed_partials<5> log_ibeta_d5(double a, double b, double x) { return detail_ibd::order<5>(a, b, x, false); }

// d^K log(1 - I_x(a, b)) / da^(K-j) db^j in c[j]
/**
 * \ingroup incomplete_beta
 * \brief First shape derivatives of \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes all mixed partial derivatives of total order 1 of the complementary logarithm with respect to the shape parameters:
 * \f[
 * c[0] = \frac{\partial}{\partial a} \log(1 - I_x(a, b)), \quad c[1] = \frac{\partial}{\partial b} \log(1 - I_x(a, b))
 * \f]
 * evaluated via symmetry \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ or composed from the lower side \f$S = I_x(a, b)\f$ using Faà di Bruno recursion.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1)\f$.
 * \return A mixed_partials of order 1 containing \f$c[0] = \frac{\partial}{\partial a} \log(1 - I_x(a, b))\f$ and \f$c[1] = \frac{\partial}{\partial b} \log(1 - I_x(a, b))\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in [0, 1)\f$: standard domain.
 * - At \f$x = 0\f$: returns all zeros in all components.
 * - At \f$x = 1\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1580 ulpN (gcc, CUDA within 1.9 times); test bound 2400.0 ulpN / 180.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibetac(), log_ibeta_d1(), log_ibetac_d2()
 */
LAPLACE_HD mixed_partials<1> log_ibetac_d1(double a, double b, double x) { return detail_ibd::order<1>(a, b, x, true); }

/**
 * \ingroup incomplete_beta
 * \brief Second shape derivatives of \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes all mixed partial derivatives of total order 2 of the complementary logarithm with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^2}{\partial a^{2-j} \partial b^j} \log(1 - I_x(a, b)), \quad j = 0, 1, 2
 * \f]
 * evaluated via symmetry \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ or composed from the lower side \f$S = I_x(a, b)\f$ using Faà di Bruno recursion.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1)\f$.
 * \return A mixed_partials of order 2 containing \f$c[j] = \frac{\partial^2}{\partial a^{2-j} \partial b^j} \log(1 - I_x(a, b))\f$ for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in [0, 1)\f$: standard domain.
 * - At \f$x = 0\f$: returns all zeros in all components.
 * - At \f$x = 1\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1340 ulpN (gcc, CUDA within 1.9 times); test bound 2050.0 ulpN / 155.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibetac(), log_ibeta_d2(), log_ibetac_d1(), log_ibetac_d3()
 */
LAPLACE_HD mixed_partials<2> log_ibetac_d2(double a, double b, double x) { return detail_ibd::order<2>(a, b, x, true); }

/**
 * \ingroup incomplete_beta
 * \brief Third shape derivatives of \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes all mixed partial derivatives of total order 3 of the complementary logarithm with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^3}{\partial a^{3-j} \partial b^j} \log(1 - I_x(a, b)), \quad j = 0, \dots, 3
 * \f]
 * evaluated via symmetry \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ or composed from the lower side \f$S = I_x(a, b)\f$ using Faà di Bruno recursion.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1)\f$.
 * \return A mixed_partials of order 3 containing \f$c[j] = \frac{\partial^3}{\partial a^{3-j} \partial b^j} \log(1 - I_x(a, b))\f$ for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in [0, 1)\f$: standard domain.
 * - At \f$x = 0\f$: returns all zeros in all components.
 * - At \f$x = 1\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1620 ulpN (gcc, CUDA within 1.9 times); test bound 2450.0 ulpN / 80.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibetac(), log_ibeta_d3(), log_ibetac_d2(), log_ibetac_d4()
 */
LAPLACE_HD mixed_partials<3> log_ibetac_d3(double a, double b, double x) { return detail_ibd::order<3>(a, b, x, true); }

/**
 * \ingroup incomplete_beta
 * \brief Fourth shape derivatives of \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes all mixed partial derivatives of total order 4 of the complementary logarithm with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^4}{\partial a^{4-j} \partial b^j} \log(1 - I_x(a, b)), \quad j = 0, \dots, 4
 * \f]
 * evaluated via symmetry \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ or composed from the lower side \f$S = I_x(a, b)\f$ using Faà di Bruno recursion.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1)\f$.
 * \return A mixed_partials of order 4 containing \f$c[j] = \frac{\partial^4}{\partial a^{4-j} \partial b^j} \log(1 - I_x(a, b))\f$ for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in [0, 1)\f$: standard domain.
 * - At \f$x = 0\f$: returns all zeros in all components.
 * - At \f$x = 1\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1370 ulpN (gcc, CUDA within 1.9 times); test bound 2950.0 ulpN / 75.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibetac(), log_ibeta_d4(), log_ibetac_d3(), log_ibetac_d5()
 */
LAPLACE_HD mixed_partials<4> log_ibetac_d4(double a, double b, double x) { return detail_ibd::order<4>(a, b, x, true); }

/**
 * \ingroup incomplete_beta
 * \brief Fifth shape derivatives of \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes all mixed partial derivatives of total order 5 of the complementary logarithm with respect to the shape parameters:
 * \f[
 * c[j] = \frac{\partial^5}{\partial a^{5-j} \partial b^j} \log(1 - I_x(a, b)), \quad j = 0, \dots, 5
 * \f]
 * evaluated via symmetry \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ or composed from the lower side \f$S = I_x(a, b)\f$ using Faà di Bruno recursion.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1)\f$.
 * \return A mixed_partials of order 5 containing \f$c[j] = \frac{\partial^5}{\partial a^{5-j} \partial b^j} \log(1 - I_x(a, b))\f$ for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, a < +\infty, b > 0, b < +\infty, x \in [0, 1)\f$: standard domain.
 * - At \f$x = 0\f$: returns all zeros in all components.
 * - At \f$x = 1\f$, or for \f$x < 0\f$, \f$x > 1\f$, \f$a \le 0\f$, \f$a = +\infty\f$, \f$b \le 0\f$, \f$b = +\infty\f$, or if any argument is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1950 ulpN (gcc, CUDA within 1.9 times); test bound 2950.0 ulpN / 690.0 ulpL.
 *
 * \see incomplete_beta, mixed_partials, log_ibetac(), log_ibeta_d5(), log_ibetac_d4()
 */
LAPLACE_HD mixed_partials<5> log_ibetac_d5(double a, double b, double x) { return detail_ibd::order<5>(a, b, x, true); }

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls (docs/numerics.md, 2026-11-09): the series
// route with the quadrature as fallback; Gauss-Laguerre rays on the host only.
using own::mixed_partials;
LAPLACE_HD_FORCEINLINE mixed_partials<1> log_ibeta_d1(double a, double b, double x) { return own::log_ibeta_d1(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<2> log_ibeta_d2(double a, double b, double x) { return own::log_ibeta_d2(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<3> log_ibeta_d3(double a, double b, double x) { return own::log_ibeta_d3(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<4> log_ibeta_d4(double a, double b, double x) { return own::log_ibeta_d4(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<5> log_ibeta_d5(double a, double b, double x) { return own::log_ibeta_d5(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<1> log_ibetac_d1(double a, double b, double x) { return own::log_ibetac_d1(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<2> log_ibetac_d2(double a, double b, double x) { return own::log_ibetac_d2(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<3> log_ibetac_d3(double a, double b, double x) { return own::log_ibetac_d3(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<4> log_ibetac_d4(double a, double b, double x) { return own::log_ibetac_d4(a, b, x); }
LAPLACE_HD_FORCEINLINE mixed_partials<5> log_ibetac_d5(double a, double b, double x) { return own::log_ibetac_d5(a, b, x); }

}  // namespace laplace::math
