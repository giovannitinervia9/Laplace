// enet_distrib (laplace::distrib): the elastic-net distribution, the density whose negative
// logarithm is the elastic-net penalty. A Laplace and a Gaussian density with the same
// location, multiplied and normalized. Parameters: the location mu (real), the penalty weight
// lambda > 0 (a rate) and the mixing weight alpha in (0, 1). With the two rates
//
//   a = lambda alpha (the Laplace rate),   c = lambda (1 - alpha) (the Gaussian precision),
//
// z = y - mu and x = a / sqrt(c),
//
//   l(mu, lambda, alpha; y) = -a |z| - c z^2 / 2 - A,
//   A = log Z = log 2 + log M(x) - log(c)/2,    M(x) = Phi(-x) / phi(x) (the Mills ratio),
//
// so f(y) = exp(-a |z| - c z^2/2) / Z with Z = 2 M(x) / sqrt(c). alpha -> 1 is the Laplace
// density with rate lambda (c -> 0, x -> infinity), alpha -> 0 the normal density with
// precision lambda (a -> 0, x -> 0). The mean is mu and the variance is -2 dA/dc.
//
// Kink. l is C^0 in mu at mu = y (and in y at y = mu): the first derivative carries
// sign(z), taken as 0 at z = 0, and the derivatives of order above two in mu vanish away
// from the kink. log F and log(1 - F) are C^1 in mu there (their second derivative in mu
// jumps by 2 a across z = 0). l is C^inf in lambda and alpha, and so are log F and log(1 - F).
//
// Derivatives of the log-density. l is bilinear in the rates, so the component with i
// derivatives in mu, j in lambda and k in alpha (i + j + k = K, packed with the sorted index
// (0^i, 1^j, 2^k), position (K - i)(K - i + 1)/2 + k) is
//   a_jk u_i + c_jk v_i - [i = 0] d^K A / dlambda^j dalpha^k,
// with u_0 = -|z|, u_1 = sign(z), v_0 = -z^2/2, v_1 = z, v_2 = -1 (all other u_i, v_i zero)
// and a_jk, c_jk the derivatives of the two rates (a_00 = lambda alpha, a_10 = alpha,
// a_01 = lambda, a_11 = 1 and c_00 = lambda (1 - alpha), c_10 = 1 - alpha, c_01 = -lambda,
// c_11 = -1; the others zero). Since l depends on y and mu through z only, d/dy = -d/dmu
// and logpdf_dy_dk<J, K> carries the sign (-1)^J, as in laplace1_distrib.
//
// The normalizing constant A(a, c). Its derivatives in (a, c) to order 6 are the joint
// cumulants of (s, s^2/2) with sign (-1)^order, s = |z| under the density; they are kept
// where no difference of large numbers forms, in two representations:
//   x < x_switch:  A = log 2 + L(x) - log(c)/2,   x = a c^(-1/2), L = log M
//                  (derivatives of L from math::log_mills_d1 .. _d6, carried to (a, c) by the
//                  chain rule in two variables, math::chain_rule_pair_dk);
//   x >= x_switch: A = log 2 - log a + F(t),      t = c / a^2 = 1/x^2, F(t) = log(x M(x))
//                  (the Laplace limit t -> 0 is smooth in t). With N(t) the integral over
//                  u > 0 of exp(-u - t u^2/2), F = log N, and F^(m)(t) is the m-th cumulant of
//                  W = -U^2/2 under the density proportional to exp(-u - t u^2/2). Its raw
//                  moments are (-1/2)^j x^(2j) m_(2j), with m_k the moments of s = u/x, from the
//                  ratios r_k = m_k/m_(k-1) = k/(x + r_(k+1)) run downward (the stable
//                  direction) in the scaled form of detail_normal::mills_cumulant.
// The derivatives in (lambda, alpha) are then the chain rule through the bilinear map
// (lambda, alpha) -> (a, c). The moment recursion and the cumulant-from-moment recursion are
// the ones of statmodels7 (pt_enet_logz.h), extended from order 4 to order 6.
//
// Expected information. The score of mu is a sign(z) + c z and, by symmetry of the density
// about mu, E[score_mu score_lambda] = E[score_mu score_alpha] = 0. By E[s'] = E[s^2] for the
// score s of a location family, I_mumu = c + 2 a f(0) = c + 2 a exp(-A) (the term 2 a f(0) is the
// point mass that the observed second derivative -c misses at the kink; the expected
// information is therefore NOT -E[l_mumu], and the test compares it with the score form of the
// quadrature). The block in (lambda, alpha) is the covariance of (|z|, z^2/2) carried through
// the bilinear map: I_lambda,lambda = A_lambda,lambda, I_lambda,alpha = A_lambda,alpha -
// A_alpha/lambda, I_alpha,alpha = A_alpha,alpha (subscripts are derivatives of A in
// (lambda, alpha); the extra term of the mixed entry comes from the bilinear part
// d^2 a / dlambda dalpha = 1 = -d^2 c / dlambda dalpha). I does not depend on mu, so every
// derivative of I with an index mu is zero; the others are components of the derivatives of A
// of order K + 2.
//
// Cdf. With D = |z| and s = sqrt(c), the probability beyond |z| on either side is
//   P(Z > D) = exp(T),   T = -a D - c D^2/2 - log 2 + A(a + c D, c) - A(a, c)
//                          = -a D - c D^2/2 - log 2 + L(x + s D) - L(x)
// (substitute z = D + v in the integral that defines the tail). Written with Phi it is
// log Phi(-(s D + x)) - log Phi(-x) - log 2, whose two terms are of size x^2/2 and cancel
// catastrophically when alpha is near 1, which is why the Mills form is used. log F is T for
// z < 0 and log(1 - e^T) = log1p(-e^T) for z >= 0 (T <= -log 2 always, so one branch of
// log(1 - e^u) is enough), and log(1 - F) the mirror. The derivatives of T in
// (mu, lambda, alpha) come from the derivatives of A at the two points (a, c) and (a + c D, c),
// d/dmu = (-sign(z)) d/dD; those of log(1 - e^T) from T by the exponential and the logarithm of
// a function of three variables (recursions over multi-indices, exp3 and log3 below). At
// z = 0 log_cdf takes the branch z >= 0 and log_ccdf the branch z <= 0.
//
// Quantile. The tail beyond D has log probability log p_tail, so D solves
// a D + c D^2/2 - (L(x + s D) - L(x)) = -log(2 p_tail) (p_tail = min(p, 1 - p)); the left side
// is convex and increasing in D, and Newton's method from the root of the quadratic part
// (which lies to the right of the root) converges monotonically.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_enet {

using math::detail_chain_pair::pair_count;
using math::detail_chain_pair::pair_index;

inline constexpr double ln2 = 0.69314718055994530942;

// from this x on (x = a / sqrt(c)) the representation in t = 1/x^2 replaces the one in x (measured against
// mpmath at x_switch = 0.5, 1, 1.5, 2 and 3: the t form loses digits below x = 1, the x form above x = 2,
// and 1.5 gave the smallest worst error at orders 5 and 6)
inline constexpr double x_switch = 1.5;

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// i!/(i - r)!
LAPLACE_HD_FORCEINLINE constexpr double fall(int i, int r) {
  double v = 1.0;
  for (int t = 0; t < r; ++t) v *= (i - t);
  return v;
}

// position of the component with i derivatives in mu and k in alpha in sym_tensor<3, K>
LAPLACE_HD_FORCEINLINE constexpr int pos3(int K, int i, int k) { return (K - i) * (K - i + 1) / 2 + k; }

// position of (n total, b in the second variable) in a pair-indexed table
LAPLACE_HD_FORCEINLINE constexpr int pn(int n, int b) { return n * (n + 1) / 2 + b; }

// d^j/dlambda^j d^k/dalpha^k of the two rates a = lambda alpha and c = lambda (1 - alpha)
LAPLACE_HD_FORCEINLINE double rate_a(int j, int k, double lam, double alp) {
  if (j == 0 && k == 0) return lam * alp;
  if (j == 1 && k == 0) return alp;
  if (j == 0 && k == 1) return lam;
  if (j == 1 && k == 1) return 1.0;
  return 0.0;
}
LAPLACE_HD_FORCEINLINE double rate_c(int j, int k, double lam, double alp) {
  if (j == 0 && k == 0) return lam * (1.0 - alp);
  if (j == 1 && k == 0) return 1.0 - alp;
  if (j == 0 && k == 1) return -lam;
  if (j == 1 && k == 1) return -1.0;
  return 0.0;
}

// d^m/dmu^m of -|z| (u) and -z^2/2 (v), z = y - mu, times the rate derivatives (j, k)
LAPLACE_HD_FORCEINLINE double poly_part(int m, int j, int k, double lam, double alp, double z) {
  double um = 0.0, vm = 0.0;
  if (m == 0) {
    um = -Kokkos::fabs(z);
    vm = -0.5 * z * z;
  } else if (m == 1) {
    um = z > 0.0 ? 1.0 : (z < 0.0 ? -1.0 : 0.0);
    vm = z;
  } else if (m == 2) {
    vm = -1.0;
  }
  return rate_a(j, k, lam, alp) * um + rate_c(j, k, lam, alp) * vm;
}

// ---- the moments and the derivatives of F(t) = log(x M(x)), t = 1/x^2 ------------------------

// cumulants kappa_1 .. kappa_n from raw moments mu_0 = 1 .. mu_n
LAPLACE_HD_FORCEINLINE void cumulants_from_moments(const double* mu, int n, double* kap) {
  for (int m = 1; m <= n; ++m) {
    double s = mu[m], binom = 1.0;  // C(m - 1, i - 1)
    for (int i = 1; i < m; ++i) {
      s -= binom * kap[i] * mu[m - i];
      binom = binom * (m - i) / i;
    }
    kap[m] = s;
  }
}

// F[m] = d^m F / dt^m, m = 1..N (N <= 6), t = 1/x^2 in [0, 1]. The scaled moments x^k m_k =
// prod_(j <= k) rho_j, rho_j = j / (1 + t rho_(j+1)), computed downward from j = K, where the
// recursion starts at the root of rho (1 + t rho) = K + 1 (its value for large j) and
// K = 2N + 20 + ceil(300 t) (as detail_normal::mills_cumulant: each step damps the error of the
// start by about j t / (1 + t rho)^2).
LAPLACE_HD void mills_f_derivs(double t, int N, double* F) {
  double r[13] = {};
  const int n = 2 * N;
  const double kk = 300.0 * t;
  const int K = n + 20 + (kk < 20000.0 ? static_cast<int>(Kokkos::ceil(kk)) : 20000);
  const double K1 = K + 1.0;
  double rho = 2.0 * K1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * K1 * t));
  for (int j = K; j >= 1; --j) {
    rho = j / (1.0 + t * rho);
    if (j <= n) r[j] = rho;
  }
  double mt[13];
  mt[0] = 1.0;
  for (int j = 1; j <= n; ++j) mt[j] = mt[j - 1] * r[j];
  double mu[7];
  mu[0] = 1.0;
  double h = 1.0;
  for (int j = 1; j <= N; ++j) {
    h *= -0.5;
    mu[j] = h * mt[2 * j];
  }
  double kap[7] = {};
  cumulants_from_moments(mu, N, kap);
  for (int m = 1; m <= N; ++m) F[m] = kap[m];
}

LAPLACE_HD void log_mills_derivs(double x, int N, double* L) {
  L[1] = math::log_mills_d1(x);
  if (N >= 2) L[2] = math::log_mills_d2(x);
  if (N >= 3) L[3] = math::log_mills_d3(x);
  if (N >= 4) L[4] = math::log_mills_d4(x);
  if (N >= 5) L[5] = math::log_mills_d5(x);
  if (N >= 6) L[6] = math::log_mills_d6(x);
}

// d^q/dc^q c^(-1/2)
LAPLACE_HD_FORCEINLINE double inv_sqrt_deriv(int q, double c) {
  double r = 1.0 / Kokkos::sqrt(c);
  for (int i = 0; i < q; ++i) r *= -(0.5 + i) / c;
  return r;
}
// d^p/da^p a^(-2)
LAPLACE_HD_FORCEINLINE double inv_square_deriv(int p, double a) {
  double r = 1.0 / (a * a);
  for (int i = 0; i < p; ++i) r *= -(2.0 + i) / a;
  return r;
}

// tab[pn(n, b)] = the order-n output of chain_rule_pair_dk<n>, n = n_first..N
template <int N, int n>
LAPLACE_HD void chain_orders(const double* G, const double* u0, const double* u1, double* tab) {
  if constexpr (n <= N) {
    double out[n + 1];
    math::chain_rule_pair_dk<n>(G, u0, u1, out);
    for (int b = 0; b <= n; ++b) tab[pn(n, b)] = out[b];
    chain_orders<N, n + 1>(G, u0, u1, tab);
  }
}

// A = log Z at (a, c)
LAPLACE_HD double log_norm_value(double a, double c) {
  const double x = a / Kokkos::sqrt(c);
  if (x < x_switch) return ln2 + math::log_mills(x) - 0.5 * math::log(c);
  return ln2 - math::log(a) + (math::log_mills(x) + math::log(x));
}

// G[pn(n, b)] = d^n A / da^(n-b) dc^b, n = 1..N (position 0 is not set), A = log 2 + L(x) - log(c)/2
template <int N>
LAPLACE_HD void ac_table_x(double a, double c, double* G) {
  constexpr int P = pair_count(N);
  for (int i = 0; i < P; ++i) G[i] = 0.0;
  double Gc[P] = {}, u0[P] = {}, u1[P] = {};
  const double x = a / Kokkos::sqrt(c);
  double L[7] = {};
  log_mills_derivs(x, N, L);
  for (int m = 1; m <= N; ++m) Gc[pair_index(m, 0)] = L[m];
  for (int q = 0; q <= N; ++q) {  // x = a c^(-1/2): d^p/da^p d^q/dc^q is a gamma_q (p = 0), gamma_q (p = 1)
    const double gam = inv_sqrt_deriv(q, c);
    u0[pair_index(0, q)] = a * gam;
    if (q + 1 <= N) u0[pair_index(1, q)] = gam;
  }
  chain_orders<N, 1>(Gc, u0, u1, G);
  double cn = 1.0;
  for (int n = 1; n <= N; ++n) {  // -log(c)/2 is pure in c
    cn *= c;
    G[pn(n, n)] += -0.5 * sign_of_power(n - 1) * fact(n - 1) / cn;
  }
}

// the same table from A = log 2 - log a + F(t), t = c/a^2 = 1/x^2
template <int N>
LAPLACE_HD void ac_table_t(double a, double c, double* G) {
  constexpr int P = pair_count(N);
  for (int i = 0; i < P; ++i) G[i] = 0.0;
  double Gc[P] = {}, u0[P] = {}, u1[P] = {};
  double F[7] = {};
  mills_f_derivs(c / (a * a), N, F);
  for (int m = 1; m <= N; ++m) Gc[pair_index(m, 0)] = F[m];
  for (int p = 0; p <= N; ++p) {  // t = c a^(-2): d^p/da^p d^q/dc^q is c nu_p (q = 0), nu_p (q = 1)
    const double nu = inv_square_deriv(p, a);
    u0[pair_index(p, 0)] = c * nu;
    if (p + 1 <= N) u0[pair_index(p, 1)] = nu;
  }
  chain_orders<N, 1>(Gc, u0, u1, G);
  double an = 1.0;
  for (int n = 1; n <= N; ++n) {  // -log(a) is pure in a
    an *= a;
    G[pn(n, 0)] += -sign_of_power(n - 1) * fact(n - 1) / an;
  }
}

// the table in the representation that does not cancel at this x = a / sqrt(c)
template <int N>
LAPLACE_HD void ac_table(double a, double c, double* G) {
  if (a / Kokkos::sqrt(c) < x_switch) ac_table_x<N>(a, c, G);
  else ac_table_t<N>(a, c, G);
}

// the derivatives of the rates in (lambda, alpha) as the inner maps of a chain rule:
// a'(lambda, alpha) = a + c D (D = 0 gives a itself) and c
template <int N>
LAPLACE_HD void rate_maps(double lam, double alp, double D, double* ua, double* uc) {
  constexpr int P = pair_count(N);
  for (int i = 0; i < P; ++i) ua[i] = uc[i] = 0.0;
  ua[pair_index(1, 0)] = alp + (1.0 - alp) * D;
  ua[pair_index(0, 1)] = lam * (1.0 - D);
  uc[pair_index(1, 0)] = 1.0 - alp;
  uc[pair_index(0, 1)] = -lam;
  if constexpr (N >= 2) {
    ua[pair_index(1, 1)] = 1.0 - D;
    uc[pair_index(1, 1)] = -1.0;
  }
}

// Atab[pn(n, b)] = d^n A / dlambda^(n-b) dalpha^b, n = 1..N
template <int N>
LAPLACE_HD void la_table(double lam, double alp, double* Atab) {
  constexpr int P = pair_count(N);
  double G[P];
  ac_table<N>(lam * alp, lam * (1.0 - alp), G);
  double ua[P], uc[P];
  rate_maps<N>(lam, alp, 0.0, ua, uc);
  for (int i = 0; i < P; ++i) Atab[i] = 0.0;
  chain_orders<N, 1>(G, ua, uc, Atab);
}

// the variance of the distribution, -2 dA/dc
LAPLACE_HD double variance_ac(double a, double c) {
  double G[3];
  ac_table<1>(a, c, G);
  return -2.0 * G[pn(1, 1)];
}

// ---- the expected information -------------------------------------------------------------

// d^p/dlambda^p d^q/dalpha^q of x = sqrt(lambda) alpha / sqrt(1 - alpha) (a product of a function of lambda
// and a function of alpha), into a pair-indexed table
template <int N>
LAPLACE_HD void x_map(double lam, double alp, double* u) {
  const double w = 1.0 - alp;
  double s[N + 1], rho[N + 1];
  s[0] = Kokkos::sqrt(lam);
  for (int p = 0; p < N; ++p) s[p + 1] = s[p] * (0.5 - p) / lam;
  double rise[N + 1];  // (1/2)(3/2) ... (1/2 + n - 1)
  rise[0] = 1.0;
  for (int n = 1; n <= N; ++n) rise[n] = rise[n - 1] * (n - 0.5);
  rho[0] = alp / Kokkos::sqrt(w);
  double wq = 1.0 / Kokkos::sqrt(w);  // w^(-1/2 - q)
  for (int q = 1; q <= N; ++q) {
    wq /= w;
    rho[q] = wq * (alp * rise[q] + q * rise[q - 1] * w);
  }
  for (int p = 0; p <= N; ++p)
    for (int q = 0; p + q <= N; ++q) u[pair_index(p, q)] = s[p] * rho[q];
}

// the same for t = c / a^2 = (1 - alpha) / (lambda alpha^2)
template <int N>
LAPLACE_HD void t_map(double lam, double alp, double* u) {
  double l[N + 1], g[N + 1];
  l[0] = 1.0 / lam;
  for (int p = 0; p < N; ++p) l[p + 1] = -l[p] * (p + 1) / lam;
  double fq = 1.0, ap = 1.0 / (alp * alp);  // q!, alpha^(-2-q)
  for (int q = 0; q <= N; ++q) {
    g[q] = ((q % 2 == 0) ? 1.0 : -1.0) * fq * ap * ((q + 1) - alp);
    fq *= (q + 1);
    ap /= alp;
  }
  for (int p = 0; p <= N; ++p)
    for (int q = 0; p + q <= N; ++q) u[pair_index(p, q)] = l[p] * g[q];
}

// tab[pn(n, b)], n = 0..K: d^n I_mumu / dlambda^(n-b) dalpha^b, for I_mumu = c + 2 a exp(-A). Written
// without exp(-A), whose table would carry the exact cancellation of the pure powers of alpha in
// exp(log alpha) (the 4th derivative of alpha is 0, as a sum of terms of size 1/alpha^4 times
// lambda/2, and lambda reaches 1e6): with r = exp(-L(x)) = 1/M(x), r' = r (r - x),
//   x < x_switch:  I_mumu = c (1 + x r(x))                       (exp(-A) = sqrt(c) r / 2),
//   x >= x_switch: I_mumu = a^2 (t + exp(-F(t)))                 (exp(-A) = a exp(-F) / 2),
// a function of the single inner variable x(lambda, alpha) or t(lambda, alpha) times a prefactor of the rates.
template <int K>
LAPLACE_HD void imm_table(double lam, double alp, double* tab) {
  constexpr int P = pair_count(K);
  const double a = lam * alp, c = lam * (1.0 - alp), x = a / Kokkos::sqrt(c);
  double H[P] = {}, Gc[P] = {}, u0[P] = {}, u1[P] = {};
  if (x < x_switch) {
    double r[K + 1];
    r[0] = math::exp(-math::log_mills(x));
    for (int n = 0; n < K; ++n) {  // r^(n+1) = sum C(n, k) r^(k) r^(n-k) - x r^(n) - n r^(n-1)
      double s = -x * r[n];
      for (int k = 0; k <= n; ++k) s += binomial_coef(n, k) * r[k] * r[n - k];
      if (n >= 1) s -= n * r[n - 1];
      r[n + 1] = s;
    }
    H[0] = 1.0 + x * r[0];
    for (int m = 1; m <= K; ++m) Gc[pair_index(m, 0)] = x * r[m] + m * r[m - 1];
    x_map<K>(lam, alp, u0);
  } else {
    const double t = c / (a * a);
    double F[K + 1] = {};
    mills_f_derivs(t, K, F);
    double h[K + 1];  // exp(-F) and its derivatives: h' = -F' h
    h[0] = math::exp(-(math::log_mills(x) + math::log(x)));
    for (int m = 1; m <= K; ++m) {
      double s = 0.0;
      for (int k = 0; k < m; ++k) s -= binomial_coef(m - 1, k) * h[k] * F[m - k];
      h[m] = s;
    }
    H[0] = t + h[0];
    for (int m = 1; m <= K; ++m) Gc[pair_index(m, 0)] = h[m] + (m == 1 ? 1.0 : 0.0);
    t_map<K>(lam, alp, u0);
  }
  chain_orders<K, 1>(Gc, u0, u1, H);
  // the prefactor: c (x form) or a^2 (t form), its derivatives in (lambda, alpha), Leibniz with H
  for (int n = 0; n <= K; ++n)
    for (int b = 0; b <= n; ++b) {
      const int j = n - b, k = b;
      double s = 0.0;
      for (int jp = 0; jp <= j; ++jp)
        for (int kp = 0; kp <= k; ++kp) {
          double pre;
          if (x < x_switch) {
            pre = rate_c(jp, kp, lam, alp);
          } else {  // a^2 = lambda^2 alpha^2
            const double fl = jp == 0 ? lam * lam : (jp == 1 ? 2.0 * lam : (jp == 2 ? 2.0 : 0.0));
            const double fa = kp == 0 ? alp * alp : (kp == 1 ? 2.0 * alp : (kp == 2 ? 2.0 : 0.0));
            pre = fl * fa;
          }
          if (pre != 0.0)
            s += binomial_coef(j, jp) * binomial_coef(k, kp) * pre * H[pn(j - jp + k - kp, k - kp)];
        }
      tab[pn(n, b)] = s;
    }
}

// the four nonzero entries of d^K I / dlambda^(K-k) dalpha^k, for every k = 0..K
template <int K>
LAPLACE_HD void info_entries(double lam, double alp, double* imm, double* ill, double* ila, double* iaa) {
  constexpr int N = K + 2;
  double At[pair_count(N)];
  la_table<N>(lam, alp, At);
  double Im[pair_count(K)];
  imm_table<K>(lam, alp, Im);
  for (int k = 0; k <= K; ++k) {
    const int j = K - k;
    ill[k] = At[pn(K + 2, k)];
    iaa[k] = At[pn(K + 2, k + 2)];
    double s = At[pn(K + 2, k + 1)];
    double dpow[K + 1];  // (-1)^m m! lambda^(-m-1): d^m/dlambda^m of 1/lambda
    double il = 1.0 / lam, fl = 1.0;
    for (int m = 0; m <= j; ++m) {
      dpow[m] = fl * il;
      fl *= -(m + 1);
      il /= lam;
    }
    for (int p = 0; p <= j; ++p) s -= binomial_coef(j, p) * dpow[j - p] * At[pn(p + k + 1, k + 1)];
    ila[k] = s;
    imm[k] = Im[pn(K, k)];
  }
}

// ---- the cdf ------------------------------------------------------------------------------

// log of the probability beyond distance D from the location, T of the header
LAPLACE_HD double tail_log(double lam, double alp, double D) {
  const double a = lam * alp, c = lam * (1.0 - alp);
  const double s = Kokkos::sqrt(c), x = a / s, w = x + s * D;
  return -a * D - 0.5 * c * D * D - ln2 + (math::log_mills(w) - math::log_mills(x));
}

// flat table of a function of (D, lambda, alpha), orders i + j + k <= K
template <int K>
LAPLACE_HD_FORCEINLINE constexpr int idx3(int i, int j, int k) {
  return (i * (K + 1) + j) * (K + 1) + k;
}

// the entries of T with i derivatives in D and n = j + k in (lambda, alpha), n = 1..K - i
template <int K, int n>
LAPLACE_HD void tail_orders(int i, double lam, double alp, double D, const double* Aab, const double* Gap,
                            const double* ua, const double* uc, const double* cp, double* T) {
  if constexpr (n <= K) {
    if (i + n <= K) {
      constexpr int P = pair_count(n);
      // the outer function c^i d^i A / da^i of (a', c): Leibniz in c, d^p/da'^p d^q/dc^q of it
      double Gh[P] = {};
      for (int p = 0; p <= n; ++p)
        for (int q = 0; p + q <= n; ++q) {
          if (p + q == 0) continue;
          double s = 0.0;
          for (int r = 0; r <= q && r <= i; ++r)
            s += binomial_coef(q, r) * fall(i, r) * cp[i - r] * Gap[pair_index(p + i, q - r)];
          Gh[pair_index(p, q)] = s;
        }
      double out[n + 1];
      math::chain_rule_pair_dk<n>(Gh, ua, uc, out);
      for (int b = 0; b <= n; ++b) {
        const int j = n - b, k = b;
        double v = out[b];
        const double aj = rate_a(j, k, lam, alp), cj = rate_c(j, k, lam, alp);
        if (i == 0) v += -aj * D - 0.5 * cj * D * D - Aab[pn(n, b)];
        else if (i == 1) v += -aj - cj * D;
        else if (i == 2) v += -cj;
        T[idx3<K>(i, j, k)] = v;
      }
    }
    tail_orders<K, n + 1>(i, lam, alp, D, Aab, Gap, ua, uc, cp, T);
  }
}

// T[i][j][k] for i + j + k <= K: the derivatives of
//   T(D, lambda, alpha) = -a D - c D^2/2 - log 2 + A(a + c D, c) - A(a, c)
// with the value Tval at (0, 0, 0), times eps^i (eps = d D / d mu = -sign(z))
template <int K>
LAPLACE_HD void tail_table(double lam, double alp, double D, double eps, double Tval, double* T) {
  constexpr int P = pair_count(K);
  const double a = lam * alp, c = lam * (1.0 - alp), ap = a + c * D;
  double Aab[P];
  la_table<K>(lam, alp, Aab);
  double Gap[P];
  ac_table<K>(ap, c, Gap);
  double ua[P], uc[P];
  rate_maps<K>(lam, alp, D, ua, uc);
  double cp[K + 1];
  cp[0] = 1.0;
  for (int i = 1; i <= K; ++i) cp[i] = cp[i - 1] * c;
  for (int q = 0; q < (K + 1) * (K + 1) * (K + 1); ++q) T[q] = 0.0;
  T[idx3<K>(0, 0, 0)] = Tval;
  for (int i = 0; i <= K; ++i) {
    if (i >= 1) {
      double v = cp[i] * Gap[pair_index(i, 0)];
      if (i == 1) v += -a - c * D;
      else if (i == 2) v += -c;
      T[idx3<K>(i, 0, 0)] = v;
    }
    tail_orders<K, 1>(i, lam, alp, D, Aab, Gap, ua, uc, cp, T);
  }
  double ei = 1.0;
  for (int i = 1; i <= K; ++i) {
    ei *= eps;
    for (int j = 0; i + j <= K; ++j)
      for (int k = 0; i + j + k <= K; ++k) T[idx3<K>(i, j, k)] *= ei;
  }
}

// E = exp(T) as tables of partial derivatives over multi-indices (i, j, k), i + j + k <= K:
// with gamma = alpha + e_r, D^gamma e^T = sum_(beta <= alpha) C(alpha, beta) D^(gamma - beta) T D^beta e^T
template <int K>
LAPLACE_HD void exp3(const double* T, double* E) {
  E[idx3<K>(0, 0, 0)] = math::exp(T[idx3<K>(0, 0, 0)]);
  for (int tot = 1; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i)
      for (int j = 0; i + j <= tot; ++j) {
        const int k = tot - i - j;
        int ai = i, aj = j, ak = k;
        if (ai > 0) --ai;
        else if (aj > 0) --aj;
        else --ak;
        double s = 0.0;
        for (int bi = 0; bi <= ai; ++bi)
          for (int bj = 0; bj <= aj; ++bj)
            for (int bk = 0; bk <= ak; ++bk)
              s += binomial_coef(ai, bi) * binomial_coef(aj, bj) * binomial_coef(ak, bk) * E[idx3<K>(bi, bj, bk)] *
                   T[idx3<K>(i - bi, j - bj, k - bk)];
        E[idx3<K>(i, j, k)] = s;
      }
}

// U = log(F) from the table of F (F_000 > 0): F = e^U, so
// F_gamma = F_0 U_gamma + sum_(0 != beta <= alpha) C(alpha, beta) F_beta U_(gamma - beta)
template <int K>
LAPLACE_HD void log3(const double* F, double* U) {
  U[idx3<K>(0, 0, 0)] = math::log(F[idx3<K>(0, 0, 0)]);
  for (int tot = 1; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i)
      for (int j = 0; i + j <= tot; ++j) {
        const int k = tot - i - j;
        int ai = i, aj = j, ak = k;
        if (ai > 0) --ai;
        else if (aj > 0) --aj;
        else --ak;
        double s = F[idx3<K>(i, j, k)];
        for (int bi = 0; bi <= ai; ++bi)
          for (int bj = 0; bj <= aj; ++bj)
            for (int bk = 0; bk <= ak; ++bk) {
              if (bi + bj + bk == 0) continue;
              s -= binomial_coef(ai, bi) * binomial_coef(aj, bj) * binomial_coef(ak, bk) * F[idx3<K>(bi, bj, bk)] *
                   U[idx3<K>(i - bi, j - bj, k - bk)];
            }
        U[idx3<K>(i, j, k)] = s / F[idx3<K>(0, 0, 0)];
      }
}

// the order-K components of a table, packed
template <int K>
LAPLACE_HD sym_tensor<3, K> gather(const double* S) {
  sym_tensor<3, K> t;
  for (int i = 0; i <= K; ++i)
    for (int k = 0; i + k <= K; ++k) t[pos3(K, i, k)] = S[idx3<K>(i, K - i - k, k)];
  return t;
}

// the derivatives of log(P(Z > D)) (near_side false) or of log(1 - P(Z > D)) (true), D = |z|,
// eps = d D / d mu
template <int K>
LAPLACE_HD sym_tensor<3, K> tail_tensor(double lam, double alp, double D, double eps, bool near_side) {
  constexpr int S = (K + 1) * (K + 1) * (K + 1);
  double T[S];
  const double Tval = tail_log(lam, alp, D);
  tail_table<K>(lam, alp, D, eps, Tval, T);
  if (!near_side) return gather<K>(T);
  double E[S], U[S];
  exp3<K>(T, E);  // e^T
  for (int q = 0; q < S; ++q) E[q] = -E[q];
  E[idx3<K>(0, 0, 0)] = -Kokkos::expm1(Tval);  // F = 1 - e^T
  log3<K>(E, U);
  return gather<K>(U);
}

// D >= 0 with log(P(Z > D)) = -log(2) - ell, ell >= 0 (the quantile equation of the header)
LAPLACE_HD double tail_distance(double lam, double alp, double ell) {
  if (!(ell > 0.0)) return 0.0;
  const double a = lam * alp, c = lam * (1.0 - alp);
  const double s = Kokkos::sqrt(c), x = a / s;
  double D = 2.0 * ell / (a + Kokkos::sqrt(a * a + 2.0 * c * ell));
  const double lx = math::log_mills(x);
  for (int it = 0; it < 60; ++it) {
    const double w = x + s * D;
    const double g = a * D + 0.5 * c * D * D - (math::log_mills(w) - lx) - ell;
    const double gp = a + c * D - s * math::log_mills_d1(w);
    const double dD = g / gp;
    D -= dD;
    if (D < 0.0) D = 0.0;
    if (Kokkos::fabs(dD) <= 8.0 * 2.220446049250313e-16 * D) break;
  }
  return D;
}

}  // namespace detail_enet

struct enet_distrib {
  // Location mu, penalty weight lambda > 0 (a = lambda alpha is the Laplace rate, c = lambda (1 - alpha) the Gaussian precision) and mixing weight alpha in (0, 1).
  double mu;
  double lambda;
  double alpha;

  static constexpr distrib_kind kind = distrib_kind::continuous;
  static constexpr int n_params = 3;
  // log f is C^0 in y at y = mu (y_kinks)
  static constexpr int y_regularity = 0;
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1},
      {"lambda", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"alpha", 0.0, 1.0, false, infinitely_smooth, false, infinitely_smooth}};

  /// The log-density.
  LAPLACE_HD double logpdf(double y) const {
    const double z = y - mu, a = lambda * alpha, c = lambda * (1.0 - alpha);
    return -a * Kokkos::fabs(z) - 0.5 * c * z * z - detail_enet::log_norm_value(a, c);
  }

  /// The derivatives of order K of the log-density in (mu, lambda, alpha), packed.
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    const double z = y - mu;
    double At[detail_enet::pair_count(K)];
    detail_enet::la_table<K>(lambda, alpha, At);
    sym_tensor<3, K> t;
    for (int i = 0; i <= K; ++i)
      for (int k = 0; i + k <= K; ++k) {
        const int j = K - i - k;
        double v = detail_enet::poly_part(i, j, k, lambda, alpha, z);
        if (i == 0) v -= At[detail_enet::pn(K, k)];
        t[detail_enet::pos3(K, i, k)] = v;
      }
    return t;
  }
  /// First-order derivatives of log f in (mu, lambda, alpha); sign(0) = 0 at y = mu.
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  // d^(J+K) l / dy^J dtheta^K: d/dy = -d/dmu, so the component (i, j, k) is (-1)^J times the
  // component with i + J derivatives in mu; the value of log f at y = mu uses sign(0) = 0.
  // J = 0 is logpdf_dk or the value.
  template <int J, int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dy_dk(double y) const {
    sym_tensor<3, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      const double z = y - mu, sj = (J % 2 == 0) ? 1.0 : -1.0;
      for (int i = 0; i <= K; ++i)
        for (int k = 0; i + k <= K; ++k)
          t[detail_enet::pos3(K, i, k)] = sj * detail_enet::poly_part(i + J, K - i - k, k, lambda, alpha, z);
      return t;
    }
  }

  /// The expected information (not -E[l_mumu]: see the header); the (mu, lambda) and (mu, alpha) entries are 0 by symmetry.
  LAPLACE_HD sym_tensor<3, 2> expected_info() const {
    double imm[1], ill[1], ila[1], iaa[1];
    detail_enet::info_entries<0>(lambda, alpha, imm, ill, ila, iaa);
    sym_tensor<3, 2> t;
    t[0] = imm[0];
    t[3] = ill[0];
    t[4] = ila[0];
    t[5] = iaa[0];
    return t;
  }
  /// The K-th derivatives of the expected information in (mu, lambda, alpha); zero for every index mu.
  template <int K>
  LAPLACE_HD expected_info_tensor<3, K> expected_info_dk() const {
    double imm[K + 1], ill[K + 1], ila[K + 1], iaa[K + 1];
    detail_enet::info_entries<K>(lambda, alpha, imm, ill, ila, iaa);
    expected_info_tensor<3, K> t;
    for (int k = 0; k <= K; ++k) {
      const int p = detail_enet::pos3(K, 0, k);
      t.pair[0][p] = imm[k];
      t.pair[3][p] = ill[k];
      t.pair[4][p] = ila[k];
      t.pair[5][p] = iaa[k];
    }
    return t;
  }
  /// First derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 4> expected_info_d4() const { return expected_info_dk<4>(); }

  /// F(y).
  LAPLACE_HD double cdf(double y) const {
    const double z = y - mu;
    if (z < 0.0) return math::exp(detail_enet::tail_log(lambda, alpha, -z));
    return -Kokkos::expm1(detail_enet::tail_log(lambda, alpha, z));
  }
  /// 1 - F(y).
  LAPLACE_HD double ccdf(double y) const {
    const double z = y - mu;
    if (z > 0.0) return math::exp(detail_enet::tail_log(lambda, alpha, z));
    return -Kokkos::expm1(detail_enet::tail_log(lambda, alpha, -z));
  }
  /// log F(y).
  LAPLACE_HD double log_cdf(double y) const {
    const double z = y - mu;
    if (z < 0.0) return detail_enet::tail_log(lambda, alpha, -z);
    return Kokkos::log1p(-math::exp(detail_enet::tail_log(lambda, alpha, z)));
  }
  /// log(1 - F(y)).
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = y - mu;
    if (z > 0.0) return detail_enet::tail_log(lambda, alpha, z);
    return Kokkos::log1p(-math::exp(detail_enet::tail_log(lambda, alpha, -z)));
  }
  // F^-1(p): the tail beyond D has probability min(p, 1 - p) (1 - p is exact above 1/2)
  /// The quantile F^-1(p).
  LAPLACE_HD double quantile(double p) const {
    if (!(p > 0.0 && p < 1.0)) {
      if (p == 0.0) return -distrib_infinity;
      if (p == 1.0) return distrib_infinity;
      return math::detail::quiet_nan();
    }
    const bool lower = p < 0.5;
    const double ptail = lower ? p : 1.0 - p;
    const double D = detail_enet::tail_distance(lambda, alpha, -math::log(2.0 * ptail));
    return lower ? mu - D : mu + D;
  }

  /// The derivatives of order K of log F in (mu, lambda, alpha); at z = 0 the branch z >= 0.
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_dk(double y) const {
    const double z = y - mu;
    if (z < 0.0) return detail_enet::tail_tensor<K>(lambda, alpha, -z, 1.0, false);
    return detail_enet::tail_tensor<K>(lambda, alpha, z, -1.0, true);
  }
  /// The derivatives of order K of log(1 - F) in (mu, lambda, alpha); at z = 0 the branch z <= 0.
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_ccdf_dk(double y) const {
    const double z = y - mu;
    if (z > 0.0) return detail_enet::tail_tensor<K>(lambda, alpha, z, -1.0, false);
    return detail_enet::tail_tensor<K>(lambda, alpha, -z, 1.0, true);
  }
  /// First derivatives of log F.
  LAPLACE_HD sym_tensor<3, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second derivatives of log F.
  LAPLACE_HD sym_tensor<3, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third derivatives of log F.
  LAPLACE_HD sym_tensor<3, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth derivatives of log F.
  LAPLACE_HD sym_tensor<3, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth derivatives of log F.
  LAPLACE_HD sym_tensor<3, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth derivatives of log F.
  LAPLACE_HD sym_tensor<3, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /// First derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth derivatives of log(1 - F).
  LAPLACE_HD sym_tensor<3, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }

  /// Lower end of the support (-infinity).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper end of the support (+infinity).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// The centre, mu (the mean and the median).
  LAPLACE_HD double center() const { return mu; }
  /// The standard deviation, sqrt(-2 dA/dc).
  LAPLACE_HD double scale() const {
    return Kokkos::sqrt(detail_enet::variance_ac(lambda * alpha, lambda * (1.0 - alpha)));
  }
  // mu: kink at y, continuous (order 0); lambda, alpha: none
  /// The non-smooth points of theta_j for the observation y: mu = y for j = 0, none for lambda and alpha.
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    distrib_kinks k;
    if (j == 0) {
      k.count = 1;
      k.point[0] = y;
    }
    return k;
  }
  /// The non-smooth point of log f in y: y = mu.
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    k.count = 1;
    k.point[0] = mu;
    return k;
  }
  // Starting value from a sample (generic_start.hpp): mu the median, alpha = 1/2, and lambda the
  // root of the variance at alpha = 1/2 (a = c = lambda/2, decreasing in lambda) equal to the sample
  // variance, found by bisection in log(lambda); all strictly inside the domain.
  enet_distrib start_value(const distrib_sample_summary& s) const {
    double target = s.variance;
    if (!(target > 1e-300)) target = 1e-300;
    if (!(target < 1e300)) target = 1e300;
    double lo = -70.0, hi = 70.0;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi), lam = Kokkos::exp(mid);
      if (detail_enet::variance_ac(0.5 * lam, 0.5 * lam) > target) lo = mid;
      else hi = mid;
    }
    return {start_inside(s.median, params[0]), start_inside(Kokkos::exp(0.5 * (lo + hi)), params[1]), 0.5};
  }
};

}  // namespace laplace::distrib
