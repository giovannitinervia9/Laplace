// betabinom_distrib (laplace::distrib): the beta-binomial of mean proportion mu in (0, 1) and
// dispersion sigma > 0 with a known size n (a trailing field of the struct, not a parameter), on
// y = 0, 1, ..., n. With a = mu / sigma and b = (1 - mu) / sigma,
//
//   l(mu, sigma; y) = log C(n, y) + log B(y + a, n - y + b) - log B(a, b).
//
// There are two routes, and every quantity takes whichever of them its own size calls for: sums
// over the integer shifts, which cost n + 1 terms and are exact at the binomial limit, and a form
// of bounded cost through Stirling's error, which holds where 10 sigma <= min(mu, 1 - mu). The
// second half of this comment describes the second; use_stirling below says which is taken where,
// with the measurements.
//
// The two log-betas are of the size of (a + b) log(a + b) while their difference is of the size of
// one, so neither is ever formed. Both shifts y and n - y are integers, so the recurrence of the
// gamma function turns the difference into a sum of logarithms, and the logarithms of sigma in it
// cancel (y + (n - y) - n = 0), which leaves
//
//   l = log C(n, y) + sum_(i<y) log(mu + i sigma) + sum_(i<n-y) log(1 - mu + i sigma)
//                   - sum_(i<n) log(1 + i sigma).
//
// Every term is regular at sigma = 0, where the sums vanish and the value is the binomial's
// log C(n, y) + y log mu + (n - y) log(1 - mu); no limit and no series is needed there. The cost is
// n + 1 logarithms for the value and n + 1 terms per order for the derivatives. Accumulating the
// first two sums as products of their positive terms, with one logarithm taken whenever the running
// product leaves [2^-512, 2^512], was measured and dropped: it saves 11 per cent of the value at
// n = 1e4 and costs two orders of magnitude of accuracy, because the relative error of a product of
// n factors is n eps while n independent logarithms accumulate sqrt(n) eps.
//
// The derivatives are the same sums differentiated term by term. Each logarithm is a logarithm of
// an affine function of (mu, sigma), so with K = p + q derivatives, p in mu and q in sigma,
//
//   d^K l = (-1)^(K-1) (K-1)! [ sum_(i<y) i^q / (mu + i sigma)^K
//                               + (-1)^p sum_(i<n-y) i^q / (1 - mu + i sigma)^K
//                               - [p = 0] sum_(i<n) i^K / (1 + i sigma)^K ],
//
// the third sum entering the pure derivatives in sigma only, since 1 + i sigma does not depend on
// mu. Each sum has terms of one sign, and whatever cancels does so between the three, where the
// derivative is genuinely small. This is written directly in (mu, sigma): the derivatives in
// (a, b) composed with a = mu / sigma carry inner factors of the size of sigma^-(K+1) against a
// result of the size of one, a relative loss of sigma^-K at the binomial limit, the same trap
// negbin1's shape chain has (negbin_core.hpp).
//
// log C(n, y) is taken in Loader's form, (1/2) log(n / (2 pi y (n - y))) + delta(n) - delta(y)
// - delta(n - y) + n H(y / n) with delta Stirling's error and n H = -y log(y/n) - (n - y) log(1 - y/n),
// so that no term is larger than the result.
//
// The support is finite, so the expected information and its derivatives come from the generic
// sum over the support (distrib_expected_info<K> of generic_info.hpp), which is exact there, and
// the cumulative function and the derivatives of log F and log(1 - F) from the generic
// summation route (generic_cdf.hpp). start_value inverts the moments: mu = mean / n and, from
// Var = n mu (1 - mu) (1 + (n - 1) sigma / (1 + sigma)), sigma = rho / (1 - rho) with
// rho = (Var / (n mu (1 - mu)) - 1) / (n - 1), floored away from both ends of its domain.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/binomial.hpp>
#include <laplace/distrib/core.hpp>
// for detail_negbin::dev_ratio_derivs, the deviance divided by its argument, D(1 + t)/t, with its
// derivatives: the only piece of that file this family reads, and the same quantity the two
// deviances of the route of bounded cost below are taken through
#include <laplace/distrib/negbin_core.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_betabinom {

// log C(n, y) in Loader's form, for 0 < y < n (0 at the two ends)
LAPLACE_HD double log_choose(double n, double y) {
  if (y <= 0.0 || y >= n) return 0.0;
  const double m = n - y, p = y / n;
  return 0.5 * math::log(n / (6.28318530717958647693 * y * m)) + math::stirling_error(n) -
         math::stirling_error(y) - math::stirling_error(m) - y * math::log(p) - m * Kokkos::log1p(-p);
}

// The three shifted sums in one pass, paired term by term. They cancel against each other, and
// each is far larger than the result once the size grows: at (mu, sigma, n) = (0.02, 0.25, 50) and
// y = 0 the second and the third are of the size of 100 while the log-mass is 0.4, and at n = 1e4
// the three are of the size of 4e4 against a result of the size of 10. Summing each over its own
// range and subtracting afterwards therefore loses the ratio: measured 651 ulp at that first point
// and 1.75e5 at the second. Paired at each i, the error of a term is relative to that term's own
// contribution, and the sums only add what is left.
//
// Two of the pairings also remove a cancellation of their own. With A = mu + i sigma,
// B = 1 - mu + i sigma and C = 1 + i sigma:
//
//  - the pure derivatives in mu of odd order need 1/A^K - 1/B^K, whose terms are equal at
//    mu = 1/2 and within (1 - 2 mu)/(i sigma) of each other at every i once sigma is large, while
//    their difference is of the size of the result (measured 3.9e15 ulp at order 5, mu = 1/2,
//    sigma = 1e3, n = 7). Since B - A = (1 - mu) - mu is exact and does not depend on i,
//        1/A^K - 1/B^K = (B - A) sum_(j=0)^(K-1) A^-(K-j) B^-(j+1),
//    a sum of terms of one sign, exactly 0 at mu = 1/2;
//  - the pure derivative in sigma takes i^K (1/A^K + 1/B^K - 1/C^K) at each i, with
//    (i/C)^K formed as a power of the ratio.
//
// Adds into acc[0..K], which hold the components with q = 0..K derivatives in sigma and K - q in mu.
template <int K>
LAPLACE_HD void shift_sums(double mu, double q1, double sigma, double y, double n, double* acc) {
  const double dba = q1 - mu;  // B - A, exact and independent of i
  for (double i = 0.0; i < n; i += 1.0) {
    const bool ha = i < y, hb = i < n - y;
    double plus = 0.0, minus = 0.0;  // the combination of the first two, for K - q even and odd
    if (ha && hb) {
      const double ia = 1.0 / (mu + i * sigma), ib = 1.0 / (q1 + i * sigma);
      double pa[K + 1], pb[K + 1];
      pa[0] = pb[0] = 1.0;
      for (int k = 1; k <= K; ++k) pa[k] = pa[k - 1] * ia, pb[k] = pb[k - 1] * ib;
      plus = pa[K] + pb[K];
      double t = 0.0;
      for (int j = 0; j < K; ++j) t += pa[K - j] * pb[j + 1];
      minus = dba * t;
    } else if (ha) {
      double w = 1.0 / (mu + i * sigma), wk = w;
      for (int k = 1; k < K; ++k) wk *= w;
      plus = wk;
      minus = wk;
    } else if (hb) {
      double w = 1.0 / (q1 + i * sigma), wk = w;
      for (int k = 1; k < K; ++k) wk *= w;
      plus = wk;
      minus = -wk;
    }
    double wc = i / (1.0 + i * sigma), wck = wc;
    for (int k = 1; k < K; ++k) wck *= wc;  // (i / C)^K
    double ip = 1.0;                        // i^q
    for (int q = 0; q < K; ++q) {
      acc[q] += (((K - q) & 1) ? minus : plus) * ip;
      ip *= i;
    }
    acc[K] += plus * ip - wck;  // ip is i^K here
  }
}

// the log-mass by the sums over the integer shifts: one logarithm a term, of the product of
// whichever of A = mu + i sigma and B = 1 - mu + i sigma is still in range over C = 1 + i sigma,
// since the three sums cancel against each other and are therefore paired at each i and not summed
// apart (the ratio is taken first, so that a size of 1e8 with a large sigma does not overflow)
LAPLACE_HD double logpmf_sums(double mu, double sigma, double n, double y) {
  const double q1 = 1.0 - mu;
  double s = log_choose(n, y);
  for (double i = 0.0; i < n; i += 1.0) {
    const double c = 1.0 + i * sigma;
    const bool ha = i < y, hb = i < n - y;
    if (ha && hb) s += math::log(((mu + i * sigma) / c) * (q1 + i * sigma));
    else if (ha) s += math::log((mu + i * sigma) / c);
    else if (hb) s += math::log((q1 + i * sigma) / c);
    else s -= Kokkos::log1p(i * sigma);
  }
  return s;
}

// the derivatives of order K by the sums, into t[0..K]
template <int K>
LAPLACE_HD void logpmf_dk_sums(double mu, double sigma, double n, double y, double* t) {
  double acc[K + 1];
  for (int q = 0; q <= K; ++q) acc[q] = 0.0;
  shift_sums<K>(mu, 1.0 - mu, sigma, y, n, acc);
  double f = 1.0;  // (K-1)!
  for (int k = 2; k < K; ++k) f *= k;
  const double c = ((K & 1) ? f : -f);
  for (int b = 0; b <= K; ++b) t[b] = c * acc[b];
}

// ------------------------------------------------- the route of bounded cost, through Stirling
//
// The sums above cost n + 1 terms and their roundings grow as (n/50)^(3/2). Substituting
// lgamma(z) = (z - 1/2) log z - z + log sqrt(2 pi) + delta(z) in the six log-gammas of the
// definition gives a form of bounded cost, with Delta = y - n mu, A = mu + y sigma,
// B = 1 - mu + (n - y) sigma and C = 1 + n sigma, so that A + B = C exactly:
//
//   l = l_binom(mu; n, y)                                 the binomial log-mass, in Loader's form
//       + Delta [rho(w) - rho(v)],   rho(t) = D(1 + t) / t,
//         w = Delta sigma / (mu C),  v = -Delta sigma / ((1 - mu) C)
//       - (1/2) [log A - log mu + log B - log(1 - mu) - log C]
//       + [delta(A/sigma) + delta(B/sigma) - delta(C/sigma)]
//       - [delta(mu/sigma) + delta((1 - mu)/sigma) - delta(1/sigma)].
//
// Checked symbolically (the six log-gammas cancel, so only an identity among logarithms is left)
// and against the definition at 60 digits, 3.9e-55 relative over 60 points, non-integer y and n
// among them (tools/reference/distrib_betabinom.py).
//
// Nothing in it is a difference of large terms. l_binom is Loader's own form (binomial.hpp), so
// the pair log C(n, y) and y log mu + (n - y) log(1 - mu), each of the size of n log n and
// cancelling to the size of one, is never formed. The two deviances are
// [bd0(A, mu C) + bd0(B, (1 - mu) C)] / sigma, taken through rho because 1 + w = A/(mu C) and
// 1 + v = B/((1 - mu) C) exactly, and w and v have opposite signs, so the difference of the two rho
// is a sum of magnitudes. Every logarithm is a logarithm of an affine form. The two bracketed
// groups of Stirling errors are each of the size of sigma and their difference of the size of
// sigma^2, which is an absolute error of the size of eps and never a relative one.
//
// The derivatives are closed except for the two rho, because every inner map is affine in
// (mu, sigma) or separable: d^(p+q) log A = (-1)^(p+q-1) (p+q-1)! y^q A^-(p+q) and likewise for
// the other four logarithms; w = (y/mu - n) sigma/C and v = ((n-y)/(1-mu) - n) sigma/C are each a
// function of mu times a function of sigma; and Delta is affine, so
// d^(p+q) (Delta F) = Delta F_(p,q) - n p F_(p-1,q). The Stirling errors are taken as the
// asymptotic series of stirling.hpp in the SMALL variable x = sigma/z, delta(z/sigma) =
// sum_j s_j x^(2j-1) with s_j = B_2j / (2j (2j-1)); composing delta's own derivatives with z/sigma
// instead would carry inner factors of the size of sigma^-(q+1) against a result of the size of
// one, the trap the (a, b) chain has. That series is the asymptotic branch of stirling.hpp, so the
// route holds where all six arguments are at least 10, that is where 10 sigma <= min(mu, 1 - mu).

// The size above which the Stirling route is taken where it applies. Two numbers, because the two
// routes' costs cross at a different size for the value than for a tensor (measured on one thread,
// ns per call, sums against Stirling): the value crosses at n = 8 (366/361 there, 1156/194 at
// n = 64), and the tensors at 120, 250, 420, 570, 700 and 800 for the orders 1 to 6, the Stirling
// side costing 214, 460, 1040, 2200, 3400 and 6600 ns at every size. Above order 1 the cut is set
// by accuracy and not by cost: at n = 160 the sums' error has already reached about 940 ulp, by the
// (n/50)^(3/2) law, where the Stirling route stays near 65 ulp at every size, so the cut is taken
// there and the price is at most five times the cost of a tensor between n = 160 and n = 800.
LAPLACE_HD_FORCEINLINE constexpr double n_stirling_cut(int K) { return K == 0 ? 8.0 : 160.0; }

// whether the route of bounded cost applies to the quantity of order K (0 the value): the six
// Stirling errors must all be in the asymptotic branch of stirling.hpp, whose argument starts at
// 10, and the size must be above the cut
LAPLACE_HD_FORCEINLINE bool use_stirling(double mu, double sigma, double n, int K) {
  const double q = mu < 1.0 - mu ? mu : 1.0 - mu;
  return n > n_stirling_cut(K) && 10.0 * sigma <= q;
}

// d^(p+q) log z / dmu^p dsigma^q for an affine z = z0 + cm mu + cs sigma, p + q >= 1:
// (-1)^(p+q-1) (p+q-1)! cm^p cs^q z^-(p+q), which is already 0 when z does not depend on one of
// the two (cm or cs zero and the matching order above zero)
LAPLACE_HD_FORCEINLINE double log_affine_dk(int p, int q, double z, double cm, double cs) {
  double f = 1.0;  // (p + q - 1)!
  for (int t = 2; t < p + q; ++t) f *= t;
  double zp = 1.0;  // z^-(p+q)
  for (int t = 0; t < p + q; ++t) zp /= z;
  double c = 1.0;
  for (int t = 0; t < p; ++t) c *= cm;
  for (int t = 0; t < q; ++t) c *= cs;
  return (((p + q) & 1) ? f : -f) * c * zp;
}

// delta(z/sigma) by the asymptotic series of stirling.hpp in x = sigma/z, and its derivatives in x
// of orders 0..K: sum_j s_j x^(2j-1) differentiated term by term, which holds for x <= 1/10
template <int K>
LAPLACE_HD void delta_series_derivs(double x, double* out) {
  const double x2 = x * x;
  for (int k = 0; k <= K; ++k) out[k] = 0.0;
  double xm = x;  // x^(2j-1)
  for (int j = 1; j <= 20; ++j, xm *= x2) {
    const int m = 2 * j - 1;
    const double sj = math::detail_stirling::asym_coef(j - 1);
    double fall = 1.0, p = xm, last = 0.0;  // m (m-1) ... (m-k+1) and x^(m-k)
    for (int k = 0; k <= K && k <= m; ++k) {
      if (k > 0) fall *= (m - k + 1), p /= x;
      const double t = sj * fall * p;
      out[k] += t;
      last = t;
    }
    // the highest order converges the slowest, by the factor m^K it carries
    if (j > 2 && Kokkos::fabs(last) <= 0x1p-56 * Kokkos::fabs(out[K < m ? K : m])) break;
  }
}

// adds sign * d^K delta(z/sigma) to acc[0..K], acc[b] holding the component with b derivatives in
// sigma and K - b in mu, from the asymptotic series term by term. With z affine in (mu, sigma),
// d^(p,q) z^-m = (-1)^(p+q) (m)_(p+q) cm^p cs^q z^-(m+p+q) where (m)_r is the rising factorial, so
// Leibniz over the two factors of sigma^m z^-m gives
//   d^(p,q) [sigma^m z^-m] = sum_i C(q,i) [m!/(m-i)! sigma^(m-i)]
//                                        [(-1)^(p+q-i) (m)_(p+q-i) cm^p cs^(q-i) z^-(m+p+q-i)],
// which is closed: no chain rule and no composition. The j term is dropped once it is below 2^-56
// of what every component has accumulated.
template <int K>
LAPLACE_HD void add_delta_dk(double sigma, double z, double cm, double cs, double sign, double* acc) {
  const double x = sigma / z, x2 = x * x, iz = 1.0 / z, isig = 1.0 / sigma;
  double cmp[K + 1], csp[K + 1];
  cmp[0] = csp[0] = 1.0;
  for (int t = 1; t <= K; ++t) cmp[t] = cmp[t - 1] * cm, csp[t] = csp[t - 1] * cs;
  double xm = x;  // x^m at m = 2j - 1
  for (int j = 1; j <= 20; ++j, xm *= x2) {
    const int m = 2 * j - 1;
    const double sj = math::detail_stirling::asym_coef(j - 1);
    // r = K - i, so the rising factorial and the power of z depend on j only through m: one pass
    double rz[K + 1], fa[K + 1];  // (-1)^r (m)_r z^-r and m!/(m-i)! sigma^-i, r = K - i
    rz[0] = 1.0;
    for (int r = 1; r <= K; ++r) rz[r] = -rz[r - 1] * (m + r - 1) * iz;
    fa[0] = 1.0;
    for (int i = 1; i <= K; ++i) fa[i] = (i <= m) ? fa[i - 1] * (m - i + 1) * isig : 0.0;
    double big = 0.0, seen = 0.0;
    for (int b = 0; b <= K; ++b) {
      const int p = K - b, q = b;
      double tot = 0.0;
      for (int i = 0; i <= q; ++i)
        tot += math::detail_chain_pair::binom(q, i) * fa[i] * rz[K - i] * csp[q - i];
      const double t = sj * xm * cmp[p] * tot;
      acc[b] += sign * t;
      big = Kokkos::fmax(big, Kokkos::fabs(t));
      seen = Kokkos::fmax(seen, Kokkos::fabs(acc[b]));
    }
    if (j > 3 && big <= 0x1p-56 * seen) break;
  }
}

// F = rho(w) - rho(v) and every partial of total order up to K, rho(t) = D(1 + t)/t. The two inner
// maps are separable, w = (y/mu - n) g(sigma) and v = ((n-y)/(1-mu) - n) g(sigma) with
// g = sigma/(1 + n sigma), so their partials are the product of the two factors' derivatives.
template <int K>
LAPLACE_HD void stirling_f_partials(double mu, double sigma, double n, double y, double* F) {
  using math::detail_chain_pair::pair_count;
  using math::detail_chain_pair::pair_index;
  const double q1 = 1.0 - mu, k1 = n - y, C = 1.0 + n * sigma;
  const double fw0 = y / mu - n, fv0 = k1 / q1 - n, g0 = sigma / C;
  double fw[K + 1], fv[K + 1], g[K + 1];
  fw[0] = fw0, fv[0] = fv0, g[0] = g0;
  double fm = 1.0 / mu, fq = 1.0 / q1, gc = 1.0 / C;
  for (int p = 1; p <= K; ++p) {
    fm *= p / mu, fq *= p / q1;                       // p! / mu^(p+1), p! / (1-mu)^(p+1)
    fw[p] = ((p & 1) ? -y : y) * fm;
    fv[p] = k1 * fq;
  }
  for (int q = 1; q <= K; ++q) {
    gc *= q / C;                                      // q! / C^(q+1), with one n^(q-1) to apply
    double np = 1.0;
    for (int t = 0; t < q - 1; ++t) np *= n;
    g[q] = ((q & 1) ? gc : -gc) * np;
  }
  const double w = fw0 * g0, v = fv0 * g0;
  double rw[K + 1], rv[K + 1];
  detail_negbin::dev_ratio_derivs<K>(w, rw);
  detail_negbin::dev_ratio_derivs<K>(v, rv);
  double Uw[pair_count(K)], Uv[pair_count(K)];
  for (int i = 0; i < pair_count(K); ++i) Uw[i] = Uv[i] = 0.0;
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q)
      Uw[pair_index(p, q)] = fw[p] * g[q], Uv[pair_index(p, q)] = fv[p] * g[q];
  F[0] = rw[0] - rv[0];
  double out[K + 1], out2[K + 1];
#define LAPLACE_BB_F(ORD)                                                     \
  if constexpr (K >= ORD) {                                                   \
    math::chain_rule_pair_scalar_dk<ORD>(rw, Uw, out);                        \
    math::chain_rule_pair_scalar_dk<ORD>(rv, Uv, out2);                       \
    for (int b = 0; b <= ORD; ++b) F[pair_index(ORD - b, b)] = out[b] - out2[b]; \
  }
  LAPLACE_BB_F(1)
  LAPLACE_BB_F(2)
  LAPLACE_BB_F(3)
  LAPLACE_BB_F(4)
  LAPLACE_BB_F(5)
  LAPLACE_BB_F(6)
#undef LAPLACE_BB_F
}

// the log-mass by the route of bounded cost
LAPLACE_HD double logpmf_stirling(double mu, double sigma, double n, double y) {
  const double q1 = 1.0 - mu, k1 = n - y;
  const double A = mu + y * sigma, B = q1 + k1 * sigma, C = 1.0 + n * sigma;
  const double Dl = Kokkos::fma(-n, mu, y);  // y - n mu
  const double w = Dl * sigma / (mu * C), v = -Dl * sigma / (q1 * C);
  double rw[1], rv[1];
  detail_negbin::dev_ratio_derivs<0>(w, rw);
  detail_negbin::dev_ratio_derivs<0>(v, rv);
  double s = binomial_distrib{mu, n}.logpmf(y) + Dl * (rw[0] - rv[0]);
  s -= 0.5 * (math::log(A / mu) + math::log(B / q1) - math::log(C));
  const double is = 1.0 / sigma;
  s += math::stirling_error(A * is) + math::stirling_error(B * is) - math::stirling_error(C * is);
  s -= math::stirling_error(mu * is) + math::stirling_error(q1 * is) - math::stirling_error(is);
  return s;
}

// its derivatives of order K, into t[0..K]
template <int K>
LAPLACE_HD void logpmf_dk_stirling(double mu, double sigma, double n, double y, double* t) {
  using math::detail_chain_pair::pair_count;
  using math::detail_chain_pair::pair_index;
  const double q1 = 1.0 - mu, k1 = n - y;
  const double A = mu + y * sigma, B = q1 + k1 * sigma, C = 1.0 + n * sigma;
  const double Dl = Kokkos::fma(-n, mu, y);
  double F[pair_count(K)];
  stirling_f_partials<K>(mu, sigma, n, y, F);
  double acc[K + 1];
  for (int b = 0; b <= K; ++b) acc[b] = 0.0;
  add_delta_dk<K>(sigma, A, 1.0, y, 1.0, acc);
  add_delta_dk<K>(sigma, B, -1.0, k1, 1.0, acc);
  add_delta_dk<K>(sigma, C, 0.0, n, -1.0, acc);
  add_delta_dk<K>(sigma, mu, 1.0, 0.0, -1.0, acc);
  add_delta_dk<K>(sigma, q1, -1.0, 0.0, -1.0, acc);
  add_delta_dk<K>(sigma, 1.0, 0.0, 0.0, 1.0, acc);
  const auto bl = binomial_distrib{mu, n}.logpmf_dk<K>(y);
  for (int b = 0; b <= K; ++b) {
    const int p = K - b, q = b;
    double s = Dl * F[pair_index(p, q)];
    if (p >= 1) s -= n * p * F[pair_index(p - 1, q)];
    s -= 0.5 * (log_affine_dk(p, q, A, 1.0, y) - log_affine_dk(p, q, mu, 1.0, 0.0) +
                log_affine_dk(p, q, B, -1.0, k1) - log_affine_dk(p, q, q1, -1.0, 0.0) -
                log_affine_dk(p, q, C, 0.0, n));
    s += acc[b];
    if (q == 0) s += bl[0];
    t[b] = s;
  }
}

}  // namespace detail_betabinom

/**
 * \ingroup distrib_betabinom
 * \brief Beta-binomial distribution with known size \f$n\f$, parameterized by the mean proportion
 * \f$\mu \in (0, 1)\f$ and the dispersion \f$\sigma > 0\f$.
 *
 * Mass \f$p(y) = \binom{n}{y} B(y + a, n - y + b) / B(a, b)\f$ on \f$y = 0, \dots, n\f$, with
 * \f$a = \mu/\sigma\f$ and \f$b = (1 - \mu)/\sigma\f$, mean \f$n\mu\f$ and variance
 * \f$n\mu(1-\mu)\left(1 + (n-1)\sigma/(1+\sigma)\right)\f$; the binomial distribution is the limit
 * \f$\sigma \to 0\f$. The log-mass and its derivatives to order 6 are evaluated by paired sums over
 * the integer shifts (regular at that limit) or, above size cuts where
 * \f$10\sigma \le \min(\mu, 1 - \mu)\f$, by a bounded-cost form through Stirling's error; the size
 * is a data constant, a trailing field not counted in \c n_params and not listed in \c params.
 * \see distrib_betabinom, distrib_core, stirling
 */
struct betabinom_distrib {
  /// Mean proportion \f$\mu \in (0, 1)\f$.
  double mu;
  /// Dispersion \f$\sigma \in (0, \infty)\f$; the binomial distribution is the limit \f$\sigma \to 0\f$.
  double sigma;
  /// Known size \f$n \ge 1\f$ (an integer held in a double), the upper end of the support.
  double size;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (2; the size is not one).
  static constexpr int n_params = 2;
  /// Descriptors of \c mu and \c sigma.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, 1.0, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_betabinom
   * \brief Evaluates the log-mass \f$\log p(y; \mu, \sigma)\f$.
   *
   * Evaluates paired sums over the integer shifts for \f$n \le 8\f$ or outside
   * \f$10\sigma \le \min(\mu, 1 - \mu)\f$, and a bounded-cost form through Stirling's error above, so
   * that the two log-betas are never formed and the value at \f$\sigma \to 0\f$ is the binomial's.
   * Returns \f$-\infty\f$ when \f$y\f$ is not an integer in \f$\{0, \dots, n\}\f$.
   * \param y Observation point.
   * \return Log-mass value.
   * \see distrib_betabinom, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    const double n = size;
    if (!(y >= 0.0 && y <= n) || y != Kokkos::floor(y)) return -distrib_infinity;
    return detail_betabinom::use_stirling(mu, sigma, n, 0) ? detail_betabinom::logpmf_stirling(mu, sigma, n, y)
                                                           : detail_betabinom::logpmf_sums(mu, sigma, n, y);
  }

  /**
   * \ingroup distrib_betabinom
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu, \sigma)\f$.
   *
   * Component \f$b\f$ of the packed tensor holds \f$\partial^K \log p / \partial\mu^{K-b}
   * \partial\sigma^{b}\f$, evaluated by the shift sums differentiated term by term (for
   * \f$n \le 160\f$ or outside \f$10\sigma \le \min(\mu, 1 - \mu)\f$) or by the bounded-cost
   * Stirling form above.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Integer observation point in \f$\{0, \dots, n\}\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpmf_dk(double y) const {
    const double n = size;
    sym_tensor<2, K> ts;
    if (detail_betabinom::use_stirling(mu, sigma, n, K)) {
      detail_betabinom::logpmf_dk_stirling<K>(mu, sigma, n, y, ts.v);
      return ts;
    }
    detail_betabinom::logpmf_dk_sums<K>(mu, sigma, n, y, ts.v);
    return ts;
  }
  /**
   * \name Derivatives of the log-mass
   * Parameter derivatives of \f$\log p(y; \mu, \sigma)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpmf_d1(double y) const { return logpmf_dk<1>(y); }
  /// Second-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpmf_d2(double y) const { return logpmf_dk<2>(y); }
  /// Third-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpmf_d3(double y) const { return logpmf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpmf_d4(double y) const { return logpmf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpmf_d5(double y) const { return logpmf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log p\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpmf_d6(double y) const { return logpmf_dk<6>(y); }
  /** @} */

  /// \ingroup distrib_betabinom
  /// \brief Lower bound of the support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_betabinom
  /// \brief Upper bound of the support (the size \f$n\f$).
  LAPLACE_HD double support_upper() const { return size; }
  /// \ingroup distrib_betabinom
  /// \brief Central location (the mean \f$n\mu\f$).
  LAPLACE_HD double center() const { return size * mu; }
  /// \ingroup distrib_betabinom
  /// \brief Typical spread \f$\sqrt{n\mu(1-\mu)(1 + (n-1)\sigma/(1+\sigma))}\f$.
  LAPLACE_HD double scale() const {
    return Kokkos::sqrt(size * mu * (1.0 - mu) * (1.0 + (size - 1.0) * sigma / (1.0 + sigma)));
  }
  /// \ingroup distrib_betabinom
  /// \brief Non-smooth points in the parameters (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }

  /**
   * \ingroup distrib_betabinom
   * \brief Starting value from a sample summary: \f$\mu = \bar{y}/n\f$ (`s.mean`) and the moment
   * estimate of \f$\sigma\f$ from the sample variance (`s.variance`).
   *
   * With \f$\rho = \left(s^2 / (n\mu(1-\mu)) - 1\right)/(n-1)\f$ the intraclass correlation inverted
   * from \f$\mathrm{Var} = n\mu(1-\mu)(1 + (n-1)\sigma/(1+\sigma))\f$, the estimate is
   * \f$\sigma = \rho/(1-\rho)\f$, with \f$\rho\f$ clamped to \f$[10^{-4}, 1 - 10^{-4}]\f$: a sample with
   * no overdispersion starts near the binomial limit instead of at an end of the domain, and a size
   * of 1 (where \f$\sigma\f$ is not identified) starts at \f$\sigma = 10^{-4}\f$.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c betabinom_distrib preserving \c size, with both parameters strictly inside their domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  betabinom_distrib start_value(const distrib_sample_summary& s) const {
    const double m = start_inside(s.mean / size, params[0]);
    const double v0 = size * m * (1.0 - m);
    double rho = size > 1.0 ? (s.variance / v0 - 1.0) / (size - 1.0) : 1e-4;
    if (!(rho > 1e-4)) rho = 1e-4;
    if (rho > 1.0 - 1e-4) rho = 1.0 - 1e-4;
    return {m, start_inside(rho / (1.0 - rho), params[1]), size};
  }
};

}  // namespace laplace::distrib
