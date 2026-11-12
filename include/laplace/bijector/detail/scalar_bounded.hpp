// Scalar bijector bounded (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   bounded: (p1, p2) -> R, g(v) = log(v - p1) - log(p2 - v), p1 < p2; p1 = lower, p2 = upper
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
// Notation, with a = p1 (lower), b = p2 (upper), s = b - a:
//   forward  p = v - a, q = b - v (both one subtraction, exact near each bound),
//            A = 1/p, B = 1/q, S = 1/s, D = p - q = 2v - a - b.
//            g = log p - log q. Derivatives in v (K >= 1), as in logit (scalar.hpp):
//              g^(K)  = (K-1)! ((-1)^(K-1) A^K + B^K), written for even K as
//                       (K-1)! D A B S_K(A, B), with S_K(A, B) = sum_j A^(K-1-j) B^j
//                       (B - A = D A B), so that no difference of two large terms is formed;
//              log|g'| = log s - log p - log q, its K-th derivative (K-1)! ((-1)^K A^K + B^K),
//                       written for odd K as (K-1)! D A B S_K(A, B).
//            The mixed partials follow from g = log p - log q with p = v - a, q = b - v:
//            d_a = -d_p, d_b = d_q, d_v = d_p - d_q, and log p depends on v and a only,
//            log q on v and b only (log s on a and b only). For K = n0 + n1 + n2 >= 1:
//              g:      [n2 = 0] (-1)^(n1+K-1) (K-1)! A^K  +  [n1 = 0] (-1)^n2 (K-1)! B^K
//              log|g'|: [n2 = 0] (-1)^n0 (K-1)! A^K  +  [n1 = 0] (-1)^n2 (K-1)! B^K
//                       +  [n0 = 0] (-1)^(n1+K-1) (K-1)! S^K
//            (the first term of each is zero where its own parameter is absent; the
//            mixed a-b derivatives of g vanish, those of log|g'| do not: the last term).
//            Two pairs of these terms cancel near a bound (B^K - S^K where A and B are
//            close to S, and the same with A), so they are written with their exact
//            factors: B - S = p S B and A - S = q S A, with geometric sums of the powers.
//            The values are log(p/q) for g (one rounding in the ratio, where log p - log q
//            would carry 18 eps near the midpoint) and log((s/p)/q) for log|g'|.
//   inverse  h(u) = a + s expit(u) = a q + b p, p = expit(u), q = expit(-u), U = p q,
//            D = q - p (D^2 = 1 - 4U); the pair p, q comes from e = exp(-|u|) without
//            cancellation. Pure: h^(K) = s sigma_K(U, D) with sigma_K the derivatives of
//            expit: U, U D, U (1 - 6U), U D (1 - 12U), U (1 - 30U + 120U^2),
//            U D (1 - 60U + 360U^2) (the same polynomials as in scalar.hpp).
//            Partials: a is affine with d_a h = 1 - expit(u) = q and d_b h = expit(u) = p,
//            second partials in (a, b) zero; so with n0 = K - n1 - n2:
//              n1 + n2 = 0: s sigma_K;
//              n1 + n2 = 1: n0 = 0: q (for n1 = 1) or p (for n2 = 1);
//                           n0 >= 1: -sigma_n0 (n1 = 1) or +sigma_n0 (n2 = 1);
//              n1 + n2 >= 2: 0.
//            log|h'| = log s + log(p q): the derivatives in u are D (order 1) and
//            -2 sigma_(K-1) (order K >= 2), of log(p q); log s carries the (a, b)
//            partials as in the forward case, with zero u-derivatives.
//   distances  b - h(u) = s q, h(u) - a = s p: products, exact to the rounding of s.
//
// Each quantity is formed from s, p and q, each one subtraction of the bounds from v or
// u, and from the reciprocals, with no difference of two large terms except where one
// was rewritten as above. The inverse log-Jacobian is log s - |u| - 2 log1p(e), a sum of
// positive terms after the first. The polynomials were derived and checked with SymPy:
// tools/reference/bijector_bounded.py checks each closed form above, for every
// multi-index of total order up to 6, against SymPy's derivative of the defining
// expression.
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>

namespace laplace::bijector::detail_scalar {

// A valid bounded map has finite a < b; otherwise every function returns NaN.
LAPLACE_HD_FORCEINLINE bool bounded_valid(double a, double b) {
  return a < b && Kokkos::isfinite(a) && Kokkos::isfinite(b);
}

// (-1)^k for k >= 0
LAPLACE_HD_FORCEINLINE double bounded_sign(int k) { return k % 2 == 0 ? 1.0 : -1.0; }

// forward quantities at v, for the bounds a, b: p = v - a, q = b - v, s = b - a,
// their reciprocals A, B, S, and D = p - q
struct bounded_fwd_parts {
  double p, q, s, A, B, S, D;
};
LAPLACE_HD_FORCEINLINE bounded_fwd_parts bounded_fwd_of(double v, double a, double b) {
  const double p = v - a, q = b - v, s = b - a;
  return {p, q, s, 1.0 / p, 1.0 / q, 1.0 / s, p - q};
}

// S_K(A, B) = sum_(j=0..K-1) A^(K-1-j) B^j, K >= 1, by products of powers. The reciprocal
// form of geo_sum of scalar_helpers gives 0 times infinity at an infinite A (on a bound).
template <int K>
LAPLACE_HD_FORCEINLINE double bounded_geo(double A, double B) {
  double pa[K], pb[K];
  pa[0] = 1.0;
  pb[0] = 1.0;
  for (int i = 1; i < K; ++i) {
    pa[i] = pa[i - 1] * A;
    pb[i] = pb[i - 1] * B;
  }
  double s = 0.0;
  for (int j = 0; j < K; ++j) s += pa[K - 1 - j] * pb[j];
  return s;
}

// expit quantities at u: p = expit(u), q = expit(-u), U = p q, D = q - p, e = exp(-|u|).
// p and q are formed from e = exp(-|u|) and 1/(1 + e), so each is accurate on its own.
struct bounded_sig_parts {
  double p, q, U, D, e;
};
LAPLACE_HD_FORCEINLINE bounded_sig_parts bounded_sig_of(double u) {
  const double au = Kokkos::fabs(u), e = m::exp(-au), d = 1.0 + e;
  const double big = 1.0 / d, small = e / d;
  const bool pos = u >= 0.0;
  return {pos ? big : small, pos ? small : big, small * big, Kokkos::copysign(Kokkos::expm1(-au) / d, -u), e};
}

// d^K expit(u) / du^K, K >= 1, from U and D
template <int K>
LAPLACE_HD_FORCEINLINE double bounded_sig_dk(double U, double D) {
  if constexpr (K == 1) return U;
  if constexpr (K == 2) return U * D;
  if constexpr (K == 3) return U * (1.0 - 6.0 * U);
  if constexpr (K == 4) return U * D * (1.0 - 12.0 * U);
  if constexpr (K == 5) {
    constexpr double c[] = {1.0, -30.0, 120.0};
    return U * poly(c, U);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {1.0, -60.0, 360.0};
    return U * D * poly(c, U);
  }
}
// the same for a runtime order k in 1..6 (0 gives 0)
LAPLACE_HD_FORCEINLINE double bounded_sig_dk_rt(int k, double U, double D) {
  switch (k) {
    case 1: return bounded_sig_dk<1>(U, D);
    case 2: return bounded_sig_dk<2>(U, D);
    case 3: return bounded_sig_dk<3>(U, D);
    case 4: return bounded_sig_dk<4>(U, D);
    case 5: return bounded_sig_dk<5>(U, D);
    case 6: return bounded_sig_dk<6>(U, D);
    default: return 0.0;
  }
}

// ---------------------------------------------------------------- pure derivatives

template <int K>
LAPLACE_HD double bounded_fwd(double x, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  const auto f = bounded_fwd_of(x, p1, p2);
  if constexpr (K == 0) return m::log(f.p / f.q);  // the ratio carries one rounding, the two logs would carry 18 eps each near the midpoint
  else {
    constexpr double fk = fact<K - 1>();
    if constexpr (K % 2 == 1) return fk * (ipow<K>(f.A) + ipow<K>(f.B));
    else return fk * f.D * f.A * f.B * bounded_geo<K>(f.A, f.B);
  }
}
template <int K>
LAPLACE_HD double bounded_inv(double x, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  const auto u = bounded_sig_of(x);
  if constexpr (K == 0) return p1 * u.q + p2 * u.p;
  else return (p2 - p1) * bounded_sig_dk<K>(u.U, u.D);
}
template <int K>
LAPLACE_HD double bounded_fwd_lj(double x, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  const auto f = bounded_fwd_of(x, p1, p2);
  if constexpr (K == 0) return m::log((f.s / f.p) / f.q);  // log s - log p - log q, without three logarithms of large size
  else {
    constexpr double fk = fact<K - 1>();
    if constexpr (K % 2 == 0) return fk * (ipow<K>(f.A) + ipow<K>(f.B));
    else return fk * f.D * f.A * f.B * bounded_geo<K>(f.A, f.B);
  }
}
template <int K>
LAPLACE_HD double bounded_inv_lj(double x, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  const auto u = bounded_sig_of(x);
  if constexpr (K == 0) return m::log(p2 - p1) - Kokkos::fabs(x) - 2.0 * Kokkos::log1p(u.e);
  else if constexpr (K == 1) return u.D;
  else return -2.0 * bounded_sig_dk<K - 1>(u.U, u.D);
}

// ---------------------------------------------------------------- partial derivatives

// index of (n1, n2) in bijector_partials<K> (scalar.hpp)
LAPLACE_HD_FORCEINLINE int bounded_index(int K, int n1, int n2) { return n2 * (K + 1) - n2 * (n2 - 1) / 2 + n1; }

template <int K>
LAPLACE_HD void bounded_fwd_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!bounded_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const auto f = bounded_fwd_of(x, p1, p2);
  constexpr double fk = fact<K - 1>();
  double pA[K + 1], pB[K + 1];
  pA[0] = 1.0;
  pB[0] = 1.0;
  for (int i = 1; i <= K; ++i) {
    pA[i] = pA[i - 1] * f.A;
    pB[i] = pB[i - 1] * f.B;
  }
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) {
      double r = 0.0;
      if (n1 == 0 && n2 == 0) r = bounded_fwd<K>(x, p1, p2);  // the pure derivative, in its own form
      else {
        if (n2 == 0) r += bounded_sign(n1 + K - 1) * fk * pA[K];
        if (n1 == 0) r += bounded_sign(n2) * fk * pB[K];
      }
      out[bounded_index(K, n1, n2)] = r;
    }
}
template <int K>
LAPLACE_HD void bounded_fwd_lj_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!bounded_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const auto f = bounded_fwd_of(x, p1, p2);
  constexpr double fk = fact<K - 1>();
  double pA[K + 1], pB[K + 1], pS[K + 1];
  pA[0] = 1.0;
  pB[0] = 1.0;
  pS[0] = 1.0;
  for (int i = 1; i <= K; ++i) {
    pA[i] = pA[i - 1] * f.A;
    pB[i] = pB[i - 1] * f.B;
    pS[i] = pS[i - 1] * f.S;
  }
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) {
      const int n0 = K - n1 - n2;
      double r = 0.0;
      if (n1 == 0 && n2 == 0) {
        r = bounded_fwd_lj<K>(x, p1, p2);
        out[bounded_index(K, n1, n2)] = r;
        continue;
      }
      // The two terms that can meet in one component cancel where they are close: at n0 = 0,
      // n1 = 0 the b-part is (-1)^K (K-1)! (B^K - S^K) with B - S = p S B exactly, and at
      // n0 = 0, n2 = 0 the a-part is (K-1)! (A^K - S^K) with A - S = q S A exactly. Each is
      // written with its exact factor (p, q are one subtraction each), so no difference of
      // two nearly equal powers is formed.
      if (n0 == 0 && n1 == 0) r = bounded_sign(K) * fk * (f.p * f.S * f.B) * bounded_geo<K>(f.B, f.S);
      else if (n0 == 0 && n2 == 0) r = fk * (f.q * f.S * f.A) * bounded_geo<K>(f.A, f.S);
      else {
        if (n2 == 0) r += bounded_sign(n0) * fk * pA[K];
        if (n1 == 0) r += bounded_sign(n2) * fk * pB[K];
        if (n0 == 0) r += bounded_sign(n1 + K - 1) * fk * pS[K];
      }
      out[bounded_index(K, n1, n2)] = r;
    }
}
template <int K>
LAPLACE_HD void bounded_inv_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!bounded_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const auto u = bounded_sig_of(x);
  const double s = p2 - p1;
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) {
      const int n0 = K - n1 - n2;
      double r = 0.0;
      if (n1 + n2 == 0) r = s * bounded_sig_dk<K>(u.U, u.D);
      else if (n1 + n2 == 1) {
        if (n0 == 0) r = n1 == 1 ? u.q : u.p;
        else {
          const double sg = bounded_sig_dk_rt(n0, u.U, u.D);
          r = n1 == 1 ? -sg : sg;
        }
      }
      out[bounded_index(K, n1, n2)] = r;
    }
}
template <int K>
LAPLACE_HD void bounded_inv_lj_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!bounded_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const auto u = bounded_sig_of(x);
  const double s = p2 - p1, S = 1.0 / s;
  constexpr double fk = fact<K - 1>();
  double pS[K + 1];
  pS[0] = 1.0;
  for (int i = 1; i <= K; ++i) pS[i] = pS[i - 1] * S;
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) {
      const int n0 = K - n1 - n2;
      double r = 0.0;
      if (n1 + n2 == 0) {
        if constexpr (K == 1) r = u.D;
        else r = -2.0 * bounded_sig_dk_rt(K - 1, u.U, u.D);
      } else if (n0 == 0) r = bounded_sign(n1 + K - 1) * fk * pS[K];
      out[bounded_index(K, n1, n2)] = r;
    }
}

// ---------------------------------------------------------------- domains and distances

LAPLACE_HD_FORCEINLINE double bounded_dom_lo(double p1, double p2) {
  return bounded_valid(p1, p2) ? p1 : nan();
}
LAPLACE_HD_FORCEINLINE double bounded_dom_hi(double p1, double p2) {
  return bounded_valid(p1, p2) ? p2 : nan();
}
LAPLACE_HD_FORCEINLINE double bounded_img_lo(double p1, double p2) {
  return bounded_valid(p1, p2) ? -inf() : nan();
}
LAPLACE_HD_FORCEINLINE double bounded_img_hi(double p1, double p2) {
  return bounded_valid(p1, p2) ? inf() : nan();
}
// b - h(u) = (b - a) expit(-u) and h(u) - a = (b - a) expit(u)
LAPLACE_HD_FORCEINLINE double bounded_inv_to_upper(double u, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  return (p2 - p1) * bounded_sig_of(u).q;
}
LAPLACE_HD_FORCEINLINE double bounded_inv_to_lower(double u, double p1, double p2) {
  if (!bounded_valid(p1, p2)) return nan();
  return (p2 - p1) * bounded_sig_of(u).p;
}

}  // namespace laplace::bijector::detail_scalar
