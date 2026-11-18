// What mcp_distrib and scad_distrib share (laplace::distrib::detail_properized):
// the pieces of the penalty q that the two laws have in common, the tables of
// partial derivatives over multi-indices used for log Z, and the Gaussian tail.
//
// Both laws have the log-density
//
//   l(mu, lambda, p, V; y) = -q(u; lambda, p, V) - log Z(lambda, p, V),
//   u = |y - mu|,  p = gamma (MCP) or a (SCAD),  V the variance of the Gaussian tail,
//
// with q a polynomial of degree two in u on each piece, whose coefficients are
// rational in the parameters, and with the same tail piece in both:
//
//   q = C(lambda, p) + (u - K)^2 / (2 V),   K = p lambda the knot, C the value of q there.
//
// Two groups of functions live here.
//
// (a) One mixed partial of q at a time. The orders are (j, i, m, n) in
//     (u, lambda, p, V); d^a/dmu^a is carried by the caller, which multiplies by
//     (-1)^a sign(y - mu)^(a + J) (the laws' logpdf_dy_dk). q is quadratic in u, so
//     every component with j >= 3 is zero inside a piece. The forms were derived and
//     checked against SymPy's own derivatives in
//     tools/reference/distrib_mcp_scad_forms.py, for every (j, i, m, n) of total
//     order up to six and at points inside every piece.
//
// (b) Tables of partial derivatives over multi-indices, for log Z. In both laws
//
//       Z = 2 [ A(lambda, p) + B(lambda, p) sqrt(pi V / 2) ],
//
//     so V enters through one factor only and the table of Z over the three
//     parameters is assembled from two tables over (lambda, p) and the derivatives
//     of V^(1/2). The tables are built by the Leibniz and chain-rule recursions over
//     multi-indices (mul2, exp2, comp2) and log Z by the recursion log3, the same
//     recursions enet_distrib uses for its own normalizing constant. The finished
//     log Z table was checked against SymPy's derivatives of Z written with erfi,
//     F(x) = sqrt(pi)/2 e^(-x^2) erfi(x), in the script named above: worst relative
//     error 1.2e-13 over the five test points at total order up to six, rising to
//     3e-9 for the sixth pure derivative in p at lambda = 0.01, where that
//     derivative is itself 2.7e-11 and the absolute error is 8e-20.
//
// The tables are local arrays of (K + 1)^2 or (K + 1)^3 doubles, so an order-six
// call of logpdf_dk holds about ten kilobytes of them; at order one or two, which
// is what a fit reads, they are a few hundred bytes.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/dawson.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib::detail_properized {

// ---------------------------------------------------------------------------
// elementary pieces
// ---------------------------------------------------------------------------

// d^j/dx^j of x^2
LAPLACE_HD_FORCEINLINE double d2_pow(int j, double x) {
  return j == 0 ? x * x : (j == 1 ? 2.0 * x : (j == 2 ? 2.0 : 0.0));
}

// d^j/dx^j of x
LAPLACE_HD_FORCEINLINE double d1_pow(int j, double x) { return j == 0 ? x : (j == 1 ? 1.0 : 0.0); }

// d^m/dr^m of 1/r = (-1)^m m! / r^(m + 1)
LAPLACE_HD_FORCEINLINE double inv_deriv(int m, double r) {
  double f = 1.0;
  for (int i = 2; i <= m; ++i) f *= double(i);
  double p = r;
  for (int i = 0; i < m; ++i) p *= r;
  return (m % 2 == 0 ? f : -f) / p;
}

// d^m/dr^m of r^(1/2) = c_m r^(1/2 - m), c_m = prod_(t < m) (1/2 - t)
LAPLACE_HD_FORCEINLINE double sqrt_deriv(int m, double r) {
  double c = 1.0;
  for (int t = 0; t < m; ++t) c *= 0.5 - double(t);
  double v = Kokkos::sqrt(r);
  for (int t = 0; t < m; ++t) v /= r;
  return c * v;
}

// log(e^a + e^b), either argument -infinity allowed
LAPLACE_HD_FORCEINLINE double log_add(double a, double b) {
  if (a < b) {
    const double t = a;
    a = b;
    b = t;
  }
  if (!(b > -distrib_infinity)) return a;  // b = -inf (or NaN, which propagates through a)
  return a + Kokkos::log1p(math::exp(b - a));
}

// ---------------------------------------------------------------------------
// (a) the pieces of q shared by the two laws
// ---------------------------------------------------------------------------

// The factor that d^a/dmu^a d^J/dy^J contribute to a derivative of q(|y - mu|):
// (-1)^a sign(y - mu)^(a + J), where sign(y - mu)^t is read as 1 for even t and as
// s = sign(y - mu) with sign(0) = 0 for odd t. At the kink the odd orders are the
// mean of the two one-sided values, which is zero, and the even orders agree on the
// two sides, so the one-sided value of the inner piece is the derivative there.
LAPLACE_HD_FORCEINLINE double u_chain_factor(int a, int t, double s) {
  const double st = (t % 2 == 0) ? 1.0 : s;
  return (a % 2 == 0) ? st : -st;
}

// d^j_u d^i_lambda of lambda u
LAPLACE_HD_FORCEINLINE double q_rate(int j, int i, double u, double lambda) {
  return d1_pow(j, u) * d1_pow(i, lambda);
}

// d^j_u d^i_lambda d^m_p of (u - p lambda)^2 / 2, the numerator of both tail pieces
LAPLACE_HD double q_tail_square(int j, int i, int m, double u, double lambda, double p) {
  double s = 0.0;
  if (i == 0 && m == 0) s += 0.5 * d2_pow(j, u);
  if (j <= 1 && i <= 1 && m <= 1) s -= d1_pow(j, u) * d1_pow(i, lambda) * d1_pow(m, p);
  if (j == 0) s += 0.5 * d2_pow(i, lambda) * d2_pow(m, p);
  return s;
}

// ---------------------------------------------------------------------------
// (b) tables of partial derivatives over multi-indices
// ---------------------------------------------------------------------------

template <int K>
LAPLACE_HD_FORCEINLINE constexpr int idx2(int i, int m) {
  return i * (K + 1) + m;
}
// A table over three variables is packed triangularly, over the multi-indices of
// total order at most K only: the blocks of equal total order t in turn, and inside
// a block the pairs (i, m) with i + m <= t, i-major. The square packing of the
// two-variable tables wastes little, while at order six the square cube would hold
// 343 entries against the 84 that exist, and the Gaussian tail of these two laws
// already asks a CUDA thread for more stack than the tests give it by default.
LAPLACE_HD_FORCEINLINE constexpr int idx3(int i, int m, int n) {
  const int t = i + m + n;
  return t * (t + 1) * (t + 2) / 6 + i * (t + 1) - i * (i - 1) / 2 + m;
}
LAPLACE_HD constexpr int tab2_size(int K) { return (K + 1) * (K + 1); }
LAPLACE_HD constexpr int tab3_size(int K) { return (K + 1) * (K + 2) * (K + 3) / 6; }

template <int K>
LAPLACE_HD void zero2(double* t) {
  for (int q = 0; q < tab2_size(K); ++q) t[q] = 0.0;
}

// Leibniz product of two tables over two variables; h must alias neither f nor g
template <int K>
LAPLACE_HD void mul2(const double* f, const double* g, double* h) {
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m) {
      double s = 0.0;
      for (int a = 0; a <= i; ++a)
        for (int b = 0; b <= m; ++b)
          s += binomial_coef(i, a) * binomial_coef(m, b) * f[idx2<K>(a, b)] * g[idx2<K>(i - a, m - b)];
      h[idx2<K>(i, m)] = s;
    }
}

// the table of exp(phi) from the table of phi: with alpha = gamma + e_r,
// D^alpha e^phi = sum_(beta <= gamma) C(gamma, beta) D^beta e^phi D^(gamma - beta + e_r) phi
template <int K>
LAPLACE_HD void exp2(const double* phi, double* e) {
  zero2<K>(e);
  e[idx2<K>(0, 0)] = math::exp(phi[idx2<K>(0, 0)]);
  for (int tot = 1; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i) {
      const int m = tot - i;
      const int gi = i > 0 ? i - 1 : i, gm = i > 0 ? m : m - 1, r = i > 0 ? 0 : 1;
      double s = 0.0;
      for (int bi = 0; bi <= gi; ++bi)
        for (int bm = 0; bm <= gm; ++bm) {
          const int di = gi - bi + (r == 0 ? 1 : 0), dm = gm - bm + (r == 1 ? 1 : 0);
          s += binomial_coef(gi, bi) * binomial_coef(gm, bm) * e[idx2<K>(bi, bm)] * phi[idx2<K>(di, dm)];
        }
      e[idx2<K>(i, m)] = s;
    }
}

// the table of f(x) from the values fd[k] = f^(k)(x) at the point and the table of x.
// The tables of f^(k)(x) are built downward from k = K, each to total order K - k:
// D^(gamma + e_r) f^(k)(x) = sum_(beta <= gamma) C(gamma, beta) D^beta f^(k+1)(x)
// D^(gamma - beta + e_r) x
// Two buffers are enough, the step from k + 1 to k being the only one that reads
// the table above: they alternate, and the one that holds order 0 at the end is
// copied out.
template <int K>
LAPLACE_HD void comp2(const double* fd, const double* x, double* out) {
  double buf[2 * tab2_size(K)];
  int cur = 0;  // buf + cur * tab2_size(K) holds the table of f^(k)(x)
  zero2<K>(buf);
  buf[idx2<K>(0, 0)] = fd[K];
  for (int k = K - 1; k >= 0; --k) {
    const double* hi = buf + cur * tab2_size(K);
    cur = 1 - cur;
    double* lo = buf + cur * tab2_size(K);
    zero2<K>(lo);
    lo[idx2<K>(0, 0)] = fd[k];
    for (int tot = 1; tot <= K - k; ++tot)
      for (int i = 0; i <= tot; ++i) {
        const int m = tot - i;
        const int gi = i > 0 ? i - 1 : i, gm = i > 0 ? m : m - 1, r = i > 0 ? 0 : 1;
        double s = 0.0;
        for (int bi = 0; bi <= gi; ++bi)
          for (int bm = 0; bm <= gm; ++bm) {
            const int di = gi - bi + (r == 0 ? 1 : 0), dm = gm - bm + (r == 1 ? 1 : 0);
            s += binomial_coef(gi, bi) * binomial_coef(gm, bm) * hi[idx2<K>(bi, bm)] * x[idx2<K>(di, dm)];
          }
        lo[idx2<K>(i, m)] = s;
      }
  }
  const double* done = buf + cur * tab2_size(K);
  for (int q = 0; q < tab2_size(K); ++q) out[q] = done[q];
}

// the table of log(F) over three variables from the table of F (F at 0 positive)
template <int K>
LAPLACE_HD void log3(const double* F, double* U) {
  const double f0 = F[idx3(0, 0, 0)];
  U[idx3(0, 0, 0)] = math::log(f0);
  for (int tot = 1; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i)
      for (int m = 0; i + m <= tot; ++m) {
        const int n = tot - i - m;
        int gi = i, gm = m, gn = n, r = 2;
        if (i > 0) {
          gi = i - 1;
          r = 0;
        } else if (m > 0) {
          gm = m - 1;
          r = 1;
        } else {
          gn = n - 1;
        }
        double s = F[idx3(i, m, n)];
        for (int bi = 0; bi <= gi; ++bi)
          for (int bm = 0; bm <= gm; ++bm)
            for (int bn = 0; bn <= gn; ++bn) {
              if (bi + bm + bn == 0) continue;
              const int di = gi - bi + (r == 0 ? 1 : 0), dm = gm - bm + (r == 1 ? 1 : 0),
                        dn = gn - bn + (r == 2 ? 1 : 0);
              s -= binomial_coef(gi, bi) * binomial_coef(gm, bm) * binomial_coef(gn, bn) *
                   F[idx3(bi, bm, bn)] * U[idx3(di, dm, dn)];
            }
        U[idx3(i, m, n)] = s / f0;
      }
}

// F^(k)(x), k = 0..K, of Dawson's integral
template <int K>
LAPLACE_HD void dawson_derivs(double x, double* fd) {
  fd[0] = math::dawson(x);
  if constexpr (K >= 1) fd[1] = math::dawson_d1(x);
  if constexpr (K >= 2) fd[2] = math::dawson_d2(x);
  if constexpr (K >= 3) fd[3] = math::dawson_d3(x);
  if constexpr (K >= 4) fd[4] = math::dawson_d4(x);
  if constexpr (K >= 5) fd[5] = math::dawson_d5(x);
  if constexpr (K >= 6) fd[6] = math::dawson_d6(x);
}

// the table over (lambda, p) of sqrt(2 r) F(lambda sqrt(r / 2)), r = gamma (MCP) or
// a - 1 (SCAD); dr/dp = 1 in both, so the derivatives in p are those in r
template <int K>
LAPLACE_HD void dawson_term(double lambda, double r, double* out) {
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  constexpr double sqrt2 = 1.41421356237309504880;
  double x[tab2_size(K)], sq[tab2_size(K)], g[tab2_size(K)];
  zero2<K>(x);
  zero2<K>(sq);
  for (int m = 0; m <= K; ++m) {
    const double h = inv_sqrt2 * sqrt_deriv(m, r);
    x[idx2<K>(0, m)] = lambda * h;
    if constexpr (K >= 1) x[idx2<K>(1, m)] = h;
    sq[idx2<K>(0, m)] = sqrt2 * sqrt_deriv(m, r);
  }
  double fd[K + 1];
  dawson_derivs<K>(x[idx2<K>(0, 0)], fd);
  comp2<K>(fd, x, g);
  mul2<K>(sq, g, out);
}

// the table over (lambda, p, V) of log Z from the two tables over (lambda, p) of
// A and B in Z = 2 [A + B sqrt(pi V / 2)]
template <int K>
LAPLACE_HD void log_z_table(const double* A, const double* B, double V, double* out) {
  constexpr double root_half_pi = 1.25331413731550025121;  // sqrt(pi / 2)
  double Z[tab3_size(K)];
  for (int q = 0; q < tab3_size(K); ++q) Z[q] = 0.0;
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m)
      for (int n = 0; i + m + n <= K; ++n) {
        double v = 2.0 * root_half_pi * B[idx2<K>(i, m)] * sqrt_deriv(n, V);
        if (n == 0) v += 2.0 * A[idx2<K>(i, m)];
        Z[idx3(i, m, n)] = v;
      }
  log3<K>(Z, out);
}

// ---------------------------------------------------------------------------
// the Gaussian tail, shared by the two laws
// ---------------------------------------------------------------------------

// log of the whole tail integral int_K^inf exp(-q) du = e^(-C) sqrt(pi V / 2),
// C the value of q at the knot; kept in logs, since e^(-C) underflows for a large knot
LAPLACE_HD_FORCEINLINE double log_tail_integral(double C, double V) {
  constexpr double log_root_half_pi = 0.22579135264472743236;  // log sqrt(pi / 2)
  return -C + log_root_half_pi + 0.5 * math::log(V);
}

// int_u^inf exp(-q) du beyond the knot, w = u - knot >= 0:
// 2 e^(-C) sqrt(pi V / 2) Phi(-w / sqrt(V))
LAPLACE_HD_FORCEINLINE double tail_upper(double w, double C, double V) {
  return 2.0 * math::exp(-C) * 1.25331413731550025121 * Kokkos::sqrt(V) * math::Phi(-w / Kokkos::sqrt(V));
}

LAPLACE_HD_FORCEINLINE double log_tail_upper(double w, double C, double V) {
  return 0.69314718055994530942 + log_tail_integral(C, V) + math::log_Phi(-w / Kokkos::sqrt(V));
}

}  // namespace laplace::distrib::detail_properized
