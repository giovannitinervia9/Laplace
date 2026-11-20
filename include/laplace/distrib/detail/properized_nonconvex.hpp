// What mcp_distrib and scad_distrib share (laplace::distrib::detail_properized):
// the pieces of the penalty q that the two laws have in common, and the tables of
// partial derivatives over multi-indices used for log Z. The properizing tail itself
// is a template parameter of the two laws and lives in distrib/properizing_tail.hpp.
//
// Both laws have the log-density
//
//   l(mu, lambda, p, theta_t; y) = -q(u; lambda, p, theta_t) - log Z(lambda, p, theta_t),
//   u = |y - mu|,  p = gamma (MCP) or a (SCAD),  theta_t the tail's own parameters,
//
// with q a polynomial of degree two in u on each inner piece, whose coefficients are
// rational in the parameters, and with the same shape of tail piece in both:
//
//   q = C(lambda, p) + H(u - K; theta_t),   K = p lambda the knot, C the value of q
//                                           there, H the tail's exponent.
//
// Two groups of functions live here.
//
// (a) One mixed partial of q at a time. The orders are (j, i, m, n1, n2) in
//     (u, lambda, p, theta_t); d^a/dmu^a is carried by the caller, which multiplies by
//     (-1)^a sign(y - mu)^(a + J) (the laws' logpdf_dy_dk). The inner pieces are
//     quadratic in u, so every component with j >= 3 is zero inside one of them; the
//     tail piece goes through q_tail_piece, which composes identity (1) of
//     properizing_tail.hpp with the tail's own components. The forms were derived and
//     checked against SymPy's own derivatives in
//     tools/reference/distrib_mcp_scad_forms.py and
//     tools/reference/properized_tail_forms.py, for every multi-index of total order
//     up to six and at points inside every piece.
//
// (b) Tables of partial derivatives over multi-indices, for log Z. In both laws
//
//       Z = 2 [ A(lambda, p) + B(lambda, p) T(theta_t) ],
//
//     so the tail's parameters enter through T only, and the table of Z over every
//     parameter is assembled from two tables over (lambda, p) and the mixed partials of
//     T. The tables are built by the Leibniz and chain-rule recursions over
//     multi-indices (mul2, exp2, comp2) and log Z by the recursion log3 for a tail with
//     one parameter or log4 for a tail with two, the same recursions enet_distrib uses
//     for its own normalizing constant. The finished log Z table was checked against
//     SymPy's derivatives of Z written with erfi, F(x) = sqrt(pi)/2 e^(-x^2) erfi(x),
//     in the first script named above.
//
// The tables are local arrays of (K + 1)^2 doubles over (lambda, p) and of
// (K+1)(K+2)(K+3)/6 or (K+1)(K+2)(K+3)(K+4)/24 doubles for log Z, so an order-six call
// of logpdf_dk holds a few kilobytes of them; at order one or two, which is what a fit
// reads, they are a few hundred bytes.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/properizing_tail.hpp>
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

// One mixed partial of the tail piece q = C(lambda, p) + H(u - p lambda; theta_t), the
// constant C excluded: identity (1) of properizing_tail.hpp carries the knot, and the
// tail supplies the components of H. The derivatives in theta_t commute through, the
// knot being free of them.
template <class Tail>
LAPLACE_HD double q_tail_piece(int j, int i, int m, int n1, int n2, double u, double lambda, double p,
                               const Tail& tail) {
  const double x = u - p * lambda;
  const int lo = i > m ? i : m;
  double s = 0.0;
  for (int r = lo; r <= i + m; ++r) {
    const double c = detail_tail::x_chain_coef(r, i, m, lambda, p);
    if (c != 0.0) s += c * detail_tail::h_mixed(tail, j + r, n1, n2, x);
  }
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
// The same ranking over four variables, for a tail with two parameters of its own: the
// blocks of equal total order in turn, and inside a block the colex rank, which for
// three variables is idx3 above. At order six it holds 210 entries against the 2401 of
// the square box.
LAPLACE_HD_FORCEINLINE constexpr int idx4(int i, int m, int n, int o) {
  const int t = i + m + n + o;
  int pos = binomial_coef(t + 3, 4);
  for (int j = 0; j < i; ++j) pos += binomial_coef(t - j + 2, 2);
  for (int j = 0; j < m; ++j) pos += binomial_coef(t - i - j + 1, 1);
  return pos + n;
}
LAPLACE_HD constexpr int tab2_size(int K) { return (K + 1) * (K + 1); }
LAPLACE_HD constexpr int tab3_size(int K) { return (K + 1) * (K + 2) * (K + 3) / 6; }
LAPLACE_HD constexpr int tab4_size(int K) { return (K + 1) * (K + 2) * (K + 3) * (K + 4) / 24; }
// the size of the log Z table: one slot per parameter of the law except the location
LAPLACE_HD constexpr int z_tab_size(int K, int n_tail_params) {
  return n_tail_params == 1 ? tab3_size(K) : tab4_size(K);
}

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

// the table of log(F) over four variables from the table of F (F at 0 positive), the
// same recursion as log3 with one index more
template <int K>
LAPLACE_HD void log4(const double* F, double* U) {
  const double f0 = F[idx4(0, 0, 0, 0)];
  U[idx4(0, 0, 0, 0)] = math::log(f0);
  for (int tot = 1; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i)
      for (int m = 0; i + m <= tot; ++m)
        for (int n = 0; i + m + n <= tot; ++n) {
          const int o = tot - i - m - n;
          int gi = i, gm = m, gn = n, go = o, r = 3;
          if (i > 0) {
            gi = i - 1;
            r = 0;
          } else if (m > 0) {
            gm = m - 1;
            r = 1;
          } else if (n > 0) {
            gn = n - 1;
            r = 2;
          } else {
            go = o - 1;
          }
          double s = F[idx4(i, m, n, o)];
          for (int bi = 0; bi <= gi; ++bi)
            for (int bm = 0; bm <= gm; ++bm)
              for (int bn = 0; bn <= gn; ++bn)
                for (int bo = 0; bo <= go; ++bo) {
                  if (bi + bm + bn + bo == 0) continue;
                  const int di = gi - bi + (r == 0 ? 1 : 0), dm = gm - bm + (r == 1 ? 1 : 0),
                            dn = gn - bn + (r == 2 ? 1 : 0), dd = go - bo + (r == 3 ? 1 : 0);
                  s -= binomial_coef(gi, bi) * binomial_coef(gm, bm) * binomial_coef(gn, bn) *
                       binomial_coef(go, bo) * F[idx4(bi, bm, bn, bo)] * U[idx4(di, dm, dn, dd)];
                }
          U[idx4(i, m, n, o)] = s / f0;
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

// N_k = int_0^(p lambda) t^k exp(-lambda t + t^2/(2 p)) dt for k = 0, 1, 2: the moments
// of the law's quadratic piece in the distance from its upper end, where the exponent's
// derivative vanishes (p = gamma for MCP, a - 1 for SCAD). With x = lambda sqrt(p/2),
// w = p lambda^2 = 2 x^2 and E = e^(-w/2),
//
//   N_0 = sqrt(2 p) F(x),  N_1 = p [E - F'(x)],  N_2 = 2 p^2 lambda E + p sqrt(p/2) F''(x),
//
// the last two by differentiating N_0 in lambda, whose upper end moves with it. Every
// term there is positive, so nothing cancels however large w grows, while the plain
// forms lambda p N_0 - p (1 - E) and (p^2 lambda^2 - p) N_0 + p^2 lambda (2 E - 1) lose
// about one digit per decade of w (7e-5 relative at w = 1e6, measured). Below w = 1
// those forms cancel the other way, by eps/w, and the double series in w serves instead:
//
//   N_k = (p lambda)^(k+1) sum_(m, n) (w/2)^m (-w)^n / [m! n! (k + 2m + n + 1)],
//
// from expanding exp(t^2/2p) and exp(-lambda t) on t = p lambda s with s in (0, 1).
// Measured against mpmath: 1e-16 to 5e-16 below w = 1, 1e-16 to 1.6e-15 above.
LAPLACE_HD void dawson_moments(double lambda, double p, double* N) {
  const double w = p * lambda * lambda, root = Kokkos::sqrt(0.5 * p);
  const double x = lambda * root;
  N[0] = 1.41421356237309504880 * Kokkos::sqrt(p) * math::dawson(x);
  if (w >= 1.0) {
    const double E = math::exp(-0.5 * w);
    N[1] = p * (E - math::dawson_d1(x));
    N[2] = 2.0 * p * p * lambda * E + p * root * math::dawson_d2(x);
    return;
  }
  const double knot = p * lambda;
  for (int k = 1; k <= 2; ++k) {
    double s = 0.0, tm = 1.0;
    for (int m = 0; m < 24; ++m) {
      double inner = 0.0, tn = 1.0;
      for (int n = 0; n < 24; ++n) {
        inner += tn / double(k + 2 * m + n + 1);
        tn *= -w / double(n + 1);
        if (Kokkos::fabs(tn) < 1.0e-22) break;
      }
      s += tm * inner;
      tm *= 0.5 * w / double(m + 1);
      if (Kokkos::fabs(tm) < 1.0e-22) break;
    }
    double kk = knot;
    for (int t = 0; t < k; ++t) kk *= knot;
    N[k] = s * kk;
  }
}

// The table over (lambda, p, theta_t...) of log Z from the two tables over (lambda, p)
// of A and B in Z = 2 [A + B T], T the tail's constant: T carries every derivative in
// the tail's own parameters, so the table over all of them is assembled from A, B and
// the mixed partials of T, and the logarithm follows by log3 or log4.
template <int K, class Tail>
LAPLACE_HD void log_z_table(const double* A, const double* B, const Tail& tail, double* out) {
  if constexpr (Tail::n_tail_params == 1) {
    double Z[tab3_size(K)];
    for (int i = 0; i <= K; ++i)
      for (int m = 0; i + m <= K; ++m)
        for (int n = 0; i + m + n <= K; ++n) {
          double v = 2.0 * B[idx2<K>(i, m)] * tail.const_mixed(n, 0);
          if (n == 0) v += 2.0 * A[idx2<K>(i, m)];
          Z[idx3(i, m, n)] = v;
        }
    log3<K>(Z, out);
  } else {
    double Z[tab4_size(K)];
    for (int i = 0; i <= K; ++i)
      for (int m = 0; i + m <= K; ++m)
        for (int n = 0; i + m + n <= K; ++n)
          for (int o = 0; i + m + n + o <= K; ++o) {
            double v = 2.0 * B[idx2<K>(i, m)] * tail.const_mixed(n, o);
            if (n == 0 && o == 0) v += 2.0 * A[idx2<K>(i, m)];
            Z[idx4(i, m, n, o)] = v;
          }
    log4<K>(Z, out);
  }
}

// the position of a component of the log Z table, whichever of the two packings it uses
template <class Tail>
LAPLACE_HD_FORCEINLINE int z_tab_index(int b, int m, int n1, int n2) {
  if constexpr (Tail::n_tail_params == 1) return idx3(b, m, n1);
  else return idx4(b, m, n1, n2);
}

}  // namespace laplace::distrib::detail_properized
