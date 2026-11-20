// The derivatives of the expected information of mcp_distrib and scad_distrib
// (laplace::distrib::detail_properized_info), orders one to four.
//
// The information of both laws does not depend on the location: it is built from Z, from
// the moments of the inner pieces and from the tail's own integrals, none of which sees
// mu. So every derivative component that carries mu vanishes, and what is left is a table
// over the smooth parameters
//
//   (lambda, p, theta_1, theta_2),   p = gamma (MCP) or a (SCAD),
//
// packed as the log Z table is, by idx3 for a tail with one parameter of its own and by
// idx4 for a tail with two (detail/properized_nonconvex.hpp). A "smooth table" below is
// one of those, holding every multi-index of total order at most K.
//
// What makes the tables cheap is that no factor of the closed form depends on both halves
// of that list. The moments N_0, N_1, N_2 and the two tables A and B of Z = 2[A + B T] are
// functions of (lambda, p) alone; the tail constant T and the tail's three integrals are
// functions of theta alone. A product of two such factors is an outer product, which
// costs one multiplication a slot and no Leibniz sum at all; the Leibniz product st_mul
// and the reciprocal st_recip are needed only where a (lambda, p) factor meets a theta
// factor already mixed, which happens once per entry, in num_ab / Z2.
//
// The derivatives of d2_ab log Z are not computed here: the log Z table of order K + 2,
// which the laws already build for their log-density, holds them at the multi-index
// C + (a, b).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/detail/properized_nonconvex.hpp>
#include <laplace/math/dawson.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib::detail_properized_info {

using namespace detail_properized;

// ---------------------------------------------------------------------------
// the smooth tables: NT is the number of the tail's own parameters
// ---------------------------------------------------------------------------

template <int K, int NT>
LAPLACE_HD constexpr int st_size() {
  return NT == 1 ? tab3_size(K) : tab4_size(K);
}

template <int K, int NT>
LAPLACE_HD_FORCEINLINE int st_idx(int i, int m, int n, int o) {
  if constexpr (NT == 1) return idx3(i, m, n);
  else return idx4(i, m, n, o);
}

template <int K, int NT>
LAPLACE_HD void st_zero(double* t) {
  for (int q = 0; q < st_size<K, NT>(); ++q) t[q] = 0.0;
}

// h = f g, the Leibniz rule over the four indices
template <int K, int NT>
LAPLACE_HD void st_mul(const double* f, const double* g, double* h) {
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m)
      for (int n = 0; i + m + n <= K; ++n)
        for (int o = 0; i + m + n + o <= K; ++o) {
          if (NT == 1 && o > 0) continue;
          double s = 0.0;
          for (int a = 0; a <= i; ++a)
            for (int b = 0; b <= m; ++b)
              for (int c = 0; c <= n; ++c)
                for (int d = 0; d <= o; ++d)
                  s += double(binomial_coef(i, a)) * double(binomial_coef(m, b)) *
                       double(binomial_coef(n, c)) * double(binomial_coef(o, d)) *
                       f[st_idx<K, NT>(a, b, c, d)] * g[st_idx<K, NT>(i - a, m - b, n - c, o - d)];
          h[st_idx<K, NT>(i, m, n, o)] = s;
        }
}

// h = 1/f, from h f = 1: h_C = [C = 0 - sum_(0 < B <= C) C(C, B) f_B h_(C-B)] / f_0
template <int K, int NT>
LAPLACE_HD void st_recip(const double* f, double* h) {
  const double f0 = f[st_idx<K, NT>(0, 0, 0, 0)];
  for (int tot = 0; tot <= K; ++tot)
    for (int i = 0; i <= tot; ++i)
      for (int m = 0; i + m <= tot; ++m)
        for (int n = 0; i + m + n <= tot; ++n) {
          const int o = tot - i - m - n;
          if (NT == 1 && o > 0) continue;
          double s = (tot == 0) ? 1.0 : 0.0;
          for (int a = 0; a <= i; ++a)
            for (int b = 0; b <= m; ++b)
              for (int c = 0; c <= n; ++c)
                for (int d = 0; d <= o; ++d) {
                  if (a + b + c + d == 0) continue;
                  s -= double(binomial_coef(i, a)) * double(binomial_coef(m, b)) *
                       double(binomial_coef(n, c)) * double(binomial_coef(o, d)) *
                       f[st_idx<K, NT>(a, b, c, d)] *
                       h[st_idx<K, NT>(i - a, m - b, n - c, o - d)];
                }
          h[st_idx<K, NT>(i, m, n, o)] = s / f0;
        }
}

// h += c f(lambda, p) g(theta), with f a square two-variable table (idx2) and g a table
// over the tail's parameters alone (idx2 as well, its second index unused for NT = 1)
template <int K, int NT>
LAPLACE_HD void st_add_outer(double* h, double c, const double* f, const double* g) {
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m)
      for (int n = 0; i + m + n <= K; ++n)
        for (int o = 0; i + m + n + o <= K; ++o) {
          if (NT == 1 && o > 0) continue;
          h[st_idx<K, NT>(i, m, n, o)] += c * f[idx2<K>(i, m)] * g[idx2<K>(n, o)];
        }
}

// h += c f(lambda, p), the theta derivatives of a factor that does not see theta
template <int K, int NT>
LAPLACE_HD void st_add_lp(double* h, double c, const double* f) {
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m) h[st_idx<K, NT>(i, m, 0, 0)] += c * f[idx2<K>(i, m)];
}

// ---------------------------------------------------------------------------
// the tables over (lambda, p) that the information reads
// ---------------------------------------------------------------------------

// The table of lambda^beta p^alpha for any real exponents: the falling factorials make
// every slot past a whole exponent vanish on their own, so one function serves the
// monomials of the numerators and the negative powers of p that the information divides
// by.
template <int K>
LAPLACE_HD void monomial2(double lambda, double p, double beta, double alpha, double* out) {
  zero2<K>(out);
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m) {
      double c = 1.0;
      for (int t = 0; t < i; ++t) c *= beta - double(t);
      for (int t = 0; t < m; ++t) c *= alpha - double(t);
      if (c == 0.0) continue;
      out[idx2<K>(i, m)] =
          c * Kokkos::pow(lambda, beta - double(i)) * Kokkos::pow(p, alpha - double(m));
    }
}

// the table of e^(-p lambda^2 / 2)
template <int K>
LAPLACE_HD void exp_half(double lambda, double p, double* out) {
  double phi[tab2_size(K)];
  zero2<K>(phi);
  phi[idx2<K>(0, 0)] = -0.5 * p * lambda * lambda;
  if constexpr (K >= 1) {
    phi[idx2<K>(1, 0)] = -p * lambda;
    phi[idx2<K>(0, 1)] = -0.5 * lambda * lambda;
  }
  if constexpr (K >= 2) {
    phi[idx2<K>(2, 0)] = -p;
    phi[idx2<K>(1, 1)] = -lambda;
  }
  if constexpr (K >= 3) phi[idx2<K>(2, 1)] = -1.0;
  exp2<K>(phi, out);
}

// The tables of the moments of the quadratic piece, N_k = int_0^(p lambda) t^k
// exp(-lambda t + t^2/(2p)) dt for k = 0, 1, 2, as functions of (lambda, p).
//
// N_0 is the Dawson term the normalizing constant already holds. N_1 = p[E - F'(x)] and
// N_2 = 2 p^2 lambda E + p sqrt(p/2) F''(x) with x = lambda sqrt(p/2) and E = e^(-w/2),
// w = p lambda^2, the forms of dawson_moments: their tables follow from the same
// composition that dawson_term uses, with the derivative array of F shifted by one and by
// two. Below w = 1 the double series serves instead, which is a polynomial in lambda and
// p, so its table is exact term by term.
template <int K>
LAPLACE_HD void moment_tables(double lambda, double p, double* N0, double* N1, double* N2) {
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  constexpr double sqrt2 = 1.41421356237309504880;
  const double w = p * lambda * lambda;
  dawson_term<K>(lambda, p, N0);
  if (w < 1.0) {
    zero2<K>(N1);
    zero2<K>(N2);
    double term[tab2_size(K)];
    for (int k = 1; k <= 2; ++k) {
      double* out = (k == 1) ? N1 : N2;
      // the coefficients carry no power of w: the monomial
      // p^(k+1+j) lambda^(k+1+2j), j = m + n, is (p lambda)^(k+1) w^j itself. The running
      // powers of w serve only to stop the two sums where a term cannot move them.
      double cm = 1.0, wm = 1.0;  // (1/2)^m/m! and (w/2)^m/m!
      for (int m = 0; m < 24; ++m) {
        double cn = 1.0, wn = 1.0;  // (-1)^n/n! and (-w)^n/n!
        for (int n = 0; n < 24; ++n) {
          const int j = m + n;
          monomial2<K>(lambda, p, k + 1 + 2 * j, k + 1 + j, term);
          const double c = cm * cn / double(k + 2 * m + n + 1);
          for (int q = 0; q < tab2_size(K); ++q) out[q] += c * term[q];
          cn /= -double(n + 1);
          wn *= -w / double(n + 1);
          if (Kokkos::fabs(wn) < 1.0e-22) break;
        }
        cm *= 0.5 / double(m + 1);
        wm *= 0.5 * w / double(m + 1);
        if (Kokkos::fabs(wm) < 1.0e-22) break;
      }
    }
    return;
  }
  // x = lambda sqrt(p/2) and the tables of F'(x) and F''(x)
  double x[tab2_size(K)], sq[tab2_size(K)], g[tab2_size(K)], e[tab2_size(K)], t1[tab2_size(K)];
  zero2<K>(x);
  for (int m = 0; m <= K; ++m) {
    const double h = inv_sqrt2 * sqrt_deriv(m, p);
    x[idx2<K>(0, m)] = lambda * h;
    if constexpr (K >= 1) x[idx2<K>(1, m)] = h;
  }
  double fd[K + 3];
  dawson_derivs<K + 2>(x[idx2<K>(0, 0)], fd);
  double fd1[K + 1], fd2[K + 1];
  for (int k = 0; k <= K; ++k) fd1[k] = fd[k + 1], fd2[k] = fd[k + 2];
  exp_half<K>(lambda, p, e);
  // N_1 = p E - p F'(x)
  comp2<K>(fd1, x, g);
  monomial2<K>(lambda, p, 0, 1, t1);  // p
  double pe[tab2_size(K)], pg[tab2_size(K)];
  mul2<K>(t1, e, pe);
  mul2<K>(t1, g, pg);
  for (int q = 0; q < tab2_size(K); ++q) N1[q] = pe[q] - pg[q];
  // N_2 = 2 p^2 lambda E + p sqrt(p/2) F''(x)
  comp2<K>(fd2, x, g);
  monomial2<K>(lambda, p, 1, 2, t1);  // lambda p^2
  double a[tab2_size(K)];
  mul2<K>(t1, e, a);
  // d^m of p sqrt(p/2) = (1/sqrt 2) d^m of p^(3/2) = (1/sqrt 2) (3/2)(1/2)... p^(3/2-m)
  zero2<K>(sq);
  for (int m = 0; m <= K; ++m) {
    double c = inv_sqrt2;
    for (int t = 0; t < m; ++t) c *= 1.5 - double(t);
    double v = c * p * Kokkos::sqrt(p);
    for (int t = 0; t < m; ++t) v /= p;
    sq[idx2<K>(0, m)] = v;
  }
  mul2<K>(sq, g, pg);
  for (int q = 0; q < tab2_size(K); ++q) N2[q] = 2.0 * a[q] + pg[q];
}

// ---------------------------------------------------------------------------
// the assembly
// ---------------------------------------------------------------------------

// the next sorted multi-index of length K over P parameters, in lexicographic order
template <int P, int K>
LAPLACE_HD_FORCEINLINE void next_sorted(int* c) {
  int i = K - 1;
  while (i >= 0 && c[i] == P - 1) --i;
  if (i < 0) return;
  ++c[i];
  for (int j = i + 1; j < K; ++j) c[j] = c[i];
}

// The derivatives of the expected information from the pieces the two laws share.
//
// Builder supplies, for one pair (i, j) of smooth parameter indices, the smooth table of
// int_0^inf d2_ij q e^-q du (build_num), and the smooth table of the numerator of
// I_mu_mu, which is lambda plus the integral of q'' (build_imm). Both are built one at a
// time so that no more than a few hundred doubles of table live at once: at order four a
// smooth table of a two-parameter tail holds 70 of them, the log Z table of order six
// 210.
//
// The location gives I_mu_mu = imm / Z2 and I_mu_theta = 0; the smooth block gives
// num_ij / Z2 + d2_ij log Z, the second term read from the log Z table of order K + 2 at
// the multi-index of the component plus one for each of the two parameters of the pair.
// Every component whose multi-index carries mu is zero, the information not depending on
// the location.
// one pair's components from its smooth table, plus the entry of the log Z table when the
// pair has one (the location's has none); a component whose multi-index carries mu is zero
template <int K, int P, int NT, class Tail>
LAPLACE_HD void fill_pair(sym_tensor<P, K>& dst, const double* t, const double* LZ,
                          const int* add) {
  int c[K] = {};
  for (int pos = 0; pos < sym_tensor<P, K>::size; ++pos, next_sorted<P, K>(c)) {
    int cnt[5] = {0, 0, 0, 0, 0};
    for (int q = 0; q < K; ++q) ++cnt[c[q]];
    if (cnt[0] > 0) {
      dst[pos] = 0.0;
      continue;
    }
    double v = t[st_idx<K, NT>(cnt[1], cnt[2], cnt[3], cnt[4])];
    if (LZ != nullptr)
      v += LZ[z_tab_index<Tail>(cnt[1] + add[0], cnt[2] + add[1], cnt[3] + add[2],
                                cnt[4] + add[3])];
    dst[pos] = v;
  }
}

template <int K, class D, class Tail, class Builder>
LAPLACE_HD void assemble_info_dk(const Builder& b, const double* invz, const double* LZ,
                                 expected_info_tensor<D::n_params, K>& out) {
  constexpr int P = D::n_params, NT = Tail::n_tail_params;
  constexpr int N = st_size<K, NT>();
  double num[N], prod[N];
  int ab[2] = {0, 0};
  b.build_imm(num);
  st_mul<K, NT>(num, invz, prod);
  fill_pair<K, P, NT, Tail>(out.pair[sym_tensor_position<P, 2>(ab)], prod, nullptr, nullptr);
  for (int bb = 1; bb < P; ++bb) {  // I_mu_theta is zero, and so is every derivative of it
    ab[0] = 0, ab[1] = bb;
    auto& dst = out.pair[sym_tensor_position<P, 2>(ab)];
    for (int pos = 0; pos < sym_tensor<P, K>::size; ++pos) dst[pos] = 0.0;
  }
  for (int a = 1; a < P; ++a)
    for (int bb = a; bb < P; ++bb) {
      const int i = a - 1, j = bb - 1;
      b.build_num(i, j, num);
      st_mul<K, NT>(num, invz, prod);
      const int add[4] = {int(i == 0) + int(j == 0), int(i == 1) + int(j == 1),
                          int(i == 2) + int(j == 2), int(i == 3) + int(j == 3)};
      ab[0] = a, ab[1] = bb;
      fill_pair<K, P, NT, Tail>(out.pair[sym_tensor_position<P, 2>(ab)], prod, LZ, add);
    }
}

}  // namespace laplace::distrib::detail_properized_info
