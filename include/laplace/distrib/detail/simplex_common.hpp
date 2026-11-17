// What the laws with a simplex_vector parameter share (dirichlet_distrib,
// multinomial_distrib; note/distrib_multivariate.md section 18): the
// derivatives of the probabilities mu and of their logarithms in the additive
// log-ratio coordinates eta (reference category the last), over every subset
// of the positions of a sorted tuple, and the partition sums of Faa di Bruno's
// formula over them.
//
// With l = log mu_ref = -log(1 + sum_b e^(eta_b)), log mu_c = eta_c + l and
// d_b mu_c = mu_c (delta_bc - mu_b). For a sorted tuple t of K eta indices and
// a subset S of its positions:
//   L[S]    = d_S l, the same for every category once S is not empty, except
//             that a category c of the tuple adds delta at a single position of
//             category c; L[S] = -d_(S \ j) mu_(cat j), j the lowest position of
//             S, so -L[S] is the cumulant kappa_S of the categorical indicators;
//   G[S]    = d_S mu_c / mu_c for every category c outside the tuple (the
//             reference included), G[{}] = 1;
//   D[c][S] = d_S mu_c for the distinct categories c of the tuple, by Leibniz
//             over log mu_c, with 1 - mu_c formed as the sum of the other
//             probabilities (exact where mu_c is near one).
// This is the recursion of bijector/alr.hpp (inverse_dk), which keeps only the
// full subset; here every subset is kept, because a partition sum reads them
// all. The probabilities enter as given (mu and log mu from the caller), so a
// tiny probability keeps its digits.
//
// Partition sums: for one category c with subset derivatives dm[S] = d_S mu_c,
//   P[S][k] = sum over the partitions of S into k blocks of prod_B dm[B],
// so that d_S f(mu_c) = sum_k f^(k)(mu_c) P[S][k]. For the categories outside
// the tuple dm[S] = mu_c G[S], hence P[S][k] = mu_c^k Q[S][k], with Q the
// partition sum of G, one table for all of them.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/detail/double_double.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib::detail_simplex {

inline constexpr int max_order = 6;
inline constexpr int max_masks = 1 << max_order;

// log mu_j from eta (d categories, the last the reference, eta_ref = 0): with c the
// category of the largest eta and t = sum_(b != c) e^(eta_b - eta_c), log mu_c =
// -log1p(t) and log mu_j = (eta_j - eta_c) - log1p(t); a probability near one keeps its
// digits, which eta_c - LSE(eta) would lose to the subtraction
LAPLACE_HD double log_probability(const double* x, int d, int j) {
  int c = d - 1;
  double xc = 0.0;
  for (int a = 0; a + 1 < d; ++a)
    if (x[a] > xc) {
      xc = x[a];
      c = a;
    }
  double t = 0.0;
  for (int b = 0; b < d; ++b)
    if (b != c) t += Kokkos::exp((b + 1 < d ? x[b] : 0.0) - xc);
  const double l1 = Kokkos::log1p(t);
  return j == c ? -l1 : ((j + 1 < d ? x[j] : 0.0) - xc) - l1;
}

LAPLACE_HD void log_probabilities(const double* x, int d, double* lmu) {
  int c = d - 1;
  double xc = 0.0;
  for (int a = 0; a + 1 < d; ++a)
    if (x[a] > xc) {
      xc = x[a];
      c = a;
    }
  double t = 0.0;
  for (int b = 0; b < d; ++b)
    if (b != c) t += Kokkos::exp((b + 1 < d ? x[b] : 0.0) - xc);
  const double l1 = Kokkos::log1p(t);
  for (int j = 0; j < d; ++j) lmu[j] = j == c ? -l1 : ((j + 1 < d ? x[j] : 0.0) - xc) - l1;
}

// The probabilities of one unit in double-double, computed once and read by every tuple:
// pt[2 j], pt[2 j + 1] = mu_j (hi, lo), pt[2 d + 2 j], pt[2 d + 2 j + 1] = 1 - mu_j formed as
// the sum of the other probabilities (exact where mu_j is near one). point_size(d) doubles.
LAPLACE_HD_FORCEINLINE int point_size(int d) { return 4 * d; }
LAPLACE_HD void fill_point(const double* x, int d, double* pt) {
  namespace dm = math::detail::ddm;
  double xm = 0.0;
  for (int a = 0; a + 1 < d; ++a) xm = x[a] > xm ? x[a] : xm;
  dm::dd s(0.0);
  for (int b = 0; b < d; ++b) {
    const dm::dd e = dm::exp(dm::two_sum(b + 1 < d ? x[b] : 0.0, -xm));
    pt[2 * b] = e.hi;
    pt[2 * b + 1] = e.lo;
    s = s + e;
  }
  for (int b = 0; b < d; ++b) {
    const dm::dd m = dm::dd(pt[2 * b], pt[2 * b + 1]) / s;
    pt[2 * b] = m.hi;
    pt[2 * b + 1] = m.lo;
  }
  for (int c = 0; c < d; ++c) {
    dm::dd r(0.0);
    for (int b = 0; b < d; ++b)
      if (b != c) r = r + dm::dd(pt[2 * b], pt[2 * b + 1]);
    pt[2 * d + 2 * c] = r.hi;
    pt[2 * d + 2 * c + 1] = r.lo;
  }
}

// log(v / mu_j) as log1p((v - mu_j)/mu_j) with mu_j in double-double: where v is close to
// mu_j (a Dirichlet draw at a large precision is close to its mean) the difference
// log v - log mu_j would lose the digits that this keeps
LAPLACE_HD double log_ratio_dd(const math::detail::ddm::dd& mu, double v) {
  namespace dm = math::detail::ddm;
  const dm::dd r = (v - mu) / mu;
  if (r.hi > -0.5 && r.hi < 1.0) return Kokkos::log1p(r.hi + r.lo);
  const dm::dd lmu = dm::log(mu);  // far from mu: no cancellation in the difference
  return Kokkos::log(v) - (lmu.hi + lmu.lo);
}
LAPLACE_HD double log_ratio(const double* pt, int j, double v) {
  return log_ratio_dd(math::detail::ddm::dd(pt[2 * j], pt[2 * j + 1]), v);
}

LAPLACE_HD_FORCEINLINE int lowest_bit(int s) {
  int j = 0;
  while (!(s & (1 << j))) ++j;
  return j;
}

// the subset tables of a sorted tuple idx[0..K) of eta indices (0 <= idx < d - 1)
struct subset_tables {
  int K = 0;
  int nc = 0;                        // number of distinct categories of the tuple
  int cats[max_order] = {};          // the distinct categories, in tuple order
  int slot[max_order] = {};          // slot[q]: index in cats of position q
  // the tables live in the caller's scratch (fill binds them), not on the thread's stack:
  // a CUDA thread has a few kilobytes of stack, and these are 4 KB
  double* L = nullptr;
  double* G = nullptr;
  double* D[max_order] = {};

  double om[max_order] = {};         // 1 - mu_c for the categories of the tuple

  // the tables from the unit's point (fill_point), computed in double-double and rounded
  // entry by entry: the cumulants cancel where 1 - 2 mu (and the other factors of their
  // polynomials) vanish, and in double they would carry an absolute error of about eps/4
  // there; here each entry is correct to its own size. ws holds scratch_doubles doubles:
  // the rounded tables (L, G, D), which stay valid while ws does, then the double-double
  // working copies.
  static constexpr int table_doubles = (2 + max_order) * max_masks;
  static constexpr int scratch_doubles = 3 * table_doubles;
  LAPLACE_HD void fill(const double* pt, int d, const int* idx, int k, double* ws) {
    using math::detail::ddm::dd;
    K = k;
    nc = 0;
    for (int q = 0; q < K; ++q) {
      if (q == 0 || idx[q] != idx[q - 1]) cats[nc++] = idx[q];
      slot[q] = nc - 1;
    }
    L = ws;
    G = ws + max_masks;
    for (int c = 0; c < max_order; ++c) D[c] = ws + (2 + c) * max_masks;
    dd* Ld = reinterpret_cast<dd*>(ws + table_doubles);
    dd* Gd = Ld + max_masks;
    dd* Dd = Gd + max_masks;  // Dd[c * max_masks + S]
    dd omd[max_order];
    for (int c = 0; c < nc; ++c) {
      Dd[c * max_masks] = dd(pt[2 * cats[c]], pt[2 * cats[c] + 1]);
      omd[c] = dd(pt[2 * d + 2 * cats[c]], pt[2 * d + 2 * cats[c] + 1]);
      om[c] = omd[c].hi;
      D[c][0] = Dd[c * max_masks].hi;
    }
    L[0] = 0.0;  // not read: d_{} l is the value, which callers take from log mu
    G[0] = 1.0;
    Ld[0] = dd(0.0);
    Gd[0] = dd(1.0);
    const int NM = 1 << K;
    for (int S = 1; S < NM; ++S) {
      const int j = lowest_bit(S), R = S ^ (1 << j);
      Ld[S] = -Dd[slot[j] * max_masks + R];
      dd g(0.0);
      for (int T = R;; T = (T - 1) & R) {
        g = g + Gd[T] * Ld[S ^ T];
        if (T == 0) break;
      }
      Gd[S] = g;
      for (int c = 0; c < nc; ++c) {
        dd v(0.0);
        for (int T = R;; T = (T - 1) & R) {
          const int Q = S ^ T;  // contains j
          const dd dl = (Q & (Q - 1)) == 0 && slot[j] == c ? omd[c] : Ld[Q];
          v = v + Dd[c * max_masks + T] * dl;
          if (T == 0) break;
        }
        Dd[c * max_masks + S] = v;
      }
      L[S] = Ld[S].hi + Ld[S].lo;
      G[S] = Gd[S].hi + Gd[S].lo;
      for (int c = 0; c < nc; ++c) D[c][S] = Dd[c * max_masks + S].hi + Dd[c * max_masks + S].lo;
    }
  }

  // d_S log mu_cat for a nonempty S, cat a category (-1 for one outside the tuple);
  // at a single position of category cat it is 1 - mu_cat, taken from om
  LAPLACE_HD double dlog(int S, int c) const {
    if ((S & (S - 1)) == 0 && c >= 0 && slot[lowest_bit(S)] == c) return om[c];
    return L[S];
  }
};

// P[S][k] for every subset S of the K positions and k = 0..K: the partition
// sums of the subset derivatives dm (dm[0] is not read). P is laid out
// P[S * (max_order + 1) + k].
LAPLACE_HD void partition_sums(const double* dm, int K, double* P) {
  constexpr int W = max_order + 1;
  const int NM = 1 << K;
  for (int k = 0; k <= K; ++k) P[k] = k == 0 ? 1.0 : 0.0;
  for (int S = 1; S < NM; ++S) {
    const int j = lowest_bit(S), R = S ^ (1 << j);
    P[S * W] = 0.0;
    for (int k = 1; k <= K; ++k) {
      double v = 0.0;
      for (int T = R;; T = (T - 1) & R) {  // the block of position j is T + j
        v += dm[T | (1 << j)] * P[(R ^ T) * W + k - 1];
        if (T == 0) break;
      }
      P[S * W + k] = v;
    }
  }
}

// iteration over the sorted tuples of order n over m coordinates in the order
// of the multivariate layout: n = 2 packed by rows (entry t1 (t1 + 1)/2 + t0),
// n != 2 lexicographic. first() sets t to the first tuple; next() advances and
// returns false after the last; position() is the entry of t.
struct tuple_walk {
  int n, m;
  int t[max_order] = {};
  LAPLACE_HD tuple_walk(int order, int coords) : n(order), m(coords) {}
  LAPLACE_HD void first() {
    for (int q = 0; q < n; ++q) t[q] = 0;
  }
  LAPLACE_HD bool next() {
    if (n == 0) return false;
    if (n == 2) {  // (t0, t1) with t0 <= t1, t1 outer
      if (t[0] < t[1]) {
        ++t[0];
        return true;
      }
      if (t[1] + 1 >= m) return false;
      ++t[1];
      t[0] = 0;
      return true;
    }
    int q = n - 1;
    while (q >= 0 && t[q] == m - 1) --q;
    if (q < 0) return false;
    const int v = t[q] + 1;
    for (int r = q; r < n; ++r) t[r] = v;
    return true;
  }
};

}  // namespace laplace::distrib::detail_simplex
