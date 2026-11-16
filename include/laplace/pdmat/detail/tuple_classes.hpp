// Classes of equal sub-masks of a sorted tuple, for the sub-tuple recursions of
// the structures (toeplitz.hpp, the generic route to M^-1 in core.hpp).
//
// For a tuple of K positions, x[S] is the derivative of a quantity along the
// positions in the mask S. Equal indices of the tuple are interchangeable, so a
// mask S has the derivative of its class (how many positions of each group of
// equal indices it holds): only the canonical mask of a class (the lowest
// positions of each group) is computed and the others copy it (rep[S] <= S).
// Leibniz's rule over the classes: a canonical S sums over the sub-classes T of
// S with weight prod_g C(c_g, a_g), c_g and a_g the sizes of S and T in group g:
// prod_g (c_g + 1) terms instead of 2^|S|. A tuple without repeated indices
// keeps the plain sum over subsets (distinct), since there every class is one
// mask and the table costs more than it saves.
#pragma once

#include <laplace/bijector/coupled.hpp>
#include <laplace/core/config.hpp>

namespace laplace::pdmat::detail_pdmat {

LAPLACE_HD_FORCEINLINE int bit_count(int m) {
  int n = 0;
  for (; m; m &= m - 1) ++n;
  return n;
}
// the n lowest set bits of m
LAPLACE_HD_FORCEINLINE int lowest_bits(int m, int n) {
  int r = 0;
  for (; n > 0; --n, m &= m - 1) r |= m & -m;
  return r;
}
LAPLACE_HD_FORCEINLINE int binomial(int n, int k) {
  int r = 1;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}
LAPLACE_HD constexpr int pow3(int K) { return K == 0 ? 1 : 3 * pow3(K - 1); }

template <int K>
struct tuple_classes {
  static constexpr int NM = 1 << K, NT = pow3(K);
  unsigned char rep[NM];
  short start[NM + 1];
  // the sub-classes of a canonical S, T[start[S]] .. T[start[S + 1] - 1], with
  // weights w, from S down to 0; left empty for a distinct tuple
  unsigned char T[NT], w[NT];
  bool distinct = false;  // no repeated index: every class is one mask, the plain subset sum is cheaper

  LAPLACE_HD explicit tuple_classes(const int* idx) {
    int gm[K > 0 ? K : 1] = {}, ng = 0;
    for (int q = 0; q < K; ++q) {
      if (q > 0 && idx[q] == idx[q - 1]) gm[ng - 1] |= 1 << q;
      else gm[ng++] = 1 << q;
    }
    distinct = ng == K;
    int n = 0;
    for (int S = 0; S < NM; ++S) {
      int r = 0;
      for (int g = 0; g < ng; ++g) r |= lowest_bits(gm[g], bit_count(S & gm[g]));
      rep[S] = (unsigned char)r;
      start[S] = (short)n;
      if (r != S || distinct) continue;
      for (int t = S;; t = (t - 1) & S) {
        int wt = 1;
        bool ok = true;
        for (int g = 0; g < ng && ok; ++g) {
          const int a = bit_count(t & gm[g]), c = bit_count(S & gm[g]);
          ok = (t & gm[g]) == lowest_bits(S & gm[g], a);
          wt *= binomial(c, a);
        }
        if (ok) T[n] = (unsigned char)t, w[n++] = (unsigned char)wt;
        if (t == 0) break;
      }
    }
    start[NM] = (short)n;
  }

  // d_S (f g) for a canonical S, f and g with all their masks filled
  LAPLACE_HD_FORCEINLINE double lb(const double* f, const double* g, int S) const {
    if (distinct) return laplace::bijector::detail_coupled::leibniz(f, g, S);
    double s = 0.0;
    for (int i = start[S]; i < start[S + 1]; ++i) s += double(w[i]) * (f[T[i]] * g[S ^ T[i]]);
    return s;
  }
};

}  // namespace laplace::pdmat::detail_pdmat
