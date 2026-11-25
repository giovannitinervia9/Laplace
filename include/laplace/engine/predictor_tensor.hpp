#pragma once

// A value of the model at one observation unit, with its derivatives in the R
// linear predictors eta_0, ..., eta_{R-1} of the unit (laplace::engine; plan,
// section 7.9, point 3).
//
// The support of a value is the set of predictors it depends on, a bit mask
// that is the same for every unit (it follows from the graph, row_model.hpp).
// Derivatives are packed on the sorted index tuples, as the distribs pack
// theirs (distrib/core.hpp): d1[a], d2 at packed_pair(a, b, R) for a <= b, d3
// at packed_triple<R>(a, b, c) and d4 at packed_quad<R>(a, b, c, d), all
// lexicographic. Entries with an index outside the support are zero; the
// chain rule below skips them. The order K of a value (the highest derivative
// it stores) is a template parameter, so a kernel of order two carries no
// third or fourth derivatives.
//
// The chain rule through a node o = f(i_1, ..., i_M), with p, q, r, s over the
// inputs and a, b, c, d over the predictors, one function per order (one term
// for each partition of the index set into blocks, sums over repeated inputs):
//
//   o_a    = f_p i_p,a
//   o_ab   = f_p i_p,ab + f_pq i_p,a i_q,b
//   o_abc  = f_p i_p,abc + f_pq (i_p,ab i_q,c + i_p,ac i_q,b + i_p,bc i_q,a)
//            + f_pqr i_p,a i_q,b i_r,c
//   o_abcd = f_p i_p,abcd
//            + f_pq (i_p,ab i_q,cd + i_p,ac i_q,bd + i_p,ad i_q,bc
//                    + i_p,abc i_q,d + i_p,abd i_q,c + i_p,acd i_q,b + i_p,bcd i_q,a)
//            + f_pqr (i_p,ab i_q,c i_r,d + i_p,ac i_q,b i_r,d + i_p,ad i_q,b i_r,c
//                     + i_p,bc i_q,a i_r,d + i_p,bd i_q,a i_r,c + i_p,cd i_q,a i_r,b)
//            + f_pqrs i_p,a i_q,b i_r,c i_s,d
//
// where f_p, f_pq, f_pqr and f_pqrs are the node's own derivatives in its
// inputs, packed on the sorted tuples of inputs. Each function writes only its
// own order, so a kernel that needs a lower order never computes a higher one.
// A term is summed only when every block of predictor indices given to an
// input lies in that input's support.

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Position of the sorted pair \f$(a, b)\f$, \f$0 \le a \le b < n\f$, in lexicographic order.
 *
 * Evaluates \f$a n - a(a - 1)/2 + (b - a)\f$, matching the packed symmetric storage of \ref distrib_core.
 *
 * \param a First index (\f$0 \le a \le b\f$).
 * \param b Second index (\f$a \le b < n\f$).
 * \param n Total number of variables.
 * \return Zero-based packed offset in \f$[0, n(n + 1)/2)\f$.
 *
 * \see predictor_value, \ref engine
 */
LAPLACE_HD constexpr int packed_pair(int a, int b, int n) { return a * n - a * (a - 1) / 2 + (b - a); }

/**
 * \ingroup engine
 * \brief Position of the sorted triple \f$(a, b, c)\f$, \f$0 \le a \le b \le c < n\f$, in lexicographic order.
 *
 * Matches the packed symmetric tensor layout `distrib::sym_tensor_position<n, 3>()` of \ref distrib_core "distrib_core".
 *
 * \tparam n Total number of variables.
 * \param a First index (\f$0 \le a \le b\f$).
 * \param b Second index (\f$a \le b \le c\f$).
 * \param c Third index (\f$b \le c < n\f$).
 * \return Zero-based packed offset in \f$[0, n(n + 1)(n + 2)/6)\f$.
 *
 * \see packed_pair(), packed_quad(), predictor_value, \ref engine
 */
template <int n>
LAPLACE_HD constexpr int packed_triple(int a, int b, int c) {
  // the triples that start below a, C(n + 2, 3) - C(n - a + 2, 3), then the pair (b, c) among the
  // n - a values from a on
  const int m = n - a;
  return (n + 2) * (n + 1) * n / 6 - (m + 2) * (m + 1) * m / 6 + packed_pair(b - a, c - a, m);
}

/**
 * \ingroup engine
 * \brief Position of the sorted quadruple \f$(a, b, c, d)\f$, \f$0 \le a \le b \le c \le d < n\f$, in lexicographic order.
 *
 * Matches the packed symmetric tensor layout `distrib::sym_tensor_position<n, 4>()` of \ref distrib_core "distrib_core".
 *
 * \tparam n Total number of variables.
 * \param a First index (\f$0 \le a \le b\f$).
 * \param b Second index (\f$a \le b \le c\f$).
 * \param c Third index (\f$b \le c \le d\f$).
 * \param d Fourth index (\f$c \le d < n\f$).
 * \return Zero-based packed offset in \f$[0, n(n + 1)(n + 2)(n + 3)/24)\f$.
 *
 * \see packed_pair(), packed_triple(), predictor_value, \ref engine
 */
template <int n>
LAPLACE_HD constexpr int packed_quad(int a, int b, int c, int d) {
  // the quadruples that start below a, C(n + 3, 4) - C(n - a + 3, 4), then the triple (b, c, d)
  // among the n - a values from a on
  const int m = n - a, b0 = b - a, c0 = c - a, d0 = d - a;
  const int m3 = m - b0;
  const int t = (m + 2) * (m + 1) * m / 6 - (m3 + 2) * (m3 + 1) * m3 / 6 + packed_pair(c0 - b0, d0 - b0, m3);
  return (n + 3) * (n + 2) * (n + 1) * n / 24 - (m + 3) * (m + 2) * (m + 1) * m / 24 + t;
}

namespace detail_tensor {
// the closed forms above against the library's position of a sorted multi-index
template <int n>
constexpr bool packed_positions_agree() {
  for (int a = 0; a < n; ++a)
    for (int b = a; b < n; ++b)
      for (int c = b; c < n; ++c) {
        const int t3[3] = {a, b, c};
        if (packed_triple<n>(a, b, c) != distrib::sym_tensor_position<n, 3>(t3)) return false;
        for (int d = c; d < n; ++d) {
          const int t4[4] = {a, b, c, d};
          if (packed_quad<n>(a, b, c, d) != distrib::sym_tensor_position<n, 4>(t4)) return false;
        }
      }
  return true;
}
static_assert(packed_positions_agree<1>() && packed_positions_agree<2>() && packed_positions_agree<3>() &&
                  packed_positions_agree<4>() && packed_positions_agree<6>(),
              "packed_triple and packed_quad must match distrib::sym_tensor_position");
}  // namespace detail_tensor

/**
 * \ingroup engine
 * \brief A value with its predictor support mask and derivatives in the \f$R\f$ linear predictors.
 *
 * Holds the scalar value `v`, a bit mask `support` whose bit \f$r\f$ is set when the value depends
 * on \f$\eta_r\f$, first derivatives `d1[R]`, second derivatives `d2[n_pairs]` packed on sorted
 * pairs \f$a \le b\f$ via packed_pair(), and, when \f$K \ge 3\f$ and \f$K \ge 4\f$, third and fourth
 * derivatives packed via packed_triple() and packed_quad(). Entries outside `support` remain zero.
 *
 * \tparam R Number of linear predictors (\f$1 \le R \le 4\f$).
 * \tparam K Highest derivative order stored (2, 3 or 4).
 * \see linear_predictor_value(), chain_rule_d1(), chain_rule_d2(), chain_rule_d3(), chain_rule_d4(), \ref engine
 */
template <int R, int K = 2>
struct predictor_value {
  static constexpr int n_pairs = R * (R + 1) / 2;
  static constexpr int n_triples = R * (R + 1) * (R + 2) / 6;
  static constexpr int n_quads = R * (R + 1) * (R + 2) * (R + 3) / 24;
  // not initialized: linear_predictor_value and chain_rule_d1 write every member, so the
  // values of a unit that no node uses are never zeroed (a default zeroing of the 12 values
  // of a unit cost a fifth of an assembly on the Gaussian model of lk.cpp)
  double v;
  unsigned support;
  double d1[R];
  double d2[n_pairs];
  double d3[K >= 3 ? n_triples : 1];  ///< used when K >= 3
  double d4[K >= 4 ? n_quads : 1];    ///< used when K >= 4
};

/**
 * \ingroup engine
 * \brief Initializes a predictor_value representing the linear predictor \f$\eta_r\f$ itself.
 *
 * Sets `v = eta`, `support = 1u << r`, `d1[r] = 1.0`, and all other derivatives to zero.
 *
 * \tparam R Number of linear predictors.
 * \tparam K Highest derivative order stored.
 * \param r Predictor index \f$0 \le r < R\f$.
 * \param eta Scalar value \f$\eta_r\f$.
 * \return Initialized predictor_value\<R, K\>.
 *
 * \see predictor_value, \ref engine, chain_rule_d1()
 */
template <int R, int K = 2>
LAPLACE_HD predictor_value<R, K> linear_predictor_value(int r, double eta) {
  using V = predictor_value<R, K>;
  V o;
  o.v = eta;
  o.support = 1u << r;
  for (int a = 0; a < R; ++a) o.d1[a] = a == r ? 1.0 : 0.0;
  for (int q = 0; q < V::n_pairs; ++q) o.d2[q] = 0.0;
  if constexpr (K >= 3)
    for (int q = 0; q < V::n_triples; ++q) o.d3[q] = 0.0;
  if constexpr (K >= 4)
    for (int q = 0; q < V::n_quads; ++q) o.d4[q] = 0.0;
  return o;
}

/**
 * \ingroup engine
 * \brief Propagates first-order derivatives through a node via the chain rule.
 *
 * Sets `o.support` to the bitwise union of `in[p]->support` for \f$0 \le p < M\f$, zeroes the
 * higher orders of `o`, and computes \f$o_a = \sum_{p=0}^{M-1} f_p\, i_{p, a}\f$ for every predictor
 * index \f$a\f$ in `o.support`.
 *
 * \tparam R Number of linear predictors.
 * \tparam M Number of node inputs.
 * \tparam K Highest derivative order stored.
 * \param in Array of \f$M\f$ pointers to input predictor_value\<R, K\> objects.
 * \param f1 Array of \f$M\f$ first partial derivatives \f$f_p\f$ with respect to the inputs.
 * \param o Output predictor_value\<R, K\> whose `support` and `d1` are written.
 *
 * \see chain_rule_d2(), predictor_value, \ref engine
 */
template <int R, int M, int K>
LAPLACE_HD void chain_rule_d1(const predictor_value<R, K>* const* in, const double* f1, predictor_value<R, K>& o) {
  using V = predictor_value<R, K>;
  unsigned s = 0u;
  for (int p = 0; p < M; ++p) s |= in[p]->support;
  o.support = s;
  for (int q = 0; q < V::n_pairs; ++q) o.d2[q] = 0.0;
  if constexpr (K >= 3)
    for (int q = 0; q < V::n_triples; ++q) o.d3[q] = 0.0;
  if constexpr (K >= 4)
    for (int q = 0; q < V::n_quads; ++q) o.d4[q] = 0.0;
  for (int a = 0; a < R; ++a) {
    o.d1[a] = 0.0;
    if (!((s >> a) & 1u)) continue;
    double t = 0.0;
    for (int p = 0; p < M; ++p) t += f1[p] * in[p]->d1[a];
    o.d1[a] = t;
  }
}

/**
 * \ingroup engine
 * \brief Propagates second-order derivatives through a node via the chain rule.
 *
 * Requires `o.support` to be set (for example by a preceding chain_rule_d1() call) and writes
 * \f$o_{ab} = \sum_{p=0}^{M-1} f_p\, i_{p, ab} + \sum_{p=0}^{M-1} \sum_{q=0}^{M-1} f_{pq}\, i_{p, a}\, i_{q, b}\f$
 * into `o.d2` for sorted pairs \f$a \le b\f$ with both \f$a\f$ and \f$b\f$ in `o.support`.
 *
 * \tparam R Number of linear predictors.
 * \tparam M Number of node inputs.
 * \tparam K Highest derivative order stored.
 * \param in Array of \f$M\f$ pointers to input predictor_value\<R, K\> objects.
 * \param f1 Array of \f$M\f$ first partial derivatives \f$f_p\f$ with respect to the inputs.
 * \param f2 Array of \f$M(M + 1)/2\f$ second partial derivatives \f$f_{pq}\f$ packed on sorted input pairs.
 * \param o Output predictor_value\<R, K\> whose `d2` is written.
 *
 * \see chain_rule_d1(), packed_pair(), \ref engine
 */
template <int R, int M, int K>
LAPLACE_HD void chain_rule_d2(const predictor_value<R, K>* const* in, const double* f1, const double* f2,
                              predictor_value<R, K>& o) {
  const unsigned s = o.support;
  for (int a = 0; a < R; ++a) {
    if (!((s >> a) & 1u)) continue;
    for (int b = a; b < R; ++b) {
      if (!((s >> b) & 1u)) continue;
      const int ab = packed_pair(a, b, R);
      double t = 0.0;
      for (int p = 0; p < M; ++p) {
        t += f1[p] * in[p]->d2[ab];
        for (int q = 0; q < M; ++q) {
          const double fpq = f2[p <= q ? packed_pair(p, q, M) : packed_pair(q, p, M)];
          t += fpq * in[p]->d1[a] * in[q]->d1[b];
        }
      }
      o.d2[ab] = t;
    }
  }
}

namespace detail_tensor {

// a node's packed derivatives of orders two to four unpacked to full arrays over its M inputs
// ponytail: M^4 numbers on the thread's stack (1296 for a distrib of six parameters); a symmetric
// sum over the sorted input tuples avoids them if a distrib with many parameters is slow here.
template <int M>
LAPLACE_HD void unpack2(const double* f2, double (&F)[M][M]) {
  for (int p = 0; p < M; ++p)
    for (int q = p; q < M; ++q) F[p][q] = F[q][p] = f2[packed_pair(p, q, M)];
}
template <int M>
LAPLACE_HD void unpack3(const double* f3, double (&F)[M][M][M]) {
  int k = 0;
  for (int p = 0; p < M; ++p)
    for (int q = p; q < M; ++q)
      for (int r = q; r < M; ++r, ++k) {
        const double v = f3[k];
        F[p][q][r] = F[p][r][q] = F[q][p][r] = F[q][r][p] = F[r][p][q] = F[r][q][p] = v;
      }
}
template <int M>
LAPLACE_HD void unpack4(const double* f4, double (&F)[M][M][M][M]) {
  int k = 0;
  for (int p = 0; p < M; ++p)
    for (int q = p; q < M; ++q)
      for (int r = q; r < M; ++r)
        for (int u = r; u < M; ++u, ++k) {
          const double v = f4[k];
          const int ix[4] = {p, q, r, u};
          for (int i0 = 0; i0 < 4; ++i0)
            for (int i1 = 0; i1 < 4; ++i1) {
              if (i1 == i0) continue;
              for (int i2 = 0; i2 < 4; ++i2) {
                if (i2 == i0 || i2 == i1) continue;
                F[ix[i0]][ix[i1]][ix[i2]][ix[6 - i0 - i1 - i2]] = v;
              }
            }
        }
}

}  // namespace detail_tensor

/**
 * \ingroup engine
 * \brief Propagates third-order derivatives through a node via the chain rule.
 *
 * Requires `o.support` to be set (chain_rule_d1()) and \f$K \ge 3\f$; writes `o.d3` for the sorted
 * triples \f$a \le b \le c\f$ inside `o.support` by the third-order formula of the header.
 *
 * \tparam R Number of linear predictors.
 * \tparam M Number of node inputs.
 * \tparam K Highest derivative order stored (\f$\ge 3\f$).
 * \param in Array of \f$M\f$ pointers to the inputs.
 * \param f1 \f$f_p\f$, \f$M\f$ numbers.
 * \param f2 \f$f_{pq}\f$ packed on sorted input pairs.
 * \param f3 \f$f_{pqr}\f$ packed on sorted input triples.
 * \param o Output whose `d3` is written.
 *
 * \see chain_rule_d2(), packed_triple(), \ref engine
 */
template <int R, int M, int K>
LAPLACE_HD void chain_rule_d3(const predictor_value<R, K>* const* in, const double* f1, const double* f2,
                              const double* f3, predictor_value<R, K>& o) {
  static_assert(K >= 3, "chain_rule_d3 needs values of order 3");
  double F2[M][M], F3[M][M][M];
  detail_tensor::unpack2<M>(f2, F2);
  detail_tensor::unpack3<M>(f3, F3);
  unsigned su[M];
  for (int p = 0; p < M; ++p) su[p] = in[p]->support;
  const unsigned os = o.support;
  for (int a = 0; a < R; ++a) {
    if (!((os >> a) & 1u)) continue;
    for (int b = a; b < R; ++b) {
      if (!((os >> b) & 1u)) continue;
      const int ab = packed_pair(a, b, R);
      for (int c = b; c < R; ++c) {
        if (!((os >> c) & 1u)) continue;
        const int ac = packed_pair(a, c, R), bc = packed_pair(b, c, R), abc = packed_triple<R>(a, b, c);
        unsigned k[M];  // bit 0, 1, 2: the input depends on a, b, c
        for (int p = 0; p < M; ++p) k[p] = ((su[p] >> a) & 1u) | (((su[p] >> b) & 1u) << 1) | (((su[p] >> c) & 1u) << 2);
        double s = 0.0;
        for (int p = 0; p < M; ++p) {
          const unsigned kp = k[p];
          if (!kp) continue;
          const predictor_value<R, K>& ip = *in[p];
          if (kp == 7u) s += f1[p] * ip.d3[abc];
          for (int q = 0; q < M; ++q) {
            const unsigned kq = k[q];
            if (!kq) continue;
            const predictor_value<R, K>& iq = *in[q];
            double t2 = 0.0;
            if ((kp & 3u) == 3u && (kq & 4u)) t2 += ip.d2[ab] * iq.d1[c];
            if ((kp & 5u) == 5u && (kq & 2u)) t2 += ip.d2[ac] * iq.d1[b];
            if ((kp & 6u) == 6u && (kq & 1u)) t2 += ip.d2[bc] * iq.d1[a];
            s += F2[p][q] * t2;
            if ((kp & 1u) && (kq & 2u)) {
              const double pq = ip.d1[a] * iq.d1[b];
              for (int r = 0; r < M; ++r)
                if (k[r] & 4u) s += F3[p][q][r] * pq * in[r]->d1[c];
            }
          }
        }
        o.d3[abc] = s;
      }
    }
  }
}

/**
 * \ingroup engine
 * \brief Propagates fourth-order derivatives through a node via the chain rule.
 *
 * Requires `o.support` to be set (chain_rule_d1()) and \f$K \ge 4\f$; writes `o.d4` for the sorted
 * quadruples \f$a \le b \le c \le d\f$ inside `o.support` by the fourth-order formula of the header.
 *
 * \tparam R Number of linear predictors.
 * \tparam M Number of node inputs.
 * \tparam K Highest derivative order stored (\f$\ge 4\f$).
 * \param in Array of \f$M\f$ pointers to the inputs.
 * \param f1 \f$f_p\f$, \f$M\f$ numbers.
 * \param f2 \f$f_{pq}\f$ packed on sorted input pairs.
 * \param f3 \f$f_{pqr}\f$ packed on sorted input triples.
 * \param f4 \f$f_{pqrs}\f$ packed on sorted input quadruples.
 * \param o Output whose `d4` is written.
 *
 * \see chain_rule_d3(), packed_quad(), \ref engine
 */
template <int R, int M, int K>
LAPLACE_HD void chain_rule_d4(const predictor_value<R, K>* const* in, const double* f1, const double* f2,
                              const double* f3, const double* f4, predictor_value<R, K>& o) {
  static_assert(K >= 4, "chain_rule_d4 needs values of order 4");
  double F2[M][M], F3[M][M][M], F4[M][M][M][M];
  detail_tensor::unpack2<M>(f2, F2);
  detail_tensor::unpack3<M>(f3, F3);
  detail_tensor::unpack4<M>(f4, F4);
  unsigned su[M];
  for (int p = 0; p < M; ++p) su[p] = in[p]->support;
  const unsigned os = o.support;
  for (int a = 0; a < R; ++a) {
    if (!((os >> a) & 1u)) continue;
    for (int b = a; b < R; ++b) {
      if (!((os >> b) & 1u)) continue;
      for (int c = b; c < R; ++c) {
        if (!((os >> c) & 1u)) continue;
        for (int d = c; d < R; ++d) {
          if (!((os >> d) & 1u)) continue;
          const int ab = packed_pair(a, b, R), ac = packed_pair(a, c, R), ad = packed_pair(a, d, R),
                    bc = packed_pair(b, c, R), bd = packed_pair(b, d, R), cd = packed_pair(c, d, R);
          const int abc = packed_triple<R>(a, b, c), abd = packed_triple<R>(a, b, d), acd = packed_triple<R>(a, c, d),
                    bcd = packed_triple<R>(b, c, d);
          unsigned k[M];  // bit 0, 1, 2, 3: the input depends on a, b, c, d
          for (int p = 0; p < M; ++p)
            k[p] = ((su[p] >> a) & 1u) | (((su[p] >> b) & 1u) << 1) | (((su[p] >> c) & 1u) << 2) |
                   (((su[p] >> d) & 1u) << 3);
          double s = 0.0;
          for (int p = 0; p < M; ++p) {
            const unsigned kp = k[p];
            if (!kp) continue;
            const predictor_value<R, K>& ip = *in[p];
            if (kp == 15u) s += f1[p] * ip.d4[packed_quad<R>(a, b, c, d)];
            for (int q = 0; q < M; ++q) {
              const unsigned kq = k[q];
              if (!kq) continue;
              const predictor_value<R, K>& iq = *in[q];
              double t2 = 0.0;
              if ((kp & 3u) == 3u && (kq & 12u) == 12u) t2 += ip.d2[ab] * iq.d2[cd];
              if ((kp & 5u) == 5u && (kq & 10u) == 10u) t2 += ip.d2[ac] * iq.d2[bd];
              if ((kp & 9u) == 9u && (kq & 6u) == 6u) t2 += ip.d2[ad] * iq.d2[bc];
              if ((kp & 7u) == 7u && (kq & 8u)) t2 += ip.d3[abc] * iq.d1[d];
              if ((kp & 11u) == 11u && (kq & 4u)) t2 += ip.d3[abd] * iq.d1[c];
              if ((kp & 13u) == 13u && (kq & 2u)) t2 += ip.d3[acd] * iq.d1[b];
              if ((kp & 14u) == 14u && (kq & 1u)) t2 += ip.d3[bcd] * iq.d1[a];
              s += F2[p][q] * t2;
              for (int r = 0; r < M; ++r) {
                const unsigned kr = k[r];
                if (!kr) continue;
                const predictor_value<R, K>& ir = *in[r];
                double t3 = 0.0;
                if ((kp & 3u) == 3u && (kq & 4u) && (kr & 8u)) t3 += ip.d2[ab] * iq.d1[c] * ir.d1[d];
                if ((kp & 5u) == 5u && (kq & 2u) && (kr & 8u)) t3 += ip.d2[ac] * iq.d1[b] * ir.d1[d];
                if ((kp & 9u) == 9u && (kq & 2u) && (kr & 4u)) t3 += ip.d2[ad] * iq.d1[b] * ir.d1[c];
                if ((kp & 6u) == 6u && (kq & 1u) && (kr & 8u)) t3 += ip.d2[bc] * iq.d1[a] * ir.d1[d];
                if ((kp & 10u) == 10u && (kq & 1u) && (kr & 4u)) t3 += ip.d2[bd] * iq.d1[a] * ir.d1[c];
                if ((kp & 12u) == 12u && (kq & 1u) && (kr & 2u)) t3 += ip.d2[cd] * iq.d1[a] * ir.d1[b];
                s += F3[p][q][r] * t3;
                if ((kp & 1u) && (kq & 2u) && (kr & 4u)) {
                  const double pqr = ip.d1[a] * iq.d1[b] * ir.d1[c];
                  for (int u = 0; u < M; ++u)
                    if (k[u] & 8u) s += F4[p][q][r][u] * pqr * in[u]->d1[d];
                }
              }
            }
          }
          o.d4[packed_quad<R>(a, b, c, d)] = s;
        }
      }
    }
  }
}

}  // namespace laplace::engine
