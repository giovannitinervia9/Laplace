// The chain rule for a composition of two variables in two variables
// (laplace::math), to total order 6:
//
//   chain_rule_pair_dk<K>(G, u0, u1)
//
// returns the symmetric tensor of order K in (x_0, x_1) of
//
//   H(x_0, x_1) = G(u_0(x_0, x_1), u_1(x_0, x_1)),
//
// given the partial derivatives of the outer function G in its own two
// arguments and those of the two inner maps u_0 and u_1 in (x_0, x_1).
//
// Why it exists: faa_di_bruno.hpp composes one function of one variable, and a
// mixed derivative in several parameters is a product of one Bell sum per
// parameter only when each inner variable depends on one parameter alone
// (gamma1: the shape 1/phi depends on phi and the ratio y/mu on mu). A
// reparametrization in which both inner variables depend on both parameters is
// not of that kind: gamma2's shape is mu^2/sigma^2, lognormal2's map is
// sigma^2 = log(1 + v/m^2), mu = log m - sigma^2/2, and the Jacobian of an
// expected information transforms the same way.
//
// Instead of the multivariate Bell polynomials this is one numeric recursion.
// With multi-indices gamma = (p, q) over the outer variables and k = (i, j)
// over the arguments of G, write D[gamma][k] = d^gamma (G_k o u), where G_k is
// the partial derivative of G of multi-index k. Differentiating once more in
// x_m and expanding by Leibniz's rule over the product,
//
//   D[gamma + e_m][k] = sum_(alpha <= gamma) C(gamma, alpha)
//                       sum_(r = 0, 1) D[alpha][k + e_r] d^(gamma - alpha + e_m) u_r,
//
// with C(gamma, alpha) = C(p, p') C(q, q') the product of two binomial
// coefficients. The table is filled by increasing |gamma| from D[0][k] = G_k,
// and only the entries with |gamma| + |k| <= K are needed, so the state is
// sum_(n <= K) (n + 1) C(K - n + 2, 2) doubles (210 at K = 6). Every
// coefficient is a positive binomial: the recursion introduces no cancellation
// of its own, and whatever cancels does so in the partial derivatives it is
// given.
//
// Layout of the inputs and the output. A multi-index (i, j) of total order
// n = i + j sits at pair_index(i, j) = n (n + 1) / 2 + j, that is by total
// order and then by the number of derivatives in the second variable; within
// one order that is the packing of sym_tensor<2, n> (the component with j
// indices equal to 1 is at position j), so a caller fills G order by order.
//   G[pair_index(i, j)]   d^(i + j) G / du_0^i du_1^j,  i + j <= K
//   u0, u1[pair_index(p, q)]  d^(p + q) u_r / dx_0^p dx_1^q, 1 <= p + q <= K
//                             (position 0, the value of u_r, is never read)
// The result is written to out[0..K], in the same packing: out[b] holds
// d^K H / dx_0^(K-b) dx_1^b, which is component b of a sym_tensor<2, K>.
#pragma once

#include <laplace/core/config.hpp>

namespace laplace::math {

namespace detail_chain_pair {

// position of the multi-index (i, j): by total order, then by j
LAPLACE_HD constexpr int pair_index(int i, int j) { return ((i + j) * (i + j + 1)) / 2 + j; }

// number of multi-indices of total order at most s
LAPLACE_HD constexpr int pair_count(int s) { return ((s + 1) * (s + 2)) / 2; }

// offset of the block of the outer order n in the state of chain_rule_pair_dk<K>
LAPLACE_HD constexpr int state_base(int K, int n) {
  int b = 0;
  for (int m = 0; m < n; ++m) b += (m + 1) * pair_count(K - m);
  return b;
}

// size of that state
LAPLACE_HD constexpr int state_size(int K) { return state_base(K, K + 1); }

// position of D[(n - b, b)][(i, j)]
LAPLACE_HD constexpr int state_index(int K, int n, int b, int i, int j) {
  return state_base(K, n) + b * pair_count(K - n) + pair_index(i, j);
}

LAPLACE_HD constexpr double binom(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  double r = 1.0;
  for (int t = 1; t <= k; ++t) r = r * (n - k + t) / t;
  return r;
}

}  // namespace detail_chain_pair

/**
 * \ingroup chain_pair
 * \brief Evaluates the order-\f$K\f$ partial derivatives of \f$H(x_0, x_1) = G(u_0(x_0, x_1), u_1(x_0, x_1))\f$ in the two variables \f$(x_0, x_1)\f$.
 *
 * Computes all partial derivatives of total order \f$K\f$ (\f$1 \le K \le 6\f$) by filling the recurrence table
 * \f$D[\gamma][k] = \partial^\gamma (G_k \circ u)\f$ by induction on increasing multi-index order \f$|\gamma|\f$ via Leibniz's rule:
 * \f[
 * D[\gamma + e_m][k] = \sum_{\alpha \le \gamma} \binom{\gamma}{\alpha} \sum_{r=0}^1 D[\alpha][k + e_r]\,
 * \partial^{\gamma - \alpha + e_m} u_r,
 * \f]
 * initialized at \f$D[0][k] = G_k\f$, where every coefficient is a positive binomial.
 *
 * Derivative multi-indices are packed by total order and then by the number of derivatives in the second variable:
 * multi-index \f$(i, j)\f$ sits at contiguous array index \f$(i + j)(i + j + 1)/2 + j\f$.
 *
 * \tparam K Total derivative order (\f$1 \le K \le 6\f$).
 * \param G Partial derivatives of the outer function \f$G\f$, where \f$\partial^{i+j} G / \partial u_0^i \partial u_1^j\f$ sits at index \f$(i + j)(i + j + 1)/2 + j\f$ for all \f$i + j \le K\f$.
 * \param u0 Partial derivatives of the first inner map \f$u_0\f$ in \f$(x_0, x_1)\f$, where \f$\partial^{p+q} u_0 / \partial x_0^p \partial x_1^q\f$ sits at index \f$(p + q)(p + q + 1)/2 + q\f$ for \f$1 \le p + q \le K\f$. Position 0 (the undifferentiated value) is never read.
 * \param u1 Partial derivatives of the second inner map \f$u_1\f$ in \f$(x_0, x_1)\f$, packed identically to \c u0. Position 0 is never read.
 * \param out Output array of \f$K + 1\f$ doubles; \c out[b] receives \f$\partial^K H / \partial x_0^{K-b} \partial x_1^b\f$ for \f$b \in \{0, \dots, K\}\f$, matching component \f$b\f$ of a symmetric tensor of order \f$K\f$ in two variables.
 * \see chain_pair, chain_rule_d1()
 */
template <int K>
LAPLACE_HD void chain_rule_pair_dk(const double* G, const double* u0, const double* u1, double* out) {
  using namespace detail_chain_pair;
  double D[state_size(K)];
  for (int i = 0; i <= K; ++i)
    for (int j = 0; i + j <= K; ++j) D[state_index(K, 0, 0, i, j)] = G[pair_index(i, j)];
  for (int n = 1; n <= K; ++n)
    for (int b = 0; b <= n; ++b) {
      // one derivative in x_1 when there is one to take, else in x_0; gamma' is what is left
      const int m = b > 0 ? 1 : 0;
      const int pp = n - b - (m == 0 ? 1 : 0), qp = b - (m == 1 ? 1 : 0);
      for (int i = 0; i + n <= K; ++i)
        for (int j = 0; i + j + n <= K; ++j) {
          double acc = 0.0;
          for (int p = 0; p <= pp; ++p)
            for (int q = 0; q <= qp; ++q) {
              const int ua = pp - p + (m == 0 ? 1 : 0), ub = qp - q + (m == 1 ? 1 : 0);
              const int iu = pair_index(ua, ub);
              const double term = D[state_index(K, p + q, q, i + 1, j)] * u0[iu] +
                                  D[state_index(K, p + q, q, i, j + 1)] * u1[iu];
              acc += binom(pp, p) * binom(qp, q) * term;
            }
          D[state_index(K, n, b, i, j)] = acc;
        }
    }
  for (int b = 0; b <= K; ++b) out[b] = D[state_index(K, K, b, 0, 0)];
}

}  // namespace laplace::math
