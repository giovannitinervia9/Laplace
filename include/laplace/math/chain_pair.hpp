// The chain rule for a composition of two variables in two variables
// (laplace::math), to total order 6:
//
//   chain_rule_pair_dk<K>(G, u0, u1)         an outer function of two variables
//   chain_rule_pair_scalar_dk<K>(f, u)       an outer function of one variable
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

// The same state for an outer function of ONE variable: D[gamma][k] with k a single order instead
// of a multi-index over two arguments. sum_(n <= K) (n + 1) (K - n + 1) doubles, 84 at K = 6
// against the 210 of the pair version, and one loop dimension fewer.
LAPLACE_HD constexpr int scalar_state_base(int K, int n) {
  int b = 0;
  for (int m = 0; m < n; ++m) b += (m + 1) * (K - m + 1);
  return b;
}

LAPLACE_HD constexpr int scalar_state_size(int K) { return scalar_state_base(K, K + 1); }

LAPLACE_HD constexpr int scalar_state_index(int K, int n, int b, int k) {
  return scalar_state_base(K, n) + b * (K - n + 1) + k;
}

// C(n, k). A local table was measured and dropped: 49 doubles have to be written to the stack at
// every call, where the product form stays in registers, and the order-6 composition went from
// 6.9 to 44 microseconds. What the scalar recursion below does instead is to take the two
// coefficients out of its innermost loop, where they do not depend on the outer order.
LAPLACE_HD constexpr double binom(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  double r = 1.0;
  for (int t = 1; t <= k; ++t) r = r * (n - k + t) / t;
  return r;
}

}  // namespace detail_chain_pair

/**
 * \ingroup chain_pair
 * \brief Evaluates all partial derivatives of total order \f$K\f$ of \f$H(x_0, x_1) = G(u_0(x_0, x_1), u_1(x_0, x_1))\f$.
 *
 * Writes \f$\partial^K H / \partial x_0^{K-b} \partial x_1^b\f$ to `out[b]` for \f$b \in \{0, \dots, K\}\f$.
 * Derivatives in `G`, `u0`, and `u1` use the packed multi-index layout defined on \ref chain_pair,
 * where multi-index \f$(i, j)\f$ sits at index \f$(i + j)(i + j + 1)/2 + j\f$.
 *
 * \tparam K Total derivative order, with unchecked precondition \f$1 \le K \le 6\f$.
 * \param G Input array of at least \f$(K + 1)(K + 2)/2\f$ doubles holding \f$\partial^{i+j} G / \partial u_0^i \partial u_1^j\f$ for \f$0 \le i + j \le K\f$ (`G[0]` does not enter the result for \f$K \ge 1\f$).
 * \param u0 Input array of at least \f$(K + 1)(K + 2)/2\f$ doubles holding \f$\partial^{p+q} u_0 / \partial x_0^p \partial x_1^q\f$ for \f$1 \le p + q \le K\f$ (`u0[0]` is never read).
 * \param u1 Input array of at least \f$(K + 1)(K + 2)/2\f$ doubles holding derivatives of \f$u_1\f$ in \f$(x_0, x_1)\f$, packed identically to `u0` (`u1[0]` is never read).
 * \param out Output array of at least \f$K + 1\f$ doubles receiving \f$\partial^K H / \partial x_0^{K-b} \partial x_1^b\f$ in `out[b]`.
 *
 * \par Domain and special values
 * Unchecked preconditions require \f$1 \le K \le 6\f$ and the buffer sizes above; any NaN in a contributing derivative propagates to the affected entries of `out`.
 *
 * \par Measured accuracy
 * Maximum error on gcc is 2.8 ulp on cancelling random derivatives and reaches 26.2 ulp at order 6 on non-linear coordinate maps; test bound 40.0 ulp on gcc and CUDA.
 *
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

/**
 * \ingroup chain_pair
 * \brief Evaluates all partial derivatives of total order \f$K\f$ of \f$H(x_0, x_1) = f(u(x_0, x_1))\f$
 * for an outer function of one variable.
 *
 * The specialization of chain_rule_pair_dk() to an outer function of a single argument, which is the
 * common case when a distribution composes a one-variable function with one map of its two
 * parameters. The recursion is the same, with the multi-index over the outer arguments replaced by
 * one derivative order, so the state is \f$\sum_{n \le K} (n+1)(K-n+1)\f$ doubles (84 at \f$K = 6\f$
 * against 210) and one loop dimension shorter. Every coefficient is a positive binomial, so the
 * recursion introduces no cancellation of its own.
 *
 * \tparam K Total derivative order, with unchecked precondition \f$1 \le K \le 6\f$.
 * \param f Input array of at least \f$K + 1\f$ doubles holding \f$f^{(k)}(u)\f$ for \f$0 \le k \le K\f$ (`f[0]` does not enter the result).
 * \param u Input array of at least \f$(K + 1)(K + 2)/2\f$ doubles holding \f$\partial^{p+q} u / \partial x_0^p \partial x_1^q\f$ for \f$1 \le p + q \le K\f$ in the packed layout of \ref chain_pair (`u[0]` is never read).
 * \param out Output array of at least \f$K + 1\f$ doubles receiving \f$\partial^K H / \partial x_0^{K-b} \partial x_1^b\f$ in `out[b]`.
 *
 * \par Domain and special values
 * Unchecked preconditions require \f$1 \le K \le 6\f$ and the buffer sizes above; any NaN in a contributing derivative propagates to the affected entries of `out`.
 *
 * \par Measured accuracy
 * Agrees with chain_rule_pair_dk() on the same inputs to within $10^{-14}$ of the scale of the order (the test checks every component of every order against it); the two do not agree bit for bit, because this one takes the two binomial coefficients out of its innermost loop and so sums the same terms in another order.
 *
 * \see chain_pair, chain_rule_pair_dk()
 */
template <int K>
LAPLACE_HD void chain_rule_pair_scalar_dk(const double* f, const double* u, double* out) {
  using namespace detail_chain_pair;
  double D[scalar_state_size(K)];
  for (int k = 0; k <= K; ++k) D[scalar_state_index(K, 0, 0, k)] = f[k];
  for (int n = 1; n <= K; ++n)
    for (int b = 0; b <= n; ++b) {
      // one derivative in x_1 when there is one to take, else in x_0; gamma' is what is left
      const int m = b > 0 ? 1 : 0;
      const int pp = n - b - (m == 0 ? 1 : 0), qp = b - (m == 1 ? 1 : 0);
      // the two binomials and the derivative of u do not depend on the outer order k, so the
      // accumulation over (p, q) carries every k at once
      double acc[K + 1];
      for (int k = 0; k + n <= K; ++k) acc[k] = 0.0;
      for (int p = 0; p <= pp; ++p)
        for (int q = 0; q <= qp; ++q) {
          const int ua = pp - p + (m == 0 ? 1 : 0), ub = qp - q + (m == 1 ? 1 : 0);
          const double c = binom(pp, p) * binom(qp, q) * u[pair_index(ua, ub)];
          const int base = scalar_state_index(K, p + q, q, 1);
          for (int k = 0; k + n <= K; ++k) acc[k] += c * D[base + k];
        }
      for (int k = 0; k + n <= K; ++k) D[scalar_state_index(K, n, b, k)] = acc[k];
    }
  for (int b = 0; b <= K; ++b) out[b] = D[scalar_state_index(K, K, b, 0)];
}

}  // namespace laplace::math
