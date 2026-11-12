// The expected information and its derivatives to total order 6 by the generic
// route (laplace::distrib; note/distrib.md section 5, point C): an expectation
// over the distribution (generic_expectation.hpp) of the observed derivatives
// of l = log f, with the measure moving.
//
//   expected_info_by_integral<K, form>(d)   K = 0: I_ab; K = 1..4: d^K I_ab / dtheta_C
//   distrib_expected_info<K>(d)              the distribution's own member when it has one,
//                                            the generic route otherwise
//
// With C = (c_1, ..., c_K) the sorted derivative indices, B_T the complete Bell
// polynomial of l over a sub-multiset T of C (B_T = e^-l d_T e^l, the sum over the
// set partitions of T of the product of the derivatives of l over the blocks;
// B_empty = 1), and l_U the derivative of l over the multiset U:
//
//   hessian form:  I_ab = -E[l_ab],      d_C I_ab = -E[ sum_(T in C) l_(ab + C-T) B_T ]
//   score form:    I_ab =  E[l_a l_b],   d_C I_ab =  E[ sum_(T in C) S_(C-T) B_T ],
//                  S_U = sum_(V in U) l_(a + V) l_(b + U-V)   (the derivatives of l_a l_b)
//
// The hessian form reads l to order K + 2, the score form to order K + 1.
//
// A parameter theta_j of finite regularity r_j (params[j].regularity) has a kink
// in y that moves with it; an a.e. derivative of l of order above r_j + 1 in
// theta_j misses a point mass there, and differentiating an integrand that jumps
// at the kink misses a boundary term. A component is computed only when neither
// happens, and is NaN otherwise:
//   hessian form: n_a + n_b + n_C <= r_j + 1 for every such j,
//   score form:   n_C = 0, or max(n_a, n_b) + n_C <= r_j + 1,
// with n_a, n_b, n_C the multiplicities of j in a, b and C. On the Laplace (r_mu
// = 0) the score form gives I and every derivative with at most one index mu in C
// and none in (a, b) when there is one in C; the hessian form gives the
// components with at most one index mu in (a, b) and C together. A component whose quadrature or sum did not converge is
// NaN as well.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/math/elementary.hpp>

#include <type_traits>

namespace laplace::distrib {

/**
 * \ingroup distrib_generic
 * \brief Formulation of expected Fisher information and its parameter derivatives.
 *
 * Defines the mathematical representation used to evaluate the expected information:
 * - `hessian`: based on \f$-\mathbb{E}[l_{ab}]\f$, requiring observed log-density derivatives to order \f$K + 2\f$.
 * - `score`: based on \f$\mathbb{E}[l_a l_b]\f$, requiring observed log-density derivatives to order \f$K + 1\f$.
 *
 * \see expected_info_by_integral(), generic_info_form()
 */
enum class info_form : int { hessian, score };

namespace detail_generic {

// the derivatives of l of orders 1..N, packed one after the other
template <int P, int N>
struct log_density_partials {
  LAPLACE_HD static constexpr int offset(int k) {
    int o = 0;
    for (int i = 1; i < k; ++i) o += binomial_coef(P + i - 1, i);
    return o;
  }
  double v[offset(N + 1)] = {};

  template <class D>
  LAPLACE_HD void fill(const D& d, double y) {
    fill_order<1>(d, y);
    if constexpr (N >= 2) fill_order<2>(d, y);
    if constexpr (N >= 3) fill_order<3>(d, y);
    if constexpr (N >= 4) fill_order<4>(d, y);
    if constexpr (N >= 5) fill_order<5>(d, y);
    if constexpr (N >= 6) fill_order<6>(d, y);
  }
  template <int K, class D>
  LAPLACE_HD void fill_order(const D& d, double y) {
    const auto t = log_density_dk<K>(d, y);
    for (int i = 0; i < sym_tensor<P, K>::size; ++i) v[offset(K) + i] = t[i];
  }
  // l_U for the k parameter indices idx[0..k-1], in any order; k = 0 gives 0 (never used)
  LAPLACE_HD double at(const int* idx, int k) const {
    int s[6];
    for (int i = 0; i < k; ++i) {  // insertion sort
      int j = i;
      for (; j > 0 && s[j - 1] > idx[i]; --j) s[j] = s[j - 1];
      s[j] = idx[i];
    }
    int pos = 0, prev = 0;  // sym_tensor_position with a run-time order
    for (int j = 0; j < k; ++j) {
      for (int w = prev; w < s[j]; ++w) pos += binomial_coef(P - w + k - j - 2, k - j - 1);
      prev = s[j];
    }
    return k == 0 ? 0.0 : v[offset(k) + pos];
  }
};

LAPLACE_HD_FORCEINLINE int popcount4(int m) { return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1) + ((m >> 3) & 1); }

// l over the multiset {head...} + {c[i] : bit i of mask}
template <class L>
LAPLACE_HD_FORCEINLINE double l_over(const L& L_, const int* head, int nh, const int* c, int mask) {
  int idx[6], k = 0;
  for (int i = 0; i < nh; ++i) idx[k++] = head[i];
  for (int i = 0; i < 4; ++i)
    if (mask >> i & 1) idx[k++] = c[i];
  return L_.at(idx, k);
}

// the integrands of d_C I_ab (without the density) for one C = c[0..K-1] and every
// pair (a, b), written to out[pair * stride]; a pair with valid[pair * stride] false
// gets 0. B_T and the derivatives of l over (a + subsets of C) are looked up once
// for all the pairs.
template <int K, int P, class L>
LAPLACE_HD void info_integrands(const L& L_, const int* c, info_form form, const bool* valid, double* out,
                                int stride) {
  constexpr int full = (1 << K) - 1;
  double B[1 << K], lc[1 << K];
  B[0] = 1.0;
  for (int U = 1; U <= full; ++U) lc[U] = l_over(L_, nullptr, 0, c, U);
  for (int mask = 1; mask <= full; ++mask) {  // B_mask = sum over the block U holding the lowest bit
    const int low = mask & -mask, rest = mask ^ low;
    double acc = 0.0;
    for (int sub = rest;; sub = (sub - 1) & rest) {
      const int U = sub | low;
      acc += lc[U] * B[mask ^ U];
      if (sub == 0) break;
    }
    B[mask] = acc;
  }
  double la[P][1 << K];  // l over a + V (score form)
  if (form == info_form::score)
    for (int a = 0; a < P; ++a)
      for (int V = 0; V <= full; ++V) la[a][V] = l_over(L_, &a, 1, c, V);
  int pair = 0;
  for (int a = 0; a < P; ++a)
    for (int b = a; b < P; ++b, ++pair) {
      double& o = out[pair * stride];
      if (!valid[pair * stride]) {
        o = 0.0;
        continue;
      }
      const int ab[2] = {a, b};
      double total = 0.0;
      for (int T = 0; T <= full; ++T) {
        const int U = full ^ T;
        double inner;
        if (form == info_form::hessian) {
          inner = -l_over(L_, ab, 2, c, U);
        } else {
          inner = 0.0;
          for (int V = U;; V = (V - 1) & U) {
            inner += la[a][V] * la[b][U ^ V];
            if (V == 0) break;
          }
        }
        total += inner * B[T];
      }
      o = total;
    }
}

// the regularity orders as a value (a static constexpr array cannot be indexed on the device)
template <class D>
struct regularity_orders {
  int r[D::n_params];
};
template <class D>
constexpr regularity_orders<D> regularities_of() {
  regularity_orders<D> o{};
  for (int j = 0; j < D::n_params; ++j) o.r[j] = D::params[j].regularity;
  return o;
}

// whether the component (a, b, C) can be computed in this form (see the header)
template <class D, int K>
LAPLACE_HD bool info_component_valid(int a, int b, const int* c, info_form form) {
  constexpr regularity_orders<D> R = regularities_of<D>();
  for (int j = 0; j < D::n_params; ++j) {
    const int r = R.r[j];
    if (r >= infinitely_smooth) continue;
    const int na = a == j, nb = b == j;
    int nc = 0;
    for (int i = 0; i < K; ++i) nc += c[i] == j;
    if (form == info_form::hessian) {
      if (na + nb + nc > r + 1) return false;
    } else if (nc > 0 && (na > nb ? na : nb) + nc > r + 1) {
      return false;
    }
  }
  return true;
}

// the next sorted multi-index C of length K over P parameters, in lexicographic order
template <int P, int K>
LAPLACE_HD_FORCEINLINE void next_multi_index(int* c) {
  int i = K - 1;
  while (i >= 0 && c[i] == P - 1) --i;
  if (i < 0) return;
  ++c[i];
  for (int j = i + 1; j < K; ++j) c[j] = c[i];
}

}  // namespace detail_generic

/**
 * \ingroup distrib_generic
 * \brief Return type for expected information matrix and its parameter derivatives.
 *
 * Resolves to `sym_tensor<P, 2>` for \f$K = 0\f$ (the \f$P \times P\f$ symmetric information matrix),
 * or to `expected_info_tensor<P, K>` for \f$K \ge 1\f$ (a derivative tensor of symmetric matrices).
 *
 * \tparam P Number of parameters in the distribution.
 * \tparam K Order of parameter derivatives (\f$0 \le K \le 4\f$).
 *
 * \see expected_info_by_integral(), distrib_expected_info()
 */
template <int P, int K>
using expected_info_result = std::conditional_t<K == 0, sym_tensor<P, 2>, expected_info_tensor<P, K>>;

// I(theta) (K = 0) or its K-th derivative (K = 1..4) by the generic route, in the given form.
// Every component is integrated on the same nodes (or summed over the same terms), so
// the derivatives of l are evaluated once a node for all of them; a level is refined
// until every component passes its own stopping test.
/**
 * \ingroup distrib_generic
 * \brief Expected Fisher information matrix or its parameter derivative by numerical integration.
 *
 * Computes \f$I_{ab}\f$ (\f$K = 0\f$) or \f$\frac{\partial^K}{\partial \theta_C} I_{ab}\f$ (\f$K = 1, \dots, 4\f$)
 * by simultaneous numerical quadrature or summation in one pass:
 * - In hessian form:
 *   \f[
 *   \frac{\partial^K}{\partial \theta_C} I_{ab} = -\mathbb{E}\left[ \sum_{T \subseteq C} l_{ab + (C \setminus T)} B_T \right]
 *   \f]
 * - In score form:
 *   \f[
 *   \frac{\partial^K}{\partial \theta_C} I_{ab} = \mathbb{E}\left[ \sum_{T \subseteq C} S_{C \setminus T} B_T \right], \quad S_U = \sum_{V \subseteq U} l_{a + V} l_{b + (U \setminus V)}
 *   \f]
 * where \f$l = \log f\f$, \f$l_U\f$ is its derivative over the multiset \f$U\f$ of parameter indices,
 * and \f$B_T\f$ is the complete Bell polynomial of these derivatives over the sub-multiset \f$T\f$ of
 * \f$C\f$ (see the module page).
 *
 * \tparam K Derivative order with respect to parameters (\f$0 \le K \le 4\f$, total order up to 6).
 * \tparam F Information representation format (info_form::hessian or info_form::score).
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \return An expected_info_result<D::n_params, K> holding the computed tensor.
 *
 * \par Domain and special values
 * - Returns quiet NaN for components where quadrature or summation fails to converge.
 * - Returns quiet NaN for components violating the finite-regularity kink condition:
 *   hessian form when \f$n_a + n_b + n_C > r_j + 1\f$; score form when \f$n_C > 0\f$ and \f$\max(n_a, n_b) + n_C > r_j + 1\f$.
 *
 * \par Measured accuracy
 * Worst error in eps of the largest component:
 * at most 3.9 eps (hessian) / 3.2 eps (score) on gaussian1;
 * 0.0–4.0 eps (hessian) / 0.7–3.2 eps (score) on gamma1;
 * 0.0–4.7 eps (hessian) / 0.0–10.9 eps (score) on poisson(3.5);
 * on poisson(40) up to 247 eps (hessian, K = 4) and 116 eps (score, K = 3);
 * on laplace, the components that are not NaN within the test bounds 64 eps (K = 0, 1), 256 (K = 2), 1024 (K = 4).
 *
 * \see distrib_expected_info(), generic_info_form(), info_form
 */
template <int K, info_form F, class D>
LAPLACE_HD expected_info_result<D::n_params, K> expected_info_by_integral(const D& d) {
  static_assert(K >= 0 && K <= 4, "the expected information has total order 6: K <= 4");
  constexpr int P = D::n_params, KC = K == 0 ? 1 : K;
  constexpr int n_pairs = sym_tensor<P, 2>::size, n_c = K == 0 ? 1 : sym_tensor<P, KC>::size, M = n_pairs * n_c;
  constexpr int N = F == info_form::hessian ? K + 2 : K + 1;
  // the components in the order of the result: the pair (a, b), then C in lexicographic order
  bool valid[M];
  {
    int m = 0;
    for (int a = 0; a < P; ++a)
      for (int b = a; b < P; ++b) {
        int c[KC] = {};
        for (int pos = 0; pos < n_c; ++pos, ++m, detail_generic::next_multi_index<P, K>(c))
          valid[m] = detail_generic::info_component_valid<D, K>(a, b, c, F);
      }
  }
  const auto r = detail_generic::expectation_vec_by_integral<M>(d, [&](double y, double* v) {
    detail_generic::log_density_partials<P, N> L_;
    L_.fill(d, y);
    int c[KC] = {};
    for (int pos = 0; pos < n_c; ++pos, detail_generic::next_multi_index<P, K>(c))
      detail_generic::info_integrands<K, P>(L_, c, F, valid + pos, v + pos, n_c);
  });
  expected_info_result<P, K> out;
  int m = 0;
  for (int pair = 0; pair < n_pairs; ++pair)
    for (int pos = 0; pos < n_c; ++pos, ++m) {
      const double v = valid[m] && r.converged[m] ? r.value[m] : math::detail::quiet_nan();
      if constexpr (K == 0) out[pair] = v;
      else out.pair[pair][pos] = v;
    }
  return out;
}

// The default form of the generic route: the score form for every distribution
// (decided 2026-11-12). It is the only form that gives I when a parameter has a
// finite regularity; on smooth distributions it ties the hessian form in accuracy
// and avoids the cancellation of -E[l_aa] seen on pig2.
/**
 * \ingroup distrib_generic
 * \brief Default representation format for the generic expected information route.
 *
 * Always returns info_form::score. The score form is the only formulation valid
 * in the presence of parameters with finite regularity, matches the hessian form in
 * accuracy on the smooth distributions tested, and \f$-\mathbb{E}[l_{aa}]\f$ can lose digits to cancellation
 * where \f$\mathbb{E}[l_a^2]\f$ does not.
 *
 * \tparam D Distribution type.
 * \return Constant info_form::score.
 *
 * \see info_form, expected_info_by_integral(), distrib_expected_info()
 */
template <class D>
LAPLACE_HD constexpr info_form generic_info_form() {
  return info_form::score;
}

// I(theta) or its K-th derivative: the distribution's member when it has one, the generic route otherwise.
/**
 * \ingroup distrib_generic
 * \brief Expected Fisher information matrix or its parameter derivative, dispatching to specialized member or generic route.
 *
 * Calls the distribution's specialized member `expected_info()` (for \f$K = 0\f$) or `expected_info_dK()`
 * (for \f$K = 1, \dots, 4\f$) if implemented; otherwise evaluates the expected information via
 * expected_info_by_integral() using generic_info_form(). The choice is made at compile time.
 *
 * \tparam K Derivative order with respect to parameters (\f$0 \le K \le 4\f$).
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \return An expected_info_result<D::n_params, K> holding the information matrix or derivative tensor.
 *
 * \par Domain and special values
 * - Follows the member function if provided, or expected_info_by_integral() (quiet NaN for non-converged or kink-excluded components).
 *
 * \par Measured accuracy
 * That of the member, or that of expected_info_by_integral().
 *
 * \see expected_info_by_integral(), generic_info_form()
 */
template <int K, class D>
LAPLACE_HD expected_info_result<D::n_params, K> distrib_expected_info(const D& d) {
  if constexpr (K == 0 && requires { d.expected_info(); }) return d.expected_info();
  else if constexpr (K == 1 && requires { d.expected_info_d1(); }) return d.expected_info_d1();
  else if constexpr (K == 2 && requires { d.expected_info_d2(); }) return d.expected_info_d2();
  else if constexpr (K == 3 && requires { d.expected_info_d3(); }) return d.expected_info_d3();
  else if constexpr (K == 4 && requires { d.expected_info_d4(); }) return d.expected_info_d4();
  else return expected_info_by_integral<K, generic_info_form<D>()>(d);
}

}  // namespace laplace::distrib
