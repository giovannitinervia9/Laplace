// The pieces shared by every univariate distribution (laplace::distrib);
// the structure is described in note/distrib.md.
//
// A distribution is a struct whose fields are its natural parameters theta
// (copied by value into kernels), with static descriptors and one const
// LAPLACE_HD member function per quantity and per order:
//
//   kind                          continuous or discrete
//   n_params, params[j]           the descriptor of theta_j (distrib_param_info)
//   logpdf(y) / logpmf(y)         the log-density or log-mass
//   logpdf_d1(y) ... _d6(y)       d^K l / dtheta^K, a sym_tensor<P, K>
//   expected_info()               I(theta) = -E[d^2 l], a sym_tensor<P, 2>
//   expected_info_d1() ... _d4()  its derivatives in theta (total order 6)
//   cdf, ccdf, log_cdf, log_ccdf  F(y), 1 - F(y) and their logarithms
//   quantile(p)                   continuous: F^-1(p)
//   support_lower(), _upper()     the ends of the support at theta
//   center(), scale()             rough location and spread, for quadratures
//   param_kinks(j, y)             points of theta_j at which l is not smooth
//
// Derivatives are symmetric tensors packed over the sorted indices
// i_1 <= ... <= i_K in lexicographic order; with two parameters the
// component with b indices equal to 1 is at position b.
#pragma once

#include <laplace/core/config.hpp>

#include <limits>

namespace laplace::distrib {

/**
 * \ingroup distrib_core
 * \brief Positive infinity for distribution support and parameter domain bounds.
 */
inline constexpr double distrib_infinity = std::numeric_limits<double>::infinity();

/**
 * \ingroup distrib_core
 * \brief Categorization of probability distributions by support and reference measure.
 *
 * Distinguishes continuous distributions with densities on \f$\mathbb{R}\f$ from
 * discrete distributions with probability mass functions on integers.
 */
enum class distrib_kind : int { continuous, discrete };

// Regularity order r_j of theta_j -> l(theta; y): C^inf, or C^r with kinks.
/**
 * \ingroup distrib_core
 * \brief Regularity order representing infinitely differentiable (\f$C^\infty\f$) parameters.
 */
inline constexpr int infinitely_smooth = 1 << 30;

// The descriptor of a parameter: name, open natural domain (lower, upper) or
// circular, regularity order of log f, whether the ends of the support move
// with it, and the regularity order of log F and log(1 - F) (what a censored,
// truncated or interval-censored observation contributes; for a location
// parameter usually one order more than log f). The non-smooth points of log F
// are those of log f (param_kinks).
/**
 * \ingroup distrib_core
 * \brief Descriptor of a distribution parameter's domain, smoothness, and support properties.
 */
struct distrib_param_info {
  /// Parameter identifier name.
  const char* name;
  /// Lower bound of the open natural parameter domain.
  double lower;
  /// Upper bound of the open natural parameter domain.
  double upper;
  /// Whether the parameter domain is circular (periodic).
  bool circular;
  /// Differentiability order of the log-density \f$\log f\f$ in this parameter (\c infinitely_smooth or non-negative integer).
  int regularity;
  /// Whether varying this parameter moves the support boundary points.
  bool moves_support;
  /// Differentiability order of the cumulative functions \f$\log F\f$ and \f$\log(1 - F)\f$ in this parameter.
  int cdf_regularity;
};

// Points at which l is not smooth in one parameter (or in y), at most 2, and
// the order up to which it stays continuous there.
/**
 * \ingroup distrib_core
 * \brief Description of non-smooth points in a parameter or in the observation argument.
 */
struct distrib_kinks {
  /// Number of non-smooth points (at most 2).
  int count = 0;
  /// Coordinates of the non-smooth points.
  double point[2] = {0.0, 0.0};
  /// Highest derivative order that remains continuous across the kink.
  int continuous_order = 0;
};

/**
 * \ingroup distrib_core
 * \brief Evaluates the binomial coefficient \f$\binom{n}{k}\f$.
 *
 * \param n Total items (\f$n \ge 0\f$).
 * \param k Selected items (\f$0 \le k \le n\f$).
 * \return Binomial coefficient \f$\binom{n}{k}\f$, or 0 if \f$k < 0\f$ or \f$k > n\f$.
 */
LAPLACE_HD constexpr int binomial_coef(int n, int k) {
  if (k < 0 || k > n) return 0;
  int r = 1;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

// Number of components of a symmetric tensor of order K in P parameters.
/**
 * \ingroup distrib_core
 * \brief Computes the number of unique components in a symmetric tensor of order \f$K\f$ in \f$P\f$ variables.
 *
 * Evaluates \f$\binom{P + K - 1}{K}\f$.
 * \param P Number of variables or parameters (\f$P \ge 1\f$).
 * \param K Tensor derivative order (\f$K \ge 0\f$).
 * \return Number of packed elements.
 */
LAPLACE_HD constexpr int sym_tensor_size(int P, int K) { return binomial_coef(P + K - 1, K); }

// Position of the sorted multi-index idx[0] <= ... <= idx[K-1] (lexicographic).
/**
 * \ingroup distrib_core
 * \brief Computes the lexicographical storage index for a sorted multi-index in a packed symmetric tensor.
 *
 * \tparam P Number of variables or parameters.
 * \tparam K Tensor derivative order.
 * \param idx Array of \f$K\f$ sorted indices \f$0 \le idx[0] \le \dots \le idx[K-1] < P\f$.
 * \return Linear index in the packed array `v`.
 */
template <int P, int K>
LAPLACE_HD constexpr int sym_tensor_position(const int (&idx)[K]) {
  int pos = 0, prev = 0;
  for (int j = 0; j < K; ++j) {
    for (int v = prev; v < idx[j]; ++v) pos += binomial_coef(P - v + K - j - 2, K - j - 1);
    prev = idx[j];
  }
  return pos;
}

// A symmetric tensor of order K in P parameters, packed.
/**
 * \ingroup distrib_core
 * \brief Fixed-size container for a packed symmetric tensor of order \f$K\f$ in \f$P\f$ variables.
 *
 * Stores unique components in an array `v` of size \f$\binom{P + K - 1}{K}\f$ ordered lexicographically
 * by sorted multi-indices.
 * \tparam P Number of variables or parameters.
 * \tparam K Tensor derivative order.
 */
template <int P, int K>
struct sym_tensor {
  /// Number of unique tensor components.
  static constexpr int size = sym_tensor_size(P, K);
  /// Array of packed tensor elements.
  double v[size] = {};
  /// Mutable element access by packed index.
  LAPLACE_HD double& operator[](int i) { return v[i]; }
  /// Const element access by packed index.
  LAPLACE_HD const double& operator[](int i) const { return v[i]; }
};

// A derivative of order K of the expected information I_ab: for each pair
// a <= b (in sym_tensor<P, 2> order) the symmetric tensor of order K.
/**
 * \ingroup distrib_core
 * \brief Container for order-\f$K\f$ derivatives of the \f$P \times P\f$ expected Fisher information matrix.
 *
 * For each pair of parameters \f$a \le b\f$ in \c sym_tensor<P, 2> order, `pair[p]` holds the
 * symmetric tensor of order \f$K\f$ representing \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\theta)\f$.
 * \tparam P Number of parameters.
 * \tparam K Derivative order of the expected information matrix.
 */
template <int P, int K>
struct expected_info_tensor {
  /// Number of unique parameter pairs \f$\binom{P + 1}{2}\f$.
  static constexpr int n_pairs = sym_tensor_size(P, 2);
  /// Array of derivative tensors, one per parameter pair.
  sym_tensor<P, K> pair[n_pairs];
};

}  // namespace laplace::distrib
