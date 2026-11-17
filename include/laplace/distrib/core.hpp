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
//   start_value(summary)          (host) a starting theta from a distrib_sample_summary
//                                 (generic_start.hpp), strictly inside the domain
//   n_atoms, atom_point(i)        (optional) points carrying mass beyond what kind implies,
//   atom_log_mass(i)              read through distrib_n_atoms() and its two companions
//   param_kinks(j, y)             points of theta_j, placed by y, at which l is not smooth
//                                 (params[j].axis_kinks holds the ones fixed on theta_j's axis)
//   y_regularity, y_kinks()       (continuous) regularity of l in y and its non-smooth points
//   logpdf_dy_dk<J, K>(y)         (continuous) J derivatives in y and the symmetric tensor of
//                                 order K in theta, computed only when called (J + K <= 6)
//
// Derivatives are symmetric tensors packed over the sorted indices
// i_1 <= ... <= i_K in lexicographic order; with two parameters the
// component with b indices equal to 1 is at position b.
#pragma once

#include <laplace/core/config.hpp>
#include <Kokkos_NumericTraits.hpp>

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

// The descriptor of a parameter: name, open natural domain (lower, upper) or
// circular, regularity order of log f, whether the ends of the support move
// with it, the regularity order of log F and log(1 - F) (what a censored,
// truncated or interval-censored observation contributes; for a location
// parameter usually one order more than log f), and the non-smooth points of
// the parameter's own axis. The non-smooth points of log F in y are those of
// log f (param_kinks).
//
// The two kinds of non-smooth point are different things and are declared
// apart. param_kinks(j, y) gives a point of theta_j whose position depends on
// the observation, where l has a kink in y as well (the Laplace and the enet
// at mu = y); regularity is the order up to which l stays differentiable in
// theta_j there, and it holds at every value of the parameter. axis_kinks
// gives the points of theta_j's own axis at which l is not smooth in theta_j
// while being C^inf everywhere else on that axis and C^inf in y throughout
// (the skew normal's gamma = 0, where the map to the classic triple runs
// through a cube root). A finite regularity cannot say this: it would make the
// generic routes drop the derivatives in that parameter at every value of it,
// not only at the one point where they do not exist.
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
  /// Points of this parameter's own axis at which \f$\log f\f$ is not smooth (none by default).
  distrib_kinks axis_kinks = {};
};

// Atoms. A law may carry mass on points that its kind does not already account
// for: a continuous family with a point mass (a zero-adjusted gamma) is neither
// continuous nor discrete in the usual reading, and a route that integrates has
// to know where that mass sits. An atom here is a point carrying mass BEYOND
// what kind implies, so a discrete family declares none, its kind already
// saying that the dominating measure counts the support. A family that has them
// declares
//
//   static constexpr int n_atoms = 1;
//   LAPLACE_HD double atom_point(int i) const;      where the mass sits
//   LAPLACE_HD double atom_log_mass(int i) const;   log P(Y = atom_point(i))
//
// and nothing else. In particular no derivative function: logpdf(y) returns the
// log mass at an atom, by the rule below, so the mass's derivatives to order
// six are logpdf_dk<K> at that point, which every family already supplies;
// atom_log_mass only spares a route the call.
//
// Two rules come with the declaration. logpdf(y) is one log-density with
// respect to Lebesgue measure plus a counting measure on the atoms, so it gives
// the log mass at an atom and the log density elsewhere; a quadrature node that
// landed on an atom would read a mass as if it were a density, and what stops
// it is that every integrand of the generic cumulative and expectation routes
// returns zero at an atom point (detail_generic::at_an_atom), the continuous
// part carrying no mass there. Reporting the atom through y_kinks() is NOT
// sufficient, and this comment said for a time that it was: a cut makes the
// atom a panel END, and tanh_sinh_quad does evaluate its integrand at a panel
// end when a node's distance to it underflows, so a cut puts the point exactly
// where a node can reach it (measured on 2026-11-17; docs/numerics.md, "The
// generic cumulative routes of a law that carries atoms"). The atom points are
// still passed as cuts, for the panel split they buy. And logpdf_dy_dk<J, K>
// with J >= 1 has no meaning at an atom, the derivative in y of a mass not
// being a quantity.
//
// Every route asks through the three functions below and never through kind,
// which is the defect to avoid: a route that asks the object what it carries
// cannot be wrong about a family it has never heard of. They answer for a
// family that declares nothing, so no family without atoms is touched.
/**
 * \ingroup distrib_core
 * \brief Number of atoms a distribution declares beyond what its kind implies (0 when it declares none).
 * \tparam D Distribution type.
 * \return `D::n_atoms` when the member exists, 0 otherwise.
 */
template <class D>
LAPLACE_HD constexpr int distrib_n_atoms() {
  if constexpr (requires { D::n_atoms; }) return D::n_atoms;
  else return 0;
}

/**
 * \ingroup distrib_core
 * \brief Position of the distribution's atom \f$i\f$.
 * \param d Distribution carrying the atom.
 * \param i Atom index, \f$0 \le i < \f$ distrib_n_atoms().
 * \return `d.atom_point(i)`, or quiet NaN when the distribution declares no atom.
 */
template <class D>
LAPLACE_HD double distrib_atom_point(const D& d, int i) {
  if constexpr (requires { d.atom_point(i); }) return d.atom_point(i);
  else return (void)d, (void)i, Kokkos::Experimental::quiet_NaN_v<double>;
}

/**
 * \ingroup distrib_core
 * \brief Log mass the distribution puts on its atom \f$i\f$.
 * \param d Distribution carrying the atom.
 * \param i Atom index, \f$0 \le i < \f$ distrib_n_atoms().
 * \return `d.atom_log_mass(i)`, or \f$-\infty\f$ when the distribution declares no atom.
 */
template <class D>
LAPLACE_HD double distrib_atom_log_mass(const D& d, int i) {
  if constexpr (requires { d.atom_log_mass(i); }) return d.atom_log_mass(i);
  else return (void)d, (void)i, -distrib_infinity;
}

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
 * \see distrib_core
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
 * \see distrib_core
 */
template <int P, int K>
struct expected_info_tensor {
  /// Number of unique parameter pairs \f$\binom{P + 1}{2}\f$.
  static constexpr int n_pairs = sym_tensor_size(P, 2);
  /// Array of derivative tensors, one per parameter pair.
  sym_tensor<P, K> pair[n_pairs];
};

/**
 * \ingroup distrib_start
 * \brief Sample summary statistics read by distribution \c start_value() members.
 *
 * Holds sample size, mean, variance (divisor \f$n\f$), type-7 quartiles, mean absolute
 * deviation from the median, extremes, and moments over positive observations (\f$y_i > 0\f$).
 * Computed on the host by summarize_sample().
 * \see summarize_sample(), distrib_start_value(), distrib_start
 */
// What the starting values of a fit read from a sample (generic_start.hpp
// computes it, each distrib's start_value member reads it). Divisor n in the
// variances; the quartiles are R's type 7; the log moments and the mean of 1/y
// are taken over y > 0 only (positive_count of them).
struct distrib_sample_summary {
  /// Total number of observations \f$n\f$.
  double count = 0.0;
  /// Sample mean \f$\bar{y} = \frac{1}{n}\sum_i y_i\f$.
  double mean = 0.0;
  /// Sample variance with divisor \f$n\f$, \f$\frac{1}{n}\sum_i (y_i - \bar{y})^2\f$.
  double variance = 0.0;
  /// Sample skewness with divisor \f$n\f$, the third central moment over the second to the power \f$3/2\f$; zero for a sample of no spread.
  double skewness = 0.0;
  /// Sample median (type-7 quantile at \f$p = 0.5\f$).
  double median = 0.0;
  /// Mean absolute deviation from the median, \f$\frac{1}{n}\sum_i |y_i - q_{0.5}|\f$.
  double mean_abs_dev_median = 0.0;  // mean of |y - median|
  /// Lower quartile (type-7 quantile at \f$p = 0.25\f$).
  double lower_quartile = 0.0;
  /// Upper quartile (type-7 quantile at \f$p = 0.75\f$).
  double upper_quartile = 0.0;
  /// Sample minimum \f$\min_i y_i\f$.
  double minimum = 0.0;
  /// Sample maximum \f$\max_i y_i\f$.
  double maximum = 0.0;
  /// Number of positive observations (\f$y_i > 0\f$).
  double positive_count = 0.0;
  /// Mean of \f$\log y_i\f$ over positive observations (\f$y_i > 0\f$).
  double mean_log = 0.0;
  /// Variance of \f$\log y_i\f$ with divisor \f$n_+\f$ over positive observations (\f$y_i > 0\f$).
  double variance_log = 0.0;
  /// Mean of \f$1 / y_i\f$ over positive observations (\f$y_i > 0\f$).
  double mean_inverse = 0.0;
};

/**
 * \ingroup distrib_start
 * \brief Moves a candidate parameter value strictly inside its open natural domain.
 *
 * At a finite lower bound \f$l\f$ (\f$v \le l\f$), returns \f$10^{-8}\max(|v|, 1)\f$ when \f$l = 0\f$
 * and \f$l + 16\varepsilon|l|\f$ otherwise. At an upper bound \f$u\f$ (\f$v \ge u\f$), returns
 * \f$10^{300}\f$ when \f$u = +\infty\f$, \f$-10^{-8}\f$ when \f$u = 0\f$, and \f$u - 16\varepsilon|u|\f$
 * otherwise, where \f$\varepsilon\f$ is double-precision machine epsilon. Values inside \f$(l, u)\f$
 * and \c NaN stay unchanged.
 * \param v Candidate parameter value.
 * \param p Parameter descriptor carrying \c lower and \c upper domain bounds.
 * \return Value strictly inside the open domain \f$(l, u)\f$, or \c NaN if \p v is \c NaN.
 * \see distrib_param_info, distrib_start_value(), distrib_start
 */
// v moved strictly inside the open domain (lower, upper) of a parameter: at a
// lower end 0 to 1e-8 max(|v|, 1), at another finite end by 16 eps relative
// (1e-8 absolute at 0), at an infinite one to 1e300; NaN stays NaN.
inline double start_inside(double v, const distrib_param_info& p) {
  constexpr double e16 = 16.0 * std::numeric_limits<double>::epsilon();
  if (v <= p.lower) {
    if (p.lower == 0.0) return 1e-8 * (v < -1.0 || v > 1.0 ? (v < 0 ? -v : v) : 1.0);
    return p.lower + (p.lower < 0 ? -p.lower : p.lower) * e16;
  }
  if (v >= p.upper) {
    if (p.upper == distrib_infinity) return 1e300;
    if (p.upper == 0.0) return -1e-8;
    return p.upper - (p.upper < 0 ? -p.upper : p.upper) * e16;
  }
  return v;
}

}  // namespace laplace::distrib
