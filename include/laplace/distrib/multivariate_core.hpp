// The pieces shared by the multivariate distributions (laplace::distrib); the
// design is in note/distrib_multivariate.md, decided with Giovanni on
// 2026-11-15, and its typed parameter list in note/multivariate_typed_params.md
// (2026-11-17). distrib/core.hpp (the univariate structure) is not changed.
//
// A multivariate distrib declares P typed parameters (params[P]): vectors
// (vector, simplex_vector), which are dense, and scalars and matrices, which
// are indexed. The coordinates of the dense parameters are concatenated, in
// declaration order, in one array x; those of the indexed parameters in one
// array theta of theta_size() doubles. A distrib holds its structure by value
// (copied into kernels) and splits the work in two:
//
//   state_size(), scratch_size()  doubles of the state and of the working scratch
//   theta_state(theta, w)         writes into w[0, state_size()) what depends on
//                                 theta only (log det, the matrix of the quadratic
//                                 form, the factor) and returns a small state that
//                                 points into w and at theta; it uses
//                                 w[state_size(), state_size() + scratch_size())
//                                 as scratch
//   logpdf(state, y, x)           the log-density of the unit y
//   logpdf_dk<K, A...>(state, y, x, idx, out, w)
//                                 K derivatives in theta along the sorted tuple
//                                 idx (ignored when K = 0) and A_j in the j-th
//                                 dense parameter (omitted trailing orders are 0),
//                                 sum A + K <= 6; out is the Kronecker product,
//                                 row-major in declaration order, of the packed
//                                 symmetric tensors of order A_j over each dense
//                                 parameter's coordinates: a scalar (order 0), a
//                                 vector (1), a packed symmetric matrix (2, by rows
//                                 of the lower triangle, entry (i, j) at
//                                 i (i + 1)/2 + j), a sym_tensor (>= 3, the sorted
//                                 tuples in lexicographic order); returns false,
//                                 out untouched, when the block is zero by
//                                 structure; w has scratch_size() doubles
//   logpdf_dy_dk<J, K, A...>      J derivatives in y, K in theta and A_j in the j-th
//                                 dense parameter, sum J + K + A <= 6; out is the
//                                 packed tensor of order J over y, Kronecker (y
//                                 first) with the layout of logpdf_dk
//   normalization                 proper, degenerate or improper (below)
//   kind                          continuous (logpdf, logpdf_dk, logpdf_dy_dk) or
//                                 discrete (logpmf, logpmf_dk; no y derivatives)
//   y_coord_size()                the number of y coordinates the y derivatives run
//                                 over, dim() unless the unit is constrained (the
//                                 Dirichlet stores d values on the simplex and
//                                 differentiates in d - 1)
//
// The state stays valid while w and theta do. With a theta shared by the units
// the state is computed once; with a theta per unit each unit computes its own.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/pdmat/core.hpp>

#include <concepts>
#include <utility>
#include <vector>

namespace laplace::distrib {

/**
 * \ingroup distrib_multivariate
 * \brief Normalization regime allowed by a multivariate distribution's matrix parameter.
 *
 * Distinguishes three regimes for the quadratic/precision or covariance matrix:
 * - `proper`: a positive definite matrix in either the covariance or precision parametrization.
 * - `degenerate`: a singular covariance matrix (a proper distribution supported on \f$\mu + \operatorname{col}(\Sigma)\f$).
 * - `improper`: a singular precision matrix (a density flat along the null space of \f$Q\f$).
 */
// What the matrix parameter allows (note/distrib_multivariate.md section 4):
// proper, a positive definite matrix in either parametrization; degenerate, a
// singular covariance (a proper law on mu + col(Sigma)); improper, a singular
// precision (flat along its null space).
enum class distrib_normalization : int { proper, degenerate, improper };

/**
 * \ingroup distrib_multivariate
 * \brief Kind of a parameter of a multivariate distribution.
 *
 * Distinguishes indexed parameters (`scalar` and `matrix`), whose coordinates
 * are concatenated in \f$\theta\f$ and differentiated by sorted index tuple,
 * from dense parameters (`vector` and `simplex_vector`), whose coordinates are
 * concatenated in `x` and differentiated over all coordinates at once. A
 * `simplex_vector` is carried by its additive log-ratio coordinates; a `matrix`
 * is carried by the parameters of its `pdmat` or `psdmat` structure.
 *
 * \see distrib_mv_param_info, mv_param_dense(), distrib_multivariate
 */
enum class distrib_mv_param_kind : int { scalar, vector, simplex_vector, matrix };

/**
 * \ingroup distrib_multivariate
 * \brief Descriptor of one parameter of a multivariate distribution.
 *
 * \see distrib_mv_param_kind, mv_param_dense(), distrib_multivariate
 */
struct distrib_mv_param_info {
  /// Null-terminated parameter identifier name.
  const char* name;
  /// Storage and differentiation kind (`scalar`, `vector`, `simplex_vector`, or `matrix`).
  distrib_mv_param_kind kind;
  /// Lower bound of the open natural domain of each coordinate (`scalar` and `vector` parameters; per coordinate through mv_coord_lower()).
  double lower;
  /// Upper bound of the open natural domain of each coordinate (`scalar` and `vector` parameters; per coordinate through mv_coord_upper()).
  double upper;
  /// Differentiability order of the log-density in this parameter (`infinitely_smooth` or a non-negative integer).
  int regularity;
};

/**
 * \ingroup distrib_multivariate
 * \brief Whether a parameter kind is dense, with its coordinates concatenated in `x`.
 *
 * \param k Parameter kind.
 * \return `true` for `vector` and `simplex_vector`; `false` for `scalar` and `matrix`.
 * \see distrib_mv_param_kind, distrib_mv_param_info, distrib_multivariate
 */
LAPLACE_HD constexpr bool mv_param_dense(distrib_mv_param_kind k) {
  return k == distrib_mv_param_kind::vector || k == distrib_mv_param_kind::simplex_vector;
}

/**
 * \ingroup distrib_multivariate
 * \brief Slot constant (\f$-1\f$) for the indexed coordinate array \f$\theta\f$.
 *
 * A slot passed to `expected_info_dk()` is either the index in `params` of a
 * dense parameter or `theta_slot` for \f$\theta\f$.
 *
 * \see y_slot, distrib_multivariate
 */
inline constexpr int theta_slot = -1;
/**
 * \ingroup distrib_multivariate
 * \brief Slot constant (\f$-2\f$) for the observation \f$y\f$ in block queries.
 *
 * \see theta_slot, mv_slot_size(), mv_block_start(), mv_block_tensor_size(), distrib_multivariate
 */
inline constexpr int y_slot = -2;

/**
 * \ingroup distrib_multivariate
 * \brief Concept satisfied by multivariate distributions in `laplace::distrib`.
 *
 * Requires the compile-time constants `normalization`, `kind` and `n_params`,
 * the parameter descriptor array `params`, an associated `state_type`,
 * dimension and size queries (`dim()`, `y_coord_size()`, `param_size()`,
 * `param_offset()`, `theta_size()`, `state_size()`, `scratch_size()`),
 * parameter precomputation (`theta_state()`), and the value with its
 * first-order derivative blocks: for a continuous law `logpdf()`,
 * `logpdf_dk<0, 1>()`, `logpdf_dk<1>()` and `logpdf_dy_dk<1, 0>()`; for a
 * discrete law `logpmf()`, `logpmf_dk<0, 1>()` and `logpmf_dk<1>()`, and no
 * derivatives in \f$y\f$.
 *
 * \tparam D Multivariate distribution type to check.
 * \see distrib_normalization, distrib_mv_param_info, distrib_multivariate
 */
template <class D>
concept multivariate_distrib =
    requires(const D d, const double* p, double* w, const int* idx, const typename D::state_type& st) {
      { D::normalization } -> std::convertible_to<distrib_normalization>;
      { D::kind } -> std::convertible_to<distrib_kind>;
      { D::n_params } -> std::convertible_to<int>;
      { D::params[0] } -> std::convertible_to<distrib_mv_param_info>;
      { d.dim() } -> std::convertible_to<int>;
      { d.y_coord_size() } -> std::convertible_to<int>;
      { d.param_size(0) } -> std::convertible_to<int>;
      { d.param_offset(0) } -> std::convertible_to<int>;
      { d.theta_size() } -> std::convertible_to<int>;
      { d.state_size() } -> std::convertible_to<int>;
      { d.scratch_size() } -> std::convertible_to<int>;
      { d.theta_state(p, w) } -> std::same_as<typename D::state_type>;
    } &&
    (D::kind == distrib_kind::continuous
         ? requires(const D d, const double* p, double* w, const int* idx, const typename D::state_type& st) {
             { d.logpdf(st, p, p) } -> std::same_as<double>;
             { d.template logpdf_dk<0, 1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
             { d.template logpdf_dk<1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
             { d.template logpdf_dy_dk<1, 0>(st, p, p, idx, w, w) } -> std::same_as<bool>;
           }
         : requires(const D d, const double* p, double* w, const int* idx, const typename D::state_type& st) {
             { d.logpmf(st, p, p) } -> std::same_as<double>;
             { d.template logpmf_dk<0, 1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
             { d.template logpmf_dk<1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
           });

/**
 * \ingroup distrib_multivariate
 * \brief A multivariate law with the optional distribution function members.
 *
 * `log_cdf`, `log_ccdf` and `log_rect_prob` (with `cdf`, `ccdf` and the sizes
 * `cdf_value_scratch_size()` and `cdf_scratch_size(K, A)`), and their derivatives
 * `log_cdf_dk`, `log_ccdf_dk`, `log_rect_prob_dk`, `log_cdf_dy_dk`,
 * `log_ccdf_dy_dk` in the layout of `logpdf_dk` and `logpdf_dy_dk`. The concept
 * itself checks the sizes, `log_cdf`, `log_ccdf`, `log_rect_prob`, `log_cdf_dk` and
 * `log_rect_prob_dk`, not `cdf`, `ccdf` or the other derivatives. They are not
 * part of `multivariate_distrib`: a law declares them when it has them, and a
 * caller asks through this named concept.
 *
 * \tparam D Multivariate distribution type to check.
 * \see multivariate_distrib, distrib_multivariate
 */
template <class D>
concept mv_has_cdf = requires(const D d, const double* p, double* w, const int* idx, const typename D::state_type& st) {
  { d.cdf_value_scratch_size() } -> std::convertible_to<int>;
  { d.cdf_scratch_size(1, 0) } -> std::convertible_to<int>;
  { d.log_cdf(st, p, p, w) } -> std::same_as<double>;
  { d.log_ccdf(st, p, p, w) } -> std::same_as<double>;
  { d.log_rect_prob(st, p, p, p, w) } -> std::same_as<double>;
  { d.template log_cdf_dk<1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
  { d.template log_rect_prob_dk<0, 1>(st, p, p, p, idx, w, w) } -> std::same_as<bool>;
};

// Coordinate atoms (Giovanni, 2026-11-17). A law may put mass on the
// hyperplanes y_i = c of single coordinates, as a product of one-dimensional
// factors with atoms does: its logpdf is then the log density with respect to
// the product over the coordinates i of (Lebesgue measure + counting measure on
// coordinate i's atoms), which gives a factor's log mass on an atom coordinate
// and its log density elsewhere. Such a law declares n_coord_atoms(i) and
// coord_atom_point(i, a); a law that declares nothing has none. Derivatives in
// a coordinate of y that sits on one of its atoms have no meaning. The index i
// runs over the dim() stored values of y (0 <= i < dim()), not over the
// y_coord_size() coordinates of the derivatives in y, which are fewer for a
// constrained unit such as the Dirichlet's.

/**
 * \ingroup distrib_multivariate
 * \brief Number of atoms on coordinate \f$i\f$ of \f$y\f$ (0 when the law declares none).
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param i Coordinate of \f$y\f$, \f$0 \le i <\f$ `d.dim()`.
 * \return `d.n_coord_atoms(i)` when `D` declares it, 0 otherwise.
 * \see mv_coord_atom_point(), distrib_multivariate
 */
template <class D>
LAPLACE_HD int mv_n_coord_atoms(const D& d, int i) {
  if constexpr (requires { d.n_coord_atoms(i); }) return d.n_coord_atoms(i);
  else return (void)d, (void)i, 0;
}

/**
 * \ingroup distrib_multivariate
 * \brief Value \f$c\f$ of atom \f$a\f$ of coordinate \f$i\f$: the law puts mass on the hyperplane \f$y_i = c\f$.
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param i Coordinate of \f$y\f$.
 * \param a Atom index, \f$0 \le a <\f$ `mv_n_coord_atoms(d, i)`.
 * \return `d.coord_atom_point(i, a)`, or quiet NaN when the law declares no atom.
 * \see mv_n_coord_atoms(), distrib_multivariate
 */
template <class D>
LAPLACE_HD double mv_coord_atom_point(const D& d, int i, int a) {
  if constexpr (requires { d.coord_atom_point(i, a); }) return d.coord_atom_point(i, a);
  else return (void)d, (void)i, (void)a, Kokkos::Experimental::quiet_NaN_v<double>;
}

/**
 * \ingroup distrib_multivariate
 * \brief Lower bound of the natural domain of coordinate \f$i\f$ of parameter \f$j\f$.
 *
 * A law whose coordinates of one parameter have different domains (a product
 * of laws with a positive mean and laws with a real one) declares
 * `coord_lower(j, i)`; otherwise every coordinate has `D::params[j].lower`.
 *
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param j Parameter index in `D::params`.
 * \param i Coordinate of the parameter, \f$0 \le i <\f$ `d.param_size(j)`.
 * \return The lower bound of the open domain of that coordinate.
 * \see mv_coord_upper(), distrib_mv_param_info, distrib_multivariate
 */
template <class D>
inline double mv_coord_lower(const D& d, int j, int i) {  // host: reads D::params at a runtime index
  if constexpr (requires { d.coord_lower(j, i); }) return d.coord_lower(j, i);
  else return (void)d, (void)i, D::params[j].lower;
}

/**
 * \ingroup distrib_multivariate
 * \brief Upper bound of the natural domain of coordinate \f$i\f$ of parameter \f$j\f$.
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param j Parameter index in `D::params`.
 * \param i Coordinate of the parameter, \f$0 \le i <\f$ `d.param_size(j)`.
 * \return `d.coord_upper(j, i)` when `D` declares it, `D::params[j].upper` otherwise.
 * \see mv_coord_lower(), distrib_mv_param_info, distrib_multivariate
 */
template <class D>
inline double mv_coord_upper(const D& d, int j, int i) {  // host: reads D::params at a runtime index
  if constexpr (requires { d.coord_upper(j, i); }) return d.coord_upper(j, i);
  else return (void)d, (void)i, D::params[j].upper;
}

// Blocks (note/multivariate_typed_params.md section 4). A distrib may split y
// and each dense parameter into G blocks, block g owning the contiguous range
// [block_start(slot, g), block_start(slot, g + 1)) of each, and then return its
// derivatives block by block (logpdf_dk_blocks, logpdf_dy_dk_blocks): the
// concatenation over g of the dense layout restricted to block g's
// coordinates. A distrib that declares no n_blocks() has G = 1, and the helpers
// below fall back to the dense calls.

/**
 * \ingroup distrib_multivariate
 * \brief Number of coordinates in a block-query slot.
 *
 * Returns `d.y_coord_size()` when `slot == y_slot`, and `d.param_size(slot)` when
 * `slot` indexes a dense parameter in `D::params`.
 *
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param slot `y_slot` or the index in `D::params` of a dense parameter.
 * \return Number of coordinates in `slot`.
 * \see y_slot, mv_block_start(), mv_block_tensor_size(), distrib_multivariate
 */
template <class D>
LAPLACE_HD int mv_slot_size(const D& d, int slot) {
  return slot == y_slot ? d.y_coord_size() : d.param_size(slot);
}

/**
 * \ingroup distrib_multivariate
 * \brief Number of coordinate blocks \f$G\f$ of a multivariate distribution.
 *
 * Returns `d.n_blocks()` when `D` declares it, and \f$1\f$ otherwise.
 *
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \return Number of blocks \f$G \ge 1\f$.
 * \see mv_block_start(), mv_block_tensor_size(), mv_logpdf_dk_blocks(), mv_logpdf_dy_dk_blocks(), distrib_multivariate
 */
template <class D>
LAPLACE_HD int mv_n_blocks(const D& d) {
  if constexpr (requires { d.n_blocks(); }) return d.n_blocks();
  else return 1;
}

/**
 * \ingroup distrib_multivariate
 * \brief Starting coordinate index of block \f$g\f$ in a slot.
 *
 * Block \f$g\f$ (\f$0 \le g < G\f$) owns the contiguous coordinate range
 * \f$[s_g, s_{g+1})\f$, where \f$s_g =\f$ `mv_block_start(d, slot, g)`. Calls
 * `d.block_start(slot, g)` when `D` declares it; otherwise (\f$G = 1\f$)
 * returns \f$0\f$ for \f$g = 0\f$ and `mv_slot_size(d, slot)` for \f$g = 1\f$.
 *
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param slot `y_slot` or the index in `D::params` of a dense parameter.
 * \param g Block index satisfying \f$0 \le g \le G\f$, where \f$G =\f$ `mv_n_blocks(d)`.
 * \return Starting coordinate of block \f$g\f$, or `mv_slot_size(d, slot)` when \f$g = G\f$.
 * \see mv_n_blocks(), mv_slot_size(), mv_block_tensor_size(), distrib_multivariate
 */
template <class D>
LAPLACE_HD int mv_block_start(const D& d, int slot, int g) {
  if constexpr (requires { d.block_start(slot, g); }) return d.block_start(slot, g);
  else return g == 0 ? 0 : mv_slot_size(d, slot);
}

namespace detail_mv {
// the order in mu of logpdf_dk<K, A...> for a distrib with one dense parameter
template <int... A>
constexpr int dense_order() {
  static_assert(sizeof...(A) <= 1, "logpdf_dk: this distrib has one dense parameter (mu)");
  return (0 + ... + A);
}
// the index in params of the a-th dense parameter (-1 if none)
template <class D>
constexpr int dense_param_index(int a) {
  for (int j = 0, c = 0; j < D::n_params; ++j)
    if (mv_param_dense(D::params[j].kind)) {
      if (c == a) return j;
      ++c;
    }
  return -1;
}
template <class D, int A>
inline constexpr int dense_param_v = dense_param_index<D>(A);
// prod_j sym_tensor_size(n_jg, A_j) over the dense parameters j, block g
template <int... A, class D, std::size_t... I>
LAPLACE_HD int block_kron_size(const D& d, int g, std::index_sequence<I...>) {
  static_assert(((dense_param_v<D, int(I)> >= 0) && ...), "more orders than dense parameters");
  return (1 * ... *
          sym_tensor_size(mv_block_start(d, dense_param_v<D, int(I)>, g + 1) - mv_block_start(d, dense_param_v<D, int(I)>, g),
                          A));
}
}  // namespace detail_mv

/**
 * \ingroup distrib_multivariate
 * \brief Number of doubles in a symmetric tensor over one slot in the block layout.
 *
 * Evaluates \f$\sum_{g=0}^{G-1}\f$ `sym_tensor_size(n_g, order)`, where
 * \f$G =\f$ `mv_n_blocks(d)` and \f$n_g =\f$
 * `mv_block_start(d, slot, g + 1) - mv_block_start(d, slot, g)` is the size
 * of block \f$g\f$ in `slot`. With `slot == y_slot` and `order == J`, this is
 * the required length of `out` for `mv_logpdf_dy_dk_blocks<J, K>()`. Order 0
 * is the scalar: \f$1\f$.
 *
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param slot `y_slot` or the index in `D::params` of a dense parameter.
 * \param order Non-negative tensor order.
 * \return Total number of doubles across all \f$G\f$ blocks.
 * \see mv_n_blocks(), mv_block_start(), mv_logpdf_dy_dk_blocks(), sym_tensor_size(), distrib_multivariate
 */
template <class D>
LAPLACE_HD int mv_block_tensor_size(const D& d, int slot, int order) {
  if (order == 0) return 1;  // the scalar, summed over the blocks
  int n = 0;
  for (int g = 0, G = mv_n_blocks(d); g < G; ++g)
    n += sym_tensor_size(mv_block_start(d, slot, g + 1) - mv_block_start(d, slot, g), order);
  return n;
}

/**
 * \ingroup distrib_multivariate
 * \brief Number of doubles in the output of `mv_logpdf_dk_blocks<K, A...>()` or, with `y_order` \f$= J\f$, of `mv_logpdf_dy_dk_blocks<J, K, A...>()`.
 *
 * Evaluates \f$\sum_{g=0}^{G-1} \prod_j\f$ `sym_tensor_size(n_{jg}, A_j)`,
 * where \f$G =\f$ `mv_n_blocks(d)` and \f$n_{jg}\f$ is the number of
 * coordinates of the \f$j\f$-th dense parameter in block \f$g\f$ (omitted
 * trailing orders default to \f$0\f$), times `sym_tensor_size(n_yg, y_order)`
 * for the \f$n_{yg}\f$ coordinates of \f$y\f$ in block \f$g\f$. Order 0
 * (`y_order` and every \f$A_j\f$ equal to 0) is the scalar: \f$1\f$.
 *
 * \tparam A Derivative orders in the dense parameters, in declaration order.
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param y_order Derivative order in \f$y\f$ (0 by default).
 * \return Total number of doubles across all \f$G\f$ blocks.
 * \see mv_n_blocks(), mv_block_start(), mv_logpdf_dk_blocks(), sym_tensor_size(), distrib_multivariate
 */
template <int... A, class D>
LAPLACE_HD int mv_block_tensor_size(const D& d, int y_order = 0) {
  if (y_order == 0 && (true && ... && (A == 0))) return 1;  // the scalar, summed over the blocks
  int n = 0;
  for (int g = 0, G = mv_n_blocks(d); g < G; ++g)
    n += sym_tensor_size(mv_block_start(d, y_slot, g + 1) - mv_block_start(d, y_slot, g), y_order) *
         detail_mv::block_kron_size<A...>(d, g, std::make_index_sequence<sizeof...(A)>{});
  return n;
}

/**
 * \ingroup distrib_multivariate
 * \brief Evaluates `logpdf_dk<K, A...>()` in the block layout.
 *
 * Calls `d.logpdf_dk_blocks<K, A...>(st, y, x, idx, out, w)` when `D`
 * declares `n_blocks()`, and falls back to
 * `d.logpdf_dk<K, A...>(st, y, x, idx, out, w)` (one block, dense layout)
 * otherwise. `out` must hold `mv_block_tensor_size<A...>(d)` doubles.
 *
 * \tparam K Derivative order in \f$\theta\f$.
 * \tparam A Derivative orders in the dense parameters, in declaration order (omitted trailing orders are \f$0\f$).
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param st Precomputed state returned by `d.theta_state()`.
 * \param y Observation vector of length `d.dim()`.
 * \param x Concatenated dense parameter coordinates.
 * \param idx Sorted tuple of \f$K\f$ indices into \f$\theta\f$ (ignored when \f$K = 0\f$).
 * \param out Output buffer of length `mv_block_tensor_size<A...>(d)` doubles; untouched on a `false` return.
 * \param w Working scratch buffer of length `d.scratch_size()` doubles.
 * \return `true` if the block may be nonzero, `false` if it vanishes identically by structure (leaving `out` untouched).
 * \see mv_n_blocks(), mv_block_tensor_size(), mv_logpdf_dy_dk_blocks(), distrib_multivariate
 */
template <int K, int... A, class D>
LAPLACE_HD bool mv_logpdf_dk_blocks(const D& d, const typename D::state_type& st, const double* y, const double* x,
                                    const int* idx, double* out, double* w) {
  if constexpr (requires { d.n_blocks(); })
    return d.template logpdf_dk_blocks<K, A...>(st, y, x, idx, out, w);
  else
    return d.template logpdf_dk<K, A...>(st, y, x, idx, out, w);
}

/**
 * \ingroup distrib_multivariate
 * \brief Evaluates `logpdf_dy_dk<J, K, A...>()` in the block layout.
 *
 * Calls `d.logpdf_dy_dk_blocks<J, K, A...>(st, y, x, idx, out, w)` when `D`
 * declares `n_blocks()`, and falls back to
 * `d.logpdf_dy_dk<J, K, A...>(st, y, x, idx, out, w)` (one block, dense layout)
 * otherwise. `out` must hold `mv_block_tensor_size<A...>(d, J)` doubles.
 *
 * \tparam J Derivative order in the observation \f$y\f$.
 * \tparam K Derivative order in \f$\theta\f$.
 * \tparam A Derivative orders in the dense parameters, in declaration order (omitted trailing orders are \f$0\f$).
 * \tparam D Multivariate distribution type.
 * \param d Multivariate distribution.
 * \param st Precomputed state returned by `d.theta_state()`.
 * \param y Observation vector of length `d.dim()`.
 * \param x Concatenated dense parameter coordinates.
 * \param idx Sorted tuple of \f$K\f$ indices into \f$\theta\f$ (ignored when \f$K = 0\f$).
 * \param out Output buffer of length `mv_block_tensor_size<A...>(d, J)` doubles; untouched on a `false` return.
 * \param w Working scratch buffer of the length that `d.logpdf_dy_dk<J, K, A...>()` asks for.
 * \return `true` if the block may be nonzero, `false` if it vanishes identically by structure (leaving `out` untouched).
 * \see mv_n_blocks(), mv_block_tensor_size(), mv_logpdf_dk_blocks(), y_slot, distrib_multivariate
 */
template <int J, int K, int... A, class D>
LAPLACE_HD bool mv_logpdf_dy_dk_blocks(const D& d, const typename D::state_type& st, const double* y, const double* x,
                                       const int* idx, double* out, double* w) {
  if constexpr (requires { d.n_blocks(); })
    return d.template logpdf_dy_dk_blocks<J, K, A...>(st, y, x, idx, out, w);
  else
    return d.template logpdf_dy_dk<J, K, A...>(st, y, x, idx, out, w);
}

/**
 * \ingroup distrib_multivariate
 * \brief Host-side sample summary used to compute starting parameter values.
 *
 * Stores the sample size, dimension, sample mean, and sample covariance matrix
 * with divisor \f$n\f$ (packed by rows of the lower triangle).
 */
// What the starting values of a fit read from a sample of n units of
// dimension d: the mean and the covariance with divisor n, packed by rows of
// the lower triangle.
struct distrib_mv_sample_summary {
  /// Number of units \f$n\f$, as a double.
  double count = 0.0;
  /// Vector dimension \f$d\f$ of each unit.
  int dim = 0;
  /// Sample mean vector of length `dim`.
  std::vector<double> mean;
  /// Sample covariance matrix with divisor \f$n\f$, packed by rows of the lower triangle (length `d*(d+1)/2`).
  std::vector<double> covariance;
};

/**
 * \ingroup distrib_multivariate
 * \brief Compute a host-side sample mean and covariance summary from stacked units.
 *
 * Uses a two-pass algorithm on host memory: first computing the sample mean
 * across the `n` units (unit `i` at `y[i * d .. i * d + d - 1]`), then
 * accumulating centred cross products with divisor `n`. With `n = 0` the
 * mean and the covariance are zero.
 *
 * \param y Pointer to `n * d` contiguous sample values (host memory).
 * \param n Number of units.
 * \param d Dimension of each unit.
 * \return Populated `distrib_mv_sample_summary`.
 */
// The summary of y, n units of d values each, unit i at y[i d .. i d + d - 1]
// (host). Two passes: the mean, then the centred cross products.
inline distrib_mv_sample_summary summarize_mv_sample(const double* y, int n, int d) {
  distrib_mv_sample_summary s;
  s.count = n;
  s.dim = d;
  s.mean.assign(d, 0.0);
  s.covariance.assign(pdmat::packed_size(d), 0.0);
  if (n == 0) return s;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) s.mean[k] += y[i * d + k];
  for (int k = 0; k < d; ++k) s.mean[k] /= n;
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < d; ++a) {
      const double ra = y[i * d + a] - s.mean[a];
      for (int b = 0; b <= a; ++b) s.covariance[pdmat::packed_index(a, b)] += ra * (y[i * d + b] - s.mean[b]);
    }
  for (double& c : s.covariance) c /= n;
  return s;
}

/**
 * \ingroup distrib_multivariate
 * \brief Concept for a positive semidefinite matrix structure whose null space is constant in \f$\theta\f$ and supplied.
 *
 * Refines `pdmat::psdmat_structure` by requiring `S::constant_null_space` and
 * `S::supplies_null_space` to be `true`, so that `null_space_basis()` returns
 * a fixed orthonormal basis of the null space. Both the degenerate
 * specialization of `mvgaussian1_distrib` and the improper specialization of
 * `mvgaussian2_distrib` exist over structures satisfying this concept (such as
 * `pdmat::weighted_sum_psdmat`, whereas `pdmat::reduced_rank_psdmat` is excluded).
 *
 * \tparam S Matrix structure type to check.
 * \see mvgaussian1_distrib, mvgaussian2_distrib, distrib_multivariate
 */
template <class S>
concept constant_null_space_psdmat = pdmat::psdmat_structure<S> && S::constant_null_space && S::supplies_null_space;

/**
 * \ingroup distrib_mvgaussian1
 * \brief Precomputed state of a rank-deficient multivariate Gaussian at fixed \f$\theta\f$.
 *
 * Written by `theta_state()` (and, for the improper specialization of
 * `mvgaussian2_distrib`, augmented by `constrained_state()`) into caller-owned
 * workspace `w`. Stores \f$\log\lvert M(\theta)\rvert_+\f$, pointers into `w`
 * for the packed quadratic-form matrix and optional shifted Cholesky factor and
 * inverse, the shift \f$c\f$, optional structure pseudo-determinant state, and
 * optional \f$\theta\f$-derivative table. Both `w` and `theta` must remain
 * alive while the state is in use.
 *
 * \see mvgaussian1_distrib, mvgaussian2_distrib, distrib_multivariate
 */
struct mvgaussian_rank_deficient_state {
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer to caller-owned matrix structure parameter vector \f$\theta\f$ of length `theta_size()`.
   */
  const double* theta;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Log pseudo-determinant \f$\log\lvert M(\theta)\rvert_+\f$ (logarithm of the product of the \f$\rho\f$ nonzero eigenvalues).
   */
  double log_det;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the packed lower-triangular matrix of the quadratic form: \f$\Sigma^+\f$ (degenerate) or \f$Q\f$ (improper).
   */
  const double* quadratic;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the packed lower Cholesky factor \f$L\f$ of \f$M + cP\f$, or `nullptr` when produced by `theta_state()` of the improper class.
   */
  const double* factor = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding packed \f$(M + cP)^{-1}\f$, or `nullptr` when produced by `theta_state()` of the improper class.
   */
  const double* shifted_inverse = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Fixed shift \f$c = \operatorname{tr} M / \rho\f$ evaluated at `theta` (`1.0` if \f$\rho = 0\f$ or \f$\operatorname{tr} M \le 0\f$).
   */
  double shift = 1.0;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the structure's evaluated log pseudo-determinant state (read by `log_pdet_dk_at()`), or `nullptr`.
   */
  double* pdet = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer to the precomputed \f$\theta\f$-derivative table attached by `with_theta_table()`, or `nullptr` if none.
   */
  const double* table = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Largest \f$\theta\f$-derivative order covered by `table` (`0` if none).
   */
  int table_order = 0;
};

namespace detail_mv {

inline constexpr double log_two_pi = 1.8378770664093454835606594728112353;

// r' P r with r = y - mu and P packed symmetric
LAPLACE_HD double packed_quadratic(int d, const double* P, const double* y, const double* mu) {
  double acc = 0.0;
  for (int i = 0; i < d; ++i) {
    const double ri = y[i] - mu[i];
    double t = 0.5 * P[pdmat::packed_index(i, i)] * ri;
    for (int j = 0; j < i; ++j) t += P[pdmat::packed_index(i, j)] * (y[j] - mu[j]);
    acc += ri * t;
  }
  return 2.0 * acc;
}

// out = c P r, r = y - mu
LAPLACE_HD void packed_times_residual(int d, const double* P, const double* y, const double* mu, double c,
                                             double* out) {
  for (int i = 0; i < d; ++i) {
    double t = 0.0;
    for (int j = 0; j < d; ++j) t += P[i >= j ? pdmat::packed_index(i, j) : pdmat::packed_index(j, i)] * (y[j] - mu[j]);
    out[i] = c * t;
  }
}

// tr(A B) for packed symmetric A and B
LAPLACE_HD double packed_trace_product(int d, const double* A, const double* B) {
  double diag = 0.0, off = 0.0;
  for (int i = 0; i < d; ++i) {
    diag += A[pdmat::packed_index(i, i)] * B[pdmat::packed_index(i, i)];
    for (int j = 0; j < i; ++j) off += A[pdmat::packed_index(i, j)] * B[pdmat::packed_index(i, j)];
  }
  return diag + 2.0 * off;
}

// The lower Cholesky factor of a packed positive definite matrix, in place, by
// rows (Cholesky-Banachiewicz).
LAPLACE_HD void packed_cholesky_in_place(int d, double* a) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = a[pdmat::packed_index(i, j)];
      for (int k = 0; k < j; ++k) s -= a[pdmat::packed_index(i, k)] * a[pdmat::packed_index(j, k)];
      a[pdmat::packed_index(i, j)] = i == j ? Kokkos::sqrt(s) : s / a[pdmat::packed_index(j, j)];
    }
}

// L with M = L L': the structure's own when it has one, else from M
template <class S>
LAPLACE_HD void structure_factor(const S& s, const double* theta, double* l, double* w) {
  if constexpr (S::closed_factor) {
    s.cholesky(theta, l, w);
  } else {
    s.matrix(theta, l, w);
    packed_cholesky_in_place(s.dim, l);
  }
}

// d^k M (Inv false) or d^k M^-1 (Inv true) along idx[0..k-1], k known at run
// time (k = 0: the matrix itself); false when zero by structure
template <bool Inv, int K, class S>
LAPLACE_HD bool structure_matrix_dk(const S& s, const double* t, const int* idx, double* out, double* w) {
  if constexpr (K == 0) {
    if constexpr (Inv) s.inverse(t, out, w);
    else s.matrix(t, out, w);
    return true;
  } else {
    int i[K];
    for (int q = 0; q < K; ++q) i[q] = idx[q];
    if constexpr (Inv) return s.template inverse_dk<K>(t, i, out, w);
    else return s.template matrix_dk<K>(t, i, out, w);
  }
}
template <bool Inv, class S>
LAPLACE_HD bool structure_matrix_runtime(const S& s, int k, const double* t, const int* idx, double* out, double* w) {
  switch (k) {
    case 0: return structure_matrix_dk<Inv, 0>(s, t, idx, out, w);
    case 1: return structure_matrix_dk<Inv, 1>(s, t, idx, out, w);
    case 2: return structure_matrix_dk<Inv, 2>(s, t, idx, out, w);
    case 3: return structure_matrix_dk<Inv, 3>(s, t, idx, out, w);
    case 4: return structure_matrix_dk<Inv, 4>(s, t, idx, out, w);
    case 5: return structure_matrix_dk<Inv, 5>(s, t, idx, out, w);
    default: return structure_matrix_dk<Inv, 6>(s, t, idx, out, w);
  }
}

template <int K, class S>
LAPLACE_HD double structure_log_det_dk(const S& s, const double* t, const int* idx, double* w) {
  if constexpr (K == 0) {
    return s.log_det(t, w);
  } else {
    int i[K];
    for (int q = 0; q < K; ++q) i[q] = idx[q];
    return s.template log_det_dk<K>(t, i, w);
  }
}
template <class S>
LAPLACE_HD double structure_log_det_runtime(const S& s, int k, const double* t, const int* idx, double* w) {
  switch (k) {
    case 0: return structure_log_det_dk<0>(s, t, idx, w);
    case 1: return structure_log_det_dk<1>(s, t, idx, w);
    case 2: return structure_log_det_dk<2>(s, t, idx, w);
    case 3: return structure_log_det_dk<3>(s, t, idx, w);
    case 4: return structure_log_det_dk<4>(s, t, idx, w);
    case 5: return structure_log_det_dk<5>(s, t, idx, w);
    default: return structure_log_det_dk<6>(s, t, idx, w);
  }
}

// merge of two sorted tuples into out (sorted), returns its length
LAPLACE_HD int merge_sorted(const int* a, int na, const int* b, int nb, int* out) {
  int i = 0, j = 0, k = 0;
  while (i < na || j < nb) out[k++] = (j >= nb || (i < na && a[i] <= b[j])) ? a[i++] : b[j++];
  return k;
}

// position of the sorted tuple t[0] <= ... <= t[k-1] over p in the layout of
// a derivative block: packed by rows of the lower triangle for k = 2,
// sym_tensor order (sym_tensor_position with p and k known at run time) else
LAPLACE_HD int block_position(int p, int k, const int* t) {
  if (k == 2) return pdmat::packed_index(t[1], t[0]);
  int pos = 0, prev = 0;
  for (int j = 0; j < k; ++j) {
    for (int v = prev; v < t[j]; ++v) pos += binomial_coef(p - v + k - j - 2, k - j - 1);
    prev = t[j];
  }
  return pos;
}

// the sorted tuple after t over p in the same layout; false after the last
LAPLACE_HD bool next_block_tuple(int p, int k, int* t) {
  if (k == 2) {  // rows of the lower triangle: (j, i) with j <= i, by i then j
    if (t[0] < t[1]) {
      ++t[0];
      return true;
    }
    if (t[1] == p - 1) return false;
    ++t[1];
    t[0] = 0;
    return true;
  }
  int i = k - 1;
  while (i >= 0 && t[i] == p - 1) --i;
  if (i < 0) return false;
  ++t[i];
  for (int j = i + 1; j < k; ++j) t[j] = t[i];
  return true;
}

// logpdf_dy_dk<J, K, A> of a law of y - mu: d_y^J d_mu^A l = (-1)^J d_mu^(J+A) l.
// The order-(J + A) tensor in mu goes to w[0, m), the call's scratch after it;
// out[(u, v)], u a sorted J-tuple over y (outer) and v a sorted A-tuple over mu
// (inner), each in the layout of its order (block_position), is (-1)^J times
// its entry at the merged tuple.
template <int J, int K, int A, class D>
LAPLACE_HD bool location_dy_dk(const D& g, const typename D::state_type& st, const double* y, const double* mu,
                               const int* idx, double* out, double* w) {
  static_assert(J >= 0 && K >= 0 && A > 0 && J + K + A <= 6, "logpdf_dy_dk: J + K + A <= 6");
  const int d = g.dim(), m = sym_tensor_size(d, J + A);
  if (!g.template logpdf_dk<K, J + A>(st, y, mu, idx, w, w + m)) return false;
  const double sign = J % 2 == 1 ? -1.0 : 1.0;
  int u[J + 1] = {}, v[A] = {}, t[J + A];
  int o = 0;
  do {
    for (int b = 0; b < A; ++b) v[b] = 0;
    do {
      merge_sorted(u, J, v, A, t);
      out[o++] = sign * w[block_position(d, J + A, t)];
    } while (next_block_tuple(d, A, v));
  } while (next_block_tuple(d, J, u));
  return true;
}

}  // namespace detail_mv
}  // namespace laplace::distrib
