// A distribution with some of its parameters held at given values and removed
// from the parameter vector (laplace::distrib). The plan is
// note/piano_wrapper.md section 4 (the table's fixed(d, ...) row); the
// composition rule is its section 5ter.
//
// Write theta = (theta_F, theta_H) for the free coordinates and the held ones,
// with theta_H = c a vector of constants. The law is
//
//   l(y; theta_F) = l(y; theta_F, c),
//
// which is a restriction of the domain and NOT a reparametrization: the
// completion theta_F -> (theta_F, c) is the identity on the free coordinates and
// carries no Jacobian. So for 1 <= K <= 6 and free indices a_1, ..., a_K,
//
//   d^K l / dtheta_a1 ... dtheta_aK (y; theta_F)
//     = d^K l / dtheta_a1 ... dtheta_aK (y; theta_F, c),
//
// the parent's symmetric tensor of order K RESTRICTED to the multi-indices that
// lie in F: a selection of components, with no chain rule and no new numerics,
// so every surviving component is bit for bit the parent's.
//
// The same selection carries the rest of the surface. The mixed derivatives in
// y are the parent's, selected on the theta part and untouched on the y part.
// The law of Y under the fixed model IS the parent's law at (theta_F, c), so the
// expected information is
//
//   I^fix_ab = -E[d_a d_b l] = I_ab(theta_F, c),   a, b in F,
//
// the free BLOCK of the parent's information, not the inverse of a submatrix of
// its inverse: nothing here profiles anything out. Its derivatives in theta_F
// are the parent's blocks selected twice, on the pair (a, b) and on the K
// differentiating indices. The value itself, the cumulative functions, the
// quantile, the atoms, the ends of the support, center, scale, y_kinks and
// y_regularity are the parent's at the completed vector, none of them being
// indexed by a parameter. A held parameter that moves the support is not
// special: the ends become constants, and the free descriptors' own
// moves_support is unchanged.
//
// What costs anything is the index surface. A free index j is mapped back to the
// parent's index before any call that is indexed by a parameter: param_kinks(j,
// y) above all, and on the multivariate side param_size, param_offset,
// theta_size, the idx tuple of logpdf_dk<K, A...>, the slots and the index pair
// of expected_info_dk, coord_lower, coord_upper and the block descriptors.
//
// A held coordinate is named by one integer that encodes the parameter's
// index in params, and the coordinate inside it or whole_slot for all of it. The
// set is a template parameter, because n_params and params are static; only the
// values are data. coord_of<D>("mu") and coord_of<D>("mu", 2) build that code
// from a name at compile time, which is how a call site reads (decided with
// Giovanni on 2026-11-18). The argument is an int and not a pair of integers
// because nvcc writes a template argument of class type into its generated host
// code as a braced initializer, which is not a template argument, and MSVC then
// ends the host pass with a bare "fatal error C1903" and no diagnostic of its
// own; measured on 2026-11-18, a class type of two integers compiled everywhere
// except inside the closure of an extended lambda, which is where every
// distribution goes:
//
//   auto d = fixed<coord_of<G>("v")>(gaussian2_distrib{0.0, 4.0}, 4.0);
//   auto p = fixed<coord_of<M>("mu")>(mvgaussian2, zeros);   // mu held at zero
//
// Cost by kind of held coordinate. Any subset of the INDEXED coordinates theta
// (a whole scalar parameter, a whole matrix parameter, single elements of
// either) costs the remap of the at-most-six idx integers plus a completed theta
// written once into the scratch, and leaves the derivative layout alone, because
// a derivative in theta is asked for by an explicit index tuple and not as a
// tensor over theta. A whole DENSE parameter (a vector or a simplex_vector) is
// passed to the parent as the pointer the wrapper holds, and the parent is
// called with order 0 for it, whose Kronecker factor is one double, so the
// layout is again unchanged. Both are free: no gather anywhere.
//
// Two refusals on the multivariate side come from the interface and not from a
// choice here. logpdf(st, y, x) takes no scratch, so the wrapper has nowhere to
// interleave free and held DENSE coordinates: a dense parameter may be held only
// whole, and only when it is the law's only dense parameter, which is the case
// of every multivariate law in Laplace. Holding part of a dense parameter, or
// one of several, needs a logpdf that takes w, which is a change to
// multivariate_core.hpp and to every multivariate law, and so Giovanni's call.
// Single elements of a MATRIX are not affected: they are indexed coordinates and
// are held like any other. Where a partially held parameter does survive it
// SHRINKS to its free coordinates rather than keeping its shape with constants
// in it (decided with Giovanni on 2026-11-18): a slot that kept its shape would
// report an information block over coordinates that are not parameters, singular
// once their rows were zeroed, and would push the mask of free coordinates into
// every caller.
//
// The held values of a whole parameter of several coordinates are a vector whose
// length is a runtime quantity, so they are held as a non-owning pointer that
// must outlive the wrapper and must be reachable where the call runs; a null
// pointer means that every coordinate takes the scalar, so a null pointer with
// the scalar left at zero is the zero vector. This follows
// penalty/distrib_penalty.hpp, which carries mu_0 the same way. A whole held
// dense parameter is the one case that needs the pointer itself, logpdf having
// no scratch to spread a scalar into, so fixed() refuses a null one there.
//
// fixed over fixed composes like any other wrapper. start_value is the one
// member that writes a parameter rather than reading one, and it writes through
// detail_fixed::write_param, which walks down a nest of fixed_distrib mapping
// each free index to the parent's before it reaches a law and writes its leading
// doubles (the convention univariate_as_multivariate_distrib reads them by, and
// which commit ed17109 states holds for the wrappers and the atom laws). A
// fixed_distrib does not satisfy that convention itself, its leading doubles
// being the parent's full vector, which is why the walk exists; for the same
// reason univariate_as_multivariate over a fixed_distrib is not supported, and
// the composition goes the other way, fixed_distrib over
// univariate_as_multivariate.
#pragma once

#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/multivariate_core.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace laplace::distrib {

/**
 * \ingroup distrib_fixed
 * \brief Element index (\f$-1\f$) standing for every coordinate of a parameter.
 * \see fixed_coord_code(), coord_of(), fixed_distrib
 */
inline constexpr int whole_slot = -1;

/**
 * \ingroup distrib_fixed
 * \brief Stride of the code of a held coordinate: the element index occupies the low digits.
 * \see fixed_coord_code(), fixed_coord_slot(), fixed_coord_element()
 */
inline constexpr int fixed_coord_stride = 1 << 24;

/**
 * \ingroup distrib_fixed
 * \brief The code of one held coordinate: a parameter index and a coordinate inside it, in one integer.
 *
 * A held coordinate is written as a template argument of \ref fixed_distrib, usually through
 * coord_of(), and the argument is an \c int rather than a pair, because nvcc writes a template
 * argument of class type into its generated host code as a braced initializer, which is not a
 * template argument, and MSVC then ends the host pass with a bare `fatal error C1903` (measured on
 * 2026-11-18; a class type of two integers worked everywhere except inside the closure of an
 * extended lambda, which is where every distribution goes).
 *
 * \param slot Index of the parameter in `D::params`.
 * \param element Coordinate inside that parameter, or \ref whole_slot for all of them, which is the
 *                default. A parameter of a univariate distribution has one coordinate, so only
 *                \ref whole_slot is accepted there.
 * \return The code, or \f$-1\f$ when \p slot is negative, which fixed_distrib refuses.
 * \see coord_of(), fixed_coord_slot(), fixed_coord_element(), fixed_distrib, fixed()
 */
LAPLACE_HD constexpr int fixed_coord_code(int slot, int element = whole_slot) {
  return slot < 0 ? -1 : slot * fixed_coord_stride + element + 1;
}

/**
 * \ingroup distrib_fixed
 * \brief The parameter index a code names.
 * \param code A code built by fixed_coord_code() or coord_of().
 * \return The index in `D::params`.
 * \see fixed_coord_code(), fixed_coord_element()
 */
LAPLACE_HD constexpr int fixed_coord_slot(int code) { return code / fixed_coord_stride; }

/**
 * \ingroup distrib_fixed
 * \brief The coordinate inside the parameter that a code names.
 * \param code A code built by fixed_coord_code() or coord_of().
 * \return The coordinate, or \ref whole_slot for all of them.
 * \see fixed_coord_code(), fixed_coord_slot()
 */
LAPLACE_HD constexpr int fixed_coord_element(int code) { return code % fixed_coord_stride - 1; }

/**
 * \ingroup distrib_fixed
 * \brief The value or values one held coordinate takes.
 *
 * A single coordinate is held at `scalar`. A whole parameter of several coordinates is held at
 * `vector`, a non-owning buffer of `param_size(slot)` doubles that must outlive the wrapper and must
 * be reachable where the call runs; a null `vector` means that every coordinate takes `scalar`, so a
 * null pointer and `scalar` zero is the zero vector.
 * \see fixed_distrib, fixed()
 */
struct fixed_value {
  /// The value every coordinate not covered by `vector` is held at.
  double scalar = 0.0;
  /// The values of a whole parameter of several coordinates, or `nullptr` for `scalar` throughout.
  const double* vector = nullptr;
};

namespace detail_fixed {

// equality of two null-terminated names, in a constant expression
constexpr bool same_name(const char* a, const char* b) {
  while (*a != '\0' && *a == *b) {
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

}  // namespace detail_fixed

/**
 * \ingroup distrib_fixed
 * \brief The code of a coordinate of `D` named at compile time.
 *
 * Reads `D::params[j].name`, which both the univariate and the multivariate descriptor carry, so the
 * same spelling serves either.
 * \tparam D Distribution type.
 * \param name Parameter name, as `D::params[j].name` gives it.
 * \param element Coordinate inside the parameter, \ref whole_slot by default.
 * \return The code, as fixed_coord_code() builds it; \f$-1\f$ when no parameter carries that name,
 *         which fixed_distrib refuses with a `static_assert`.
 * \see fixed_coord_code(), fixed_distrib, fixed()
 */
template <class D>
constexpr int coord_of(const char* name, int element = whole_slot) {
  for (int j = 0; j < D::n_params; ++j)
    if (detail_fixed::same_name(D::params[j].name, name)) return fixed_coord_code(j, element);
  return -1;
}

namespace detail_fixed {

// The questions the pack answers are all folds over the template arguments, so
// no table is read at a runtime index, which on the device would need a
// definition the host side does not provide.

// parameter j is named with whole_slot
template <int... C>
LAPLACE_HD constexpr bool whole_held(int j) {
  return ((fixed_coord_element(C) == whole_slot && fixed_coord_slot(C) == j) || ...);
}
// coordinate e of parameter j is named on its own
template <int... C>
LAPLACE_HD constexpr bool element_held(int j, int e) {
  return ((fixed_coord_element(C) >= 0 && fixed_coord_slot(C) == j && fixed_coord_element(C) == e) || ...);
}
// coordinate e of parameter j is held, either way
template <int... C>
LAPLACE_HD constexpr bool coord_held(int j, int e) {
  return whole_held<C...>(j) || element_held<C...>(j, e);
}
// some coordinates of parameter j are named on their own, and not the whole of it
template <int... C>
LAPLACE_HD constexpr bool partly_held(int j) {
  return !whole_held<C...>(j) && ((fixed_coord_element(C) >= 0 && fixed_coord_slot(C) == j) || ...);
}
// how many parameters the pack names with whole_slot
template <int... C>
constexpr int n_whole_held() {
  return (0 + ... + (fixed_coord_element(C) == whole_slot ? 1 : 0));
}

// the free parameter indices of P, in the parent's order
template <int P, int... C>
LAPLACE_HD void write_free_slots(int* fr) {
  int n = 0;
  for (int j = 0; j < P; ++j)
    if (!whole_held<C...>(j)) fr[n++] = j;
}

// the pack is well formed over P parameters
template <int P, int... C>
constexpr bool coords_valid() {
  constexpr int H = int(sizeof...(C));
  const int code[H] = {C...};
  for (int i = 0; i < H; ++i) {
    if (code[i] < 0) return false;  // coord_of() found no parameter of that name
    const int si = fixed_coord_slot(code[i]), ei = fixed_coord_element(code[i]);
    if (si < 0 || si >= P || ei < whole_slot) return false;
    for (int j = 0; j < H; ++j) {
      if (i == j || code[j] < 0 || fixed_coord_slot(code[j]) != si) continue;
      const int ej = fixed_coord_element(code[j]);
      if (ei == ej) return false;                                  // the same coordinate twice
      if (ei == whole_slot || ej == whole_slot) return false;       // a whole parameter and one of its coordinates
    }
  }
  return true;
}

// out[pos], over the sorted K-tuples of the M free coordinates in the packed
// order, is t at the parent's tuple; fr maps a free index to the parent's
template <int P, int M, int K>
LAPLACE_HD sym_tensor<M, K> select_tensor(const sym_tensor<P, K>& t, const int* fr) {
  static_assert(K >= 1 && M >= 1, "a tensor of order one or more over one free coordinate or more");
  sym_tensor<M, K> out;
  int idx[K] = {};
  for (int pos = 0; pos < out.size; ++pos) {
    int pidx[K];
    for (int k = 0; k < K; ++k) pidx[k] = fr[idx[k]];
    out[pos] = t.v[sym_tensor_position<P, K>(pidx)];
    int k = K - 1;
    while (k >= 0 && idx[k] == M - 1) --k;
    if (k < 0) break;
    const int v = idx[k] + 1;
    for (int j = k; j < K; ++j) idx[j] = v;
  }
  return out;
}

// the same selection on a derivative of the expected information: the pair
// (a, b) and the K differentiating indices are both restricted to the free set
template <int P, int M, int K>
LAPLACE_HD expected_info_tensor<M, K> select_info(const expected_info_tensor<P, K>& t, const int* fr) {
  expected_info_tensor<M, K> out;
  int ab[2] = {0, 0};
  for (int q = 0; q < out.n_pairs; ++q) {
    const int pab[2] = {fr[ab[0]], fr[ab[1]]};
    out.pair[q] = select_tensor<P, M, K>(t.pair[sym_tensor_position<P, 2>(pab)], fr);
    if (ab[1] == M - 1) {
      ++ab[0];
      ab[1] = ab[0];
    } else {
      ++ab[1];
    }
  }
  return out;
}

// the descriptors of the free parameters, in the parent's order, for either
// descriptor type (host: D::params read at a runtime index, as everywhere)
template <class D, class Info, int M, int... C>
constexpr std::array<Info, M> free_params_of() {
  std::array<Info, M> p{};
  int n = 0;
  for (int j = 0; j < D::n_params; ++j)
    if (!whole_held<C...>(j)) p[n++] = D::params[j];
  return p;
}

// The optional members of a parent are detected through NAMED concepts and never
// through a requires-expression written inline in a member's constraint, which
// nvcc evaluates to false where gcc evaluates it to true
// (univariate_as_multivariate.hpp met the same trap): on 2026-11-18 the five
// errors of the first CUDA build of test_fixed.cpp were all the inline form of
// has_logpdf_dy_dk below.
template <class D>
concept has_logpdf_dy_dk = requires(const D d, double v) { d.template logpdf_dy_dk<1, 0>(v); };
template <class D>
concept has_prox_y = requires(const D d, double a) { d.prox_y(a, a); };
template <class D>
concept has_start_from_sample = requires(const D d, const std::vector<double>& v) { d.start_value(v); };
template <class D>
concept has_info_scratch_size = requires(const D d) { d.info_scratch_size(); };
template <class D>
concept has_coord_atoms = requires(const D d) {
  d.n_coord_atoms(0);
  d.coord_atom_point(0, 0);
};
template <class D>
concept has_blocks = requires(const D d) {
  d.n_blocks();
  d.block_start(0, 0);
};
template <class D>
concept has_mv_rng = requires(const D d, const typename D::state_type& s, const double* v, double* o) {
  d.rng(s, v, 0ull, 0ull, o);
};
template <class D>
concept has_mean = requires(const D d, const double* v, double* o) { d.mean(v, o); };
template <class D>
concept has_mv_start = requires(const D d, const distrib_mv_sample_summary& s, double* o) { d.start_value(s, o); };
template <class D>
concept has_mv_prox_y = requires(const D d, const typename D::state_type& s, const double* q, double* o) {
  d.prox_y(s, q, q, 0.0, o, o);
};
template <class D>
concept has_expected_info_dk =
    requires(const D d, const typename D::state_type& s, const int* i, const double* v, double* o) {
      d.template expected_info_dk<0>(s, 0, 0, i, v, i, o, o);
    };

// y_regularity is INHERITED rather than declared, as atom_mixture_distrib
// inherits it: a discrete parent has none, and nvcc instantiates every static
// member of a class eagerly where gcc waits for a use, so declaring it
// unconditionally is an error on one compiler and not on the other.
template <class D, distrib_kind K>
struct forwarded_y_regularity {};

template <class D>
struct forwarded_y_regularity<D, distrib_kind::continuous> {
  /// Differentiability order of \f$\log f\f$ in the observation, the parent's.
  static constexpr int y_regularity = D::y_regularity;
};

}  // namespace detail_fixed

// A distribution with some of its parameters held at given values, specialized
// below for a univariate parent and for a multivariate one; each specialization
// carries its own documentation, and this declaration none, so that the manual
// shows one entry and not an empty second one.
template <class D, int... C>
struct fixed_distrib;

namespace detail_fixed {

// whether T is a fixed_distrib, and its parent type
template <class T>
struct is_fixed : std::false_type {};
template <class D, int... C>
struct is_fixed<fixed_distrib<D, C...>> : std::true_type {};

// Writes v into parameter j of d. A nest of fixed_distrib is walked down, each
// free index mapped to its parent's, until a law is reached, whose parameters
// are its leading doubles.
template <class T>
void write_param(T& d, int j, double v) {
  if constexpr (is_fixed<T>::value) {
    write_param(d.parent, T::parent_slot(j), v);
  } else {
    double p[T::n_params];
    std::memcpy(static_cast<void*>(p), static_cast<const void*>(&d), sizeof p);
    p[j] = v;
    std::memcpy(static_cast<void*>(&d), static_cast<const void*>(p), sizeof p);
  }
}

}  // namespace detail_fixed

// ---------------------------------------------------------------------------
// a univariate parent
// ---------------------------------------------------------------------------

/**
 * \ingroup distrib_fixed
 * \brief A univariate distribution with some of its parameters held at given values.
 *
 * Every member indexed by a parameter is the parent's re-indexed, and every derivative in the
 * parameters is the parent's tensor restricted to the free indices, so each surviving component is bit
 * for bit the parent's. Build it with fixed(); the aggregate checks nothing, so that one built on the
 * host can be copied into a kernel.
 * \tparam D Parent distribution type.
 * \tparam C The held coordinates, each with `element` equal to \ref whole_slot.
 */
template <class D, int... C>
  requires(!multivariate_distrib<D>)
struct fixed_distrib<D, C...> : detail_fixed::forwarded_y_regularity<D, D::kind> {
  static_assert(sizeof...(C) >= 1, "fixed_distrib holds at least one parameter");
  static_assert(detail_fixed::coords_valid<D::n_params, C...>(),
                "fixed_distrib: a held coordinate names a parameter that does not exist, or names one twice");
  static_assert(((fixed_coord_element(C) == whole_slot) && ...),
                "fixed_distrib over a univariate parent: a parameter has one coordinate, so write whole_slot and not "
                "an element index");
  static_assert(D::n_params - int(sizeof...(C)) >= 1,
                "fixed_distrib leaves at least one free parameter: a law with none is not fitted, and a symmetric "
                "tensor over no coordinate has no components");

  /// The wrapped distribution, carrying the held parameters at their values.
  D parent;
  /// The held values, in the order of the template arguments. fixed() writes them into \c parent as well,
  /// which is where every read takes them from; they are kept here so that start_value() can put them back
  /// after the parent has restarted every parameter. An aggregate built inside a kernel rather than by
  /// fixed() carries the held values in \c parent itself, and may leave these at zero unless it calls
  /// start_value(), which is a host member.
  double held[sizeof...(C)] = {};

  /// Number of held parameters.
  static constexpr int n_held = int(sizeof...(C));
  /// Number of natural parameters: the parent's less the held ones.
  static constexpr int n_params = D::n_params - n_held;
  /// Distribution kind, the parent's.
  static constexpr distrib_kind kind = D::kind;
  /// Number of atoms, the parent's: holding a parameter moves no mass.
  static constexpr int n_atoms = distrib_n_atoms<D>();

 private:
  static constexpr std::array<distrib_param_info, n_params> params_ =
      detail_fixed::free_params_of<D, distrib_param_info, n_params, C...>();

 public:
  /// Descriptors of the free parameters, in the parent's order. \c y_regularity is inherited from the
  /// parent when it is continuous, a discrete law declaring none.
  static constexpr const distrib_param_info* params = params_.data();

  /// \brief Writes the free parameter indices. \param fr Output buffer of \c n_params integers.
  LAPLACE_HD static void free_slots(int* fr) { detail_fixed::write_free_slots<D::n_params, C...>(fr); }
  /// \brief The parent's index of a free parameter. \param j Free parameter index. \return The parent's index.
  LAPLACE_HD static int parent_slot(int j) {
    int fr[n_params];
    free_slots(fr);
    return fr[j];
  }

  /// \brief Evaluates the log-density, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double logpdf(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return parent.logpdf(y);
  }
  /// \brief Evaluates the log-mass, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double logpmf(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return parent.logpmf(y);
  }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of the log-density or log-mass in the free parameters.
   *
   * The parent's tensor of order \f$K\f$ restricted to the free indices: each component is one of the
   * parent's, read unchanged.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \return The packed symmetric tensor of order \f$K\f$ over the free parameters.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_value_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    int fr[n_params];
    free_slots(fr);
    return detail_fixed::select_tensor<D::n_params, n_params, K>(detail_generic::log_density_dk<K>(parent, y), fr);
  }

  /// \brief Derivatives of order \f$K\f$ of the log-density. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dk(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<K>(y);
  }
  /// \brief Derivatives of order \f$K\f$ of the log-mass. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpmf_dk(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<K>(y);
  }

  /// \brief First derivatives of the log-density. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> logpdf_d1(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<1>(y);
  }
  /// \brief Second derivatives of the log-density. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> logpdf_d2(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<2>(y);
  }
  /// \brief Third derivatives of the log-density. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> logpdf_d3(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<3>(y);
  }
  /// \brief Fourth derivatives of the log-density. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> logpdf_d4(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<4>(y);
  }
  /// \brief Fifth derivatives of the log-density. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> logpdf_d5(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<5>(y);
  }
  /// \brief Sixth derivatives of the log-density. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> logpdf_d6(double y) const
    requires(D::kind == distrib_kind::continuous)
  {
    return log_value_dk<6>(y);
  }
  /// \brief First derivatives of the log-mass. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> logpmf_d1(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<1>(y);
  }
  /// \brief Second derivatives of the log-mass. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> logpmf_d2(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<2>(y);
  }
  /// \brief Third derivatives of the log-mass. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> logpmf_d3(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<3>(y);
  }
  /// \brief Fourth derivatives of the log-mass. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> logpmf_d4(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<4>(y);
  }
  /// \brief Fifth derivatives of the log-mass. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> logpmf_d5(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<5>(y);
  }
  /// \brief Sixth derivatives of the log-mass. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> logpmf_d6(double y) const
    requires(D::kind == distrib_kind::discrete)
  {
    return log_value_dk<6>(y);
  }

  /**
   * \brief Evaluates \f$J\f$ derivatives in \f$y\f$ and the tensor of order \f$K\f$ in the free parameters.
   *
   * The parent's own, selected on the \f$\theta\f$ part and untouched on the \f$y\f$ part.
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in the parameters, \f$J + K \le 6\f$.
   * \param y Observation point.
   * \return The packed symmetric tensor of order \f$K\f$ over the free parameters.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dy_dk(double y) const
    requires(D::kind == distrib_kind::continuous && detail_fixed::has_logpdf_dy_dk<D>)
  {
    static_assert(J >= 0 && K >= 0 && J + K <= 6, "total order at most six");
    if constexpr (K == 0) {
      sym_tensor<n_params, 0> out;
      out[0] = parent.template logpdf_dy_dk<J, 0>(y)[0];
      return out;
    } else {
      int fr[n_params];
      free_slots(fr);
      return detail_fixed::select_tensor<D::n_params, n_params, K>(parent.template logpdf_dy_dk<J, K>(y), fr);
    }
  }

  /// \brief Evaluates \f$F(y)\f$, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double cdf(double y) const { return distrib_cdf(parent, y); }
  /// \brief Evaluates \f$1 - F(y)\f$, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double ccdf(double y) const { return distrib_ccdf(parent, y); }
  /// \brief Evaluates \f$\log F(y)\f$, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double log_cdf(double y) const { return distrib_log_cdf(parent, y); }
  /// \brief Evaluates \f$\log(1 - F(y))\f$, the parent's. \param y Observation point. \return The value.
  LAPLACE_HD double log_ccdf(double y) const { return distrib_log_ccdf(parent, y); }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of \f$\log F\f$ or of \f$\log(1 - F)\f$ in the free parameters.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The packed symmetric tensor of order \f$K\f$ over the free parameters.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_cdf_tail_dk(double y, bool upper_tail) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    int fr[n_params];
    free_slots(fr);
    const auto t = upper_tail ? distrib_log_ccdf_dk<K>(parent, y) : distrib_log_cdf_dk<K>(parent, y);
    return detail_fixed::select_tensor<D::n_params, n_params, K>(t, fr);
  }

  /// \brief Derivatives of order \f$K\f$ of \f$\log F\f$. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_cdf_dk(double y) const {
    return log_cdf_tail_dk<K>(y, false);
  }
  /// \brief Derivatives of order \f$K\f$ of \f$\log(1 - F)\f$. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_ccdf_dk(double y) const {
    return log_cdf_tail_dk<K>(y, true);
  }

  /// \brief First derivatives of \f$\log F\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_cdf_d1(double y) const { return log_cdf_tail_dk<1>(y, false); }
  /// \brief Second derivatives of \f$\log F\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_cdf_d2(double y) const { return log_cdf_tail_dk<2>(y, false); }
  /// \brief Third derivatives of \f$\log F\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_cdf_d3(double y) const { return log_cdf_tail_dk<3>(y, false); }
  /// \brief Fourth derivatives of \f$\log F\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_cdf_d4(double y) const { return log_cdf_tail_dk<4>(y, false); }
  /// \brief Fifth derivatives of \f$\log F\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_cdf_d5(double y) const { return log_cdf_tail_dk<5>(y, false); }
  /// \brief Sixth derivatives of \f$\log F\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_cdf_d6(double y) const { return log_cdf_tail_dk<6>(y, false); }
  /// \brief First derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_ccdf_d1(double y) const { return log_cdf_tail_dk<1>(y, true); }
  /// \brief Second derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_ccdf_d2(double y) const { return log_cdf_tail_dk<2>(y, true); }
  /// \brief Third derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_ccdf_d3(double y) const { return log_cdf_tail_dk<3>(y, true); }
  /// \brief Fourth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_ccdf_d4(double y) const { return log_cdf_tail_dk<4>(y, true); }
  /// \brief Fifth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_ccdf_d5(double y) const { return log_cdf_tail_dk<5>(y, true); }
  /// \brief Sixth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_ccdf_d6(double y) const { return log_cdf_tail_dk<6>(y, true); }

  /// \brief Evaluates the quantile, the parent's. \param p Probability. \return The quantile.
  LAPLACE_HD double quantile(double p) const { return distrib_quantile(parent, p); }
  /// \brief Evaluates the quantile from \f$\log p\f$, the parent's. \param log_p Log probability. \return The quantile.
  LAPLACE_HD double quantile_from_log(double log_p) const { return distrib_quantile_from_log(parent, log_p); }
  /// \brief Evaluates the upper quantile from \f$\log(1 - p)\f$, the parent's. \param log_q Log upper probability. \return The quantile.
  LAPLACE_HD double quantile_from_log_upper(double log_q) const {
    return distrib_quantile_from_log_upper(parent, log_q);
  }

  /**
   * \brief The expected information over the free parameters.
   *
   * The free block of the parent's: the law of \f$Y\f$ is the parent's at the completed vector, so each
   * entry is the parent's entry, and not an entry of the inverse of a submatrix of its inverse.
   * \return The packed symmetric tensor of order 2 over the free parameters.
   */
  LAPLACE_HD sym_tensor<n_params, 2> expected_info() const {
    int fr[n_params];
    free_slots(fr);
    return detail_fixed::select_tensor<D::n_params, n_params, 2>(distrib_expected_info<0>(parent), fr);
  }

  /**
   * \brief Derivatives of order \f$K\f$ of the expected information in the free parameters.
   * \tparam K Derivative order, \f$1 \le K \le 4\f$.
   * \return The parent's blocks selected on the pair and on the differentiating indices.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<n_params, K> expected_info_dk() const {
    static_assert(K >= 1 && K <= 4, "orders one to four");
    int fr[n_params];
    free_slots(fr);
    return detail_fixed::select_info<D::n_params, n_params, K>(distrib_expected_info<K>(parent), fr);
  }

  /// \brief Derivatives of order 1 of the expected information. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// \brief Derivatives of order 2 of the expected information. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// \brief Derivatives of order 3 of the expected information. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// \brief Derivatives of order 4 of the expected information. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 4> expected_info_d4() const { return expected_info_dk<4>(); }

  /// \brief Position of atom \f$i\f$, the parent's. \param i Atom index. \return The point.
  LAPLACE_HD double atom_point(int i) const
    requires(distrib_n_atoms<D>() > 0)
  {
    return distrib_atom_point(parent, i);
  }
  /// \brief Log mass on atom \f$i\f$, the parent's. \param i Atom index. \return The log mass.
  LAPLACE_HD double atom_log_mass(int i) const
    requires(distrib_n_atoms<D>() > 0)
  {
    return distrib_atom_log_mass(parent, i);
  }

  /// \brief Lower end of the support, the parent's. \return The value.
  LAPLACE_HD double support_lower() const { return parent.support_lower(); }
  /// \brief Upper end of the support, the parent's. \return The value.
  LAPLACE_HD double support_upper() const { return parent.support_upper(); }
  /// \brief Rough location, for placing a quadrature. \return The parent's.
  LAPLACE_HD double center() const { return parent.center(); }
  /// \brief Rough spread, for placing a quadrature. \return The parent's.
  LAPLACE_HD double scale() const { return parent.scale(); }
  /// \brief Non-smooth points of \f$\log f\f$ in \f$y\f$, the parent's. \return The kinks.
  LAPLACE_HD distrib_kinks y_kinks() const { return parent.y_kinks(); }
  /**
   * \brief Non-smooth points of a free parameter placed by the observation.
   * \param j Free parameter index.
   * \param y Observation point.
   * \return The parent's, asked of the parameter that \p j stands for.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const { return parent.param_kinks(parent_slot(j), y); }

  /// \brief Proximal operator in \f$y\f$, the parent's. \param v Point. \param t Step. \return The minimizer.
  LAPLACE_HD double prox_y(double v, double t) const
    requires detail_fixed::has_prox_y<D>
  {
    return parent.prox_y(v, t);
  }

  /**
   * \brief Returns a starting value: the parent's own, with the held parameters put back at their values.
   *
   * The parent reads the sample and starts every one of its parameters; the held ones are then written
   * over again, so a start never moves a parameter the model does not estimate.
   * \param s The sample summary.
   * \return A distribution of the same shape with its free parameters set.
   */
  fixed_distrib start_value(const distrib_sample_summary& s) const {
    fixed_distrib out = *this;
    out.parent = distrib_start_value(parent, s);
    out.write_held();
    return out;
  }
  /**
   * \brief Returns a starting value read from the sample itself, for a parent that asks for it.
   * \param y The sample.
   * \return A distribution of the same shape with its free parameters set.
   */
  fixed_distrib start_value(const std::vector<double>& y) const
    requires detail_fixed::has_start_from_sample<D>
  {
    fixed_distrib out = *this;
    out.parent = distrib_start_value(parent, y);
    out.write_held();
    return out;
  }

  /**
   * \brief Writes the held values over the parent's parameters (host).
   *
   * Through detail_fixed::write_param, which walks down a nest of fixed_distrib before it writes a law's
   * leading doubles. fixed() calls it when it builds the wrapper, and start_value() after the parent has
   * restarted every parameter.
   */
  void write_held() {
    int h = 0;
    for (const int code : {C...}) detail_fixed::write_param(parent, fixed_coord_slot(code), held[h++]);
  }
};

// ---------------------------------------------------------------------------
// a multivariate parent
// ---------------------------------------------------------------------------

namespace detail_fixed {

// The I-th held coordinate, at a COMPILE-TIME index: the local array is folded
// away. A range-for over the braced list {C...} would build an
// std::initializer_list, whose storage is static and has no definition in device
// code; on 2026-11-18 one such loop in a LAPLACE_HD member made MSVC end the host
// pass of nvcc's output with "fatal error C1903: unable to recover from previous
// error(s)" and no error of its own printed before it.
template <std::size_t I, int... C>
LAPLACE_HD constexpr int coord_at() {
  constexpr int a[sizeof...(C)] = {C...};
  return a[I];
}

// Whether the parent's parameter j is dense. A fold over the parameter indices,
// so that D::params is read only at COMPILE-TIME indices: a static constexpr
// table read at a runtime index has no definition in device code, and nvcc
// stopped the first CUDA build of test_fixed.cpp with nine such errors
// ("identifier ...::params_ is undefined in device code"). The same reason makes
// mv_coord_lower() and mv_coord_upper() host functions.
template <class D, std::size_t... I>
LAPLACE_HD constexpr bool dense_slot_fold(int j, std::index_sequence<I...>) {
  return ((int(I) == j && mv_param_dense(D::params[I].kind)) || ...);
}
template <class D>
LAPLACE_HD constexpr bool dense_slot(int j) {
  return dense_slot_fold<D>(j, std::make_index_sequence<D::n_params>{});
}

// how many dense parameters D declares
template <class D>
constexpr int n_dense() {
  int n = 0;
  for (int j = 0; j < D::n_params; ++j)
    if (mv_param_dense(D::params[j].kind)) ++n;
  return n;
}
// whether the pack names a dense parameter, whole or in part
template <class D, int... C>
constexpr bool any_dense_held() {
  for (int j = 0; j < D::n_params; ++j)
    if (mv_param_dense(D::params[j].kind) && (whole_held<C...>(j) || partly_held<C...>(j))) return true;
  return false;
}
// whether the pack names a dense parameter in part
template <class D, int... C>
constexpr bool any_dense_partly_held() {
  for (int j = 0; j < D::n_params; ++j)
    if (mv_param_dense(D::params[j].kind) && partly_held<C...>(j)) return true;
  return false;
}
// whether the pack names an indexed coordinate, whole or in part
template <class D, int... C>
constexpr bool any_indexed_held() {
  for (int j = 0; j < D::n_params; ++j)
    if (!mv_param_dense(D::params[j].kind) && (whole_held<C...>(j) || partly_held<C...>(j))) return true;
  return false;
}
// the index in D::params of the position of the held value of the dense
// parameter, or -1 when no dense parameter is held
template <class D, int... C>
constexpr int held_dense_rank() {
  int h = 0;
  for (const int code : {C...}) {
    if (mv_param_dense(D::params[fixed_coord_slot(code)].kind)) return h;
    ++h;
  }
  return -1;
}

}  // namespace detail_fixed

/**
 * \ingroup distrib_fixed
 * \brief A multivariate distribution with some of its parameters held at given values.
 *
 * The typed parameter list is rebuilt: a wholly held parameter leaves `params`, and an indexed one held
 * in part keeps its place with fewer coordinates, so `param_size`, `param_offset` and `theta_size()` all
 * shrink and the free coordinates of \f$\theta\f$ and of \f$x\f$ stay contiguous. The parent is called
 * at the completed \f$\theta\f$, written once into the scratch at `parent.scratch_size()`, and at the
 * held pointer for a dense parameter; the \f$\theta\f$ index tuple is mapped back to the parent's, and a
 * held dense parameter is passed order 0, whose Kronecker factor is one double, so the derivative layout
 * is the parent's. Build it with fixed(); the aggregate checks nothing.
 *
 * A dense parameter (a vector or a simplex_vector) may be held only whole, and only when it is the law's
 * only dense parameter: `logpdf(st, y, x)` takes no scratch, so there is nowhere to interleave free and
 * held dense coordinates. Single coordinates of a scalar or of a matrix are indexed and are not
 * affected.
 * \tparam D Parent distribution type, multivariate.
 * \tparam C The held coordinates, usually written `coord_of<D>("name")` or `coord_of<D>("name", i)`.
 */
template <class D, int... C>
  requires multivariate_distrib<D>
struct fixed_distrib<D, C...> {
  static_assert(sizeof...(C) >= 1, "fixed_distrib holds at least one parameter");
  static_assert(detail_fixed::coords_valid<D::n_params, C...>(),
                "fixed_distrib: a held coordinate names a parameter that does not exist, names one twice, or names "
                "both a whole parameter and one of its coordinates");
  static_assert(!detail_fixed::any_dense_partly_held<D, C...>(),
                "fixed_distrib: a dense parameter (a vector or a simplex_vector) is held whole or not at all. "
                "logpdf(st, y, x) takes no scratch, so the wrapper has nowhere to interleave its free and held "
                "coordinates; single coordinates of a scalar or of a matrix are indexed and may be held");
  static_assert(!detail_fixed::any_dense_held<D, C...>() || detail_fixed::n_dense<D>() == 1,
                "fixed_distrib: a dense parameter may be held only when it is the law's only dense parameter, for the "
                "same reason: the parent's x would otherwise interleave free and held blocks");
  static_assert(D::n_params - detail_fixed::n_whole_held<C...>() >= 1,
                "fixed_distrib leaves at least one free parameter");

  /// The wrapped distribution.
  D parent;
  /// The held values, in the order of the template arguments.
  fixed_value held[sizeof...(C)] = {};

  /// Number of held coordinates or parameters named.
  static constexpr int n_held = int(sizeof...(C));
  /// Normalization regime, the parent's.
  static constexpr distrib_normalization normalization = D::normalization;
  /// Distribution kind, the parent's.
  static constexpr distrib_kind kind = D::kind;
  /// Number of parameters: the parent's less the wholly held ones.
  static constexpr int n_params = D::n_params - detail_fixed::n_whole_held<C...>();
  /// The precomputed state, the parent's.
  using state_type = typename D::state_type;
  /// Whether an indexed coordinate is held, so that \f$\theta\f$ is completed in the scratch.
  static constexpr bool completes_theta = detail_fixed::any_indexed_held<D, C...>();
  /// The position in `held` of the value of the dense parameter, or \f$-1\f$ when none is held.
  static constexpr int dense_rank = detail_fixed::held_dense_rank<D, C...>();

 private:
  static constexpr std::array<distrib_mv_param_info, n_params> params_ =
      detail_fixed::free_params_of<D, distrib_mv_param_info, n_params, C...>();

 public:
  /// Descriptors of the free parameters, in the parent's order.
  static constexpr const distrib_mv_param_info* params = params_.data();

  /// \brief Writes the free parameter indices. \param fr Output buffer of \c n_params integers.
  LAPLACE_HD static void free_slots(int* fr) { detail_fixed::write_free_slots<D::n_params, C...>(fr); }
  /// \brief The parent's index of a free parameter. \param j Free parameter index. \return The parent's index.
  LAPLACE_HD static int parent_slot(int j) {
    int fr[n_params];
    free_slots(fr);
    return fr[j];
  }
  /// \brief The parent's slot of a block query. \param slot A free parameter index, \ref theta_slot or \ref y_slot. \return The parent's.
  LAPLACE_HD static int parent_query_slot(int slot) { return slot < 0 ? slot : parent_slot(slot); }

  /// \brief Dimension of a unit, the parent's. \return `parent.dim()`.
  LAPLACE_HD int dim() const { return parent.dim(); }
  /// \brief Number of \f$y\f$ coordinates the derivatives in \f$y\f$ run over, the parent's. \return `parent.y_coord_size()`.
  LAPLACE_HD int y_coord_size() const { return parent.y_coord_size(); }

  /**
   * \brief Number of free coordinates of parameter \p j.
   * \param j Free parameter index.
   * \return The parent's count for the parameter \p j stands for, less the coordinates held on their own.
   */
  LAPLACE_HD int param_size(int j) const {
    const int s = parent_slot(j);
    const int n = parent.param_size(s);
    int m = n;
    for (int e = 0; e < n; ++e)
      if (detail_fixed::element_held<C...>(s, e)) --m;
    return m;
  }

  /**
   * \brief Where the free coordinates of parameter \p j start in their array.
   * \param j Free parameter index.
   * \return The sum of the free sizes of the earlier free parameters of the same class, dense or indexed.
   */
  LAPLACE_HD int param_offset(int j) const {
    const bool dense = detail_fixed::dense_slot<D>(parent_slot(j));
    int off = 0;
    for (int k = 0; k < j; ++k)
      if (detail_fixed::dense_slot<D>(parent_slot(k)) == dense) off += param_size(k);
    return off;
  }

  /// \brief Number of free indexed coordinates. \return The length of \f$\theta\f$.
  LAPLACE_HD int theta_size() const {
    int n = 0;
    for (int j = 0; j < n_params; ++j)
      if (!detail_fixed::dense_slot<D>(parent_slot(j))) n += param_size(j);
    return n;
  }
  /// \brief Number of free dense coordinates. \return The length of \f$x\f$, zero when the dense parameter is held.
  LAPLACE_HD int dense_size() const {
    int n = 0;
    for (int j = 0; j < n_params; ++j)
      if (detail_fixed::dense_slot<D>(parent_slot(j))) n += param_size(j);
    return n;
  }

  /// \brief Doubles of the state, the parent's. \return `parent.state_size()`.
  LAPLACE_HD int state_size() const { return parent.state_size(); }
  /**
   * \brief Doubles of the working scratch: the parent's, then the completed \f$\theta\f$ when one is needed.
   * \return `parent.scratch_size()`, plus the parent's \f$\theta\f$ length when an indexed coordinate is held.
   */
  LAPLACE_HD int scratch_size() const {
    return parent.scratch_size() + (completes_theta ? parent.theta_size() : 0);
  }
  /// \brief Doubles of the scratch of expected_info_dk(), the parent's. \return `parent.info_scratch_size()`.
  LAPLACE_HD int info_scratch_size() const
    requires detail_fixed::has_info_scratch_size<D>
  {
    return parent.info_scratch_size();
  }

  /// \brief Whether the parent's indexed coordinate \p t is held. \param t Index into the parent's \f$\theta\f$. \return The answer.
  LAPLACE_HD bool theta_coord_held(int t) const {
    for (int j = 0; j < D::n_params; ++j) {
      if (detail_fixed::dense_slot<D>(j)) continue;
      const int o = parent.param_offset(j), n = parent.param_size(j);
      if (t >= o && t < o + n) return detail_fixed::coord_held<C...>(j, t - o);
    }
    return false;
  }

  /**
   * \brief Completes \f$\theta\f$ with the held values, in the parent's layout.
   * \param theta The free indexed coordinates, length `theta_size()`.
   * \param w Working scratch of `scratch_size()` doubles; the completion is written at `parent.scratch_size()`.
   * \return \p theta itself when no indexed coordinate is held, the completion otherwise.
   */
  LAPLACE_HD const double* complete_theta(const double* theta, double* w) const {
    if constexpr (!completes_theta) {
      return (void)w, theta;
    } else {
      double* t = w + parent.scratch_size();
      int s = 0;
      for (int j = 0; j < D::n_params; ++j) {
        if (detail_fixed::dense_slot<D>(j)) continue;
        const int o = parent.param_offset(j), n = parent.param_size(j);
        for (int e = 0; e < n; ++e) t[o + e] = value_at(j, e, theta, s);
      }
      return t;
    }
  }

  /**
   * \brief The parent's \f$x\f$: the held pointer when its dense parameter is held, the caller's otherwise.
   * \param x The free dense coordinates, length `dense_size()`.
   * \return The parent's dense coordinates.
   */
  LAPLACE_HD const double* parent_dense(const double* x) const {
    if constexpr (dense_rank < 0) return x;
    else return (void)x, held[dense_rank].vector;
  }

  /**
   * \brief Maps a sorted tuple of free \f$\theta\f$ indices to the parent's.
   *
   * One pass over the parent's \f$\theta\f$, skipping the held coordinates and matching the free ones
   * against the tuple in order.
   * \param idx The sorted tuple of \f$K\f$ free indices.
   * \param K Length of the tuple.
   * \param out Output buffer of \f$K\f$ integers.
   */
  LAPLACE_HD void map_theta_idx(const int* idx, int K, int* out) const {
    if constexpr (!completes_theta) {
      for (int k = 0; k < K; ++k) out[k] = idx[k];
    } else {
      int s = 0, k = 0;
      for (int t = 0, n = parent.theta_size(); t < n && k < K; ++t) {
        if (theta_coord_held(t)) continue;
        while (k < K && idx[k] == s) out[k++] = t;
        ++s;
      }
      for (; k < K; ++k) out[k] = parent.theta_size();  // the caller passed an index out of range
    }
  }

  /**
   * \brief Precomputes what depends on \f$\theta\f$ only, the parent's at the completed vector.
   * \param theta The free indexed coordinates, length `theta_size()`.
   * \param w Working scratch of `state_size() + scratch_size()` doubles.
   * \return The parent's state; it stays valid while \p w and \p theta do.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const {
    // The first state_size() doubles of w are the state and the rest the scratch,
    // so the parent takes w itself; the completion goes into the scratch PAST the
    // parent's own, where the derivative calls, which receive the scratch base,
    // read it again. Written as sc = w + state_size() and passed to the parent in
    // place of w, the parent's own scratch overlapped the completion.
    return parent.theta_state(complete_theta(theta, w + state_size()), w);
  }

  /// \brief Evaluates the log-density, the parent's. \param st State. \param y Unit. \param x Free dense coordinates. \return The value.
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* x) const
    requires(D::kind == distrib_kind::continuous)
  {
    return parent.logpdf(st, y, parent_dense(x));
  }
  /// \brief Evaluates the log-mass, the parent's. \param st State. \param y Unit. \param x Free dense coordinates. \return The value.
  LAPLACE_HD double logpmf(const state_type& st, const double* y, const double* x) const
    requires(D::kind == distrib_kind::discrete)
  {
    return parent.logpmf(st, y, parent_dense(x));
  }

  /**
   * \brief Derivatives of order \f$K\f$ in \f$\theta\f$ along \p idx and \f$A_j\f$ in the free dense parameters.
   *
   * The parent's own: \p idx is mapped back to its \f$\theta\f$, and a held dense parameter takes order 0,
   * whose Kronecker factor is one double, so the layout of \p out is this wrapper's.
   * \tparam K Derivative order in \f$\theta\f$.
   * \tparam A Derivative orders in the free dense parameters, in declaration order.
   * \param st State from theta_state().
   * \param y Unit.
   * \param x Free dense coordinates.
   * \param idx Sorted tuple of \f$K\f$ free indices into \f$\theta\f$.
   * \param out Output buffer, the parent's layout at these orders.
   * \param w Working scratch of `scratch_size()` doubles, holding the completed \f$\theta\f$.
   * \return What the parent returns: \c false and \p out untouched when the block is zero by structure.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                            double* w) const
    requires(D::kind == distrib_kind::continuous)
  {
    static_assert(dense_rank < 0 || sizeof...(A) == 0,
                  "fixed_distrib: the dense parameter is held, so it takes no derivative order");
    int pidx[K > 0 ? K : 1];
    map_theta_idx(idx, K, pidx);
    return parent.template logpdf_dk<K, A...>(st, y, parent_dense(x), pidx, out, w);
  }

  /**
   * \brief Derivatives of order \f$K\f$ in \f$\theta\f$ along \p idx and \f$A_j\f$ in the free dense parameters, discrete.
   * \tparam K Derivative order in \f$\theta\f$.
   * \tparam A Derivative orders in the free dense parameters.
   * \param st State from theta_state().
   * \param y Unit.
   * \param x Free dense coordinates.
   * \param idx Sorted tuple of \f$K\f$ free indices into \f$\theta\f$.
   * \param out Output buffer.
   * \param w Working scratch of `scratch_size()` doubles.
   * \return What the parent returns.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpmf_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                            double* w) const
    requires(D::kind == distrib_kind::discrete)
  {
    static_assert(dense_rank < 0 || sizeof...(A) == 0,
                  "fixed_distrib: the dense parameter is held, so it takes no derivative order");
    int pidx[K > 0 ? K : 1];
    map_theta_idx(idx, K, pidx);
    return parent.template logpmf_dk<K, A...>(st, y, parent_dense(x), pidx, out, w);
  }

  /**
   * \brief Derivatives of order \f$J\f$ in \f$y\f$, \f$K\f$ in \f$\theta\f$ and \f$A_j\f$ in the free dense parameters.
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in \f$\theta\f$.
   * \tparam A Derivative orders in the free dense parameters.
   * \param st State from theta_state().
   * \param y Unit.
   * \param x Free dense coordinates.
   * \param idx Sorted tuple of \f$K\f$ free indices into \f$\theta\f$.
   * \param out Output buffer.
   * \param w Working scratch of `scratch_size()` doubles.
   * \return What the parent returns.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                               double* w) const
    requires(D::kind == distrib_kind::continuous)
  {
    static_assert(dense_rank < 0 || sizeof...(A) == 0,
                  "fixed_distrib: the dense parameter is held, so it takes no derivative order");
    int pidx[K > 0 ? K : 1];
    map_theta_idx(idx, K, pidx);
    return parent.template logpdf_dy_dk<J, K, A...>(st, y, parent_dense(x), pidx, out, w);
  }

  /**
   * \brief The expected information block of two free parameters, or its derivative of order \f$K\f$.
   *
   * The free block of the parent's: the slots and the index pair are mapped back, and the block is the
   * parent's, not an entry of the inverse of a submatrix of its inverse.
   * \tparam K Derivative order in \f$\theta\f$.
   * \tparam A Derivative orders in the free dense parameters.
   * \param st State from theta_state().
   * \param slot_a Free index of the row parameter, or \ref theta_slot.
   * \param slot_b Free index of the column parameter, or \ref theta_slot.
   * \param ab The index pair inside \f$(\theta, \theta)\f$, in free indices.
   * \param x Free dense coordinates.
   * \param idx Sorted tuple of \f$K\f$ free indices into \f$\theta\f$.
   * \param out Output buffer.
   * \param w Working scratch of `info_scratch_size()` doubles.
   * \return What the parent returns.
   */
  template <int K, int... A>
  LAPLACE_HD bool expected_info_dk(const state_type& st, int slot_a, int slot_b, const int* ab, const double* x,
                                   const int* idx, double* out, double* w) const
    requires detail_fixed::has_expected_info_dk<D>
  {
    int pidx[K > 0 ? K : 1];
    map_theta_idx(idx, K, pidx);
    int pab[2] = {0, 0};
    if (ab != nullptr) map_theta_idx(ab, 2, pab);
    return parent.template expected_info_dk<K, A...>(st, parent_query_slot(slot_a), parent_query_slot(slot_b), pab,
                                                     parent_dense(x), pidx, out, w);
  }

  /// \brief Number of coordinate blocks, the parent's. \return `parent.n_blocks()`.
  LAPLACE_HD int n_blocks() const
    requires detail_fixed::has_blocks<D>
  {
    return parent.n_blocks();
  }
  /**
   * \brief Starting coordinate of block \p g in a slot, the parent's.
   * \param slot A free parameter index, \ref theta_slot or \ref y_slot.
   * \param g Block index.
   * \return `parent.block_start()` at the parent's slot; a dense parameter is held whole or not at all, so
   *         its blocks are unchanged.
   */
  LAPLACE_HD int block_start(int slot, int g) const
    requires detail_fixed::has_blocks<D>
  {
    return parent.block_start(parent_query_slot(slot), g);
  }

  /// \brief Number of atoms on coordinate \p i of \f$y\f$, the parent's. \param i Coordinate. \return The count.
  LAPLACE_HD int n_coord_atoms(int i) const
    requires detail_fixed::has_coord_atoms<D>
  {
    return parent.n_coord_atoms(i);
  }
  /// \brief Value of atom \p a of coordinate \p i, the parent's. \param i Coordinate. \param a Atom index. \return The value.
  LAPLACE_HD double coord_atom_point(int i, int a) const
    requires detail_fixed::has_coord_atoms<D>
  {
    return parent.coord_atom_point(i, a);
  }

  /**
   * \brief Lower bound of the natural domain of free coordinate \p i of free parameter \p j.
   * \param j Free parameter index.
   * \param i Free coordinate inside it.
   * \return The parent's, at the coordinate the pair stands for.
   */
  inline double coord_lower(int j, int i) const { return mv_coord_lower(parent, parent_slot(j), parent_element(j, i)); }
  /**
   * \brief Upper bound of the natural domain of free coordinate \p i of free parameter \p j.
   * \param j Free parameter index.
   * \param i Free coordinate inside it.
   * \return The parent's, at the coordinate the pair stands for.
   */
  inline double coord_upper(int j, int i) const { return mv_coord_upper(parent, parent_slot(j), parent_element(j, i)); }

  /// \brief The parent's coordinate of free coordinate \p i of free parameter \p j. \param j Free parameter index. \param i Free coordinate. \return The parent's coordinate.
  LAPLACE_HD int parent_element(int j, int i) const {
    const int s = parent_slot(j);
    int free_seen = 0;
    for (int e = 0, n = parent.param_size(s); e < n; ++e) {
      if (detail_fixed::element_held<C...>(s, e)) continue;
      if (free_seen == i) return e;
      ++free_seen;
    }
    return parent.param_size(s);
  }

  /**
   * \brief Draws one unit, the parent's at the held dense parameter.
   * \param st State from theta_state().
   * \param x Free dense coordinates.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output unit of length `dim()`.
   */
  LAPLACE_HD void rng(const state_type& st, const double* x, std::uint64_t seed, std::uint64_t unit, double* y) const
    requires detail_fixed::has_mv_rng<D>
  {
    parent.rng(st, parent_dense(x), seed, unit, y);
  }

  /// \brief Writes the mean, the parent's. \param x Free dense coordinates. \param out Output of length `dim()`.
  LAPLACE_HD void mean(const double* x, double* out) const
    requires detail_fixed::has_mean<D>
  {
    parent.mean(parent_dense(x), out);
  }

  /**
   * \brief Writes starting values of the free indexed coordinates (host).
   *
   * The parent starts every one of its indexed coordinates from the sample; the held ones are then
   * dropped, so a start never moves a coordinate the model does not estimate.
   * \param summary The sample summary.
   * \param theta Output of length `theta_size()`.
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const
    requires detail_fixed::has_mv_start<D>
  {
    std::vector<double> full(static_cast<std::size_t>(parent.theta_size()), 0.0);
    parent.start_value(summary, full.data());
    int s = 0;
    for (int t = 0, n = parent.theta_size(); t < n; ++t)
      if (!theta_coord_held(t)) theta[s++] = full[static_cast<std::size_t>(t)];
  }

  /**
   * \brief Proximal operator in \f$y\f$, the parent's at the held dense parameter.
   * \param st State from theta_state().
   * \param v Point of length `dim()`.
   * \param x Free dense coordinates.
   * \param t Step \f$t > 0\f$.
   * \param out Minimizer of length `dim()`.
   * \param w Working scratch.
   */
  LAPLACE_HD void prox_y(const state_type& st, const double* v, const double* x, double t, double* out,
                         double* w) const
    requires detail_fixed::has_mv_prox_y<D>
  {
    parent.prox_y(st, v, parent_dense(x), t, out, w);
  }

 private:
  // Whether the I-th held coordinate is coordinate e of the parent's parameter j,
  // and then its value. The pack is read at a compile-time index, never through a
  // braced list, which in device code would need an std::initializer_list.
  template <std::size_t I>
  LAPLACE_HD bool value_from_pack(int j, int e, double& out) const {
    constexpr int code = detail_fixed::coord_at<I, C...>();
    constexpr int slot = fixed_coord_slot(code), element = fixed_coord_element(code);
    if (slot != j) return false;
    if constexpr (element == whole_slot) {
      out = held[I].vector == nullptr ? held[I].scalar : held[I].vector[e];
      return true;
    } else {
      if (element != e) return false;
      out = held[I].scalar;
      return true;
    }
  }
  template <std::size_t... I>
  LAPLACE_HD bool value_from_pack_fold(int j, int e, double& out, std::index_sequence<I...>) const {
    return (value_from_pack<I>(j, e, out) || ...);
  }

  // the value of coordinate e of the parent's parameter j: the held one, or the
  // next free coordinate of the caller's array, s being advanced
  LAPLACE_HD double value_at(int j, int e, const double* free_values, int& s) const {
    double v = 0.0;
    if (value_from_pack_fold(j, e, v, std::make_index_sequence<sizeof...(C)>{})) return v;
    return free_values[s++];
  }
};

// ---------------------------------------------------------------------------
// the builders
// ---------------------------------------------------------------------------

namespace detail_fixed {

inline fixed_value as_fixed_value(double v) { return fixed_value{v, nullptr}; }
inline fixed_value as_fixed_value(const double* v) { return fixed_value{0.0, v}; }
inline fixed_value as_fixed_value(fixed_value v) { return v; }

}  // namespace detail_fixed

/**
 * \ingroup distrib_fixed
 * \brief Holds some parameters of a univariate distribution at given values, checking that each is one.
 *
 * Refuses a value outside the open natural domain of its parameter, or sitting exactly on a non-smooth
 * point of that parameter's own axis, where the law is not differentiable in it and the derivatives in
 * the free parameters need not exist either. The values are written into the parent, so the wrapper and
 * its parent agree.
 * \tparam C The held coordinates, usually written `coord_of<D>("name")`.
 * \tparam D Parent distribution type, univariate.
 * \param parent The wrapped distribution.
 * \param values One value per held parameter, in the order of \p C.
 * \return The wrapper.
 * \exception std::invalid_argument When a value is refused.
 */
template <int... C, class D, class... V>
  requires(!multivariate_distrib<D> && sizeof...(V) == sizeof...(C))
fixed_distrib<D, C...> fixed(const D& parent, V... values) {
  fixed_distrib<D, C...> f{.parent = parent, .held = {static_cast<double>(values)...}};
  int h = 0;
  for (const int code : {C...}) {
    const distrib_param_info& p = D::params[fixed_coord_slot(code)];
    const double v = f.held[h++];
    if (!(v > p.lower && v < p.upper))
      throw std::invalid_argument("fixed: the value " + std::to_string(v) + " of the parameter " + p.name +
                                  " is not inside its open domain (" + std::to_string(p.lower) + ", " +
                                  std::to_string(p.upper) + ")");
    for (int i = 0; i < p.axis_kinks.count; ++i)
      if (v == p.axis_kinks.point[i])
        throw std::invalid_argument("fixed: the value " + std::to_string(v) + " of the parameter " + p.name +
                                    " sits on a non-smooth point of its own axis, where the law is not "
                                    "differentiable in it");
  }
  f.write_held();
  return f;
}

/**
 * \ingroup distrib_fixed
 * \brief Holds some parameters of a multivariate distribution at given values, checking that each is one.
 *
 * Each value is a double, for one coordinate or for every coordinate of a parameter, or a pointer to the
 * coordinates of a whole parameter (not copied, and it must be reachable where the call runs). A whole
 * held dense parameter needs the pointer: `logpdf(st, y, x)` takes no scratch, so there is nowhere to
 * spread a scalar over its coordinates. Every coordinate is checked against the parent's own bounds,
 * which it must lie strictly inside.
 * \tparam C The held coordinates, usually written `coord_of<D>("name")` or `coord_of<D>("name", i)`.
 * \tparam D Parent distribution type, multivariate.
 * \param parent The wrapped distribution.
 * \param values One value or pointer per held coordinate, in the order of \p C.
 * \return The wrapper.
 * \exception std::invalid_argument When a value is refused.
 */
template <int... C, class D, class... V>
  requires(multivariate_distrib<D> && sizeof...(V) == sizeof...(C))
fixed_distrib<D, C...> fixed(const D& parent, V... values) {
  fixed_distrib<D, C...> f{.parent = parent, .held = {detail_fixed::as_fixed_value(values)...}};
  int h = 0;
  for (const int code : {C...}) {
    const int slot = fixed_coord_slot(code), element = fixed_coord_element(code);
    const fixed_value& fv = f.held[h++];
    const char* name = D::params[slot].name;
    const bool dense = mv_param_dense(D::params[slot].kind);
    if (dense && fv.vector == nullptr)
      throw std::invalid_argument(std::string("fixed: the dense parameter ") + name +
                                  " is held at a pointer to its coordinates, there being no scratch in logpdf to "
                                  "spread one value over them");
    // The bounds of a matrix parameter are not per-coordinate intervals: what
    // constrains it is the definiteness of its structure, and its descriptor
    // carries a placeholder (distrib_mv_param_info, lower and upper being for a
    // scalar or a vector). So only the other kinds are checked here.
    const int lo = element == whole_slot ? 0 : element;
    const int hi = element == whole_slot ? parent.param_size(slot) : element + 1;
    for (int e = lo; e < hi && D::params[slot].kind != distrib_mv_param_kind::matrix; ++e) {
      const double v = fv.vector == nullptr ? fv.scalar : fv.vector[e - lo];
      const double l = mv_coord_lower(parent, slot, e), u = mv_coord_upper(parent, slot, e);
      if (!(v > l && v < u))
        throw std::invalid_argument("fixed: the value " + std::to_string(v) + " of coordinate " + std::to_string(e) +
                                    " of the parameter " + name + " is not inside its open domain (" +
                                    std::to_string(l) + ", " + std::to_string(u) + ")");
    }
  }
  return f;
}

}  // namespace laplace::distrib
