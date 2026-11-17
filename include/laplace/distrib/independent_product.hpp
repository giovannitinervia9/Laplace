// independent_product_distrib<M_1, ..., M_n>: the product of independent
// blocks (note/penalties.md section 5a). Component k is a multivariate distrib
// M_k of dimension d_k repeated G_k times ("copies"); copy c of component k is
// one block of y and of mu, and its parameters are those of the set
// h_k(c) in [0, H_k):
//
//   log p(y | mu, theta) = sum_k sum_c log p_{M_k}(y_kc | mu_kc, theta_k,h_k(c))
//
// with h_k shared (H_k = 1), by copy (H_k = G_k, h_k(c) = c) or by group (an
// index vector). y and mu are concatenated over the blocks, component by
// component and copy by copy. Each M_k has one dense parameter, its location.
//
// theta is param-major (Giovanni, 2026-11-17): for each component k and each
// of its indexed parameters j, the H_k copies of parameter j side by side. The
// product declares one typed parameter per (k, j), with M_k's kind, domain and
// regularity, of size H_k times that of M_k's parameter j. theta_state gathers
// each set's theta into w and calls M_k::theta_state on it.
//
// Derivatives: a theta tuple whose indices belong to two sets is zero (false);
// otherwise only the copies of that set contribute. The block route
// (logpdf_dy_dk_blocks, n_blocks, block_start) returns, block after block, the
// dense output of M_k for that copy; the dense route scatters the blocks into
// the packed tensors over the whole dimension (for tests and small d).
#pragma once

#include <laplace/distrib/core.hpp>
#include <laplace/distrib/multivariate_core.hpp>

#include <cstring>
#include <type_traits>
#include <utility>

namespace laplace::distrib {

/**
 * \ingroup distrib_independent_product
 * \brief One component of `independent_product_distrib`: a law, its number of copies and the map from copies to parameter sets.
 *
 * Copy \f$c\f$ (\f$0 \le c <\f$ `copies`) uses parameter set \f$h(c)\f$:
 * `set_of_copy[c]` when the map is given; otherwise 0 when `n_sets` is 1
 * (shared) and \f$c\f$ when `n_sets` equals `copies` (one set per copy).
 *
 * \tparam M Multivariate distribution whose only dense parameter is its location.
 */
template <class M>
struct independent_product_component {
  /// The law of one copy.
  M distribution{};
  /// Number of copies \f$G\f$.
  int copies = 1;
  /// Number of parameter sets \f$H\f$.
  int n_sets = 1;
  /// Map from copies to sets, length `copies` (not copied), or `nullptr` for shared or one set per copy.
  const int* set_of_copy = nullptr;

  /// \param c Copy index. \return Its parameter set \f$h(c)\f$.
  LAPLACE_HD int set(int c) const { return set_of_copy != nullptr ? set_of_copy[c] : (n_sets == 1 ? 0 : c); }
};

/**
 * \ingroup distrib_independent_product
 * \brief State of `independent_product_distrib`: a pointer to the per-set states stored in `w`.
 */
struct independent_product_state {
  /// Start of the region written by `theta_state()`.
  const double* regions;
};

namespace detail_product {

// The components, head and tail, visited in order with their compile-time index.
template <class... M>
struct component_list {
  template <int I, class F>
  LAPLACE_HD void visit(F&) const {}
};
template <class M, class... R>
struct component_list<M, R...> {
  independent_product_component<M> head;
  component_list<R...> tail;
  component_list() = default;
  LAPLACE_HD component_list(const independent_product_component<M>& h, const independent_product_component<R>&... r)
      : head(h), tail(r...) {}
  template <int I, class F>
  LAPLACE_HD void visit(F& f) const {
    f(head, std::integral_constant<int, I>{});
    tail.template visit<I + 1>(f);
  }
};

template <class M>
constexpr void check_component() {
  static_assert(detail_mv::dense_param_index<M>(0) == 0 && detail_mv::dense_param_index<M>(1) < 0,
                "independent_product_distrib: a component must have one dense parameter, its location, first");
  static_assert(std::is_trivially_copyable_v<typename M::state_type>,
                "independent_product_distrib: a component's state must be trivially copyable");
}

template <class... M>
constexpr distrib_normalization product_normalization() {
  bool improper = false, degenerate = false;
  ((improper = improper || M::normalization == distrib_normalization::improper), ...);
  ((degenerate = degenerate || M::normalization == distrib_normalization::degenerate), ...);
  return improper ? distrib_normalization::improper
                  : (degenerate ? distrib_normalization::degenerate : distrib_normalization::proper);
}

template <class... M>
struct param_table {
  distrib_mv_param_info v[1 + ((M::n_params - 1) + ... + 0)];
};
template <class... M>
constexpr param_table<M...> make_param_table() {
  param_table<M...> t{};
  int mu_regularity = infinitely_smooth, p = 1;
  auto add = [&](const auto& params, int n) {
    if (params[0].regularity < mu_regularity) mu_regularity = params[0].regularity;
    for (int j = 1; j < n; ++j) t.v[p++] = params[j];
  };
  (add(M::params, M::n_params), ...);
  t.v[0] = {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, mu_regularity};
  return t;
}
template <class... M>
inline constexpr param_table<M...> params_of = make_param_table<M...>();

// Doubles that hold a state struct.
template <class T>
inline constexpr int doubles_of = int((sizeof(T) + sizeof(double) - 1) / sizeof(double));

// Position of the sorted tuple t[0] <= ... <= t[K-1] over n variables (lexicographic).
LAPLACE_HD int sorted_tuple_position(const int* t, int K, int n) {
  int pos = 0, prev = 0;
  for (int j = 0; j < K; ++j) {
    for (int v = prev; v < t[j]; ++v) pos += binomial_coef(n - v + K - j - 2, K - j - 1);
    prev = t[j];
  }
  return pos;
}
// The next sorted tuple over n variables, in lexicographic order.
LAPLACE_HD void next_sorted_tuple(int* t, int K, int n) {
  int i = K - 1;
  while (i > 0 && t[i] == n - 1) --i;
  ++t[i];
  for (int j = i + 1; j < K; ++j) t[j] = t[i];
}
// The same in the layout of the multivariate outputs: order 2 is the packed lower
// triangle by rows, the sorted pair (j, i), j <= i, at i (i + 1)/2 + j, which
// enumerates the pairs in colexicographic order; other orders are lexicographic.
LAPLACE_HD int output_position(const int* t, int K, int n) {
  return K == 2 ? t[1] * (t[1] + 1) / 2 + t[0] : sorted_tuple_position(t, K, n);
}
LAPLACE_HD void next_output_tuple(int* t, int K, int n) {
  if (K != 2) return next_sorted_tuple(t, K, n);
  if (t[0] < t[1]) {
    ++t[0];
  } else {
    t[0] = 0;
    ++t[1];
  }
}

}  // namespace detail_product

/**
 * \ingroup distrib_independent_product
 * \brief Product of independent blocks: component \f$k\f$ is the law `M_k` repeated over its copies.
 *
 * \f$\log p(y \mid \mu, \theta) = \sum_k \sum_c \log p_{M_k}(y_{kc} \mid \mu_{kc}, \theta_{k, h_k(c)})\f$,
 * with \f$y\f$ and \f$\mu\f$ concatenated over the copies, component by
 * component, and \f$\theta\f$ param-major: for each component \f$k\f$ and each
 * indexed parameter \f$j\f$ of `M_k`, the \f$H_k\f$ sets side by side. Each
 * copy is one block. Satisfies `multivariate_distrib`.
 *
 * \tparam M Multivariate distributions whose only dense parameter is the location.
 */
template <class... M>
struct independent_product_distrib {
  static_assert(sizeof...(M) >= 1, "independent_product_distrib: at least one component");
  static_assert((true && ... && (detail_product::check_component<M>(), true)));

  /// The components.
  detail_product::component_list<M...> components;

  /// Normalization: improper if a component is, degenerate if a component is, proper otherwise.
  static constexpr distrib_normalization normalization = detail_product::product_normalization<M...>();
  /// Number of typed parameters: \f$\mu\f$, then the indexed parameters of each component.
  static constexpr int n_params = 1 + ((M::n_params - 1) + ... + 0);
  /// Parameter descriptors: \f$\mu\f$ (vector), then those of each `M_k` after its location.
  static constexpr const distrib_mv_param_info (&params)[n_params] = detail_product::params_of<M...>.v;
  /// State type.
  using state_type = independent_product_state;

  /// Default constructor.
  independent_product_distrib() = default;
  /**
   * \brief Construct from the components, in the order of `M`.
   * \param c One `independent_product_component<M_k>` per component.
   */
  LAPLACE_HD explicit independent_product_distrib(const independent_product_component<M>&... c) : components(c...) {}

  /// \return The dimension \f$\sum_k G_k d_k\f$.
  LAPLACE_HD int dim() const {
    int n = 0;
    each([&](const auto& c, auto) { n += c.copies * c.distribution.dim(); });
    return n;
  }
  /// \return The number of blocks, \f$\sum_k G_k\f$.
  LAPLACE_HD int n_blocks() const {
    int n = 0;
    each([&](const auto& c, auto) { n += c.copies; });
    return n;
  }
  /**
   * \brief First coordinate of block `g` in \f$y\f$ or \f$\mu\f$ (the two share their blocks).
   * \param slot `y_slot` or 0 (\f$\mu\f$).
   * \param g Block index, \f$0 \le g \le\f$ `n_blocks()`.
   * \return The start of block `g`, `dim()` for `g == n_blocks()`.
   */
  LAPLACE_HD int block_start(int slot, int g) const {
    (void)slot;
    int s = 0, done = 0;
    each([&](const auto& c, auto) {
      if (done) return;
      const int dk = c.distribution.dim();
      if (g < c.copies) {
        s += g * dk;
        done = 1;
      } else {
        s += c.copies * dk;
        g -= c.copies;
      }
    });
    return s;
  }
  /// \param j Parameter index. \return `dim()` for \f$\mu\f$, \f$H_k\f$ times the size of `M_k`'s parameter otherwise.
  LAPLACE_HD int param_size(int j) const {
    if (j == 0) return dim();
    int n = 0;
    each([&](const auto& c, auto) {
      using Mk = std::remove_cvref_t<decltype(c.distribution)>;
      if (j >= 1 && j < Mk::n_params) n = c.n_sets * c.distribution.param_size(j);
      j -= Mk::n_params - 1;
    });
    return n;
  }
  /// \param j Parameter index. \return 0 for \f$\mu\f$, the start of parameter `j` in \f$\theta\f$ otherwise.
  LAPLACE_HD int param_offset(int j) const {
    int off = 0;
    for (int i = 1; i < j; ++i) off += param_size(i);
    return off;
  }
  /// \return The length of \f$\theta\f$, \f$\sum_k H_k q_k\f$ with \f$q_k\f$ that of `M_k`.
  LAPLACE_HD int theta_size() const {
    int n = 0;
    each([&](const auto& c, auto) { n += c.n_sets * c.distribution.theta_size(); });
    return n;
  }
  /// \return Doubles of the state: for each set, its state struct, its gathered \f$\theta\f$ and the state of `M_k`.
  LAPLACE_HD int state_size() const {
    int n = 0;
    each([&](const auto& c, auto) { n += c.n_sets * region_size(c); });
    return n;
  }
  /// \return Doubles of scratch: the largest scratch of the components.
  LAPLACE_HD int scratch_size() const {
    int n = 0;
    each([&](const auto& c, auto) { n = c.distribution.scratch_size() > n ? c.distribution.scratch_size() : n; });
    return n;
  }

  /**
   * \brief Gather the \f$\theta\f$ of each set and compute the state of `M_k` on it.
   * \param theta Parameters, length `theta_size()`, param-major.
   * \param w Workspace of `state_size() + scratch_size()` doubles.
   * \return The state, pointing into `w`.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const {
    double* r = w;
    int toff = 0;
    each([&](const auto& c, auto) {
      using Mk = std::remove_cvref_t<decltype(c.distribution)>;
      const int size = region_size(c);
      for (int h = 0; h < c.n_sets; ++h, r += size) {
        double* th = r + detail_product::doubles_of<typename Mk::state_type>;
        gather_set_theta(c, theta + toff, h, th);
        // M_k's scratch follows its state, over the regions not yet written
        const typename Mk::state_type st = c.distribution.theta_state(th, th + c.distribution.theta_size());
        std::memcpy(r, &st, sizeof st);
      }
      toff += c.n_sets * c.distribution.theta_size();
    });
    return state_type{w};
  }

  /**
   * \brief Copy the \f$\theta\f$ of set `h` of a component out of its param-major slice.
   * \tparam Mk Component distribution type.
   * \param c The component.
   * \param theta_k Start of the component's slice of \f$\theta\f$.
   * \param h Set index.
   * \param out The parameters of `M_k`, length `M_k::theta_size()`.
   */
  template <class Mk>
  LAPLACE_HD static void gather_set_theta(const independent_product_component<Mk>& c, const double* theta_k, int h,
                                          double* out) {
    const double* src = theta_k;
    for (int j = 1; j < Mk::n_params; ++j) {
      const int n = c.distribution.param_size(j);
      const int o = c.distribution.param_offset(j);
      for (int i = 0; i < n; ++i) out[o + i] = src[h * n + i];
      src += c.n_sets * n;
    }
  }

  /**
   * \brief The log-density, the sum over the copies.
   * \param st State from `theta_state()`.
   * \param y Observation, length `dim()`.
   * \param x Location \f$\mu\f$, length `dim()`.
   * \return \f$\log p(y \mid \mu, \theta)\f$.
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* x) const {
    double s = 0.0;
    int off = 0;
    const double* r = st.regions;
    each([&](const auto& c, auto) {
      const int dk = c.distribution.dim();
      for (int g = 0; g < c.copies; ++g, off += dk)
        s += c.distribution.logpdf(set_state(c, r, c.set(g)), y + off, x + off);
      r += c.n_sets * region_size(c);
    });
    return s;
  }

  /**
   * \brief Derivatives `J` in \f$y\f$, `K` in \f$\theta\f$ along `idx` and `A` in \f$\mu\f$, block by block.
   *
   * Writes, block after block, the output of `M_k::logpdf_dy_dk<J, K, A...>` for
   * that copy (\f$\binom{d_k + J - 1}{J} \binom{d_k + A - 1}{A}\f$ doubles),
   * zeros for a copy outside the set of `idx` or reported zero by `M_k`. Order
   * 0 (\f$J = A = 0\f$) is the scalar, the sum over the copies.
   *
   * \tparam J Order in \f$y\f$.
   * \tparam K Order in \f$\theta\f$.
   * \tparam A Order in \f$\mu\f$ (omitted means 0).
   * \param st State from `theta_state()`.
   * \param y Observation, length `dim()`.
   * \param x Location, length `dim()`.
   * \param idx Sorted tuple of `K` indices into \f$\theta\f$ (unused when `K == 0`).
   * \param out Output of `mv_block_tensor_size<A...>(*this, J)` doubles.
   * \param w Scratch: what the components ask (`scratch_size()`, plus \f$\binom{d_k + J + A - 1}{J + A}\f$ for a Gaussian with \f$A > 0\f$).
   * \return `false`, `out` untouched, when `idx` spans two sets; `true` otherwise.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk_blocks(const state_type& st, const double* y, const double* x, const int* idx,
                                      double* out, double* w) const {
    constexpr int a = detail_mv::dense_order<A...>();
    int kset = -1, hset = -1;
    int local[K > 0 ? K : 1];
    if (!decode_tuple<K>(idx, kset, hset, local)) return false;
    // order 0 is the scalar: each copy writes into one double, summed into out[0]
    constexpr bool scalar = J == 0 && a == 0;
    double one = 0.0, sum = 0.0;
    int off = 0;
    const double* r = st.regions;
    each([&](const auto& c, auto kc) {
      const int dk = c.distribution.dim();
      const int n = scalar ? 0 : sym_tensor_size(dk, J) * sym_tensor_size(dk, a);
      for (int g = 0; g < c.copies; ++g, off += dk, out += n) {
        const int h = c.set(g);
        double* o = scalar ? &one : out;
        if ((K > 0 && (int(kc) != kset || h != hset)) ||
            !c.distribution.template logpdf_dy_dk<J, K, A...>(set_state(c, r, h), y + off, x + off, local, o, w))
          for (int i = 0; i < (scalar ? 1 : n); ++i) o[i] = 0.0;
        sum += one;
      }
      r += c.n_sets * region_size(c);
    });
    if constexpr (scalar) out[0] = sum;
    return true;
  }

  /**
   * \brief Derivatives `K` in \f$\theta\f$ and `A` in \f$\mu\f$, block by block: `logpdf_dy_dk_blocks<0, K, A...>`.
   * \tparam K Order in \f$\theta\f$.
   * \tparam A Order in \f$\mu\f$ (omitted means 0).
   * \param st State. \param y Observation. \param x Location. \param idx Sorted tuple over \f$\theta\f$ (unused when `K == 0`).
   * \param out Output of `mv_block_tensor_size<A...>(*this)` doubles. \param w Scratch as for `logpdf_dy_dk_blocks()`.
   * \return As `logpdf_dy_dk_blocks()`.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk_blocks(const state_type& st, const double* y, const double* x, const int* idx,
                                   double* out, double* w) const {
    return logpdf_dy_dk_blocks<0, K, A...>(st, y, x, idx, out, w);
  }

  /**
   * \brief Derivatives `J` in \f$y\f$, `K` in \f$\theta\f$ and `A` in \f$\mu\f$ in the dense layout over `dim()`.
   *
   * The packed tensor of order `J` over \f$y\f$, Kronecker (\f$y\f$ first) with
   * that of order `A` over \f$\mu\f$, every entry that couples two blocks zero;
   * for \f$J = A = 0\f$ the scalar, as in the block layout. Written for tests
   * and small dimensions: the size grows as \f$d^{J + A}\f$.
   *
   * \tparam J Order in \f$y\f$. \tparam K Order in \f$\theta\f$. \tparam A Order in \f$\mu\f$ (omitted means 0).
   * \param st State. \param y Observation. \param x Location. \param idx Sorted tuple over \f$\theta\f$ (unused when `K == 0`).
   * \param out Output of \f$\binom{d + J - 1}{J}\binom{d + A - 1}{A}\f$ doubles.
   * \param w Scratch as for `logpdf_dy_dk_blocks()`, followed by `mv_block_tensor_size<A...>(*this, J)` doubles.
   * \return As `logpdf_dy_dk_blocks()`.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                               double* w) const {
    constexpr int a = detail_mv::dense_order<A...>();
    if constexpr (J == 0 && a == 0) return logpdf_dy_dk_blocks<J, K, A...>(st, y, x, idx, out, w);  // the scalar
    const int d = dim();
    const int na = sym_tensor_size(d, a), nt = sym_tensor_size(d, J) * na;
    // the components' scratch first, the blocks after it
    int need = scratch_size();
    if constexpr (a > 0) each([&](const auto& c, auto) {
        const int m = c.distribution.scratch_size() + sym_tensor_size(c.distribution.dim(), J + a);
        need = m > need ? m : need;
      });
    double* buf = w + need;
    if (!logpdf_dy_dk_blocks<J, K, A...>(st, y, x, idx, buf, w)) return false;
    for (int i = 0; i < nt; ++i) out[i] = 0.0;
    int ty[J > 0 ? J : 1], ta[a > 0 ? a : 1], gy[J > 0 ? J : 1], ga[a > 0 ? a : 1];
    const double* b = buf;
    for (int g = 0, G = n_blocks(); g < G; ++g) {
      const int s = block_start(y_slot, g), m = block_start(y_slot, g + 1) - s;
      const int sy = sym_tensor_size(m, J), sa = sym_tensor_size(m, a);
      for (int i = 0; i < J; ++i) ty[i] = 0;
      for (int iy = 0; iy < sy; ++iy) {
        for (int i = 0; i < J; ++i) gy[i] = ty[i] + s;
        const int py = detail_product::output_position(gy, J, d);
        for (int i = 0; i < a; ++i) ta[i] = 0;
        for (int ia = 0; ia < sa; ++ia) {
          for (int i = 0; i < a; ++i) ga[i] = ta[i] + s;
          out[py * na + detail_product::output_position(ga, a, d)] = *b++;
          if (a > 0) detail_product::next_output_tuple(ta, a, m);
        }
        if (J > 0) detail_product::next_output_tuple(ty, J, m);
      }
    }
    return true;
  }

  /**
   * \brief Derivatives `K` in \f$\theta\f$ and `A` in \f$\mu\f$, dense layout: `logpdf_dy_dk<0, K, A...>`.
   * \tparam K Order in \f$\theta\f$. \tparam A Order in \f$\mu\f$ (omitted means 0).
   * \param st State. \param y Observation. \param x Location. \param idx Sorted tuple over \f$\theta\f$ (unused when `K == 0`).
   * \param out Output, the packed tensor of order `A` over `dim()`. \param w Scratch as for `logpdf_dy_dk()`.
   * \return As `logpdf_dy_dk_blocks()`.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                            double* w) const {
    return logpdf_dy_dk<0, K, A...>(st, y, x, idx, out, w);
  }

  /**
   * \brief Visit the components in order: `f(component, std::integral_constant<int, k>)`.
   * \tparam F Callable type.
   * \param f Callable.
   */
  template <class F>
  LAPLACE_HD void each(F&& f) const {
    components.template visit<0>(f);
  }

 private:
  template <class Mk>
  LAPLACE_HD static int region_size(const independent_product_component<Mk>& c) {
    return detail_product::doubles_of<typename Mk::state_type> + c.distribution.theta_size() + c.distribution.state_size();
  }
  // the state of set h of a component whose regions start at r
  template <class Mk>
  LAPLACE_HD static typename Mk::state_type set_state(const independent_product_component<Mk>& c, const double* r,
                                                      int h) {
    typename Mk::state_type st;
    std::memcpy(static_cast<void*>(&st), r + h * region_size(c), sizeof st);
    return st;
  }
  // maps a sorted tuple over theta to one (component, set) and M_k's own indices;
  // false when it spans two sets
  template <int K>
  LAPLACE_HD bool decode_tuple(const int* idx, int& kset, int& hset, int* local) const {
    for (int i = 0; i < K; ++i) {
      int t = idx[i], k = -1, h = -1, l = -1;
      each([&](const auto& c, auto kc) {
        using Mk = std::remove_cvref_t<decltype(c.distribution)>;
        if (k >= 0) return;
        const int q = c.n_sets * c.distribution.theta_size();
        if (t >= q) {
          t -= q;
          return;
        }
        for (int j = 1; j < Mk::n_params; ++j) {
          const int n = c.distribution.param_size(j);
          if (t < c.n_sets * n) {
            k = int(kc);
            h = t / n;
            l = c.distribution.param_offset(j) + t % n;
            return;
          }
          t -= c.n_sets * n;
        }
      });
      if (i == 0) {
        kset = k;
        hset = h;
      } else if (k != kset || h != hset) {
        return false;
      }
      local[i] = l;
    }
    return true;
  }
};

}  // namespace laplace::distrib
