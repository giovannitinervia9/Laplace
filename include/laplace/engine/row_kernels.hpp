#pragma once

// The row kernels of the general route (laplace::engine; plan, sections 7.8
// and 7.9): one kernel for each quantity, which computes that quantity and
// nothing above it, a template on the distrib D and on the number R of linear
// predictors, chosen on the host by a switch over a list of distribs given
// by the caller (distrib_list) and over R = 1, ..., 4.
//
//   value                  f(x) = -sum_i w_i l_i
//   gradient               also grad f
//   observed_information   also the Hessian of f
//   expected_information   also sum_i w_i J_i' I(theta_i) J_i, J_i = d theta_i / d x
//
// with l_i the log-density (or log-mass) of unit i at theta_i, the
// parameters of the distrib, and w_i its weight. For one unit, with eta the
// linear predictors and theta = theta(eta) through the nodes,
//
//   d l / d eta_a        = l_j theta_j,a
//   d2 l / d eta_a eta_b = l_j theta_j,ab + l_jk theta_j,a theta_k,b
//   E[-d2 l / d eta_a eta_b] = I_jk theta_j,a theta_k,b
//
// (sums over the parameters j, k): the second derivatives of the nodes
// multiply the score, whose expectation is zero, so the expected information
// needs the nodes to order one only. A linear predictor is a stored design
// row times x, so the derivatives in x are those in eta times the row.
//
// The output (row_assembly) is in the layout of the arrowhead (row_model.hpp):
// gradient in the numbering of x; the information as the global block (lower
// triangle by rows), then for each group its square block (lower triangle by
// rows) and its cross block with the globals (group_width x n_global, by
// rows).

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/engine/predictor_tensor.hpp>
#include <laplace/engine/row_model.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Selects what quantity a row kernel evaluates.
 *
 * Each value computes the requested quantity and no higher-order node derivatives:
 * `value` (\f$f\f$, order 0), `gradient` (also \f$\nabla f\f$, order 1),
 * `observed_information` (also the observed Hessian \f$\nabla^2 f\f$, order 2), and
 * `expected_information` (also \f$\sum_i w_i J_i^\top I(\theta_i) J_i\f$, order 1).
 *
 * \see assemble_rows(), assemble_row_model(), \ref engine
 */
enum class row_quantity : int { value = 0, gradient = 1, observed_information = 2, expected_information = 3 };

/**
 * \ingroup engine
 * \brief Type list of distributions selectable at runtime by row_model_spec::distrib.
 *
 * \tparam Ds Distribution types satisfying \ref distrib_core.
 * \see assemble_row_model(), row_model_objective, \ref engine
 */
template <class... Ds>
struct distrib_list {};

/**
 * \ingroup engine
 * \brief Index mapping for the arrowhead layout of row_assembly::information.
 *
 * Stores the lower triangle of the \f$n_{\mathrm{global}} \times n_{\mathrm{global}}\f$ global block
 * by rows, followed for each group \f$g \in [0, G)\f$ (`n_groups`) by its
 * \f$d_{\mathrm{grp}} \times d_{\mathrm{grp}}\f$ (`group_width`) square block lower triangle by rows
 * and its \f$d_{\mathrm{grp}} \times n_{\mathrm{global}}\f$ cross block with the globals by rows.
 *
 * \see row_assembly, assemble_rows(), \ref engine
 */
struct arrowhead_layout {
  int n_global = 0, n_groups = 0, group_width = 0;
  /**
   * \ingroup engine
   * \brief Number of lower-triangular entries in the global block.
   * \return \f$n_{\mathrm{global}}(n_{\mathrm{global}} + 1)/2\f$.
   */
  LAPLACE_HD int global_size() const { return n_global * (n_global + 1) / 2; }
  /**
   * \ingroup engine
   * \brief Number of entries per group (square block lower triangle plus cross block).
   * \return \f$d_{\mathrm{grp}}(d_{\mathrm{grp}} + 1)/2 + d_{\mathrm{grp}} n_{\mathrm{global}}\f$.
   */
  LAPLACE_HD int group_block_size() const { return group_width * (group_width + 1) / 2 + group_width * n_global; }
  /**
   * \ingroup engine
   * \brief Total number of stored entries in the arrowhead layout.
   * \return `global_size() + n_groups * group_block_size()`.
   */
  LAPLACE_HD int size() const { return global_size() + n_groups * group_block_size(); }
  /**
   * \ingroup engine
   * \brief Offset of entry \f$(r, c)\f$ in the global block (\f$0 \le c \le r < n_{\mathrm{global}}\f$).
   * \param r Row index.
   * \param c Column index.
   * \return Zero-based offset in `row_assembly::information`.
   */
  LAPLACE_HD int global(int r, int c) const { return r * (r + 1) / 2 + c; }
  /**
   * \ingroup engine
   * \brief Offset of entry \f$(j, l)\f$ in the square block of group \f$g\f$ (\f$0 \le l \le j < d_{\mathrm{grp}}\f$).
   * \param g Group index.
   * \param j Row index within group \f$g\f$.
   * \param l Column index within group \f$g\f$.
   * \return Zero-based offset in `row_assembly::information`.
   */
  LAPLACE_HD int group(int g, int j, int l) const { return global_size() + g * group_block_size() + j * (j + 1) / 2 + l; }
  /**
   * \ingroup engine
   * \brief Offset of entry \f$(j, c)\f$ in the cross block between group \f$g\f$ and global coordinate \f$c\f$.
   * \param g Group index.
   * \param j Within-group coordinate index.
   * \param c Global coordinate index.
   * \return Zero-based offset in `row_assembly::information`.
   */
  LAPLACE_HD int cross(int g, int j, int c) const {
    return global_size() + g * group_block_size() + group_width * (group_width + 1) / 2 + j * n_global + c;
  }
};

/**
 * \ingroup engine
 * \brief Host result of a row kernel evaluation.
 *
 * Holds the objective value \f$f(x) = -\sum_i w_i \ell_i(x)\f$, and when requested by `row_quantity`,
 * the gradient in coordinate order and the information matrix in arrowhead_layout order.
 *
 * \see arrowhead_layout, assemble_row_model(), \ref engine
 */
struct row_assembly {
  double value = 0.0;
  std::vector<double> gradient;
  std::vector<double> information;
};

namespace detail_rows {

template <class D, std::size_t... I>
LAPLACE_HD D make_distrib(const double* th, std::index_sequence<I...>) {
  return D{th[I]...};
}

template <class D>
LAPLACE_HD double log_density(const D& d, double y) {
  if constexpr (D::kind == distrib::distrib_kind::continuous) return d.logpdf(y);
  else return d.logpmf(y);
}
template <class D>
LAPLACE_HD auto log_density_d1(const D& d, double y) {
  if constexpr (D::kind == distrib::distrib_kind::continuous) return d.logpdf_d1(y);
  else return d.logpmf_d1(y);
}
template <class D>
LAPLACE_HD auto log_density_d2(const D& d, double y) {
  if constexpr (D::kind == distrib::distrib_kind::continuous) return d.logpdf_d2(y);
  else return d.logpmf_d2(y);
}

template <class D>
LAPLACE_HD auto log_density_d3(const D& d, double y) {
  if constexpr (D::kind == distrib::distrib_kind::continuous) return d.logpdf_d3(y);
  else return d.logpmf_d3(y);
}
template <class D>
LAPLACE_HD auto log_density_d4(const D& d, double y) {
  if constexpr (D::kind == distrib::distrib_kind::continuous) return d.logpdf_d4(y);
  else return d.logpmf_d4(y);
}

// The storage order of the values of a kernel that needs derivatives to order K
template <int K>
inline constexpr int value_order = K > 2 ? K : 2;

// The walk of the pointwise nodes of unit i, to derivative order K.
template <int R, int K>
LAPLACE_HD void walk_nodes(const row_model_view& M, int i, predictor_value<R, value_order<K>>* val) {
  using V = predictor_value<R, value_order<K>>;
  for (int k = 0; k < M.n_nodes; ++k) {
    const pointwise_node& nd = M.nodes[k];
    V& o = val[M.R + k];
    switch (nd.kind) {
      case pointwise_node_kind::link: {
        const V* in[1] = {&val[nd.input[0]]};
        const double u = in[0]->v;
        o.v = nd.link.inverse(u);
        if constexpr (K >= 1) {
          const double f1 = nd.link.inverse_d1(u);
          chain_rule_d1<R, 1>(in, &f1, o);
          if constexpr (K >= 2) {
            const double f2 = nd.link.inverse_d2(u);
            chain_rule_d2<R, 1>(in, &f1, &f2, o);
            if constexpr (K >= 3) {
              const double f3 = nd.link.inverse_d3(u);
              chain_rule_d3<R, 1>(in, &f1, &f2, &f3, o);
              if constexpr (K >= 4) {
                const double f4 = nd.link.inverse_d4(u);
                chain_rule_d4<R, 1>(in, &f1, &f2, &f3, &f4, o);
              }
            }
          }
        }
        break;
      }
      case pointwise_node_kind::logistic_curve: {
        const V* in[3] = {&val[nd.input[0]], &val[nd.input[1]], &val[nd.input[2]]};
        const double A = in[0]->v, xm = in[1]->v, s = in[2]->v, t = M.data(i, nd.data_column);
        o.v = logistic_curve_value(A, xm, s, t);
        if constexpr (K >= 1) {
          double f1[3];
          logistic_curve_d1(A, xm, s, t, f1);
          chain_rule_d1<R, 3>(in, f1, o);
          if constexpr (K >= 2) {
            double f2[6];
            logistic_curve_d2(A, xm, s, t, f2);
            chain_rule_d2<R, 3>(in, f1, f2, o);
            if constexpr (K >= 3) {
              double f3[10];
              logistic_curve_d3(A, xm, s, t, f3);
              chain_rule_d3<R, 3>(in, f1, f2, f3, o);
              if constexpr (K >= 4) {
                double f4[15];
                logistic_curve_d4(A, xm, s, t, f4);
                chain_rule_d4<R, 3>(in, f1, f2, f3, f4, o);
              }
            }
          }
        }
        break;
      }
    }
  }
}

LAPLACE_HD constexpr int lower_index(int r, int c) { return r * (r + 1) / 2 + c; }

// The walk of the pointwise nodes of unit i for the values alone (the value kernel): plain numbers,
// no derivatives and no supports.
LAPLACE_HD void walk_node_values(const row_model_view& M, int i, double* v) {
  for (int k = 0; k < M.n_nodes; ++k) {
    const pointwise_node& nd = M.nodes[k];
    switch (nd.kind) {
      case pointwise_node_kind::link: v[M.R + k] = nd.link.inverse(v[nd.input[0]]); break;
      case pointwise_node_kind::logistic_curve:
        v[M.R + k] = logistic_curve_value(v[nd.input[0]], v[nd.input[1]], v[nd.input[2]], M.data(i, nd.data_column));
        break;
    }
  }
}

template <class D, int R, row_quantity Q>
struct kernel_sizes {
  static constexpr int P = D::n_params;
  static constexpr int K = Q == row_quantity::value ? 0 : (Q == row_quantity::observed_information ? 2 : 1);
  static constexpr bool want_g = Q != row_quantity::value;
  static constexpr bool want_h = Q == row_quantity::observed_information || Q == row_quantity::expected_information;
  static constexpr int LC = row_local_capacity;
  static constexpr int NH = want_h ? LC * (LC + 1) / 2 : 1;
  static constexpr int NP = R * (R + 1) / 2;
};

}  // namespace detail_rows

/**
 * \ingroup engine
 * \brief Executes the Kokkos row kernel for distribution `D`, \f$R\f$ predictors, and quantity `Q`, adding into `acc`.
 *
 * Launches one work item per unit block over `M.n_blocks` on `row_model_view::exec` using
 * `Kokkos::Experimental::ScatterView`. Evaluates pointwise nodes only up to order 0 (`value`),
 * 1 (`gradient` or `expected_information`), or 2 (`observed_information`). `acc` must have length
 * \f$1\f$ (`value`), \f$1 + N\f$ (`gradient`), or \f$1 + N + \f$ `arrowhead_layout::size()`
 * (`observed_information` or `expected_information`), where \f$N = \f$ `M.n_coordinates()`.
 *
 * \tparam D Distribution type satisfying \ref distrib_core (`D::n_params <= row_param_capacity`).
 * \tparam R Number of linear predictors (\f$1 \le R \le 4\f$).
 * \tparam Q Quantity to assemble (`row_quantity`).
 * \param M Built device model view.
 * \param x Device view of coordinates of length \f$N\f$.
 * \param acc Device accumulator view initialized by the caller; contributions are added into it.
 *
 * \see assemble_row_model(), row_quantity, arrowhead_layout, \ref engine
 */
template <class D, int R, row_quantity Q>
void assemble_rows(const row_model_view& M, const Kokkos::View<const double*, row_model_view::mem>& x,
                   const row_model_view::dview& acc) {
  using exec = row_model_view::exec;
  using SV = Kokkos::Experimental::ScatterView<double*, typename row_model_view::dview::array_layout, exec>;
  // the sizes are members of a type, not local constants: nvcc refuses a local constant that an
  // extended lambda first reads inside if constexpr
  using S = detail_rows::kernel_sizes<D, R, Q>;
  static_assert(S::P <= row_param_capacity, "the distrib has more parameters than row_param_capacity");
  const int N = M.n_coordinates();
  const arrowhead_layout lay{M.n_global, M.n_groups, M.group_width};
  SV sv(acc);
  Kokkos::parallel_for(
      "laplace::engine::assemble_rows", Kokkos::RangePolicy<exec>(0, M.n_blocks), KOKKOS_LAMBDA(const int b) {
        auto a = sv.access();
        (void)N, (void)lay;  // captured here: nvcc refuses a first capture inside if constexpr
        const int g = M.block_group(b), L = M.n_local();
        double xl[S::LC];
        for (int s = 0; s < L; ++s) xl[s] = x(M.global_index(s, g));
        double f = 0.0, gl[S::want_g ? S::LC : 1], Hl[S::NH];
        if constexpr (S::want_g)
          for (int s = 0; s < L; ++s) gl[s] = 0.0;
        if constexpr (S::want_h)
          for (int s = 0; s < L * (L + 1) / 2; ++s) Hl[s] = 0.0;
        for (int i = M.block_begin(b); i < M.block_end(b); ++i) {
          double eta[R];
          for (int r = 0; r < R; ++r) eta[r] = M.has_offset ? M.offset(i, r) : 0.0;
          for (int e = 0; e < M.width; ++e) {
            const int s = M.design_slot(i, e);
            if (s >= 0) eta[M.entry_predictor(e)] += M.design_value(i, e) * xl[s];
          }
          const double y = M.y(i), wi = M.has_weight ? M.weight(i) : 1.0;
          double th[S::P];
          if constexpr (!S::want_g) {
            double v[row_value_capacity];
            for (int r = 0; r < R; ++r) v[r] = eta[r];
            detail_rows::walk_node_values(M, i, v);
            for (int j = 0; j < S::P; ++j) th[j] = v[M.param_value[j]];
            const D d = detail_rows::make_distrib<D>(th, std::make_index_sequence<S::P>{});
            f -= wi * detail_rows::log_density(d, y);
          } else {
            predictor_value<R> val[row_value_capacity];
            for (int r = 0; r < R; ++r) val[r] = linear_predictor_value<R>(r, eta[r]);
            detail_rows::walk_nodes<R, S::K>(M, i, val);
            const predictor_value<R>* pin[S::P];
            for (int j = 0; j < S::P; ++j) {
              pin[j] = &val[M.param_value[j]];
              th[j] = pin[j]->v;
            }
            const D d = detail_rows::make_distrib<D>(th, std::make_index_sequence<S::P>{});
            f -= wi * detail_rows::log_density(d, y);
            const auto l1 = detail_rows::log_density_d1(d, y);
            predictor_value<R> out;
            chain_rule_d1<R, S::P>(pin, l1.v, out);
            for (int e = 0; e < M.width; ++e) {
              const int s = M.design_slot(i, e);
              if (s >= 0) gl[s] -= wi * M.design_value(i, e) * out.d1[M.entry_predictor(e)];
            }
            if constexpr (S::want_h) {
              double info[S::NP];  // the information of the unit in eta, packed on sorted pairs
              if constexpr (Q == row_quantity::observed_information) {
                const auto l2 = detail_rows::log_density_d2(d, y);
                chain_rule_d2<R, S::P>(pin, l1.v, l2.v, out);
                for (int q = 0; q < S::NP; ++q) info[q] = -wi * out.d2[q];
              } else {
                const auto I = d.expected_info();
                for (int ra = 0; ra < R; ++ra)
                  for (int rb = ra; rb < R; ++rb) {
                    double t = 0.0;
                    for (int j = 0; j < S::P; ++j)
                      for (int k = 0; k < S::P; ++k)
                        t += I[j <= k ? packed_pair(j, k, S::P) : packed_pair(k, j, S::P)] * pin[j]->d1[ra] * pin[k]->d1[rb];
                    info[packed_pair(ra, rb, R)] = wi * t;
                  }
              }
              for (int e1 = 0; e1 < M.width; ++e1) {
                const int s1 = M.design_slot(i, e1);
                if (s1 < 0) continue;
                const double v1 = M.design_value(i, e1);
                const int p1 = M.entry_predictor(e1);
                for (int e2 = 0; e2 <= e1; ++e2) {
                  const int s2 = M.design_slot(i, e2);
                  if (s2 < 0) continue;
                  const int p2 = M.entry_predictor(e2);
                  double c = v1 * M.design_value(i, e2) * info[p1 <= p2 ? packed_pair(p1, p2, R) : packed_pair(p2, p1, R)];
                  if (e1 != e2 && s1 == s2) c *= 2.0;  // both orders of the pair fall on the diagonal
                  Hl[s1 >= s2 ? detail_rows::lower_index(s1, s2) : detail_rows::lower_index(s2, s1)] += c;
                }
              }
            }
          }
        }
        a(0) += f;
        if constexpr (S::want_g)
          for (int s = 0; s < L; ++s) a(1 + M.global_index(s, g)) += gl[s];
        if constexpr (S::want_h) {
          const int G0 = M.n_global;
          for (int r = 0; r < L; ++r)
            for (int c = 0; c <= r; ++c) {
              const double h = Hl[detail_rows::lower_index(r, c)];
              const int pos = r < G0   ? lay.global(r, c)
                              : c < G0 ? lay.cross(g, r - G0, c)
                                       : lay.group(g, r - G0, c - G0);
              a(1 + N + pos) += h;
            }
        }
      });
  Kokkos::Experimental::contribute(acc, sv);
}

namespace detail_rows {

template <class D, int R>
void run_quantity(const row_model& m, const Kokkos::View<const double*, row_model_view::mem>& x, row_quantity q,
                  const row_model_view::dview& acc) {
  switch (q) {
    case row_quantity::value: assemble_rows<D, R, row_quantity::value>(m.view, x, acc); break;
    case row_quantity::gradient: assemble_rows<D, R, row_quantity::gradient>(m.view, x, acc); break;
    case row_quantity::observed_information:
      assemble_rows<D, R, row_quantity::observed_information>(m.view, x, acc);
      break;
    case row_quantity::expected_information:
      assemble_rows<D, R, row_quantity::expected_information>(m.view, x, acc);
      break;
  }
}

template <class D>
void run_distrib(const row_model& m, const Kokkos::View<const double*, row_model_view::mem>& x, row_quantity q,
                 const row_model_view::dview& acc) {
  if (m.view.P != D::n_params) throw std::invalid_argument("assemble_row_model: the model was built for another distrib");
  switch (m.view.R) {
    case 1: run_quantity<D, 1>(m, x, q, acc); break;
    case 2: run_quantity<D, 2>(m, x, q, acc); break;
    case 3: run_quantity<D, 3>(m, x, q, acc); break;
    case 4: run_quantity<D, 4>(m, x, q, acc); break;
    default: throw std::invalid_argument("assemble_row_model: R must be 1 to 4");
  }
}

}  // namespace detail_rows

/**
 * \ingroup engine
 * \brief Runs the row kernel of quantity `q` at device coordinates `x` and copies the result to the host.
 *
 * Switches on `m.distrib` within `List` and on `m.view.R` in \f$\{1, 2, 3, 4\}\f$, allocates a zeroed
 * device accumulator, launches assemble_rows(), and copies the assembled value, gradient, and
 * arrowhead information into `out`. Throws `std::invalid_argument` if `m.distrib` is out of range,
 * `m.view.P != D::n_params`, or `m.view.R` is outside \f$[1, 4]\f$. Runs on the host.
 *
 * \tparam List `distrib_list` of distribution types that `m.distrib` may index.
 * \param m Built model from build_row_model().
 * \param x Device view of coordinates of length `m.n_coordinates()`.
 * \param q Quantity to evaluate.
 * \param out Host row_assembly populated with the kernel output.
 *
 * \see assemble_rows(), distrib_list, row_assembly, \ref engine
 */
template <class List>
void assemble_row_model(const row_model& m, const Kokkos::View<const double*, row_model_view::mem>& x, row_quantity q,
                        row_assembly& out) {
  const int N = m.n_coordinates();
  const arrowhead_layout lay{m.view.n_global, m.view.n_groups, m.view.group_width};
  const bool g = q != row_quantity::value;
  const bool h = q == row_quantity::observed_information || q == row_quantity::expected_information;
  const int size = 1 + (g ? N : 0) + (h ? lay.size() : 0);
  row_model_view::dview acc("laplace::engine::row_assembly", size);
  bool found = false;
  [&]<class... Ds>(distrib_list<Ds...>) {
    int k = 0;
    ((k++ == m.distrib ? (detail_rows::run_distrib<Ds>(m, x, q, acc), found = true) : false), ...);
  }(List{});
  if (!found) throw std::invalid_argument("assemble_row_model: the distrib index is outside the list");
  auto hacc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc);
  out.value = hacc(0);
  out.gradient.assign(g ? N : 0, 0.0);
  for (int j = 0; g && j < N; ++j) out.gradient[j] = hacc(1 + j);
  out.information.assign(h ? lay.size() : 0, 0.0);
  for (int j = 0; h && j < lay.size(); ++j) out.information[j] = hacc(1 + N + j);
}

}  // namespace laplace::engine
