#pragma once

// A model as data for the row kernels (laplace::engine; plan, sections 7.8
// and 7.9): the response, the weights, the stored design rows of the linear
// nodes (row_design.hpp), the nonlinear pointwise nodes (pointwise_node.hpp)
// and which value feeds each parameter of the distrib.
//
// Values of a unit are numbered: the R linear predictors 0, ..., R - 1, then
// the output of node k as value R + k. The nodes are listed in an order in
// which every input is an earlier value, so one walk evaluates them.
//
// Coordinates: x = (the n_global global ones, then n_groups blocks of
// group_width), the block of group g at n_global + g group_width. A unit
// reads the global coordinates and those of its own group only, so the
// Hessian is an arrowhead: a dense global block, one group_width square block
// per group and the cross blocks between a group and the globals.
//
// build_row_model sorts the units by group (keeping the order within a
// group), cuts them into blocks of at most rows_per_block units of one group,
// rewrites every coefficient as a local slot of the block (the global
// coordinate itself, or n_global + j for coefficient j of the unit's group)
// and copies everything to the device. The support of a value (the
// predictors it depends on) is set by the chain rule (predictor_tensor.hpp). A block accumulates in its local slots and
// adds its sums once.

#include <laplace/core/config.hpp>
#include <laplace/engine/pointwise_node.hpp>
#include <laplace/engine/row_design.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Largest number of linear predictors of a unit; row kernels are instantiated for \f$R = 1, \dots, 4\f$.
 * \see row_model_spec, \ref engine
 */
inline constexpr int row_predictor_capacity = 4;
/**
 * \ingroup engine
 * \brief Largest number of pointwise nodes allowed in a model.
 * \see row_model_spec, \ref engine
 */
inline constexpr int row_node_capacity = 8;
/**
 * \ingroup engine
 * \brief Largest number of values of a unit (`row_predictor_capacity + row_node_capacity`).
 * \see row_model_spec, \ref engine
 */
inline constexpr int row_value_capacity = row_predictor_capacity + row_node_capacity;
/**
 * \ingroup engine
 * \brief Largest number of natural parameters of a distribution supported by the row kernels.
 * \see row_model_spec, \ref engine
 */
inline constexpr int row_param_capacity = 6;
/**
 * \ingroup engine
 * \brief Largest number of local coordinate slots of a block (`n_global + group_width`).
 * \see row_model_spec, build_row_model(), \ref engine
 */
// ponytail: the local Hessian of a block lives in the thread's registers or stack (528 numbers at 32
// slots); more slots need team scratch memory.
inline constexpr int row_local_capacity = 32;

/**
 * \ingroup engine
 * \brief Host specification of a model for build_row_model().
 *
 * Collects the response `y`, optional observation weights and offsets, global and group coordinate
 * layout, fixed-width design blocks, data columns, topologically ordered `pointwise_node` entries,
 * parameter value indices `param_value`, and the distribution index `distrib` into a `distrib_list`.
 *
 * \see build_row_model(), row_model, design_group_effect(), \ref engine
 */
struct row_model_spec {
  int n_predictors = 1;                         ///< \f$R\f$
  std::vector<double> y;                        ///< response, \f$n\f$
  std::vector<double> weight;                   ///< \f$n\f$, or empty for 1
  std::vector<std::vector<double>> offset;      ///< per predictor, \f$n\f$ each, or empty for 0
  int n_global = 0;                             ///< global coordinates
  std::vector<int> group;                       ///< group of each unit in \f$[0, G)\f$, or empty
  int n_groups = 0, group_width = 0;            ///< \f$G\f$ and the coordinates of one group
  std::vector<design_entries> design;
  std::vector<std::vector<double>> data_column; ///< columns read by the nodes, \f$n\f$ each
  std::vector<pointwise_node> nodes;
  std::vector<int> param_value;                 ///< the value that feeds each parameter of the distrib
  int distrib = 0;                              ///< index of the distrib in the dispatch list (row_kernels.hpp)
};

/**
 * \ingroup engine
 * \brief Builds a width-1 design block for a group-level coordinate.
 *
 * Unit \f$i\f$ contributes `v[i]` to linear predictor `predictor` at coordinate
 * `m.n_global + m.group[i] * m.group_width + j`. Requires `m.group` to be populated. Runs on the host.
 *
 * \param m Model specification carrying `n_global`, `group`, and `group_width`.
 * \param predictor Index of the linear predictor \f$r\f$ fed by this group effect.
 * \param v Design values of length \f$n\f$.
 * \param j Within-group coordinate index, \f$0 \le j < d_{\mathrm{grp}}\f$ (`group_width`).
 * \return A width-1 design_entries block referencing each unit's own group coordinate.
 *
 * \see design_entries, row_model_spec, \ref engine
 */
inline design_entries design_group_effect(const row_model_spec& m, int predictor, const std::vector<double>& v,
                                          int j) {
  design_entries d;
  d.predictor = predictor;
  d.value = v;
  d.coefficient.resize(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) d.coefficient[i] = m.n_global + m.group[i] * m.group_width + j;
  return d;
}

/**
 * \ingroup engine
 * \brief Device-space representation of a built model, copied by value into row kernels.
 *
 * Holds Kokkos Views in `Kokkos::DefaultExecutionSpace::memory_space` for the group-sorted units,
 * block boundaries, local-slot design indices, and copied `pointwise_node` and `param_value` arrays.
 *
 * \see row_model, build_row_model(), assemble_rows(), \ref engine
 */
struct row_model_view {
  using exec = Kokkos::DefaultExecutionSpace;
  using mem = exec::memory_space;
  using dview = Kokkos::View<double*, mem>;
  using iview = Kokkos::View<int*, mem>;
  using d2view = Kokkos::View<double**, Kokkos::LayoutRight, mem>;
  using i2view = Kokkos::View<int**, Kokkos::LayoutRight, mem>;

  int n = 0, R = 1, P = 0, n_nodes = 0, width = 0;
  int n_global = 0, n_groups = 0, group_width = 0, n_blocks = 0;
  bool has_weight = false, has_offset = false;
  dview y, weight;
  d2view offset;        ///< \f$n \times R\f$
  d2view design_value;  ///< \f$n \times W\f$
  i2view design_slot;   ///< \f$n \times W\f$: local slot, \f$-1\f$ for none
  iview entry_predictor;///< \f$W\f$
  d2view data;          ///< \f$n \times C\f$
  iview block_begin, block_end, block_group;
  pointwise_node nodes[row_node_capacity];
  int param_value[row_param_capacity] = {};

  /**
   * \ingroup engine
   * \brief Number of local coordinate slots per block (`n_global + group_width`).
   * \return Local slot count \f$L\f$.
   */
  LAPLACE_HD int n_local() const { return n_global + group_width; }
  /**
   * \ingroup engine
   * \brief Maps local slot \f$s\f$ in a block of group \f$g\f$ to its global coordinate index.
   * \param s Local slot index in \f$[0, L)\f$, where \f$L = \f$ `n_local()`.
   * \param g Group index in \f$[0, G)\f$, where \f$G = \f$ `n_groups` (ignored when \f$s < \f$ `n_global`).
   * \return Global coordinate index in \f$[0, N)\f$, where \f$N = \f$ `n_coordinates()`.
   */
  LAPLACE_HD int global_index(int s, int g) const {
    return s < n_global ? s : n_global + g * group_width + (s - n_global);
  }
  /**
   * \ingroup engine
   * \brief Total number of model coordinates (`n_global + n_groups * group_width`).
   * \return Coordinate count \f$N\f$.
   */
  LAPLACE_HD int n_coordinates() const { return n_global + n_groups * group_width; }
};

/**
 * \ingroup engine
 * \brief Built model combining the device view, unit permutation, and distribution index.
 *
 * Returned by build_row_model() and consumed by assemble_row_model() and row_model_objective.
 *
 * \see row_model_view, build_row_model(), \ref engine
 */
struct row_model {
  row_model_view view;
  std::vector<int> order;  ///< unit of the original data at each sorted position
  int distrib = 0;
  /**
   * \ingroup engine
   * \brief Total number of model coordinates (`view.n_coordinates()`).
   * \return Coordinate count \f$N\f$.
   */
  int n_coordinates() const { return view.n_coordinates(); }
};

namespace detail_row_model {

[[noreturn]] inline void refuse(const std::string& what) { throw std::invalid_argument("build_row_model: " + what); }

}  // namespace detail_row_model

/**
 * \ingroup engine
 * \brief Validates a model specification, sorts and blocks units by group, and copies data to the device.
 *
 * Checks predictor, node, parameter, and local-slot capacities, topological node ordering, array lengths,
 * and group coordinate bounds, throwing `std::invalid_argument` if any check fails. Units are stably
 * sorted by group, partitioned into contiguous single-group blocks of at most `rows_per_block` units,
 * mapped to local coordinate slots, and mirrored to `Kokkos::DefaultExecutionSpace::memory_space`.
 * Runs on the host.
 *
 * \param s Host model specification.
 * \param P Number of parameters of the distribution (`D::n_params`, \f$1 \le P \le 6\f$).
 * \param rows_per_block Maximum number of units per block (must be positive; default 16).
 * \return Built row_model ready for assemble_row_model() and row_model_objective.
 *
 * \see row_model_spec, row_model, row_model_view, \ref engine
 */
inline row_model build_row_model(const row_model_spec& s, int P, int rows_per_block = 16) {
  using detail_row_model::refuse;
  using V = row_model_view;
  const int n = static_cast<int>(s.y.size()), R = s.n_predictors, nn = static_cast<int>(s.nodes.size());
  if (R < 1 || R > row_predictor_capacity) refuse("the number of linear predictors must be 1 to 4");
  if (nn > row_node_capacity) refuse("more than " + std::to_string(row_node_capacity) + " nodes");
  if (P < 1 || P > row_param_capacity || static_cast<int>(s.param_value.size()) != P)
    refuse("param_value must name one value for each of the distrib's parameters");
  if (rows_per_block < 1) refuse("rows_per_block must be positive");
  const bool grouped = !s.group.empty();
  if (grouped && (static_cast<int>(s.group.size()) != n || s.n_groups < 1 || s.group_width < 1))
    refuse("a grouping needs a group for every unit, n_groups >= 1 and group_width >= 1");
  if (!grouped && (s.n_groups != 0 || s.group_width != 0)) refuse("group coordinates without a grouping");
  if (s.n_global + s.group_width > row_local_capacity)
    refuse("more than " + std::to_string(row_local_capacity) + " local slots (global coordinates plus one group's)");
  if (!s.weight.empty() && static_cast<int>(s.weight.size()) != n) refuse("weight must have n entries");
  if (!s.offset.empty() && static_cast<int>(s.offset.size()) != R) refuse("offset must have one column per predictor");
  for (const auto& o : s.offset)
    if (!o.empty() && static_cast<int>(o.size()) != n) refuse("an offset column must have n entries");
  for (const auto& c : s.data_column)
    if (static_cast<int>(c.size()) != n) refuse("a data column must have n entries");
  const int n_values = R + nn;
  for (int k = 0; k < nn; ++k) {
    const auto& nd = s.nodes[k];
    for (int p = 0; p < nd.n_inputs(); ++p)
      if (nd.input[p] < 0 || nd.input[p] >= R + k) refuse("node " + std::to_string(k) + " reads a value that is not earlier");
    if (nd.kind == pointwise_node_kind::logistic_curve &&
        (nd.data_column < 0 || nd.data_column >= static_cast<int>(s.data_column.size())))
      refuse("node " + std::to_string(k) + " reads no data column");
  }
  for (int v : s.param_value)
    if (v < 0 || v >= n_values) refuse("param_value names a value that does not exist");

  // the units sorted by group, and the blocks
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  if (grouped) {
    for (int i = 0; i < n; ++i)
      if (s.group[i] < 0 || s.group[i] >= s.n_groups) refuse("a group outside [0, n_groups)");
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s.group[a] < s.group[b]; });
  }
  std::vector<int> bb, be, bg;
  for (int i = 0; i < n;) {
    const int g = grouped ? s.group[order[i]] : 0;
    int j = i;
    while (j < n && j - i < rows_per_block && (!grouped || s.group[order[j]] == g)) ++j;
    bb.push_back(i), be.push_back(j), bg.push_back(g);
    i = j;
  }

  row_model m;
  m.order = order;
  m.distrib = s.distrib;
  V& v = m.view;
  v.n = n, v.R = R, v.P = P, v.n_nodes = nn;
  v.n_global = s.n_global, v.n_groups = s.n_groups, v.group_width = s.group_width;
  v.n_blocks = static_cast<int>(bb.size());
  int W = 0;
  for (const auto& d : s.design) {
    if (d.predictor < 0 || d.predictor >= R) refuse("a design block feeds a predictor that does not exist");
    if (d.width < 1 || d.value.size() != static_cast<std::size_t>(n) * d.width || d.coefficient.size() != d.value.size())
      refuse("a design block must have n x width values and coefficients");
    W += d.width;
  }
  v.width = W;

  auto y = Kokkos::View<double*, Kokkos::HostSpace>("y", n);
  auto w = Kokkos::View<double*, Kokkos::HostSpace>("weight", s.weight.empty() ? 0 : n);
  auto off = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("offset", s.offset.empty() ? 0 : n, R);
  auto dv = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("design_value", n, W);
  auto ds = Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>("design_slot", n, W);
  auto ep = Kokkos::View<int*, Kokkos::HostSpace>("entry_predictor", W);
  const int C = static_cast<int>(s.data_column.size());
  auto dc = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>("data", n, C);
  for (int i = 0; i < n; ++i) {
    const int u = order[i], g = grouped ? s.group[u] : 0;
    y(i) = s.y[u];
    if (!s.weight.empty()) w(i) = s.weight[u];
    if (!s.offset.empty())
      for (int r = 0; r < R; ++r) off(i, r) = s.offset[r].empty() ? 0.0 : s.offset[r][u];
    for (int c = 0; c < C; ++c) dc(i, c) = s.data_column[c][u];
    int e = 0;
    for (const auto& d : s.design)
      for (int j = 0; j < d.width; ++j, ++e) {
        const int c = d.coefficient[static_cast<std::size_t>(u) * d.width + j];
        int slot = -1;
        if (c >= 0 && c < s.n_global) slot = c;
        else if (c >= 0) {
          const int first = s.n_global + g * s.group_width;
          if (!grouped || c < first || c >= first + s.group_width)
            refuse("unit " + std::to_string(u) + " reads coordinate " + std::to_string(c) +
                   ", which is neither global nor of its group");
          slot = s.n_global + (c - first);
        }
        dv(i, e) = d.value[static_cast<std::size_t>(u) * d.width + j];
        ds(i, e) = slot;
        if (i == 0) ep(e) = d.predictor;
      }
  }
  v.has_weight = !s.weight.empty();
  v.has_offset = !s.offset.empty();
  v.y = Kokkos::create_mirror_view_and_copy(V::mem(), y);
  v.weight = Kokkos::create_mirror_view_and_copy(V::mem(), w);
  v.offset = Kokkos::create_mirror_view_and_copy(V::mem(), off);
  v.design_value = Kokkos::create_mirror_view_and_copy(V::mem(), dv);
  v.design_slot = Kokkos::create_mirror_view_and_copy(V::mem(), ds);
  v.entry_predictor = Kokkos::create_mirror_view_and_copy(V::mem(), ep);
  v.data = Kokkos::create_mirror_view_and_copy(V::mem(), dc);
  auto to_dev = [](const std::vector<int>& a, const char* name) {
    Kokkos::View<int*, Kokkos::HostSpace> h(name, a.size());
    for (std::size_t k = 0; k < a.size(); ++k) h(k) = a[k];
    return Kokkos::create_mirror_view_and_copy(V::mem(), h);
  };
  v.block_begin = to_dev(bb, "block_begin");
  v.block_end = to_dev(be, "block_end");
  v.block_group = to_dev(bg, "block_group");

  for (int k = 0; k < nn; ++k) v.nodes[k] = s.nodes[k];
  for (int j = 0; j < P; ++j) v.param_value[j] = s.param_value[j];
  return m;
}

}  // namespace laplace::engine
