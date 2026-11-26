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
// Coordinates: x = (the n_global global ones, then for each grouping k its
// n_groups_k blocks of width_k), the first grouping given by group, n_groups
// and group_width and the further ones, crossed with it, by crossed. A unit
// reads the global coordinates and, in each grouping, those of its own group
// only. With one grouping the Hessian is an arrowhead: a dense global block,
// one square block per group and the cross blocks between a group and the
// globals. With several, the groups of two groupings that share a unit are
// coupled too: one block of width_l x width_k for each such pair of groups.
//
// The information is stored in that layout (information_position): the
// global block by rows of its lower triangle; for each grouping, group after
// group, the square block by rows of its lower triangle and the cross block
// with the globals by rows; then for each pair of groupings (l < k) the
// blocks of the pairs of groups met, by rows (the coordinate of l first).
// With one grouping this is the arrowhead layout of earlier versions.
//
// build_row_model sorts the units by their tuple of groups (keeping the order
// within a tuple), cuts them into blocks of at most rows_per_block units of
// one tuple, rewrites every coefficient as a local slot of the block (the
// global coordinate itself, then the slots of each grouping in order) and
// copies everything to the device. The support of a value (the predictors it
// depends on) is set by the chain rule (predictor_tensor.hpp). A block
// accumulates in its local slots and adds its sums once.

#include <laplace/core/config.hpp>
#include <laplace/engine/pointwise_node.hpp>
#include <laplace/engine/row_design.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>
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
 * \brief Largest number of groupings of a model (the first and those crossed with it).
 * \see row_model_spec, row_grouping, \ref engine
 */
inline constexpr int row_grouping_capacity = 4;
/**
 * \ingroup engine
 * \brief Largest number of local coordinate slots of a block (`n_global` plus the width of every grouping).
 * \see row_model_spec, build_row_model(), \ref engine
 */
// ponytail: the local Hessian of a block lives in the thread's registers or stack (528 numbers at 32
// slots); more slots need team scratch memory.
inline constexpr int row_local_capacity = 32;

/**
 * \ingroup engine
 * \brief A grouping crossed with the first one of a row_model_spec: the group of each unit, the number of groups and their width.
 * \see row_model_spec, design_crossed_effect(), \ref engine
 */
struct row_grouping {
  std::vector<int> group;  ///< group of each unit in \f$[0, G)\f$
  int n_groups = 0;        ///< \f$G\f$
  int width = 0;           ///< coordinates of one group
};

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
  std::vector<row_grouping> crossed;            ///< further groupings, their coordinates after the first's
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
 * \brief The first coordinate of the groups of grouping `k` (0 the first grouping, \f$k \ge 1\f$ `m.crossed[k - 1]`).
 * \param m Model specification.
 * \param k Grouping.
 * \return `n_global` plus the coordinates of the groupings before `k`.
 * \see design_crossed_effect(), \ref engine
 */
inline int grouping_first_coordinate(const row_model_spec& m, int k) {
  int c = m.n_global + (k > 0 ? m.n_groups * m.group_width : 0);
  for (int l = 1; l < k; ++l) c += m.crossed[l - 1].n_groups * m.crossed[l - 1].width;
  return c;
}

/**
 * \ingroup engine
 * \brief Builds a width-1 design block for a coordinate of a crossed grouping.
 *
 * Unit \f$i\f$ contributes `v[i]` to linear predictor `predictor` at coordinate `j` of its group in
 * grouping `k` (\f$k \ge 1\f$, `m.crossed[k - 1]`). Runs on the host.
 *
 * \param m Model specification.
 * \param k Grouping, \f$1 \le k \le\f$ `m.crossed.size()`.
 * \param predictor Index of the linear predictor fed.
 * \param v Design values of length \f$n\f$.
 * \param j Within-group coordinate index.
 * \return A width-1 design_entries block.
 * \see design_group_effect(), row_grouping, \ref engine
 */
inline design_entries design_crossed_effect(const row_model_spec& m, int k, int predictor, const std::vector<double>& v,
                                            int j) {
  const auto& gr = m.crossed.at(k - 1);
  const int first = grouping_first_coordinate(m, k);
  design_entries d;
  d.predictor = predictor;
  d.value = v;
  d.coefficient.resize(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) d.coefficient[i] = first + gr.group[i] * gr.width + j;
  return d;
}

/**
 * \ingroup engine
 * \brief Device-space representation of a built model, copied by value into row kernels.
 *
 * Holds Kokkos Views in `Kokkos::DefaultExecutionSpace::memory_space` for the sorted units, block
 * boundaries and groups, local-slot design indices, and copied `pointwise_node` and `param_value`
 * arrays, with the layout of the information (information_position()).
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
  /// The pairs \f$(l, k)\f$, \f$l < k\f$, of groupings.
  static constexpr int pair_capacity = row_grouping_capacity * (row_grouping_capacity - 1) / 2;

  int n = 0, R = 1, P = 0, n_nodes = 0, width = 0;
  int n_global = 0, n_groups = 0, group_width = 0, n_blocks = 0;  ///< n_groups, group_width: the first grouping's
  int n_groupings = 0;                                           ///< 0 without groups
  int grouping_groups[row_grouping_capacity] = {};               ///< groups of each grouping
  int grouping_width[row_grouping_capacity] = {};                ///< coordinates of one group
  int grouping_coordinate[row_grouping_capacity] = {};           ///< its first coordinate
  int grouping_slot[row_grouping_capacity] = {};                 ///< its first local slot
  int grouping_information[row_grouping_capacity] = {};          ///< its first position in the information
  int pair_information[pair_capacity] = {};                      ///< first position of each pair of groupings
  int information_size = 0;                                       ///< total number of entries in the information layout
  bool has_weight = false, has_offset = false;
  dview y, weight;
  d2view offset;        ///< \f$n \times R\f$
  d2view design_value;  ///< \f$n \times W\f$
  i2view design_slot;   ///< \f$n \times W\f$: local slot, \f$-1\f$ for none
  iview entry_predictor;///< \f$W\f$
  d2view data;          ///< \f$n \times C\f$
  iview block_begin, block_end;
  i2view block_groups;  ///< blocks \f$\times\f$ groupings: the group of the block's units in each grouping
  i2view block_pairs;   ///< blocks \f$\times\f$ pairs: the index of its pair of groups in each pair of groupings
  pointwise_node nodes[row_node_capacity];
  int param_value[row_param_capacity] = {};

  /**
   * \ingroup engine
   * \brief The index of the pair of groupings \f$(l, k)\f$, \f$l < k\f$.
   * \param l First grouping.
   * \param k Second grouping.
   * \return Index in \f$[0,\f$ pair_capacity\f$)\f$.
   */
  LAPLACE_HD static int pair_index(int l, int k) { return k * (k - 1) / 2 + l; }
  /**
   * \ingroup engine
   * \brief Number of local coordinate slots per block (`n_global` plus every grouping's width).
   * \return Local slot count \f$L\f$.
   */
  LAPLACE_HD int n_local() const {
    int L = n_global;
    for (int k = 0; k < n_groupings; ++k) L += grouping_width[k];
    return L;
  }
  /**
   * \ingroup engine
   * \brief The grouping of a local slot.
   * \param s Local slot, at least `n_global`.
   * \return The grouping \f$k\f$ whose slots hold `s`.
   */
  LAPLACE_HD int slot_grouping(int s) const {
    int k = 0;
    while (k + 1 < n_groupings && s >= grouping_slot[k + 1]) ++k;
    return k;
  }
  /**
   * \ingroup engine
   * \brief Maps local slot \f$s\f$ of block \f$b\f$ to its global coordinate index.
   * \param s Local slot index in \f$[0, L)\f$, where \f$L = \f$ `n_local()`.
   * \param b Block.
   * \return Global coordinate index in \f$[0, N)\f$, where \f$N = \f$ `n_coordinates()`.
   */
  LAPLACE_HD int global_index(int s, int b) const {
    if (s < n_global) return s;
    const int k = slot_grouping(s);
    return grouping_coordinate[k] + block_groups(b, k) * grouping_width[k] + (s - grouping_slot[k]);
  }
  /**
   * \ingroup engine
   * \brief The position in the information of the entry \f$(r, c)\f$, \f$r \ge c\f$, of the local slots of block \f$b\f$.
   * \param b Block.
   * \param r Local slot of the row.
   * \param c Local slot of the column, \f$c \le r\f$.
   * \return Position in \f$[0,\f$ `information_size`\f$)\f$.
   */
  LAPLACE_HD int information_position(int b, int r, int c) const {
    if (r < n_global) return r * (r + 1) / 2 + c;
    const int k = slot_grouping(r), j = r - grouping_slot[k], w = grouping_width[k];
    const int start = grouping_information[k] + block_groups(b, k) * (w * (w + 1) / 2 + w * n_global);
    if (c < n_global) return start + w * (w + 1) / 2 + j * n_global + c;
    const int l = slot_grouping(c), jl = c - grouping_slot[l];
    if (l == k) return start + j * (j + 1) / 2 + jl;
    const int p = pair_index(l, k);
    return pair_information[p] + block_pairs(b, p) * grouping_width[l] * w + jl * w + j;
  }
  /**
   * \ingroup engine
   * \brief Total number of model coordinates.
   * \return Coordinate count \f$N\f$.
   */
  LAPLACE_HD int n_coordinates() const {
    int N = n_global;
    for (int k = 0; k < n_groupings; ++k) N += grouping_groups[k] * grouping_width[k];
    return N;
  }
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
  std::vector<int> information_row;     ///< the coordinate of the row of each position of the information
  std::vector<int> information_column;  ///< and of its column (row \f$\ge\f$ column)
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
 * sorted by their tuple of groups (one per grouping), partitioned into contiguous blocks of one tuple
 * of at most `rows_per_block` units, mapped to local coordinate slots, and mirrored to
 * `Kokkos::DefaultExecutionSpace::memory_space`. The layout of the information and the coordinates of
 * each of its positions (`row_model::information_row`, `information_column`) are set here. Runs on the host.
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
  if (!grouped && !s.crossed.empty()) refuse("a crossed grouping without a first grouping");
  // the groupings: the first, then the crossed ones
  std::vector<row_grouping> gr;
  if (grouped) gr.push_back({s.group, s.n_groups, s.group_width});
  for (const auto& c : s.crossed) {
    if (static_cast<int>(c.group.size()) != n || c.n_groups < 1 || c.width < 1)
      refuse("a crossed grouping needs a group for every unit, n_groups >= 1 and width >= 1");
    gr.push_back(c);
  }
  const int K = static_cast<int>(gr.size());
  if (K > row_grouping_capacity) refuse("more than " + std::to_string(row_grouping_capacity) + " groupings");
  int n_local = s.n_global;
  for (const auto& c : gr) n_local += c.width;
  if (n_local > row_local_capacity)
    refuse("more than " + std::to_string(row_local_capacity) + " local slots (global coordinates plus one group of each grouping)");
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

  // the units sorted by their tuple of groups, and the blocks of one tuple
  for (const auto& c : gr)
    for (int i = 0; i < n; ++i)
      if (c.group[i] < 0 || c.group[i] >= c.n_groups) refuse("a group outside [0, n_groups)");
  auto same_tuple = [&](int a, int b) {
    for (const auto& c : gr)
      if (c.group[a] != c.group[b]) return false;
    return true;
  };
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    for (const auto& c : gr)
      if (c.group[a] != c.group[b]) return c.group[a] < c.group[b];
    return false;
  });
  std::vector<int> bb, be;
  for (int i = 0; i < n;) {
    int j = i;
    while (j < n && j - i < rows_per_block && same_tuple(order[i], order[j])) ++j;
    bb.push_back(i), be.push_back(j);
    i = j;
  }
  const int nb = static_cast<int>(bb.size());

  row_model m;
  m.order = order;
  m.distrib = s.distrib;
  V& v = m.view;
  v.n = n, v.R = R, v.P = P, v.n_nodes = nn;
  v.n_global = s.n_global, v.n_groups = s.n_groups, v.group_width = s.group_width;
  v.n_blocks = nb;
  // the groupings, their slots, coordinates and regions of the information
  v.n_groupings = K;
  {
    int coord = s.n_global, slot = s.n_global, info = s.n_global * (s.n_global + 1) / 2;
    for (int k = 0; k < K; ++k) {
      const int w = gr[k].width;
      v.grouping_groups[k] = gr[k].n_groups, v.grouping_width[k] = w;
      v.grouping_coordinate[k] = coord, v.grouping_slot[k] = slot, v.grouping_information[k] = info;
      coord += gr[k].n_groups * w, slot += w, info += gr[k].n_groups * (w * (w + 1) / 2 + w * s.n_global);
    }
    v.information_size = info;
  }
  // the pairs of groups met by a block, for each pair of groupings
  const int npair = K * (K - 1) / 2;
  auto bgr = Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>("block_groups", nb, std::max(K, 1));
  auto bpr = Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>("block_pairs", nb, std::max(npair, 1));
  std::vector<std::vector<std::pair<int, int>>> pairs(V::pair_capacity);
  {
    std::vector<std::map<std::pair<int, int>, int>> seen(V::pair_capacity);
    for (int b = 0; b < nb; ++b) {
      const int u = order[bb[b]];
      for (int k = 0; k < K; ++k) bgr(b, k) = gr[k].group[u];
      for (int k = 1; k < K; ++k)
        for (int l = 0; l < k; ++l) {
          const int p = V::pair_index(l, k);
          const auto key = std::make_pair(gr[l].group[u], gr[k].group[u]);
          auto it = seen[p].find(key);
          if (it == seen[p].end()) {
            it = seen[p].emplace(key, static_cast<int>(pairs[p].size())).first;
            pairs[p].push_back(key);
          }
          bpr(b, p) = it->second;
        }
    }
    for (int k = 1; k < K; ++k)
      for (int l = 0; l < k; ++l) {
        const int p = V::pair_index(l, k);
        v.pair_information[p] = v.information_size;
        v.information_size += static_cast<int>(pairs[p].size()) * gr[l].width * gr[k].width;
      }
  }
  // the coordinates of every position of the information
  m.information_row.assign(v.information_size, -1), m.information_column.assign(v.information_size, -1);
  {
    int q = 0;
    for (int r = 0; r < s.n_global; ++r)
      for (int c = 0; c <= r; ++c, ++q) m.information_row[q] = r, m.information_column[q] = c;
    for (int k = 0; k < K; ++k) {
      const int w = gr[k].width;
      for (int g = 0; g < gr[k].n_groups; ++g) {
        const int first = v.grouping_coordinate[k] + g * w;
        for (int j = 0; j < w; ++j)
          for (int l = 0; l <= j; ++l, ++q) m.information_row[q] = first + j, m.information_column[q] = first + l;
        for (int j = 0; j < w; ++j)
          for (int c = 0; c < s.n_global; ++c, ++q) m.information_row[q] = first + j, m.information_column[q] = c;
      }
    }
    for (int k = 1; k < K; ++k)
      for (int l = 0; l < k; ++l)
        for (const auto& [gl, gk] : pairs[V::pair_index(l, k)])
          for (int jl = 0; jl < gr[l].width; ++jl)
            for (int jk = 0; jk < gr[k].width; ++jk, ++q) {
              // the coordinate of k is the later one: it is the row
              m.information_row[q] = v.grouping_coordinate[k] + gk * gr[k].width + jk;
              m.information_column[q] = v.grouping_coordinate[l] + gl * gr[l].width + jl;
            }
  }
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
    const int u = order[i];
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
          for (int k = 0; k < K && slot < 0; ++k) {
            const int first = v.grouping_coordinate[k] + gr[k].group[u] * gr[k].width;
            if (c >= first && c < first + gr[k].width) slot = v.grouping_slot[k] + (c - first);
          }
          if (slot < 0)
            refuse("unit " + std::to_string(u) + " reads coordinate " + std::to_string(c) +
                   ", which is neither global nor of one of its groups");
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
  v.block_groups = Kokkos::create_mirror_view_and_copy(V::mem(), bgr);
  v.block_pairs = Kokkos::create_mirror_view_and_copy(V::mem(), bpr);

  for (int k = 0; k < nn; ++k) v.nodes[k] = s.nodes[k];
  for (int j = 0; j < P; ++j) v.param_value[j] = s.param_value[j];
  return m;
}

}  // namespace laplace::engine
