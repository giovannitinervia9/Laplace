#pragma once

// The inner fit of a smooth model (laplace::engine; plan, section 7.9): the
// objective that the library's optimizers minimize,
//
//   f(x) = -sum_i w_i l_i(x) + sum_b (1/2) beta_b' S_b beta_b,
//
// the row kernels (row_kernels.hpp) for the first term and quadratic
// penalties with fixed hyperparameters for the second (a Gaussian prior on
// group effects, a roughness penalty on a smooth): quadratic_penalty_term
// puts the same block_size x block_size matrix S on each consecutive block of
// a range of coordinates.
//
// row_model_objective is a curvature_objective (optimizers/objective.hpp)
// whose curvature is dense (dense_curvature_matrix) or sparse
// (sparse_curvature_matrix, in the pattern of the arrowhead of row_model.hpp).
// The curvature is the observed information, the expected one, or the
// observed one replaced by the expected one when it is not positive definite
// (observed_else_expected, the rule for a sparse H: newton_optimizer corrects
// the eigenvalues of a dense H itself, and cannot for a sparse one).
//
// Held coordinates keep their value: their gradient is zero and their row
// and column of H are those of the identity.
//
// The objective keeps its last assembly: a call at the point of the previous
// one that needs no more than it computed runs no kernel (plan, 7.9 point 6).
// It counts the kernels it runs and the time spent in them.

#include <laplace/engine/row_kernels.hpp>
#include <laplace/optimizers/dense_curvature_solver.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/objective.hpp>
#include <laplace/optimizers/sparse_curvature_solver.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Selects the curvature matrix provided by row_model_objective.
 *
 * `observed` supplies the observed Hessian; `expected` supplies the expected Fisher information
 * \f$\sum_i w_i J_i^\top I(\theta_i) J_i\f$ plus penalties; `observed_else_expected` tries the
 * observed Hessian and replaces it with the expected information when factorization is not
 * `positive_definite`.
 *
 * \see row_model_objective, \ref engine
 */
enum class row_curvature : int { observed = 0, expected = 1, observed_else_expected = 2 };

/**
 * \ingroup engine
 * \brief Fixed quadratic penalty \f$\tfrac12 \beta_b^\top S \beta_b\f$ on blocks of coordinates.
 *
 * Applies the symmetric \f$m \times m\f$ column-major matrix \f$S\f$ (`block_size` \f$= m\f$) to
 * each consecutive block \f$\beta_b\f$ of size \f$m\f$ over `[offset, offset + dim)`, where `dim`
 * is a multiple of `block_size` and no block crosses between the global block and a group or
 * between two groups.
 *
 * \see row_model_objective, \ref engine
 */
struct quadratic_penalty_term {
  int offset = 0;
  int dim = 0;
  int block_size = 0;          ///< \f$m\f$; dim is a multiple of it
  std::vector<double> matrix;  ///< \f$S\f$, \f$m \times m\f$, symmetric, column-major
  /**
   * \ingroup engine
   * \brief Index of \f$\log\lambda\f$ in the outer vector of laplace_marginal_criterion, multiplying \f$S\f$ by \f$\lambda\f$, or \f$-1\f$ for a fixed penalty.
   */
  int hyperparameter = -1;
};

/**
 * \ingroup engine
 * \brief Counts of row-kernel calls, elapsed assembly time, fallbacks, and cache hits on a row_model_objective.
 *
 * `calls` is indexed by `static_cast<int>(row_quantity)`: `[0]` value, `[1]` gradient,
 * `[2]` observed information, and `[3]` expected information. `seconds` accumulates wall-clock
 * assembly time including host-device transfers.
 *
 * \see row_model_objective, inner_fit_report, \ref engine
 */
struct row_kernel_counts {
  long calls[4] = {0, 0, 0, 0};
  double seconds = 0.0;
  long expected_fallbacks = 0;  ///< observed_else_expected: times the expected information replaced the observed
  long reused = 0;              ///< calls answered from the last assembly
};

/**
 * \ingroup engine
 * \brief Inner objective \f$F(x) = -\sum_i w_i \ell_i(x) + \sum_t \sum_q \tfrac12 \beta_{t,q}^\top S_t \beta_{t,q}\f$ for the library's optimizers.
 *
 * Satisfies `optimizers::curvature_objective` on the host, assembling the log-likelihood term on
 * `Kokkos::DefaultExecutionSpace` via assemble_row_model(), adding fixed quadratic penalties and
 * zeroing gradient/off-diagonal Hessian rows and columns for held coordinates, and caching the
 * last evaluation point and assembly.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \tparam Curvature optimizers::dense_curvature_matrix or optimizers::sparse_curvature_matrix.
 * \see fit_inner(), row_curvature, quadratic_penalty_term, \ref optimizers "Optimizers", \ref engine
 */
template <class List, class Curvature = optimizers::dense_curvature_matrix>
class row_model_objective {
 public:
  /**
   * \ingroup engine
   * \brief Curvature matrix type exposed to `optimizers::curvature_objective`.
   */
  using curvature_type = Curvature;
  /**
   * \ingroup engine
   * \brief `true` when `Curvature` is `optimizers::sparse_curvature_matrix`, `false` for dense.
   */
  static constexpr bool sparse = std::is_same_v<Curvature, optimizers::sparse_curvature_matrix>;

  /**
   * \ingroup engine
   * \brief Constructs an inner objective from a built model, quadratic penalties, curvature choice, and held indices.
   *
   * Validates held indices in \f$[0, N)\f$ and penalty shapes and block boundaries (throwing
   * `std::invalid_argument` on error), and builds the sparse arrowhead pattern once when `sparse` is `true`.
   *
   * \param m Built model (must outlive the objective).
   * \param penalties Quadratic penalty terms with fixed hyperparameters.
   * \param curvature Curvature selection (defaults to `observed_else_expected` if `sparse`, else `observed`).
   * \param held Coordinate indices held fixed at their starting values.
   */
  row_model_objective(const row_model& m, std::vector<quadratic_penalty_term> penalties,
                      row_curvature curvature = sparse ? row_curvature::observed_else_expected : row_curvature::observed,
                      std::vector<int> held = {})
      : model_(&m), penalties_(std::move(penalties)), curvature_(curvature), n_(m.n_coordinates()),
        scale_(penalties_.size(), 1.0), held_(n_, 0), xd_("laplace::engine::row_model_objective::x", n_), check_(n_),
        layout_{m.view.n_global, m.view.n_groups, m.view.group_width} {
    for (int h : held) {
      if (h < 0 || h >= n_) throw std::invalid_argument("row_model_objective: a held coordinate outside [0, n)");
      held_[h] = 1;
    }
    for (const auto& p : penalties_) {
      if (p.block_size < 1 || p.dim % p.block_size != 0 || p.offset < 0 || p.offset + p.dim > n_ ||
          static_cast<int>(p.matrix.size()) != p.block_size * p.block_size)
        throw std::invalid_argument("row_model_objective: a penalty with a wrong shape");
      for (int b = p.offset; b < p.offset + p.dim; b += p.block_size)
        if (block_of(b) != block_of(b + p.block_size - 1))
          throw std::invalid_argument("row_model_objective: a penalty block crosses the global block or a group");
    }
    if constexpr (sparse) build_pattern();
  }

  /**
   * \ingroup engine
   * \brief Returns the accumulated row-kernel call counts, wall time, fallbacks, and cache hits.
   * \return Const reference to `row_kernel_counts`.
   */
  const row_kernel_counts& counts() const { return counts_; }
  /**
   * \ingroup engine
   * \brief Resets all counters in counts() to zero without clearing the cached assembly.
   */
  void reset_counts() { counts_ = row_kernel_counts{}; }
  /**
   * \ingroup engine
   * \brief Returns the last unpenalized row_assembly in arrowhead layout.
   * \return Const reference to the cached `row_assembly`.
   */
  const row_assembly& last_assembly() const { return last_; }
  /**
   * \ingroup engine
   * \brief The penalty terms, without their scales.
   * \return Const reference to the penalty terms given to the constructor.
   */
  const std::vector<quadratic_penalty_term>& penalties() const { return penalties_; }
  /**
   * \ingroup engine
   * \brief The factor \f$\lambda_t\f$ that multiplies the matrix of each penalty term (1 by default).
   * \return Const reference to the scales, one per penalty term.
   */
  const std::vector<double>& penalty_scales() const { return scale_; }
  /**
   * \ingroup engine
   * \brief Sets the factor \f$\lambda_t\f$ of each penalty term; the cached assembly stays valid.
   * \param scale One positive number per penalty term.
   */
  void set_penalty_scales(const std::vector<double>& scale) {
    if (scale.size() != penalties_.size()) throw std::invalid_argument("row_model_objective: one scale per penalty term");
    scale_ = scale;
  }

  /**
   * \ingroup engine
   * \brief Evaluates the penalized objective \f$F(x)\f$, reusing the cached assembly if `x` matches.
   * \param x Host coordinate vector of length \f$N\f$.
   * \return Objective value \f$F(x)\f$.
   */
  double value(const optimizers::optimizer_point& x) {
    assemble(x, 0);
    return last_.value + penalty_value(x);
  }

  /**
   * \ingroup engine
   * \brief Evaluates \f$F(x)\f$ and writes \f$\nabla F(x)\f$ into `g`, zeroing held coordinates.
   * \param x Host coordinate vector of length \f$N\f$.
   * \param g Writable host vector of length \f$N\f$ receiving the gradient.
   * \return Objective value \f$F(x)\f$.
   */
  double value_gradient(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g) {
    assemble(x, 1);
    write_gradient(x, g);
    return last_.value + penalty_value(x);
  }

  /**
   * \ingroup engine
   * \brief Evaluates \f$F(x)\f$, \f$\nabla F(x)\f$, and curvature `H`, applying `row_curvature` and held identity rows/columns.
   *
   * When `row_curvature::observed_else_expected` is active and the observed matrix fails
   * `positive_definite` factorization, re-assembles `expected_information` and increments
   * `counts().expected_fallbacks`.
   *
   * \param x Host coordinate vector of length \f$N\f$.
   * \param g Writable host vector of length \f$N\f$ receiving the gradient.
   * \param H Curvature matrix populated in dense or sparse arrowhead CSC format.
   * \return Objective value \f$F(x)\f$.
   */
  double value_gradient_curvature(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g,
                                  curvature_type& H) {
    assemble(x, curvature_ == row_curvature::expected ? 3 : 2);
    write_gradient(x, g);
    write_curvature(H);
    if (curvature_ == row_curvature::observed_else_expected && last_level_ == 2 &&
        check_.factorize(H) != optimizers::curvature_factorization::positive_definite) {
      assemble(x, 3);
      ++counts_.expected_fallbacks;
      write_curvature(H);
    }
    return last_.value + penalty_value(x);
  }

  /**
   * \ingroup engine
   * \brief As value_gradient_curvature(), with the observed information whatever curvature was chosen.
   * \param x Host coordinate vector of length \f$N\f$.
   * \param g Writable host vector of length \f$N\f$ receiving the gradient.
   * \param H Curvature matrix receiving the observed Hessian with identity rows and columns for held coordinates.
   * \return Objective value \f$F(x)\f$.
   */
  double observed_value_gradient_curvature(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g,
                                           curvature_type& H) {
    assemble(x, 2);
    write_gradient(x, g);
    write_curvature(H);
    return last_.value + penalty_value(x);
  }

 private:
  const row_model* model_;
  std::vector<quadratic_penalty_term> penalties_;
  row_curvature curvature_;
  int n_;
  std::vector<double> scale_;  // lambda of each penalty term
  std::vector<int> held_;
  row_model_view::dview xd_;
  typename optimizers::curvature_solver_for<Curvature>::type check_;
  arrowhead_layout layout_;
  row_assembly last_;
  std::vector<double> x_last_;
  int last_level_ = -1;  // 0 value, 1 gradient, 2 observed, 3 expected
  row_kernel_counts counts_;
  std::vector<int> col_ptr_, row_idx_, pos_;  // sparse: the pattern, and the position of each arrowhead entry

  // -1 for a global coordinate, the group otherwise
  int block_of(int c) const { return c < layout_.n_global ? -1 : (c - layout_.n_global) / layout_.group_width; }

  // runs the kernel of the given level unless the last assembly at x already has it
  void assemble(const optimizers::optimizer_point& x, int level) {
    bool same = last_level_ >= 0 && static_cast<int>(x_last_.size()) == n_;
    for (int j = 0; same && j < n_; ++j) same = x(j) == x_last_[j];
    const bool enough = level <= 1 ? last_level_ >= level : last_level_ == level;
    if (same && enough) {
      ++counts_.reused;
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto h = Kokkos::create_mirror_view(xd_);
    for (int j = 0; j < n_; ++j) h(j) = x(j);
    Kokkos::deep_copy(xd_, h);
    assemble_row_model<List>(*model_, xd_, static_cast<row_quantity>(level), last_);
    counts_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++counts_.calls[level];
    x_last_.assign(x.data(), x.data() + n_);
    last_level_ = level;
  }

  double penalty_value(const optimizers::optimizer_point& x) const {
    double v = 0.0;
    for (std::size_t t = 0; t < penalties_.size(); ++t) {
      const auto& p = penalties_[t];
      const int m = p.block_size;
      for (int b = p.offset; b < p.offset + p.dim; b += m)
        for (int j = 0; j < m; ++j)
          for (int k = 0; k < m; ++k) v += 0.5 * scale_[t] * x(b + j) * p.matrix[j + m * k] * x(b + k);
    }
    return v;
  }

  void write_gradient(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g) const {
    for (int j = 0; j < n_; ++j) g(j) = last_.gradient[j];
    for (std::size_t t = 0; t < penalties_.size(); ++t) {
      const auto& p = penalties_[t];
      const int m = p.block_size;
      for (int b = p.offset; b < p.offset + p.dim; b += m)
        for (int j = 0; j < m; ++j) {
          double s = 0.0;
          for (int k = 0; k < m; ++k) s += p.matrix[j + m * k] * x(b + k);
          g(b + j) += scale_[t] * s;
        }
    }
    for (int j = 0; j < n_; ++j)
      if (held_[j]) g(j) = 0.0;
  }

  // the arrowhead position of entry (r, c), r >= c, or -1 when it is structurally zero
  int arrowhead_position(int r, int c) const {
    const int G0 = layout_.n_global;
    if (r < G0) return layout_.global(r, c);
    const int g = block_of(r), j = r - G0 - g * layout_.group_width;
    if (c < G0) return layout_.cross(g, j, c);
    if (block_of(c) != g) return -1;
    return layout_.group(g, j, c - G0 - g * layout_.group_width);
  }

  // the lower triangle of the arrowhead, column by column, the diagonal first
  void build_pattern() {
    const int G0 = layout_.n_global, k = layout_.group_width;
    col_ptr_.assign(1, 0);
    for (int c = 0; c < n_; ++c) {
      const int last = c < G0 ? n_ : G0 + (block_of(c) + 1) * k;
      for (int r = c; r < last; ++r) {
        row_idx_.push_back(r);
        pos_.push_back(arrowhead_position(r, c));
      }
      col_ptr_.push_back(static_cast<int>(row_idx_.size()));
    }
  }

  void write_curvature(curvature_type& H) const {
    const auto& A = last_.information;
    if constexpr (sparse) {
      if (H.row_idx.size() != row_idx_.size()) {
        H.n = n_;
        H.col_ptr = col_ptr_;
        H.row_idx = row_idx_;
      }
      H.values.resize(row_idx_.size());
      for (std::size_t q = 0; q < row_idx_.size(); ++q) H.values[q] = A[pos_[q]];
      for (std::size_t t = 0; t < penalties_.size(); ++t) {
        const auto& p = penalties_[t];
        const int m = p.block_size;
        for (int b = p.offset; b < p.offset + p.dim; b += m)
          for (int k = 0; k < m; ++k)
            for (int j = k; j < m; ++j) H.values[find(b + j, b + k)] += scale_[t] * p.matrix[j + m * k];
      }
      for (int c = 0; c < n_; ++c)
        for (int q = col_ptr_[c]; q < col_ptr_[c + 1]; ++q)
          if (held_[c] || held_[row_idx_[q]]) H.values[q] = row_idx_[q] == c ? 1.0 : 0.0;
    } else {
      for (int c = 0; c < n_; ++c)
        for (int r = c; r < n_; ++r) {
          const int q = arrowhead_position(r, c);
          H(r, c) = H(c, r) = q >= 0 ? A[q] : 0.0;
        }
      for (std::size_t t = 0; t < penalties_.size(); ++t) {
        const auto& p = penalties_[t];
        const int m = p.block_size;
        for (int b = p.offset; b < p.offset + p.dim; b += m)
          for (int k = 0; k < m; ++k)
            for (int j = 0; j < m; ++j) H(b + j, b + k) += scale_[t] * p.matrix[j + m * k];
      }
      for (int j = 0; j < n_; ++j)
        if (held_[j])
          for (int i = 0; i < n_; ++i) H(i, j) = H(j, i) = i == j ? 1.0 : 0.0;
    }
  }

  // sparse: the position of entry (r, c), r >= c, in the pattern
  int find(int r, int c) const {
    for (int q = col_ptr_[c]; q < col_ptr_[c + 1]; ++q)
      if (row_idx_[q] == r) return q;
    throw std::logic_error("row_model_objective: an entry outside the pattern");
  }
};

/**
 * \ingroup engine
 * \brief Result of an inner fit combining the optimizer's report and the kernel counts.
 *
 * \see fit_inner(), row_kernel_counts, \ref engine
 */
struct inner_fit_report {
  optimizers::optimizer_report optimizer;
  row_kernel_counts kernels;
};

/**
 * \ingroup engine
 * \brief Minimizes `objective` from starting point `x0` using Newton's method on the host.
 *
 * Calls `objective.reset_counts()`, runs `optimizers::minimize(newton, objective, x0)`, and
 * returns the resulting `optimizer_report` together with `objective.counts()`.
 *
 * \tparam Objective Curvature objective type (typically row_model_objective).
 * \tparam Newton Newton optimizer type (defaults to `optimizers::newton_optimizer<>`).
 * \param objective Objective instance whose counts are reset and updated during the solve.
 * \param x0 Starting coordinate vector of length \f$N\f$ (left unchanged).
 * \param newton Configured `newton_optimizer` options.
 * \return An inner_fit_report holding the optimizer termination report and kernel statistics.
 *
 * \see row_model_objective, inner_fit_report, \ref optimizers_second_order "Newton and trust region", \ref engine
 */
template <class Objective, class Newton = optimizers::newton_optimizer<>>
inner_fit_report fit_inner(Objective& objective, const optimizers::optimizer_point& x0, const Newton& newton = {}) {
  objective.reset_counts();
  inner_fit_report r;
  r.optimizer = optimizers::minimize(newton, objective, x0);
  r.kernels = objective.counts();
  return r;
}

}  // namespace laplace::engine
