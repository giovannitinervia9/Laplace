#pragma once

// The inner fit of a model (laplace::engine; plan, sections 7.9 and 7.12): the
// objective that the library's optimizers minimize,
//
//   f(x) = -sum_i w_i l_i(x) + sum_t sum_r pen_t(x[C_tr]; theta_t),
//
// the row kernels (row_kernels.hpp) for the first term and the penalty terms
// (penalty_term.hpp) for the second: penalty t applied to each of its
// replicates, the gathered coordinate lists C_tr, at its hyperparameters
// theta_t, whose free ones are read from the vector of natural hyperparameter
// values that the objective holds (set_hyperparameters).
//
// row_model_objective is a curvature_objective (optimizers/objective.hpp)
// whose curvature is dense (dense_curvature_matrix) or sparse
// (sparse_curvature_matrix, in the pattern of the information of
// row_model.hpp, an arrowhead with one grouping, joined with the pairs of
// coordinates of every replicate). The curvature is
// the observed information, the expected one, or the observed one replaced by
// the expected one when it is not positive definite (observed_else_expected,
// the rule for a sparse H: newton_optimizer corrects the eigenvalues of a
// dense H itself, and cannot for a sparse one). The penalties add their
// Hessian in beta to every one of them; away from a kink a penalty with kinks
// has every derivative, and at a kink its derivatives are one-sided.
//
// Held coordinates keep their value: their gradient is zero and their row
// and column of H are those of the identity.
//
// The objective keeps its last assembly: a call at the point of the previous
// one that needs no more than it computed runs no kernel (plan, 7.9 point 6).
// It counts the kernels it runs and the time spent in them.

#include <laplace/engine/penalty_term.hpp>
#include <laplace/engine/row_kernels.hpp>
#include <laplace/optimizers/dense_curvature_solver.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/objective.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/optimizers/sparse_curvature_solver.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <memory>
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
 * \brief The penalty terms of a model with their current hyperparameters, evaluated on the host.
 *
 * Holds one evaluator per term, keeps it at the natural hyperparameters \f$\theta_t\f$ (the fixed
 * values and the free ones read from the vector given to set_hyperparameters()), and sums the
 * value, the gradient and the Hessian in \f$\beta\f$ of every replicate.
 *
 * \see penalty_term, row_model_objective, \ref engine
 */
class penalty_set {
 public:
  /**
   * \ingroup engine
   * \brief Constructs an empty penalty set with no terms or hyperparameters.
   */
  penalty_set() = default;
  /**
   * \ingroup engine
   * \brief Takes the terms and the natural values of the outer hyperparameters they read.
   * \param terms The penalty terms.
   * \param hyperparameters One natural value per outer hyperparameter (at least one more than the largest source).
   * \param n Number of model coordinates, for the checks.
   */
  penalty_set(std::vector<penalty_term> terms, std::vector<double> hyperparameters, int n)
      : terms_(std::move(terms)), hyper_(std::move(hyperparameters)) {
    for (const auto& t : terms_) {
      if (!t.penalty) throw std::invalid_argument("penalty_set: a term without a penalty");
      const int d = t.penalty->dim();
      for (const auto& r : t.replicates) {
        if (static_cast<int>(r.size()) != d) throw std::invalid_argument("penalty_set: a replicate of the wrong length");
        std::vector<int> s = r;
        std::sort(s.begin(), s.end());
        if (std::adjacent_find(s.begin(), s.end()) != s.end())
          throw std::invalid_argument("penalty_set: a replicate reads a coordinate twice");
        for (int c : r)
          if (c < 0 || c >= n) throw std::invalid_argument("penalty_set: a replicate outside [0, n)");
      }
      for (int j : t.theta_source)
        if (j >= static_cast<int>(hyper_.size()))
          throw std::invalid_argument("penalty_set: a hyperparameter source without a value");
      eval_.push_back(t.penalty->clone());
      theta_.emplace_back(t.theta_value);
    }
    refresh();
  }
  penalty_set(const penalty_set& o) : terms_(o.terms_), hyper_(o.hyper_), theta_(o.theta_) {
    for (const auto& e : o.eval_) eval_.push_back(e->clone());
  }
  penalty_set& operator=(const penalty_set& o) {
    if (this != &o) {
      penalty_set c(o);
      std::swap(terms_, c.terms_), std::swap(hyper_, c.hyper_), std::swap(theta_, c.theta_), std::swap(eval_, c.eval_);
    }
    return *this;
  }
  penalty_set(penalty_set&&) = default;
  penalty_set& operator=(penalty_set&&) = default;

  /**
   * \ingroup engine
   * \brief Returns the penalty terms held by this set.
   * \return Const reference to the vector of `penalty_term`.
   */
  const std::vector<penalty_term>& terms() const { return terms_; }
  /**
   * \ingroup engine
   * \brief Returns the natural values of the outer hyperparameters.
   * \return Const reference to the hyperparameter vector.
   */
  const std::vector<double>& hyperparameters() const { return hyper_; }
  /**
   * \ingroup engine
   * \brief Returns the current natural hyperparameters of term `t`.
   * \param t Term index in \f$[0,\f$ `terms().size()`\f$)\f$.
   * \return Const reference to term `t`'s natural hyperparameter vector.
   */
  const std::vector<double>& theta(int t) const { return theta_[t]; }
  /**
   * \ingroup engine
   * \brief Returns the host evaluator of term `t` at its current hyperparameters.
   * \param t Term index in \f$[0,\f$ `terms().size()`\f$)\f$.
   * \return Reference to the term's `penalty_evaluator`.
   */
  penalty_evaluator& evaluator(int t) const { return *eval_[t]; }
  /**
   * \ingroup engine
   * \brief Sets the natural values of the outer hyperparameters and recomputes the states.
   * \param h One value per outer hyperparameter.
   */
  void set_hyperparameters(const std::vector<double>& h) {
    if (h.size() != hyper_.size()) throw std::invalid_argument("penalty_set: one value per outer hyperparameter");
    hyper_ = h;
    refresh();
  }

  /**
   * \ingroup engine
   * \brief Sums the penalty values over all terms and replicates at `x`.
   * \tparam X Callable supporting `x(j)` for coordinate index `j`.
   * \param x Model coordinate vector.
   * \return \f$\sum_t \sum_r \mathrm{pen}_t(x_{C_{tr}}; \theta_t)\f$.
   */
  template <class X>
  double value(const X& x) const {
    double v = 0.0;
    std::vector<double> b;
    for (std::size_t t = 0; t < terms_.size(); ++t)
      for (const auto& r : terms_[t].replicates) {
        gather(x, r, b);
        v += eval_[t]->value(b.data());
      }
    return v;
  }
  /**
   * \ingroup engine
   * \brief Accumulates the gradient in \f$x\f$ of all penalty terms into `g`.
   * \tparam X Callable supporting `x(j)` for coordinate index `j`.
   * \tparam G Callable `g(j, v)` adding `v` to coordinate `j` of the gradient.
   * \param x Model coordinate vector.
   * \param g Gradient accumulator callback.
   */
  template <class X, class G>
  void add_gradient(const X& x, G&& g) const {
    std::vector<double> b, d;
    for (std::size_t t = 0; t < terms_.size(); ++t) {
      auto& e = *eval_[t];
      d.resize(e.tensor_size(1));
      for (const auto& r : terms_[t].replicates) {
        gather(x, r, b);
        if (!e.derivative(1, 0, nullptr, b.data(), d.data())) continue;
        for_each_penalty_entry(e, 1, d.data(), [&](const int* c, double v) { g(r[c[0]], v); });
      }
    }
  }
  /**
   * \ingroup engine
   * \brief Invokes `h(row, column, value)` for every nonzero Hessian entry of the penalties, visiting both triangles.
   * \tparam X Callable supporting `x(j)` for coordinate index `j`.
   * \tparam H Callable `h(row, column, value)` receiving model coordinates and Hessian entry.
   * \param x Model coordinate vector.
   * \param h Hessian entry callback.
   */
  template <class X, class H>
  void for_each_hessian_entry(const X& x, H&& h) const {
    std::vector<double> b, d;
    for (std::size_t t = 0; t < terms_.size(); ++t) {
      auto& e = *eval_[t];
      d.resize(e.tensor_size(2));
      for (const auto& r : terms_[t].replicates) {
        gather(x, r, b);
        if (!e.derivative(2, 0, nullptr, b.data(), d.data())) continue;
        for_each_penalty_entry(e, 2, d.data(), [&](const int* c, double v) { h(r[c[0]], r[c[1]], v); });
      }
    }
  }

  /**
   * \ingroup engine
   * \brief Gathers the coordinates `r` from `x` into `b`.
   * \tparam X Callable supporting `x(j)` for coordinate index `j`.
   * \param x Model coordinate vector.
   * \param r Replicate coordinate list.
   * \param b Output buffer resized to `r.size()`.
   */
  template <class X>
  static void gather(const X& x, const std::vector<int>& r, std::vector<double>& b) {
    b.resize(r.size());
    for (std::size_t k = 0; k < r.size(); ++k) b[k] = x(r[k]);
  }

 private:
  std::vector<penalty_term> terms_;
  std::vector<double> hyper_;
  std::vector<std::vector<double>> theta_;
  std::vector<std::unique_ptr<penalty_evaluator>> eval_;

  void refresh() {
    for (std::size_t t = 0; t < terms_.size(); ++t) {
      const auto& s = terms_[t].theta_source;
      for (std::size_t j = 0; j < s.size(); ++j)
        if (s[j] >= 0) theta_[t][j] = hyper_[s[j]];
      eval_[t]->set_theta(theta_[t].data());
    }
  }
};

/**
 * \ingroup engine
 * \brief Inner objective \f$F(x) = -\sum_i w_i \ell_i(x) + \sum_t \sum_r \mathrm{pen}_t(x_{C_{tr}}; \theta_t)\f$ for the library's optimizers.
 *
 * Satisfies `optimizers::curvature_objective` on the host, assembling the log-likelihood term on
 * `Kokkos::DefaultExecutionSpace` via assemble_row_model(), adding the penalty terms on the host,
 * zeroing gradient/off-diagonal Hessian rows and columns for held coordinates, and caching the
 * last evaluation point and assembly.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \tparam Curvature optimizers::dense_curvature_matrix or optimizers::sparse_curvature_matrix.
 * \see fit_inner(), row_curvature, penalty_term, \ref optimizers "Optimizers", \ref engine
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
   * \brief Constructs an inner objective from a built model, penalty terms, curvature choice, and held indices.
   *
   * Validates held indices in \f$[0, N)\f$ and the penalty terms (throwing `std::invalid_argument`
   * on error), and builds the sparse pattern once when `sparse` is `true`.
   *
   * \param m Built model (must outlive the objective).
   * \param penalties Penalty terms.
   * \param hyperparameters Natural values of the outer hyperparameters the terms read.
   * \param curvature Curvature selection (defaults to `observed_else_expected` if `sparse`, else `observed`).
   * \param held Coordinate indices held fixed at their starting values.
   */
  row_model_objective(const row_model& m, std::vector<penalty_term> penalties, std::vector<double> hyperparameters = {},
                      row_curvature curvature = sparse ? row_curvature::observed_else_expected : row_curvature::observed,
                      std::vector<int> held = {})
      : model_(&m), n_(m.n_coordinates()), pen_(std::move(penalties), std::move(hyperparameters), n_),
        curvature_(curvature), held_(n_, 0), xd_("laplace::engine::row_model_objective::x", n_), check_(n_) {
    for (int h : held) {
      if (h < 0 || h >= n_) throw std::invalid_argument("row_model_objective: a held coordinate outside [0, n)");
      held_[h] = 1;
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
   * \brief Returns the last unpenalized row_assembly, its information in the layout of row_model.hpp.
   * \return Const reference to the cached `row_assembly`.
   */
  const row_assembly& last_assembly() const { return last_; }
  /**
   * \ingroup engine
   * \brief The penalty terms at their current hyperparameters.
   * \return Const reference to the penalty_set.
   */
  const penalty_set& penalties() const { return pen_; }
  /**
   * \ingroup engine
   * \brief Sets the natural values of the outer hyperparameters; the cached assembly stays valid.
   * \param h One value per outer hyperparameter.
   */
  void set_hyperparameters(const std::vector<double>& h) { pen_.set_hyperparameters(h); }

  /**
   * \ingroup engine
   * \brief Evaluates the penalized objective \f$F(x)\f$, reusing the cached assembly if `x` matches.
   * \param x Host coordinate vector of length \f$N\f$.
   * \return Objective value \f$F(x)\f$.
   */
  double value(const optimizers::optimizer_point& x) {
    assemble(x, 0);
    return last_.value + pen_.value(x);
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
    return last_.value + pen_.value(x);
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
   * \param H Curvature matrix populated in dense or sparse CSC format.
   * \return Objective value \f$F(x)\f$.
   */
  double value_gradient_curvature(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g,
                                  curvature_type& H) {
    check_current_ = false;
    assemble(x, curvature_ == row_curvature::expected ? 3 : 2);
    write_gradient(x, g);
    write_curvature(x, H);
    if (curvature_ == row_curvature::observed_else_expected && last_level_ == 2) {
      if (check_.factorize(H) == optimizers::curvature_factorization::positive_definite) {
        check_current_ = true;
      } else {
        assemble(x, 3);
        ++counts_.expected_fallbacks;
        write_curvature(x, H);
      }
    }
    return last_.value + pen_.value(x);
  }

  /**
   * \ingroup engine
   * \brief The positive definite factor of the curvature written by the last value_gradient_curvature(), or `nullptr`.
   *
   * Not null only when that call checked the observed information (`row_curvature::observed_else_expected`)
   * and found it positive definite: the factor is then of exactly the matrix written into `H`, and
   * \ref laplace::optimizers::newton_optimizer "newton_optimizer" solves with it instead of factorizing `H` again.
   * Null after a fallback to the expected information, with the other curvature choices, and after
   * observed_value_gradient_curvature(). Valid while the caller leaves that `H` unchanged.
   * \return Pointer to the solver holding the factor, or `nullptr`.
   * \see optimizers::factorized_curvature_objective
   */
  typename optimizers::curvature_solver_for<Curvature>::type* factorized_curvature() {
    return check_current_ ? &check_ : nullptr;
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
    check_current_ = false;
    assemble(x, 2);
    write_gradient(x, g);
    write_curvature(x, H);
    return last_.value + pen_.value(x);
  }

  /**
   * \ingroup engine
   * \brief The position of entry \f$(r, c)\f$, \f$r \ge c\f$, in the sparse pattern (sparse objectives only).
   * \param r Row.
   * \param c Column.
   * \return Index into the values of the pattern; throws `std::logic_error` when the entry is outside it.
   */
  int pattern_position(int r, int c) const {
    const auto b = row_idx_.begin() + col_ptr_[c], e = row_idx_.begin() + col_ptr_[c + 1];
    const auto it = std::lower_bound(b, e, r);
    if (it == e || *it != r) throw std::logic_error("row_model_objective: an entry outside the pattern");
    return static_cast<int>(it - row_idx_.begin());
  }

 private:
  const row_model* model_;
  int n_;
  penalty_set pen_;
  row_curvature curvature_;
  std::vector<int> held_;
  row_model_view::dview xd_;
  typename optimizers::curvature_solver_for<Curvature>::type check_;
  bool check_current_ = false;  // check_ holds the factor of the H last written by value_gradient_curvature()
  row_assembly last_;
  std::vector<double> x_last_;
  int last_level_ = -1;  // 0 value, 1 gradient, 2 observed, 3 expected
  row_kernel_counts counts_;
  std::vector<int> col_ptr_, row_idx_, pos_;  // sparse: the pattern, and the information position of each entry (-1 if none)

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

  void write_gradient(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g) const {
    for (int j = 0; j < n_; ++j) g(j) = last_.gradient[j];
    pen_.add_gradient(x, [&](int j, double v) { g(j) += v; });
    for (int j = 0; j < n_; ++j)
      if (held_[j]) g(j) = 0.0;
  }

  // the lower triangle of the information joined with the pairs of every replicate and the diagonal,
  // column by column, the rows sorted (the diagonal first)
  void build_pattern() {
    const auto& ir = model_->information_row;
    const auto& ic = model_->information_column;
    std::unordered_map<long long, int> where;
    std::vector<std::vector<int>> rows(n_);
    auto key = [&](int r, int c) { return static_cast<long long>(r) * n_ + c; };
    for (std::size_t q = 0; q < ir.size(); ++q) {
      where[key(ir[q], ic[q])] = static_cast<int>(q);
      rows[ic[q]].push_back(ir[q]);
    }
    for (int c = 0; c < n_; ++c) rows[c].push_back(c);
    for (const auto& t : pen_.terms())
      for (const auto& r : t.replicates)
        for (int a : r)
          for (int b : r)
            if (a > b) rows[b].push_back(a);
    col_ptr_.assign(1, 0);
    for (int c = 0; c < n_; ++c) {
      auto& rc = rows[c];
      std::sort(rc.begin(), rc.end());
      rc.erase(std::unique(rc.begin(), rc.end()), rc.end());
      for (int r : rc) {
        row_idx_.push_back(r);
        const auto it = where.find(key(r, c));
        pos_.push_back(it == where.end() ? -1 : it->second);
      }
      col_ptr_.push_back(static_cast<int>(row_idx_.size()));
    }
  }

  void write_curvature(const optimizers::optimizer_point& x, curvature_type& H) const {
    const auto& A = last_.information;
    if constexpr (sparse) {
      if (H.row_idx.size() != row_idx_.size()) {
        H.n = n_;
        H.col_ptr = col_ptr_;
        H.row_idx = row_idx_;
      }
      H.values.resize(row_idx_.size());
      for (std::size_t q = 0; q < row_idx_.size(); ++q) H.values[q] = pos_[q] >= 0 ? A[pos_[q]] : 0.0;
      pen_.for_each_hessian_entry(x, [&](int r, int c, double v) {
        if (r >= c) H.values[pattern_position(r, c)] += v;
      });
      for (int c = 0; c < n_; ++c)
        for (int q = col_ptr_[c]; q < col_ptr_[c + 1]; ++q)
          if (held_[c] || held_[row_idx_[q]]) H.values[q] = row_idx_[q] == c ? 1.0 : 0.0;
    } else {
      for (int c = 0; c < n_; ++c)
        for (int r = 0; r < n_; ++r) H(r, c) = 0.0;
      for (std::size_t q = 0; q < A.size(); ++q) {
        const int r = model_->information_row[q], c = model_->information_column[q];
        H(r, c) = H(c, r) = A[q];
      }
      pen_.for_each_hessian_entry(x, [&](int r, int c, double v) { H(r, c) += v; });
      for (int j = 0; j < n_; ++j)
        if (held_[j])
          for (int i = 0; i < n_; ++i) H(i, j) = H(j, i) = i == j ? 1.0 : 0.0;
    }
  }
};

/**
 * \ingroup engine
 * \brief Inner objective with penalties that have kinks, as a composite objective for `optimizers::proximal_newton_optimizer`.
 *
 * Splits the penalty terms by their kinks: the smooth ones go with the likelihood into the smooth
 * part, a dense row_model_objective, which gives the value, the gradient and the curvature; those
 * with kinks form the non-smooth part, with their value and proximal operator. Each coordinate of a
 * separable kinked penalty (a lasso through a product of univariate laws, SCAD, MCP) is a block of its
 * own, so the optimizer runs coordinate descent on its subproblem; any other kinked penalty (a group
 * lasso) is one block per replicate, and the optimizer runs FISTA on the subproblem.
 *
 * Refused (std::invalid_argument): a kinked penalty without a proximal operator, two kinked
 * replicates that share a coordinate, and a held coordinate that a kinked penalty reads.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \see fit_inner(), row_model_objective, penalty_term, \ref optimizers_proximal "Proximal methods", \ref engine
 */
template <class List>
class row_model_composite_objective {
 public:
  /**
   * \ingroup engine
   * \brief Curvature matrix type of the smooth part (dense, as required by `optimizers::proximal_newton_optimizer`).
   */
  using curvature_type = optimizers::dense_curvature_matrix;

  /**
   * \ingroup engine
   * \brief Builds the composite objective by splitting smooth and kinked penalty terms.
   * \param m Built model (must outlive the objective).
   * \param penalties Penalty terms, with and without kinks.
   * \param hyperparameters Natural values of the outer hyperparameters the terms read.
   * \param curvature Curvature of the smooth part (observed by default; proximal Newton corrects an indefinite one).
   * \param held Coordinates held at their starting values.
   */
  row_model_composite_objective(const row_model& m, const std::vector<penalty_term>& penalties,
                                std::vector<double> hyperparameters = {}, row_curvature curvature = row_curvature::observed,
                                std::vector<int> held = {})
      : smooth_(m, split(penalties, false), hyperparameters, curvature, held),
        kinked_(split(penalties, true), hyperparameters, m.n_coordinates()) {
    const int n = m.n_coordinates();
    std::vector<int> owner(n, 0);
    for (int h : held) owner[h] = -1;
    const auto& terms = kinked_.terms();
    for (int t = 0; t < static_cast<int>(terms.size()); ++t) {
      const auto& e = kinked_.evaluator(t);
      if (!e.has_prox()) throw std::invalid_argument("row_model_composite_objective: a penalty with kinks has no proximal operator");
      const bool sep = e.structure() == penalty::penalty_structure::separable;
      for (int r = 0; r < static_cast<int>(terms[t].replicates.size()); ++r) {
        const auto& rep = terms[t].replicates[r];
        for (int c : rep) {
          if (owner[c] < 0) throw std::invalid_argument("row_model_composite_objective: a held coordinate read by a penalty with kinks");
          if (owner[c] > 0) throw std::invalid_argument("row_model_composite_objective: two replicates with kinks share a coordinate");
          owner[c] = 1;
        }
        if (sep)
          for (int k = 0; k < static_cast<int>(rep.size()); ++k) blocks_.push_back({t, r, k, rep[k]});
        else
          blocks_.push_back({t, r, -1, rep.empty() ? 0 : rep[0]});
      }
    }
  }

  /**
   * \ingroup engine
   * \brief Returns the row-kernel counts of the smooth objective.
   * \return Const reference to `row_kernel_counts`.
   */
  const row_kernel_counts& counts() const { return smooth_.counts(); }
  /**
   * \ingroup engine
   * \brief Resets the row-kernel counts of the smooth objective without clearing its cached assembly.
   */
  void reset_counts() { smooth_.reset_counts(); }
  /**
   * \ingroup engine
   * \brief Returns the smooth part of the composite objective.
   * \return Const reference to the dense `row_model_objective`.
   */
  const row_model_objective<List, curvature_type>& smooth() const { return smooth_; }
  /**
   * \ingroup engine
   * \brief Returns the penalty set holding the kinked penalty terms.
   * \return Const reference to `penalty_set`.
   */
  const penalty_set& kinked() const { return kinked_; }
  /**
   * \ingroup engine
   * \brief Sets the natural values of the outer hyperparameters in both the smooth and kinked parts.
   * \param h One value per outer hyperparameter.
   */
  void set_hyperparameters(const std::vector<double>& h) {
    smooth_.set_hyperparameters(h);
    kinked_.set_hyperparameters(h);
  }

  /**
   * \ingroup engine
   * \brief Evaluates the smooth part at `x`.
   * \param x Host coordinate vector of length \f$N\f$.
   * \return Value of the likelihood plus smooth penalties.
   */
  double value(const optimizers::optimizer_point& x) { return smooth_.value(x); }
  /**
   * \ingroup engine
   * \brief Evaluates the smooth part at `x` and writes its gradient into `g`.
   * \param x Host coordinate vector of length \f$N\f$.
   * \param g Writable host vector of length \f$N\f$ receiving the gradient.
   * \return Value of the smooth part.
   */
  double value_gradient(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g) {
    return smooth_.value_gradient(x, g);
  }
  /**
   * \ingroup engine
   * \brief Evaluates the smooth part at `x`, its gradient in `g`, and its dense curvature in `H`.
   * \param x Host coordinate vector of length \f$N\f$.
   * \param g Writable host vector of length \f$N\f$ receiving the gradient.
   * \param H Dense curvature matrix populated for the smooth part.
   * \return Value of the smooth part.
   */
  double value_gradient_curvature(const optimizers::optimizer_point& x, const optimizers::optimizer_vector& g,
                                  curvature_type& H) {
    return smooth_.value_gradient_curvature(x, g, H);
  }
  /**
   * \ingroup engine
   * \brief Evaluates the sum of the kinked penalties at `x`.
   * \param x Host coordinate vector of length \f$N\f$.
   * \return Non-smooth penalty value.
   */
  double nonsmooth_value(const optimizers::optimizer_point& x) { return kinked_.value(x); }
  /**
   * \ingroup engine
   * \brief Applies the proximal operator of the kinked penalties to `v`, leaving unpenalized coordinates unchanged.
   * \param v Input host point of length \f$N\f$.
   * \param t Proximal step size.
   * \param z Writable host vector of length \f$N\f$ receiving the result.
   */
  void nonsmooth_prox(const optimizers::optimizer_point& v, double t, const optimizers::optimizer_vector& z) {
    for (std::size_t j = 0; j < v.extent(0); ++j) z(j) = v(j);
    std::vector<double> b, out;
    const auto& terms = kinked_.terms();
    for (int k = 0; k < static_cast<int>(terms.size()); ++k)
      for (const auto& rep : terms[k].replicates) {
        penalty_set::gather(v, rep, b);
        out.resize(b.size());
        kinked_.evaluator(k).prox(b.data(), t, out.data());
        for (std::size_t q = 0; q < rep.size(); ++q) z(rep[q]) = out[q];
      }
  }
  /**
   * \ingroup engine
   * \brief Number of non-smooth blocks exposed to `optimizers::proximal_newton_optimizer`.
   * \return One block per coordinate for separable kinked penalties, one block per replicate otherwise.
   */
  int nonsmooth_block_count() const { return static_cast<int>(blocks_.size()); }
  /**
   * \ingroup engine
   * \brief Weak-convexity constant \f$\rho = \sup(-\mathrm{pen}'')\f$ of the penalty of block `b`.
   * \param b Block index in \f$[0,\f$ `nonsmooth_block_count()`\f$)\f$.
   * \return Weak-convexity constant at the current hyperparameters.
   */
  double weak_convexity_constant(int b) { return kinked_.evaluator(blocks_[b].term).weak_convexity(); }
  /**
   * \ingroup engine
   * \brief Coordinate range `(start, length)` of block `b`: `(coordinate, 1)` when separable, length `0` otherwise.
   *
   * A block of a penalty that is not separable reads gathered coordinates, which a range cannot
   * name; it reports length 0, and the optimizer, which then has a non-separable block, takes the
   * route that reads no range (FISTA on the subproblem).
   * \param b Block index in \f$[0,\f$ `nonsmooth_block_count()`\f$)\f$.
   * \return Pair `(coordinate, length)`.
   */
  std::pair<int, int> nonsmooth_block_range(int b) const {
    return blocks_[b].position >= 0 ? std::pair<int, int>{blocks_[b].coordinate, 1} : std::pair<int, int>{blocks_[b].coordinate, 0};
  }
  /**
   * \ingroup engine
   * \brief Reports whether block `b` is a single coordinate of a separable kinked penalty.
   * \param b Block index in \f$[0,\f$ `nonsmooth_block_count()`\f$)\f$.
   * \return `true` for a separable coordinate block, `false` for a multi-coordinate replicate block.
   */
  bool nonsmooth_block_separable(int b) const { return blocks_[b].position >= 0; }
  /**
   * \ingroup engine
   * \brief Evaluates the proximal operator of block `b` on its single coordinate.
   * \param b Separable block index.
   * \param j Coordinate offset within the block (unused, always `0`).
   * \param v Scalar input value.
   * \param t Proximal step size.
   * \return Minimizer for that coordinate.
   */
  // ponytail: the prox of the whole replicate for one coordinate, O(d) per coordinate; a per-coordinate
  // prox in penalty_evaluator if a separable penalty with many coordinates per replicate is slow.
  double nonsmooth_coordinate_prox(int b, int j, double v, double t) {
    (void)j;
    const auto& bl = blocks_[b];
    auto& e = kinked_.evaluator(bl.term);
    const int d = e.dim();
    std::vector<double> in(d, 0.0), out(d);
    in[bl.position] = v;
    e.prox(in.data(), t, out.data());
    return out[bl.position];
  }

 private:
  struct block {
    int term, replicate, position, coordinate;  // position -1: the whole replicate
  };
  row_model_objective<List, curvature_type> smooth_;
  penalty_set kinked_;
  std::vector<block> blocks_;

  static std::vector<penalty_term> split(const std::vector<penalty_term>& p, bool kinked) {
    std::vector<penalty_term> o;
    for (const auto& t : p)
      if ((t.penalty->kinks() != penalty::penalty_kink_geometry::none) == kinked) o.push_back(t);
    return o;
  }
};

/**
 * \ingroup engine
 * \brief Result of an inner fit combining the optimizer's report and the kernel counts.
 *
 * \see fit_inner(), \ref engine
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
