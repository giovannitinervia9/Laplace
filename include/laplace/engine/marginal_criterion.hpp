#pragma once

// The Laplace marginal criterion of a smooth model (laplace::engine; plan,
// sections 7.3 and 7.10), with its exact gradient and Hessian.
//
// The inner objective (inner_fit.hpp) is
//
//   F(x; rho) = -sum_i w_i l_i(x) + sum_t (lambda_t / 2) sum_blocks beta' S_t beta,
//
// with lambda_t = exp(rho_k) for a penalty term whose hyperparameter is k
// (quadratic_penalty_term::hyperparameter) and lambda_t = 1 for a fixed one.
// The outer vector is u = (rho_0, ..., rho_{K-1}, psi), where psi are
// coordinates of x estimated on the criterion (held in the inner fit). Every
// other coordinate is free, except those the caller fixes, and is integrated
// by the Laplace approximation. With x^ the inner minimum at u and H the
// Hessian of F in the free coordinates at x^, the criterion minimized is
//
//   V(u) = F(x^; rho) + (1/2) log|H| - (1/2) log|S_lambda|_+,
//
// log|S_lambda|_+ = sum_t n_t (r_t log lambda_t + log|S_t|_+), with n_t the
// number of blocks of term t, r_t the rank of S_t and log|.|_+ the logarithm
// of the product of the positive eigenvalues (the terms must not share
// coordinates). V is minus the logarithm of the Laplace approximation of the
// marginal likelihood, up to a constant.
//
// Its derivatives. Let v_a = dx^/du_a: for rho_k, v_a = -Hi s_k with
// s_k = sum_{t: k} lambda_t S_t x; for psi at coordinate c,
// v_a = e_c - Hi H_full e_c, where Hi is the inverse of H padded with zeros
// on held rows and columns and H_full the Hessian of F in all coordinates.
// With E_a = sum_{t: k} lambda_t S_t (zero for psi), T and Q the third and
// fourth derivatives of the first term of F, and tau, T[v_a], T[v_a, v_b],
// q_ab the passes of outer_rows.hpp,
//
//   dH_a = E_a + T[v_a]
//   g_a  = dF_a + (1/2) (tr(Hi E_a) + tau' v_a) - (1/2) sum_{t: k} n_t r_t
//   H_ab = F2_ab + (1/2) (-tr(Hi dH_b Hi dH_a) + delta_ab tr(Hi E_a) + q_ab + tau' x^_ab)
//
// where dF_a is (1/2) sum_{t: k} lambda_t beta' S_t beta for rho_k and the
// gradient of F at coordinate c for psi; F2_ab = delta_ab dF_a [a is rho] +
// s_a' v_b + s_b' v_a + v_a' H_full v_b; and
// x^_ab = -Hi (T[v_a, v_b] + [b is rho] E_b v_a + [a is rho] E_a v_b + delta_ab [a is rho] s_a).
//
// The Hessian H in log|H| is the observed one (decision E1 of 2026-11-25):
// where it is not positive definite at the inner minimum the criterion is +inf
// and the evaluation is counted. Hi is a dense N x N matrix (decision D1): the
// columns of the sparse factorization solved for every free coordinate.

#include <laplace/dense/dense.hpp>
#include <laplace/engine/inner_fit.hpp>
#include <laplace/engine/outer_rows.hpp>
#include <laplace/sparse/sparse_ldlt.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace laplace::engine {

namespace detail_marginal {

// X_a = Hi dH_a (N x N, dense) for the A matrices dH_a given in compressed columns (cp: A blocks of
// N + 1 offsets into rows and vals), then tr(X_a X_b) = tr(Hi dH_a Hi dH_b) for the pairs a <= b in
// lexicographic order, as Kokkos kernels (free functions: nvcc refuses an extended lambda in a private
// member function)
inline void trace_products(const Kokkos::View<double**, Kokkos::LayoutLeft, row_model_view::mem>& Hi, int N, int A,
                           const std::vector<int>& cp, const std::vector<int>& rows, const std::vector<double>& vals,
                           std::vector<double>& out) {
  using mem = row_model_view::mem;
  using exec = row_model_view::exec;
  auto to_dev = [](const auto& v, const char* name) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(v.data(), v.size());
    Kokkos::View<T*, mem> d(name, v.size());
    Kokkos::deep_copy(d, h);
    return d;
  };
  const auto dcp = to_dev(cp, "laplace::engine::dH_cols");
  const auto drow = to_dev(rows, "laplace::engine::dH_rows");
  const auto dval = to_dev(vals, "laplace::engine::dH_vals");
  Kokkos::View<double***, Kokkos::LayoutLeft, mem> X("laplace::engine::X", N, N, A);
  Kokkos::parallel_for(
      "laplace::engine::hi_times_dH", Kokkos::MDRangePolicy<exec, Kokkos::Rank<3>>({0, 0, 0}, {N, N, A}),
      KOKKOS_LAMBDA(const int i, const int j, const int a) {
        const int* c = &dcp(a * (N + 1));
        double s = 0.0;
        for (int q = c[j]; q < c[j + 1]; ++q) s += Hi(i, drow(q)) * dval(q);
        X(i, j, a) = s;
      });
  out.assign(A * (A + 1) / 2, 0.0);
  int ab = 0;
  for (int a = 0; a < A; ++a)
    for (int b = a; b < A; ++b, ++ab) {
      double t = 0.0;
      Kokkos::parallel_reduce(
          "laplace::engine::trace_XX", Kokkos::MDRangePolicy<exec, Kokkos::Rank<2>>({0, 0}, {N, N}),
          KOKKOS_LAMBDA(const int i, const int j, double& acc) { acc += X(i, j, a) * X(j, i, b); }, t);
      out[ab] = t;
    }
}

}  // namespace detail_marginal

/**
 * \ingroup engine
 * \brief Counts of a laplace_marginal_criterion: evaluations, inner fits, refusals, outer passes and time.
 * \see laplace_marginal_criterion, \ref engine
 */
struct marginal_criterion_counts {
  long evaluations[3] = {0, 0, 0};  ///< value, value and gradient, value gradient and Hessian
  long inner_fits = 0;
  long inner_failures = 0;           ///< inner fits that did not converge: the criterion is +inf there
  long inner_stalls = 0;             ///< inner fits whose step fell below rounding: their point is used
  optimizers::optimizer_status last_inner_failure = optimizers::optimizer_status::converged;  ///< why the last one failed
  long not_positive_definite = 0;    ///< points where the observed H was not positive definite: +inf
  long outer_passes[2] = {0, 0};     ///< order 3, order 4
  row_kernel_counts inner_kernels;   ///< the inner objective's kernels, summed over every inner fit
  double seconds_inner = 0.0, seconds_algebra = 0.0, seconds_passes = 0.0;
};

/**
 * \ingroup engine
 * \brief The Laplace marginal criterion \f$V(u)\f$ of a smooth model, with exact gradient and Hessian.
 *
 * A `curvature_objective` (dense) over the outer vector \f$u = (\rho, \psi)\f$ for the library's
 * optimizers; each new \f$u\f$ runs an inner fit from the previous inner minimum.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \see fit_marginal(), marginal_criterion_counts, \ref engine
 */
template <class List>
class laplace_marginal_criterion {
 public:
  /**
   * \ingroup engine
   * \brief Outer curvature matrix type exposed to `optimizers::curvature_objective`.
   */
  using curvature_type = optimizers::dense_curvature_matrix;
  /**
   * \ingroup engine
   * \brief Sparse inner objective type minimized at each outer vector \f$u\f$.
   */
  using inner_objective = row_model_objective<List, optimizers::sparse_curvature_matrix>;

  /**
   * \ingroup engine
   * \brief Builds the criterion.
   * \param m Built model (must outlive the criterion).
   * \param penalties Quadratic penalty terms; those with `hyperparameter >= 0` are scaled by \f$e^{\rho_k}\f$.
   * \param outer_coordinates Coordinates estimated on the criterion (\f$\psi\f$), held in the inner fit.
   * \param fixed Coordinates held at their value in `x0` and not estimated.
   * \param x0 Start of the first inner fit, length \f$N\f$.
   * \param inner_newton Options of the inner Newton fits.
   */
  laplace_marginal_criterion(const row_model& m, std::vector<quadratic_penalty_term> penalties,
                             std::vector<int> outer_coordinates, std::vector<int> fixed, std::vector<double> x0,
                             optimizers::newton_optimizer<> inner_newton = {})
      : model_(&m), n_(m.n_coordinates()), psi_(std::move(outer_coordinates)),
        inner_(m, penalties, row_curvature::observed_else_expected, held_list(psi_, fixed)),
        newton_(inner_newton), xhat_(std::move(x0)), held_(n_, 0),
        layout_{m.view.n_global, m.view.n_groups, m.view.group_width} {
    if (static_cast<int>(xhat_.size()) != n_) throw std::invalid_argument("laplace_marginal_criterion: x0 must have N entries");
    for (int c : psi_) held_[c] = 1;
    for (int c : fixed) held_[c] = 1;
    const auto& pen = inner_.penalties();
    n_hyper_ = 0;
    for (const auto& p : pen) n_hyper_ = std::max(n_hyper_, p.hyperparameter + 1);
    std::vector<int> seen(n_hyper_, 0), owner(n_, -1);
    for (std::size_t t = 0; t < pen.size(); ++t) {
      const auto& p = pen[t];
      if (p.hyperparameter >= 0) seen[p.hyperparameter] = 1;
      for (int c = p.offset; c < p.offset + p.dim; ++c) {
        if (owner[c] >= 0) throw std::invalid_argument("laplace_marginal_criterion: two penalty terms share a coordinate");
        owner[c] = static_cast<int>(t);
      }
    }
    for (int k = 0; k < n_hyper_; ++k)
      if (!seen[k]) throw std::invalid_argument("laplace_marginal_criterion: a hyperparameter index that no term uses");
    // the rank and log pseudo-determinant of each S_t
    // ponytail: eigenvalues above 1e-9 max|w| count as positive; a rank given by the caller if a
    // penalty is ill conditioned enough to make this ambiguous.
    for (const auto& p : pen) {
      const int mm = p.block_size;
      Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> S("S", mm, mm), U("U", mm, mm);
      Kokkos::View<double*, Kokkos::HostSpace> w("w", mm);
      for (int j = 0; j < mm; ++j)
        for (int k = 0; k < mm; ++k) S(j, k) = p.matrix[j + mm * k];
      const auto st = dense::symmetric_eigen_decomposition(S, w, U);
      if (!st.success) throw std::invalid_argument("laplace_marginal_criterion: the eigenvalues of a penalty matrix failed");
      double wmax = 0.0;
      for (int j = 0; j < mm; ++j) wmax = std::max(wmax, std::abs(w(j)));
      int r = 0;
      double ld = 0.0;
      for (int j = 0; j < mm; ++j)
        if (w(j) > 1e-9 * wmax) ++r, ld += std::log(w(j));
      rank_.push_back(r);
      logdet_.push_back(ld);
    }
    if (n_outer() < 1) throw std::invalid_argument("laplace_marginal_criterion: an empty outer vector");
  }

  /**
   * \ingroup engine
   * \brief Length \f$K + |\psi|\f$ of the outer vector \f$u\f$.
   * \return Number of outer coordinates.
   */
  int n_outer() const { return n_hyper_ + static_cast<int>(psi_.size()); }
  /**
   * \ingroup engine
   * \brief Number \f$K\f$ of log hyperparameters \f$\rho\f$ in the outer vector.
   * \return Number of hyperparameters.
   */
  int n_hyperparameters() const { return n_hyper_; }
  /**
   * \ingroup engine
   * \brief Last inner minimum \f$\hat x\f$ computed by the criterion.
   * \return Const reference to the inner coordinate vector of length \f$N\f$.
   */
  const std::vector<double>& inner_point() const { return xhat_; }
  /**
   * \ingroup engine
   * \brief Accumulated evaluation, inner-fit, refusal, outer-pass, and timing counters.
   * \return Const reference to `marginal_criterion_counts`.
   */
  const marginal_criterion_counts& counts() const { return counts_; }
  /**
   * \ingroup engine
   * \brief Inner sparse objective holding the penalty terms, current scales, and last assembly.
   * \return Const reference to `inner_objective`.
   */
  const inner_objective& inner() const { return inner_; }

  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$, returning \f$+\infty\f$ if the inner fit fails or \f$H\f$ is not positive definite.
   * \param u Outer coordinate vector of length `n_outer()`.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value(const optimizers::optimizer_point& u) {
    ++counts_.evaluations[0];
    evaluate(u, 0);
    return V_;
  }
  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$ and writes its exact gradient into `g`.
   *
   * If \f$V(u) = +\infty\f$, fills `g` with quiet NaN. Reuses cached results when called at the
   * same `u` as the previous evaluation.
   *
   * \param u Outer coordinate vector of length `n_outer()`.
   * \param g Writable host vector of length `n_outer()` receiving \f$\nabla V(u)\f$.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value_gradient(const optimizers::optimizer_point& u, const optimizers::optimizer_vector& g) {
    ++counts_.evaluations[1];
    evaluate(u, 1);
    for (int a = 0; a < n_outer(); ++a) g(a) = std::isfinite(V_) ? grad_[a] : std::numeric_limits<double>::quiet_NaN();
    return V_;
  }
  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$, its exact gradient `g`, and its exact dense Hessian `H`.
   *
   * Throws `std::invalid_argument` if `n_outer() > row_outer_direction_capacity`. If \f$V(u) = +\infty\f$,
   * fills `g` and `H` with quiet NaN. Reuses cached results when called at the same `u` as the
   * previous evaluation.
   *
   * \param u Outer coordinate vector of length `n_outer()`.
   * \param g Writable host vector of length `n_outer()` receiving \f$\nabla V(u)\f$.
   * \param H Dense outer curvature matrix of size `n_outer()` by `n_outer()` receiving \f$\nabla^2 V(u)\f$.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value_gradient_curvature(const optimizers::optimizer_point& u, const optimizers::optimizer_vector& g,
                                  curvature_type& H) {
    ++counts_.evaluations[2];
    evaluate(u, 2);
    const int A = n_outer();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int a = 0; a < A; ++a) {
      g(a) = std::isfinite(V_) ? grad_[a] : nan;
      for (int b = 0; b < A; ++b) H(a, b) = std::isfinite(V_) ? hess_[a + A * b] : nan;
    }
    return V_;
  }

 private:
  using clock = std::chrono::steady_clock;
  using dev_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, row_model_view::mem>;

  const row_model* model_;
  int n_;
  std::vector<int> psi_;
  inner_objective inner_;
  optimizers::newton_optimizer<> newton_;
  std::vector<double> xhat_;
  std::vector<int> held_;
  arrowhead_layout layout_;
  int n_hyper_ = 0;
  std::vector<int> rank_;
  std::vector<double> logdet_;
  marginal_criterion_counts counts_;
  // the state at the last point
  std::vector<double> u_last_;
  int level_ = -1;  // -1 nothing, 0 value, 1 gradient, 2 Hessian
  double V_ = 0.0;
  std::vector<double> grad_, hess_;
  sparse::sparse_ldlt ldlt_;
  optimizers::sparse_curvature_matrix Hs_;
  std::vector<double> Hi_;  // dense N x N, column-major
  bool have_hi_ = false, hid_current_ = false;
  dev_matrix Hid_;  // Hi on the device

  static std::vector<int> held_list(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> h = a;
    h.insert(h.end(), b.begin(), b.end());
    return h;
  }

  int n_blocks(const quadratic_penalty_term& p) const { return p.dim / p.block_size; }

  void evaluate(const optimizers::optimizer_point& u, int level) {
    const int A = n_outer();
    bool same = level_ >= 0 && static_cast<int>(u_last_.size()) == A;
    for (int a = 0; same && a < A; ++a) same = u(a) == u_last_[a];
    if (same && level_ >= level) return;
    if (!same) {
      u_last_.assign(u.data(), u.data() + A);
      have_hi_ = false, hid_current_ = false;
      level_ = 0;
      fit_value();
    }
    if (level >= 1 && std::isfinite(V_)) derivatives(level);
    level_ = std::max(level_, level);
  }

  void fit_value() {
    const auto& pen = inner_.penalties();
    std::vector<double> scale(pen.size(), 1.0);
    for (std::size_t t = 0; t < pen.size(); ++t)
      if (pen[t].hyperparameter >= 0) scale[t] = std::exp(u_last_[pen[t].hyperparameter]);
    inner_.set_penalty_scales(scale);
    optimizers::optimizer_vector x0("x0", n_);
    for (int j = 0; j < n_; ++j) x0(j) = xhat_[j];
    for (std::size_t k = 0; k < psi_.size(); ++k) x0(psi_[k]) = u_last_[n_hyper_ + k];
    auto t0 = clock::now();
    const auto r = fit_inner(inner_, x0, newton_);
    counts_.seconds_inner += std::chrono::duration<double>(clock::now() - t0).count();
    ++counts_.inner_fits;
    for (int k = 0; k < 4; ++k) counts_.inner_kernels.calls[k] += r.kernels.calls[k];
    counts_.inner_kernels.seconds += r.kernels.seconds;
    counts_.inner_kernels.expected_fallbacks += r.kernels.expected_fallbacks;
    counts_.inner_kernels.reused += r.kernels.reused;
    // a stall (the step no longer moves the point) is the inner minimum to the precision of x; it
    // happens from a start very near the minimum, before the decrement falls below its tolerance
    const bool stalled = r.optimizer.status == optimizers::optimizer_status::stalled;
    if (stalled) ++counts_.inner_stalls;
    if (!optimizers::optimizer_succeeded(r.optimizer.status) && !stalled) {
      ++counts_.inner_failures;
      counts_.last_inner_failure = r.optimizer.status;
      V_ = std::numeric_limits<double>::infinity();
      return;
    }
    for (int j = 0; j < n_; ++j) xhat_[j] = r.optimizer.point(j);
    optimizers::optimizer_vector g("g", n_);
    const auto xv = point_view();
    const double F = inner_.observed_value_gradient_curvature(xv, g, Hs_);
    t0 = clock::now();
    const auto out = ldlt_.factorize(n_, Hs_.col_ptr.data(), Hs_.row_idx.data(), Hs_.values.data());
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    if (out != sparse::sparse_ldlt_outcome::positive_definite) {
      ++counts_.not_positive_definite;
      V_ = std::numeric_limits<double>::infinity();
      return;
    }
    double logS = 0.0;
    for (std::size_t t = 0; t < pen.size(); ++t)
      logS += n_blocks(pen[t]) * (rank_[t] * std::log(scale[t]) + logdet_[t]);
    V_ = F + 0.5 * ldlt_.log_abs_determinant() - 0.5 * logS;
  }

  optimizers::optimizer_vector point_view() const {
    optimizers::optimizer_vector x("xhat", n_);
    for (int j = 0; j < n_; ++j) x(j) = xhat_[j];
    return x;
  }

  // the global (row, column) of each entry of the arrowhead layout
  void arrowhead_entry(int q, int& r, int& c) const {
    const int G0 = layout_.n_global, w = layout_.group_width;
    if (q < layout_.global_size()) {
      r = 0;
      while ((r + 1) * (r + 2) / 2 <= q) ++r;
      c = q - r * (r + 1) / 2;
      return;
    }
    q -= layout_.global_size();
    const int g = q / layout_.group_block_size();
    q -= g * layout_.group_block_size();
    const int sq = w * (w + 1) / 2;
    if (q < sq) {
      int j = 0;
      while ((j + 1) * (j + 2) / 2 <= q) ++j;
      r = G0 + g * w + j, c = G0 + g * w + (q - j * (j + 1) / 2);
    } else {
      q -= sq;
      r = G0 + g * w + q / G0, c = q % G0;
    }
  }

  void derivatives(int level) {
    const int N = n_, A = n_outer(), K = n_hyper_;
    if (level == 2 && A > row_outer_direction_capacity)
      throw std::invalid_argument("laplace_marginal_criterion: more outer coordinates than row_outer_direction_capacity");
    const auto& pen = inner_.penalties();
    const auto& scale = inner_.penalty_scales();
    auto t0 = clock::now();
    // H^-1, padded with zeros on held rows and columns
    // ponytail: dense N x N (decision D1): memory N^2, time N solves; selected inversion when a
    // model with many groups is measured.
    if (!have_hi_) {
      Hi_.assign(static_cast<std::size_t>(N) * N, 0.0);
      std::vector<double> b(N);
      for (int j = 0; j < N; ++j) {
        if (held_[j]) continue;
        std::fill(b.begin(), b.end(), 0.0);
        b[j] = 1.0;
        ldlt_.solve(b.data());
        for (int i = 0; i < N; ++i) Hi_[i + static_cast<std::size_t>(N) * j] = held_[i] ? 0.0 : b[i];
      }
      have_hi_ = true;
    }
    auto hi = [&](int i, int j) { return Hi_[i + static_cast<std::size_t>(N) * j]; };
    auto hi_times = [&](const std::vector<double>& w) {
      std::vector<double> o(N, 0.0);
      for (int j = 0; j < N; ++j)
        if (w[j] != 0.0)
          for (int i = 0; i < N; ++i) o[i] += Hi_[i + static_cast<std::size_t>(N) * j] * w[j];
      return o;
    };
    // the full Hessian and gradient of F at x^ (held coordinates included)
    const row_assembly& as = inner_.last_assembly();
    std::vector<double> Hf(static_cast<std::size_t>(N) * N, 0.0), gf = as.gradient;
    for (int q = 0; q < layout_.size(); ++q) {
      int r, c;
      arrowhead_entry(q, r, c);
      Hf[r + static_cast<std::size_t>(N) * c] = Hf[c + static_cast<std::size_t>(N) * r] = as.information[q];
    }
    // per hyperparameter: s_k, dF_k, tr(Hi E_k), sum n_t r_t
    std::vector<std::vector<double>> s(K, std::vector<double>(N, 0.0));
    std::vector<double> dF(K, 0.0), trE(K, 0.0), nr(K, 0.0);
    for (std::size_t t = 0; t < pen.size(); ++t) {
      const auto& p = pen[t];
      const int mm = p.block_size, k = p.hyperparameter;
      for (int b0 = p.offset; b0 < p.offset + p.dim; b0 += mm)
        for (int j = 0; j < mm; ++j)
          for (int l = 0; l < mm; ++l) {
            const double e = scale[t] * p.matrix[j + mm * l];
            Hf[(b0 + j) + static_cast<std::size_t>(N) * (b0 + l)] += e;
            gf[b0 + j] += e * xhat_[b0 + l];
            if (k < 0) continue;
            s[k][b0 + j] += e * xhat_[b0 + l];
            dF[k] += 0.5 * xhat_[b0 + j] * e * xhat_[b0 + l];
            trE[k] += hi(b0 + j, b0 + l) * e;
          }
      if (k >= 0) nr[k] += n_blocks(p) * rank_[t];
    }
    // the directions v_a
    std::vector<std::vector<double>> v(A);
    for (int a = 0; a < A; ++a) {
      if (a < K) {
        v[a] = hi_times(s[a]);
        for (double& z : v[a]) z = -z;
      } else {
        const int c = psi_[a - K];
        std::vector<double> col(Hf.begin() + static_cast<std::size_t>(N) * c, Hf.begin() + static_cast<std::size_t>(N) * (c + 1));
        v[a] = hi_times(col);
        for (double& z : v[a]) z = -z;
        v[a][c] += 1.0;
      }
    }
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    // the pass over the units
    t0 = clock::now();
    dev_matrix Vd("laplace::engine::V", N, level == 2 ? A : 1);
    if (!hid_current_) {
      if (static_cast<int>(Hid_.extent(0)) != N) Hid_ = dev_matrix("laplace::engine::Hi", N, N);
      Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(Hi_.data(), N, N);
      Kokkos::deep_copy(Hid_, h);
      hid_current_ = true;
    }
    {
      auto hv = Kokkos::create_mirror_view(Vd);
      for (int a = 0; level == 2 && a < A; ++a)
        for (int i = 0; i < N; ++i) hv(i, a) = v[a][i];
      Kokkos::deep_copy(Vd, hv);
    }
    row_model_view::dview xd("laplace::engine::xhat", N);
    {
      auto h = Kokkos::create_mirror_view(xd);
      for (int j = 0; j < N; ++j) h(j) = xhat_[j];
      Kokkos::deep_copy(xd, h);
    }
    std::vector<double> pass;
    assemble_outer_row_model<List>(*model_, level == 2 ? 4 : 3, xd, Hid_, Vd, level == 2 ? A : 0, pass);
    ++counts_.outer_passes[level == 2 ? 1 : 0];
    counts_.seconds_passes += std::chrono::duration<double>(clock::now() - t0).count();
    t0 = clock::now();
    const outer_pass_layout lo{N, level == 2 ? A : 0, layout_.size()};
    const double* tau = pass.data() + lo.tau();
    auto dot = [&](const double* a, const std::vector<double>& b) {
      double z = 0.0;
      for (int i = 0; i < N; ++i) z += a[i] * b[i];
      return z;
    };
    grad_.assign(A, 0.0);
    for (int a = 0; a < A; ++a) {
      const double tv = dot(tau, v[a]);
      grad_[a] = a < K ? dF[a] + 0.5 * (trE[a] + tv) - 0.5 * nr[a] : gf[psi_[a - K]] + 0.5 * tv;
    }
    if (level == 2) hessian(pass, lo, Hf, s, dF, trE, v);
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
  }

  void hessian(const std::vector<double>& pass, const outer_pass_layout& lo, const std::vector<double>& Hf,
               const std::vector<std::vector<double>>& s, const std::vector<double>& dF, const std::vector<double>& trE,
               const std::vector<std::vector<double>>& v) {
    const int N = n_, A = n_outer(), K = n_hyper_;
    const auto& pen = inner_.penalties();
    const auto& scale = inner_.penalty_scales();
    auto hi = [&](int i, int j) { return Hi_[i + static_cast<std::size_t>(N) * j]; };
    // E_k w on the blocks of the terms of hyperparameter k
    auto E_times = [&](int k, const std::vector<double>& w) {
      std::vector<double> o(N, 0.0);
      for (std::size_t t = 0; t < pen.size(); ++t) {
        const auto& p = pen[t];
        if (p.hyperparameter != k) continue;
        const int mm = p.block_size;
        for (int b0 = p.offset; b0 < p.offset + p.dim; b0 += mm)
          for (int j = 0; j < mm; ++j)
            for (int l = 0; l < mm; ++l) o[b0 + j] += scale[t] * p.matrix[j + mm * l] * w[b0 + l];
      }
      return o;
    };
    // dH_a in compressed columns (both triangles), for X_a = Hi dH_a and its traces on the device
    // ponytail: A dense N x N products (A N^2 memory, N nnz(dH) time); the arrowhead block formula
    // when a model with many groups is measured (decision D1).
    std::vector<int> cp(static_cast<std::size_t>(A) * (N + 1), 0), rows;
    std::vector<double> vals;
    {
      std::vector<std::vector<std::pair<int, double>>> col(N);
      for (int a = 0; a < A; ++a) {
        for (auto& c : col) c.clear();
        const double* T = pass.data() + lo.third_direction(a);
        for (int q = 0; q < layout_.size(); ++q) {
          if (T[q] == 0.0) continue;
          int r, c;
          arrowhead_entry(q, r, c);
          col[c].push_back({r, T[q]});
          if (r != c) col[r].push_back({c, T[q]});
        }
        if (a < K)
          for (std::size_t t = 0; t < pen.size(); ++t) {
            const auto& p = pen[t];
            if (p.hyperparameter != a) continue;
            const int mm = p.block_size;
            for (int b0 = p.offset; b0 < p.offset + p.dim; b0 += mm)
              for (int j = 0; j < mm; ++j)
                for (int l = 0; l < mm; ++l) col[b0 + l].push_back({b0 + j, scale[t] * p.matrix[j + mm * l]});
          }
        int* cpa = cp.data() + static_cast<std::size_t>(a) * (N + 1);
        cpa[0] = static_cast<int>(rows.size());
        for (int j = 0; j < N; ++j) {
          for (const auto& e : col[j]) rows.push_back(e.first), vals.push_back(e.second);
          cpa[j + 1] = static_cast<int>(rows.size());
        }
      }
    }
    std::vector<double> trxx_all;
    detail_marginal::trace_products(Hid_, N, A, cp, rows, vals, trxx_all);
    const double* tau = pass.data() + lo.tau();
    std::vector<double> ht(N, 0.0);  // Hi tau
    for (int j = 0; j < N; ++j)
      if (tau[j] != 0.0)
        for (int i = 0; i < N; ++i) ht[i] += hi(i, j) * tau[j];
    std::vector<std::vector<double>> Hv(A, std::vector<double>(N, 0.0));  // H_full v_b
    for (int b = 0; b < A; ++b)
      for (int j = 0; j < N; ++j)
        if (v[b][j] != 0.0)
          for (int i = 0; i < N; ++i) Hv[b][i] += Hf[i + static_cast<std::size_t>(N) * j] * v[b][j];
    hess_.assign(static_cast<std::size_t>(A) * A, 0.0);
    int ab = 0;
    for (int a = 0; a < A; ++a)
      for (int b = a; b < A; ++b, ++ab) {
        // F2
        double f2 = (a == b && a < K) ? dF[a] : 0.0;
        for (int i = 0; i < N; ++i) {
          f2 += v[a][i] * Hv[b][i];
          if (a < K) f2 += s[a][i] * v[b][i];
          if (b < K) f2 += s[b][i] * v[a][i];
        }
        const double trxx = trxx_all[ab];  // tr(Hi dH_a Hi dH_b)
        // x^_ab
        std::vector<double> rhs(pass.begin() + lo.third_pair(ab), pass.begin() + lo.third_pair(ab) + N);
        if (b < K) {
          const auto e = E_times(b, v[a]);
          for (int i = 0; i < N; ++i) rhs[i] += e[i];
        }
        if (a < K) {
          const auto e = E_times(a, v[b]);
          for (int i = 0; i < N; ++i) rhs[i] += e[i];
          if (a == b)
            for (int i = 0; i < N; ++i) rhs[i] += s[a][i];
        }
        double tx = 0.0;  // tau' x^_ab = -(Hi tau)' rhs
        for (int j = 0; j < N; ++j) tx -= ht[j] * rhs[j];
        const double q = pass[lo.fourth_pair(ab)];
        const double h = f2 + 0.5 * (-trxx + ((a == b && a < K) ? trE[a] : 0.0) + q + tx);
        hess_[a + A * b] = hess_[b + A * a] = h;
      }
  }
};

/**
 * \ingroup engine
 * \brief Result of fit_marginal(): the outer optimizer's report, the counts and the final inner point.
 * \see fit_marginal(), \ref engine
 */
struct marginal_fit_report {
  optimizers::optimizer_report optimizer;
  marginal_criterion_counts counts;
  std::vector<double> inner_point;
};

/**
 * \ingroup engine
 * \brief Minimizes the Laplace marginal criterion from `u0` with Newton's method on its exact Hessian.
 * \tparam Criterion Marginal criterion type (typically laplace_marginal_criterion).
 * \tparam Newton Outer Newton optimizer type (defaults to `optimizers::newton_optimizer<>`).
 * \param criterion The criterion (its counts accumulate).
 * \param u0 Start of the outer vector.
 * \param newton Options of the outer Newton optimizer.
 * \return marginal_fit_report.
 * \see laplace_marginal_criterion, \ref engine
 */
template <class Criterion, class Newton = optimizers::newton_optimizer<>>
marginal_fit_report fit_marginal(Criterion& criterion, const optimizers::optimizer_point& u0, const Newton& newton = {}) {
  marginal_fit_report r;
  r.optimizer = optimizers::minimize(newton, criterion, u0);
  r.counts = criterion.counts();
  r.inner_point = criterion.inner_point();
  return r;
}

}  // namespace laplace::engine
