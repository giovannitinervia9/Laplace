// The log-Cholesky bijector between the symmetric positive definite p x p
// matrices and R^(p(p+1)/2) (note/bijectors.md section 4, decisions 17-19;
// interface in coupled.hpp).
//
//   forward   Sigma = L L^T (Cholesky), u_ii = log L_ii, u_ij = L_ij (i > j)
//   inverse   L_ii = exp(u_ii), L_ij = u_ij, Sigma = L L^T
//   log |det J_h(u)| = p log 2 + sum_i (p - i + 1) u_ii   (0-based i)
//
// u and Sigma are packed lower triangles by rows (n = m = p(p+1)/2). The
// reference coordinates of Sigma are its lower triangle with the diagonal;
// moving the coordinate (i, j), i > j, moves Sigma_ij and Sigma_ji together.
//
// Derivatives:
//   inverse, closed: Sigma_ij = sum_(l<=j) L_il L_jl, each term a product of
//   two functions of one variable (exp on the diagonal, x off it), so a
//   tuple is nonzero only on at most two variables of one column; with
//   one variable (r, c) it reaches Sigma_rr and Sigma_(r,q) for every row
//   q >= c, with two (r1, c), (r2, c) only Sigma_(r1,r2).
//   inverse log-Jacobian: linear, nonzero at order 1 on the diagonal only.
//   forward: the Cholesky factor's derivatives over the sub-tuples
//   (cholesky_subtuple in coupled.hpp), then log on the diagonal.
//   forward log-Jacobian: -(p log 2 + sum_i (p - i + 1) log L_ii).
#pragma once

#include <laplace/bijector/coupled.hpp>

namespace laplace::bijector {

/**
 * \ingroup bijector_coupled
 * \brief The log-Cholesky bijector between the \f$p \times p\f$ positive definite matrices and \f$\mathbb{R}^{p(p+1)/2}\f$.
 *
 * Maps a symmetric positive definite matrix \f$\Sigma\f$ stored as a packed lower triangle to an unconstrained
 * vector \f$u\f$ of \f$p(p+1)/2\f$ free coordinates via the Cholesky factor \f$\Sigma = L L^\top\f$,
 * setting \f$u_{ii} = \log L_{ii}\f$ on the diagonal and \f$u_{ij} = L_{ij}\f$ for \f$i > j\f$.
 *
 * Reference coordinates for forward derivatives and log-Jacobians are the lower triangle of \f$\Sigma\f$
 * with its diagonal.
 *
 * \see cholesky_bijector_orders, bijector_coupled
 */
struct log_cholesky_bijector : cholesky_bijector_orders<log_cholesky_bijector> {
  /**
   * \ingroup bijector_coupled
   * \brief Matrix dimension \f$p \ge 1\f$.
   */
  int dim = 1;  // p

  /**
   * \ingroup bijector_coupled
   * \brief Default constructor, initializing with dimension \f$p = 1\f$.
   */
  LAPLACE_HD log_cholesky_bijector() = default;

  /**
   * \ingroup bijector_coupled
   * \brief Constructs a log-Cholesky bijector for \f$p \times p\f$ matrices.
   *
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit log_cholesky_bijector(int p) : dim(p) {}

  /**
   * \ingroup bijector_coupled
   * \brief Number of free unconstrained coordinates, \f$n = p(p+1)/2\f$.
   *
   * \return Length of free vector \f$u\f$.
   */
  LAPLACE_HD int free_size() const { return dim * (dim + 1) / 2; }

  /**
   * \ingroup bijector_coupled
   * \brief Number of stored constrained coordinates, \f$m = p(p+1)/2\f$.
   *
   * \return Number of packed elements in \f$\Sigma\f$.
   */
  LAPLACE_HD int value_size() const { return dim * (dim + 1) / 2; }

  /**
   * \ingroup bijector_coupled
   * \brief Required capacity of caller scratch buffer in doubles.
   *
   * \return Number of doubles required by scratch buffer \c w.
   */
  // the factor's derivatives over the 64 sub-tuples of order 6, plus log L_ii
  LAPLACE_HD int scratch_size() const { return (free_size() + 1) * detail_coupled::max_masks; }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the lower Cholesky factor \f$L(u)\f$ with positive diagonal entries.
   *
   * Computes \f$L_{ii} = \exp(u_{ii})\f$ on the diagonal and \f$L_{ij} = u_{ij}\f$ for \f$i > j\f$.
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param l Output packed lower triangle of \f$L\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   *
   * Measured accuracy: within 0.44 ulpN on gcc and CUDA.
   * \see inverse(), inverse_cholesky_dk()
   */
  LAPLACE_HD void inverse_cholesky(const double* u, double* l, [[maybe_unused]] double* w) const {
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) {
        const int e = detail_coupled::packed(i, j);
        l[e] = i == j ? detail_coupled::m::exp(u[e]) : u[e];
      }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse map \f$\Sigma = h(u) = L L^\top\f$.
   *
   * Computes the packed lower triangle of \f$\Sigma\f$ from \f$L(u)\f$:
   * \f[
   * \Sigma_{ij} = \sum_{l \le j} L_{il} L_{jl}
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param v Output packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 1.03 ulpN on gcc / 0.95 ulpN on CUDA.
   * \see forward(), inverse_cholesky(), inverse_log_jacobian(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* u, double* v, double* w) const {
    inverse_cholesky(u, w, w);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int l = 0; l <= j; ++l) s += w[detail_coupled::packed(i, l)] * w[detail_coupled::packed(j, l)];
        v[detail_coupled::packed(i, j)] = s;
      }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse log-Jacobian \f$\log|\det J_h(u)|\f$.
   *
   * Computes
   * \f[
   * \log\lvert\det J_h(u)\rvert = p \log 2 + \sum_{i=0}^{p-1} (p - i + 1) u_{ii}
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return Value of inverse log-Jacobian.
   *
   * Measured accuracy: within 0.90 ulp on gcc and CUDA.
   * \see inverse(), forward_log_jacobian(), inverse_log_jacobian_dk()
   */
  LAPLACE_HD double inverse_log_jacobian(const double* u, [[maybe_unused]] double* w) const {
    double s = dim * detail_scalar::log_2;
    for (int i = 0; i < dim; ++i) s += (dim - i + 1) * u[detail_coupled::packed(i, i)];
    return s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward map \f$u = g(v)\f$.
   *
   * Computes the Cholesky factor \f$\Sigma = L L^\top\f$ via sub-tuple evaluation, setting
   * \f$u_{ii} = \log L_{ii}\f$ on the diagonal and \f$u_{ij} = L_{ij}\f$ for \f$i > j\f$.
   *
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param u Output free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 1984 ulpN on gcc / 1982 ulpN on CUDA.
   * \see inverse(), forward_log_jacobian(), forward_dk()
   */
  LAPLACE_HD void forward(const double* v, double* u, double* w) const {
    detail_coupled::cholesky_subtuple<0>(dim, v, nullptr, false, w);
    const int M = detail_coupled::max_masks;
    for (int e = 0; e < free_size(); ++e) u[e] = w[e * M];
    for (int i = 0; i < dim; ++i) {
      const int e = detail_coupled::packed(i, i);
      u[e] = detail_coupled::m::log(u[e]);
    }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward log-Jacobian \f$\log|\det J_g(v)|\f$ in reference coordinates.
   *
   * Computes
   * \f[
   * \log\lvert\det J_g(v)\rvert = -\left(p \log 2 + \sum_{i=0}^{p-1} (p - i + 1) \log L_{ii}\right)
   * \f]
   *
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of forward log-Jacobian.
   *
   * Measured accuracy: within 1984 ulp on gcc / 1982 ulp on CUDA.
   * \see forward(), inverse_log_jacobian(), forward_log_jacobian_dk()
   */
  LAPLACE_HD double forward_log_jacobian(const double* v, double* w) const {
    detail_coupled::cholesky_subtuple<0>(dim, v, nullptr, false, w);
    double s = dim * detail_scalar::log_2;
    for (int i = 0; i < dim; ++i)
      s += (dim - i + 1) * detail_coupled::m::log(w[detail_coupled::packed(i, i) * detail_coupled::max_masks]);
    return -s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Evaluates the closed-form derivative of \f$\Sigma = L L^\top\f$. Returns \c false and leaves \c out
   * untouched when the tuple contains variables in more than one column or more than two distinct variables,
   * or exceeds polynomial degrees.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * Measured accuracy: within 1.03 ulpN on gcc / 0.95 ulpN on CUDA.
   * \see inverse(), inverse_log_jacobian_dk(), inverse_cholesky_dk(), inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* u, const int (&idx)[K], double* out, [[maybe_unused]] double* w) const {
    int r1, c1, r2 = -1, c2 = -1, m1 = 0, m2 = 0;
    detail_coupled::unpack(idx[0], r1, c1);
    for (int q = 0; q < K; ++q) {
      if (idx[q] == idx[0]) ++m1;
      else if (r2 < 0 || idx[q] == detail_coupled::packed(r2, c2)) {
        detail_coupled::unpack(idx[q], r2, c2);
        ++m2;
      }
      else return false;  // three distinct variables
    }
    if (m2 > 0) {
      // two variables of one column, each through exp (diagonal) or x (order 1 only)
      if (c1 != c2 || (r1 != c1 && m1 > 1) || (r2 != c2 && m2 > 1)) return false;
      zero(out);
      const double f1 = r1 == c1 ? detail_coupled::m::exp(u[idx[0]]) : 1.0;
      const double f2 = r2 == c2 ? detail_coupled::m::exp(u[detail_coupled::packed(r2, c2)]) : 1.0;
      out[detail_coupled::packed(r2 > r1 ? r2 : r1, r2 > r1 ? r1 : r2)] = f1 * f2;
      return true;
    }
    // one variable x = (r, c) of multiplicity K: L_rc^2 in Sigma_rr, L_rc L_qc in Sigma_(r,q)
    const bool diag = r1 == c1;
    if (!diag && K > 2) return false;
    zero(out);
    const double x = u[idx[0]];
    const double ex = diag ? detail_coupled::m::exp(x) : 0.0;
    // d^K of the square: diagonal 2^K e^(2x), off it 2x then 2
    out[detail_coupled::packed(r1, r1)] = diag ? detail_scalar::ipow<K>(2.0) * ex * ex : (K == 1 ? 2.0 * x : 2.0);
    if (diag || K == 1) {
      const double f = diag ? ex : 1.0;  // d^K of L_rc
      for (int q = c1; q < dim; ++q) {
        if (q == r1) continue;
        const int lq = detail_coupled::packed(q, c1);
        const double Lq = q == c1 ? detail_coupled::m::exp(u[lq]) : u[lq];
        out[detail_coupled::packed(q > r1 ? q : r1, q > r1 ? r1 : q)] = f * Lq;
      }
    }
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse log-Jacobian partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * The inverse log-Jacobian is linear: returns \f$p - i + 1\f$ for order 1 on a diagonal coordinate \f$(i, i)\f$,
   * and \f$0.0\f$ for off-diagonal coordinates and all orders \f$K \ge 2\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ (unused).
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param w Caller scratch buffer (unused).
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 0.90 ulp on gcc and CUDA (exact beyond order 0).
   * \see inverse_log_jacobian(), inverse_dk(), inverse_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double inverse_log_jacobian_dk([[maybe_unused]] const double* u, const int (&idx)[K], [[maybe_unused]] double* w) const {
    if constexpr (K > 1) return 0.0;
    int i, j;
    detail_coupled::unpack(idx[0], i, j);
    return i == j ? double(dim - i + 1) : 0.0;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates Cholesky factor partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Evaluates the derivative of \f$L(u)\f$. Returns \c false and leaves \c l untouched unless the tuple is pure
   * (all indices equal) with \f$K = 1\f$ for off-diagonal coordinates.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param l Output packed lower triangle of \f$L\f$ of at least \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * Measured accuracy: within 0.44 ulpN on gcc and CUDA.
   * \see inverse_cholesky(), inverse_dk(), inverse_cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_cholesky_dk(const double* u, const int (&idx)[K], double* l, [[maybe_unused]] double* w) const {
    if (idx[0] != idx[K - 1]) return false;
    int i, j;
    detail_coupled::unpack(idx[0], i, j);
    if (i != j && K > 1) return false;
    zero(l);
    l[idx[0]] = i == j ? detail_coupled::m::exp(u[idx[0]]) : 1.0;
    return true;
  }

  // the forward derivatives are never structurally zero: L_(p-1,p-1) depends on every entry
  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx.
   *
   * Differentiates the Cholesky factor of \f$\Sigma = L L^\top\f$ over the subsets of the tuple's positions,
   * then the logarithm of the diagonal entries.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true (forward derivatives are never structurally zero).
   *
   * Measured accuracy: within 1984 ulpN on gcc / 1982 ulpN on CUDA.
   * \see forward(), forward_log_jacobian_dk(), forward_d1()
   */
  template <int K>
  LAPLACE_HD bool forward_dk(const double* v, const int (&idx)[K], double* out, double* w) const {
    constexpr int full = (1 << K) - 1, M = detail_coupled::max_masks;
    detail_coupled::cholesky_subtuple<K>(dim, v, idx, false, w);
    double* y = w + free_size() * M;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) {
        const int e = detail_coupled::packed(i, j);
        if (i != j) {
          out[e] = w[e * M + full];
          continue;
        }
        detail_coupled::log_subtuple<K>(w + e * M, y);
        out[e] = y[full];
      }
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward log-Jacobian partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx.
   *
   * Computes
   * \f[
   * \frac{\partial^K \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\cdots\partial v_{i_K}} = -\sum_{i=0}^{p-1} (p - i + 1) y_i[S_{\mathrm{full}}]
   * \f]
   * where \f$y_i = \log L_{ii}\f$ over the sub-tuples.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 1984 ulp on gcc / 1982 ulp on CUDA.
   * \see forward_log_jacobian(), forward_dk(), forward_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double forward_log_jacobian_dk(const double* v, const int (&idx)[K], double* w) const {
    constexpr int full = (1 << K) - 1, M = detail_coupled::max_masks;
    detail_coupled::cholesky_subtuple<K>(dim, v, idx, false, w);
    double* y = w + free_size() * M;
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
      detail_coupled::log_subtuple<K>(w + detail_coupled::packed(i, i) * M, y);
      s -= (dim - i + 1) * y[full];
    }
    return s;
  }

 private:
  LAPLACE_HD void zero(double* out) const {
    for (int e = 0; e < value_size(); ++e) out[e] = 0.0;
  }
};

}  // namespace laplace::bijector
