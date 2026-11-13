// The partial correlation bijector between the p x p correlation matrices and
// R^(p(p-1)/2), through the canonical partial correlations z_ab = tanh u_ab
// (the onion / LKJ construction; note/bijectors.md section 4, decisions
// 17-19; interface in coupled.hpp).
//
//   inverse   L_ab = tanh(u_ab) prod_(l<b) sech(u_al)  (a > b),
//             L_aa = prod_(l<a) sech(u_al), R = L L^T
//   forward   R = L L^T (Cholesky, unit diagonal),
//             u_ab = asinh(L_ab / Q_ab), Q_ab^2 = sum_(l=b+1..a) L_al^2,
//             which is atanh(z_ab) without forming 1 - z^2
//   log |det J_h(u)| = -sum_(a>b) (p - b) log cosh u_ab   (0-based b)
//
// u is a strict lower triangle by rows (n = p(p-1)/2, (a, b) at
// a(a-1)/2 + b); R is stored as a packed lower triangle with its unit
// diagonal (m = p(p+1)/2). The reference coordinates of R are its strict
// lower triangle (moving (a, b) moves R_ab and R_ba together); the forward
// derivative indices are strict positions.
//
// Derivatives:
//   inverse, closed: each L_ab is a product of functions of one variable of
//   row a, so a derivative is a product of tanh^(k) and sech^(k); R_ij =
//   sum_(l<=j) L_il L_jl is nonzero only for tuples on at most two rows, and
//   on two rows a1 < a2 only R_(a2,a1), when the columns of row a2 in the
//   tuple are <= a1.
//   inverse log-Jacobian: separable, -(p - b) tanh^(k-1)(u_ab) on a tuple of
//   one variable, 0 on mixed tuples.
//   forward: the Cholesky factor's derivatives over the sub-tuples
//   (cholesky_subtuple in coupled.hpp), then Q, w = L/Q, r = sqrt(1 + w^2)
//   = cosh u and u from r du = dw, all over the sub-tuples.
//   forward log-Jacobian: sum (p - b) log r_ab.
#pragma once

#include <laplace/bijector/coupled.hpp>

namespace laplace::bijector {

/**
 * \ingroup bijector_coupled
 * \brief The partial correlation bijector between the \f$p \times p\f$ correlation matrices and \f$\mathbb{R}^{p(p-1)/2}\f$.
 *
 * Maps a symmetric positive definite \f$p \times p\f$ correlation matrix \f$R\f$ (\f$R_{ii} = 1\f$) to an
 * unconstrained vector \f$u\f$ of \f$p(p-1)/2\f$ free coordinates via canonical partial correlations
 * \f$z_{ab} = \tanh(u_{ab})\f$ using the onion construction.
 *
 * The free vector \f$u\f$ is stored as a strict lower triangle by rows at index \f$a(a-1)/2 + b\f$.
 * Reference coordinates for forward derivatives and log-Jacobians are the strict lower triangle of \f$R\f$.
 *
 * \see cholesky_bijector_orders, bijector_coupled
 */
struct partial_corr_bijector : cholesky_bijector_orders<partial_corr_bijector> {
  /**
   * \ingroup bijector_coupled
   * \brief Matrix dimension \f$p \ge 1\f$.
   */
  int dim = 1;  // p

  /**
   * \ingroup bijector_coupled
   * \brief Default constructor, initializing with dimension \f$p = 1\f$.
   */
  LAPLACE_HD partial_corr_bijector() = default;

  /**
   * \ingroup bijector_coupled
   * \brief Constructs a partial correlation bijector for \f$p \times p\f$ correlation matrices.
   *
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit partial_corr_bijector(int p) : dim(p) {}

  /**
   * \ingroup bijector_coupled
   * \brief Number of free unconstrained coordinates, \f$n = p(p-1)/2\f$.
   *
   * \return Length of free strict lower-triangular vector \f$u\f$.
   */
  LAPLACE_HD int free_size() const { return dim * (dim - 1) / 2; }

  /**
   * \ingroup bijector_coupled
   * \brief Number of stored constrained coordinates, \f$m = p(p+1)/2\f$.
   *
   * \return Number of packed elements in \f$R\f$ (including the unit diagonal).
   */
  LAPLACE_HD int value_size() const { return dim * (dim + 1) / 2; }

  /**
   * \ingroup bijector_coupled
   * \brief Required capacity of caller scratch buffer in doubles.
   *
   * \return Number of doubles required by scratch buffer \c w.
   */
  // the factor's derivatives over the 64 sub-tuples of order 6, plus Q, w, r, u
  LAPLACE_HD int scratch_size() const { return (value_size() + 4) * detail_coupled::max_masks; }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the lower Cholesky factor \f$L(u)\f$ of correlation matrix \f$R\f$.
   *
   * Computes
   * \f[
   * L_{ab} = \tanh(u_{ab}) \prod_{l<b} \operatorname{sech}(u_{al}) \quad (a > b), \quad
   * L_{aa} = \prod_{l<a} \operatorname{sech}(u_{al})
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param l Output packed lower triangle of \f$L\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   *
   * Measured accuracy: within 6.2 ulpN on gcc / 4.0 ulpN on CUDA.
   * \see inverse(), inverse_cholesky_dk()
   */
  LAPLACE_HD void inverse_cholesky(const double* u, double* l, [[maybe_unused]] double* w) const {
    for (int a = 0; a < dim; ++a) {
      double pr = 1.0;
      for (int b = 0; b < a; ++b) {
        const double x = u[detail_coupled::strict(a, b)];
        l[detail_coupled::packed(a, b)] = detail_coupled::tanh_d(0, x) * pr;
        pr *= detail_coupled::sech_d(0, x);
      }
      l[detail_coupled::packed(a, a)] = pr;
    }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse map \f$R = h(u) = L L^\top\f$ with exact unit diagonal.
   *
   * Computes \f$R = L L^\top\f$ from \f$L(u)\f$, setting \f$R_{ii} = 1.0\f$ exactly:
   * \f[
   * R_{ij} = \sum_{l \le j} L_{il} L_{jl} \quad (i > j), \quad R_{ii} = 1
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param v Output packed lower triangle of \f$R\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 2839 ulpN on gcc (K=5) / 366 ulpN on CUDA (K=5).
   * \see forward(), inverse_cholesky(), inverse_log_jacobian(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* u, double* v, double* w) const {
    inverse_cholesky(u, w, w);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < i; ++j) {
        double s = 0.0;
        for (int l = 0; l <= j; ++l) s += w[detail_coupled::packed(i, l)] * w[detail_coupled::packed(j, l)];
        v[detail_coupled::packed(i, j)] = s;
      }
      v[detail_coupled::packed(i, i)] = 1.0;
    }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse log-Jacobian \f$\log|\det J_h(u)|\f$.
   *
   * Computes
   * \f[
   * \log\lvert\det J_h(u)\rvert = -\sum_{a>b} (p - b) \log \cosh u_{ab}
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return Value of inverse log-Jacobian.
   *
   * Measured accuracy: within 2839 ulp on gcc (K=6) / 1959 ulp on CUDA (K=5).
   * \see inverse(), forward_log_jacobian(), inverse_log_jacobian_dk()
   */
  LAPLACE_HD double inverse_log_jacobian(const double* u, [[maybe_unused]] double* w) const {
    double s = 0.0;
    for (int a = 0; a < dim; ++a)
      for (int b = 0; b < a; ++b) s -= (dim - b) * detail_scalar::log_cosh(u[detail_coupled::strict(a, b)]);
    return s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward map \f$u = g(v)\f$.
   *
   * Computes the unit-diagonal Cholesky factor \f$R = L L^\top\f$, and for each \f$a > b\f$:
   * \f[
   * Q_{ab}^2 = \sum_{l=b+1}^a L_{al}^2, \quad u_{ab} = \operatorname{asinh}\left(\frac{L_{ab}}{Q_{ab}}\right)
   * \f]
   *
   * \param v Input packed lower triangle of \f$R\f$ of \c value_size() doubles.
   * \param u Output free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 7305 ulpN on gcc / 7304 ulpN on CUDA.
   * \see inverse(), forward_log_jacobian(), forward_dk()
   */
  LAPLACE_HD void forward(const double* v, double* u, double* w) const {
    detail_coupled::cholesky_subtuple<0>(dim, v, nullptr, true, w);
    const int M = detail_coupled::max_masks;
    for (int a = 1; a < dim; ++a) {
      double q2 = 0.0;
      for (int b = a - 1; b >= 0; --b) {
        const double lb1 = w[detail_coupled::packed(a, b + 1) * M];
        q2 += lb1 * lb1;
        u[detail_coupled::strict(a, b)] = Kokkos::asinh(w[detail_coupled::packed(a, b) * M] / Kokkos::sqrt(q2));
      }
    }
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward log-Jacobian \f$\log|\det J_g(v)|\f$ in strict lower reference coordinates.
   *
   * Computes
   * \f[
   * \log\lvert\det J_g(v)\rvert = \sum_{a>b} (p - b) \log r_{ab}
   * \f]
   * where \f$r_{ab} = \sqrt{1 + (L_{ab}/Q_{ab})^2}\f$.
   *
   * \param v Input packed lower triangle of \f$R\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of forward log-Jacobian.
   *
   * Measured accuracy: within 7356 ulp on gcc / 7361 ulp on CUDA.
   * \see forward(), inverse_log_jacobian(), forward_log_jacobian_dk()
   */
  LAPLACE_HD double forward_log_jacobian(const double* v, double* w) const {
    detail_coupled::cholesky_subtuple<0>(dim, v, nullptr, true, w);
    const int M = detail_coupled::max_masks;
    double s = 0.0;
    for (int a = 1; a < dim; ++a) {
      double q2 = 0.0;
      for (int b = a - 1; b >= 0; --b) {
        const double lb1 = w[detail_coupled::packed(a, b + 1) * M];
        q2 += lb1 * lb1;
        s += 0.5 * (dim - b) * detail_scalar::log1p_sq(w[detail_coupled::packed(a, b) * M] / Kokkos::sqrt(q2));
      }
    }
    return s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Closed-form derivative of \f$R = L L^\top\f$. Returns \c false and leaves \c out untouched when
   * tuple coordinates span more than two rows or violate row/column constraints.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * Measured accuracy: within 2839 ulpN on gcc (K=5) / 366 ulpN on CUDA (K=5).
   * \see inverse(), inverse_cholesky_dk(), inverse_log_jacobian_dk(), inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* u, const int (&idx)[K], double* out, double* w) const {
    int rows[2], nr = 0;
    if (!tuple_rows<K>(idx, rows, nr)) return false;
    double* L = w;
    double* d1 = w + value_size();
    double* d2 = d1 + dim;
    if (nr == 2 && max_col<K>(idx, rows[1]) > rows[0]) return false;
    inverse_cholesky(u, L, w);
    row_derivative<K>(u, idx, rows[0], d1);
    for (int e = 0; e < value_size(); ++e) out[e] = 0.0;
    if (nr == 2) {
      row_derivative<K>(u, idx, rows[1], d2);
      double s = 0.0;
      for (int l = 0; l <= rows[0]; ++l) s += d2[l] * d1[l];
      out[detail_coupled::packed(rows[1], rows[0])] = s;
      return true;
    }
    const int a = rows[0];
    for (int j = 0; j < dim; ++j) {
      if (j == a) continue;
      const int lo = j < a ? j : a;
      double s = 0.0;
      for (int l = 0; l <= lo; ++l) s += d1[l] * L[detail_coupled::packed(j, l)];
      out[detail_coupled::packed(j > a ? j : a, j > a ? a : j)] = s;
    }
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates Cholesky factor partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Computes derivatives of \f$L(u)\f$. Returns \c false and leaves \c l untouched when tuple coordinates
   * span more than one row.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param l Output packed lower triangle of \f$L\f$ of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * Measured accuracy: within 6.2 ulpN on gcc / 4.0 ulpN on CUDA.
   * \see inverse_cholesky(), inverse_dk(), inverse_cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_cholesky_dk(const double* u, const int (&idx)[K], double* l, double* w) const {
    int rows[2], nr = 0;
    if (!tuple_rows<K>(idx, rows, nr) || nr != 1) return false;
    row_derivative<K>(u, idx, rows[0], w);
    for (int e = 0; e < value_size(); ++e) l[e] = 0.0;
    for (int c = 0; c <= rows[0]; ++c) l[detail_coupled::packed(rows[0], c)] = w[c];
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse log-Jacobian partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * The inverse log-Jacobian is separable: computes \f$-(p - b) \tanh^{(K-1)}(u_{ab})\f$ for pure tuples
   * on coordinate \f$(a, b)\f$, and returns \f$0.0\f$ for any mixed tuple.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param w Caller scratch buffer (unused).
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 2839 ulp on gcc (K=6) / 1959 ulp on CUDA (K=5).
   * \see inverse_log_jacobian(), inverse_dk(), inverse_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double inverse_log_jacobian_dk(const double* u, const int (&idx)[K], [[maybe_unused]] double* w) const {
    if (idx[0] != idx[K - 1]) return 0.0;
    int a, b;
    detail_coupled::unstrict(idx[0], a, b);
    return -(dim - b) * detail_coupled::tanh_d(K - 1, u[idx[0]]);
  }

  // the forward derivatives are never structurally zero: u_(p-1,0) depends on every entry
  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward partial derivatives of order \f$K\f$ along strict reference tuple \c idx via sub-tuple recursion.
   *
   * Differentiates the unit-diagonal Cholesky factor \f$R = L L^\top\f$, \f$Q\f$, \f$W = L/Q\f$, \f$R = \sqrt{1 + W^2}\f$,
   * and \f$u\f$ over sub-tuples.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$R\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ strict reference coordinate indices.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true (forward derivatives are never structurally zero).
   *
   * Measured accuracy: within 7305 ulpN on gcc / 7304 ulpN on CUDA.
   * \see forward(), forward_log_jacobian_dk(), forward_d1()
   */
  template <int K>
  LAPLACE_HD bool forward_dk(const double* v, const int (&idx)[K], double* out, double* w) const {
    constexpr int full = (1 << K) - 1;
    forward_subtuple<K>(v, idx, w, [&](int a, int b, const double*, const double* uu) {
      out[detail_coupled::strict(a, b)] = uu[full];
    });
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward log-Jacobian partial derivatives of order \f$K\f$ along strict reference tuple \c idx.
   *
   * Computes
   * \f[
   * \frac{\partial^K \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\cdots\partial v_{i_K}} = \sum_{a>b} (p - b) \log r_{ab}[S_{\mathrm{full}}]
   * \f]
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$R\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ strict reference coordinate indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 7356 ulp on gcc / 7361 ulp on CUDA.
   * \see forward_log_jacobian(), forward_dk(), forward_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double forward_log_jacobian_dk(const double* v, const int (&idx)[K], double* w) const {
    constexpr int full = (1 << K) - 1;
    double s = 0.0;
    forward_subtuple<K>(v, idx, w, [&](int, int b, const double* r, const double* y) {
      // y is free again here: log r over the sub-tuples
      double* lr = const_cast<double*>(y);
      detail_coupled::log_subtuple<K>(r, lr);
      s += (dim - b) * lr[full];
    });
    return s;
  }

 private:
  // the distinct rows of the variables of a tuple (sorted, so rows ascend); false beyond two
  template <int K>
  LAPLACE_HD static bool tuple_rows(const int (&idx)[K], int (&rows)[2], int& nr) {
    nr = 0;
    for (int q = 0; q < K; ++q) {
      int a, b;
      detail_coupled::unstrict(idx[q], a, b);
      if (nr > 0 && rows[nr - 1] == a) continue;
      if (nr == 2) return false;
      rows[nr++] = a;
    }
    return true;
  }
  template <int K>
  LAPLACE_HD static int max_col(const int (&idx)[K], int row) {
    int c = -1;
    for (int q = 0; q < K; ++q) {
      int a, b;
      detail_coupled::unstrict(idx[q], a, b);
      if (a == row && b > c) c = b;
    }
    return c;
  }
  // d[l] = derivative of L_(a,l), l = 0..a, along the variables of the tuple in row a
  template <int K>
  LAPLACE_HD void row_derivative(const double* u, const int (&idx)[K], int a, double* d) const {
    int mult[detail_coupled::max_order + 1] = {};  // multiplicity per column; a row of the tuple has <= K columns
    int cols[K], nc = 0;
    for (int q = 0; q < K; ++q) {
      int r, b;
      detail_coupled::unstrict(idx[q], r, b);
      if (r != a) continue;
      if (nc == 0 || cols[nc - 1] != b) cols[nc++] = b;
      ++mult[nc - 1];
    }
    const int top = cols[nc - 1];
    double pr = 1.0;
    int c = 0;
    for (int l = 0; l < a; ++l) {
      const int k = c < nc && cols[c] == l ? mult[c++] : 0;
      const double x = u[detail_coupled::strict(a, l)];
      d[l] = l >= top ? pr * detail_coupled::tanh_d(k, x) : 0.0;
      pr *= detail_coupled::sech_d(k, x);
    }
    d[a] = pr;
  }

  // the forward recursion: Cholesky over the sub-tuples, then for each (a, b)
  // Q, w = L/Q, r = sqrt(1 + w^2) and u (r du = dw); calls f(a, b, r, u)
  template <int K, class F>
  LAPLACE_HD void forward_subtuple(const double* v, const int (&idx)[K], double* w, F&& f) const {
    constexpr int NM = 1 << K, M = detail_coupled::max_masks;
    int var[K];
    for (int q = 0; q < K; ++q) {
      int a, b;
      detail_coupled::unstrict(idx[q], a, b);
      var[q] = detail_coupled::packed(a, b);
    }
    detail_coupled::cholesky_subtuple<K>(dim, v, var, true, w);
    double* Q = w + value_size() * M;
    double* W = Q + M;
    double* R = W + M;
    double* U = R + M;
    for (int a = 1; a < dim; ++a)
      for (int b = 0; b < a; ++b) {
        const double* Lab = w + detail_coupled::packed(a, b) * M;
        for (int S = 0; S < NM; ++S) {
          double q2 = 0.0;
          for (int l = b + 1; l <= a; ++l) {
            const double* x = w + detail_coupled::packed(a, l) * M;
            q2 += detail_coupled::leibniz(x, x, S);
          }
          Q[S] = q2;  // sqrt_solve reads Q[S] before it writes it
          detail_coupled::sqrt_solve(Q, Q, S);
        }
        for (int S = 0; S < NM; ++S) detail_coupled::div_solve(Lab, Q, W, S);
        for (int S = 0; S < NM; ++S) {
          R[S] = (S == 0 ? 1.0 : 0.0) + detail_coupled::leibniz(W, W, S);
          detail_coupled::sqrt_solve(R, R, S);
        }
        U[0] = Kokkos::asinh(W[0]);
        for (int S = 1; S < NM; ++S) detail_coupled::pivot_solve(W, R, U, S);
        f(a, b, R, U);
      }
  }
};

}  // namespace laplace::bijector
