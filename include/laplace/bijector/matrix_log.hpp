// The matrix logarithm bijector between the symmetric positive definite p x p
// matrices and R^(p(p+1)/2) (note/bijectors.md section 4, decisions 17-20;
// interface in coupled.hpp).
//
//   forward   S = log Sigma, u = the lower triangle of S
//   inverse   Sigma = exp(S)
//   log |det J_h(u)| = sum_(i<=j) log f[s_i, s_j],  f[x, y] = (e^x - e^y)/(x - y),
//             f[x, x] = e^x, s_1..s_p the eigenvalues of S; the forward one is
//             minus it at the eigenvalues log sigma_i of Sigma
//
// u, S and Sigma are packed lower triangles by rows (n = m = p(p+1)/2), and the
// reference coordinates of Sigma are its lower triangle with the diagonal, as
// for log_cholesky_bijector.
//
// Values and order 0: a cyclic Jacobi eigendecomposition, S = V diag(s) V^T,
// with f[x, y] = e^max(x, y) exprel(-|x - y|) (laplace::math::exprel).
//
// Derivatives: the truncated Taylor arithmetic of the tuple. Its distinct
// coordinates v (multiplicities m_v) get symbols e_v with e_v^(m_v + 1) = 0;
// an element is one p x p matrix per monomial prod e_v^(t_v), t_v <= m_v,
// indexed in mixed radix, and a product sums A[U] B[T - U] over U <= T
// componentwise (prod_v (m_v + 1)(m_v + 2)/2 matrix products). The derivative
// along the tuple is the top coefficient times prod_v m_v!.
//   inverse: exp(S + sum_v e_v E_v), E_v the symmetric unit direction of v, by
//     scaling and squaring with a Taylor polynomial of degree 12 + K after
//     scaling the constant part to norm <= 1/4.
//   forward: X = log(Sigma + sum_v e_v E_v) by K Newton steps
//     X += L^-1 [Sigma(e) - exp X], L^-1 the inverse derivative of exp at
//     X_0 = log Sigma, which in the eigenbasis divides entry (i, j) by
//     f[s_i, s_j]; each step makes one more order exact.
//   log-Jacobians: log det J(S(e)), J the n x n Jacobian of exp in packed
//     coordinates, column f from exp(S(e) + d E_f) with one more symbol d
//     (d^2 = 0); log det by LU with partial pivoting on the constant parts, in
//     the scalar arithmetic. The forward one is minus that at X(e) = log Sigma(e).
// The cost of a derivative grows like prod_v (m_v + 1)(m_v + 2)/2 times p^3
// (3^K p^3 for K distinct coordinates), the log-Jacobians n times more; the
// Daleckii-Krein sum over the K! orderings and p^(K+1) eigenvalue chains is not
// used (20 ms at p = 4, order 6, and up to 1554 ulp in the prototype).
#pragma once

#include <laplace/bijector/coupled.hpp>
#include <laplace/math/exprel.hpp>

namespace laplace::bijector {

namespace detail_matrix_log {

namespace m = laplace::math;
constexpr int max_vars = detail_coupled::max_order + 1;  // the tuple's coordinates and d
constexpr int max_terms = 2 * detail_coupled::max_masks;  // prod (m_v + 1) with d, order 6

// the monomials of the truncated arithmetic, in mixed radix
struct taylor_index {
  int nv = 0, size = 1;
  int mult[max_vars], radix[max_vars], coord[max_vars];
  template <int K>
  LAPLACE_HD void from_tuple(const int (&idx)[K]) {
    nv = 0;
    for (int q = 0; q < K; ++q)
      if (q == 0 || idx[q] != idx[q - 1]) coord[nv] = idx[q], mult[nv++] = 1;
      else ++mult[nv - 1];
    finish();
  }
  LAPLACE_HD void add(int c, int mlt) {
    coord[nv] = c, mult[nv++] = mlt;
    finish();
  }
  LAPLACE_HD void finish() {
    size = 1;
    for (int v = 0; v < nv; ++v) radix[v] = size, size *= mult[v] + 1;
  }
  LAPLACE_HD int digit(int T, int v) const { return (T / radix[v]) % (mult[v] + 1); }
  // f(U) for every U <= T componentwise (T - U is then the complementary monomial)
  template <class F>
  LAPLACE_HD void below(int T, F&& f) const {
    int u[max_vars] = {}, U = 0;
    for (;;) {
      f(U);
      int v = 0;
      for (; v < nv; ++v) {
        if (u[v] < digit(T, v)) {
          ++u[v], U += radix[v];
          break;
        }
        U -= u[v] * radix[v], u[v] = 0;
      }
      if (v == nv) return;
    }
  }
  LAPLACE_HD double factorials() const {
    double f = 1.0;
    for (int v = 0; v < nv; ++v)
      for (int k = 2; k <= mult[v]; ++k) f *= k;
    return f;
  }
};

LAPLACE_HD int packed(int i, int j) { return detail_coupled::packed(i, j); }

// symmetric unit direction of packed coordinate k, added into a p x p matrix
LAPLACE_HD void add_direction(int p, int k, double* M) {
  int a, b;
  detail_coupled::unpack(k, a, b);
  M[a * p + b] += 1.0;
  if (a != b) M[b * p + a] += 1.0;
}

// cyclic Jacobi: A (p x p, destroyed) = V diag(lam) V^T
LAPLACE_HD void jacobi(int p, double* A, double* V, double* lam) {
  for (int i = 0; i < p * p; ++i) V[i] = 0.0;
  for (int i = 0; i < p; ++i) V[i * p + i] = 1.0;
  for (int sweep = 0; sweep < 64; ++sweep) {
    double off = 0.0, dia = 0.0;
    for (int i = 0; i < p; ++i) {
      dia += A[i * p + i] * A[i * p + i];
      for (int j = i + 1; j < p; ++j) off += A[i * p + j] * A[i * p + j];
    }
    if (!(off > 1e-34 * dia) || off < 1e-300) break;
    for (int i = 0; i < p; ++i)
      for (int j = i + 1; j < p; ++j) {
        const double aij = A[i * p + j];
        if (aij == 0.0) continue;
        const double th = (A[j * p + j] - A[i * p + i]) / (2.0 * aij);
        const double t = (th >= 0.0 ? 1.0 : -1.0) / (Kokkos::fabs(th) + Kokkos::sqrt(th * th + 1.0));
        const double c = 1.0 / Kokkos::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < p; ++k) {
          const double aki = A[k * p + i], akj = A[k * p + j];
          A[k * p + i] = c * aki - s * akj, A[k * p + j] = s * aki + c * akj;
        }
        for (int k = 0; k < p; ++k) {
          const double aik = A[i * p + k], ajk = A[j * p + k];
          A[i * p + k] = c * aik - s * ajk, A[j * p + k] = s * aik + c * ajk;
        }
        for (int k = 0; k < p; ++k) {
          const double vki = V[k * p + i], vkj = V[k * p + j];
          V[k * p + i] = c * vki - s * vkj, V[k * p + j] = s * vki + c * vkj;
        }
      }
  }
  for (int i = 0; i < p; ++i) lam[i] = A[i * p + i];
}

// f[x, y] of exp and its logarithm
LAPLACE_HD double dd_exp(double x, double y) {
  const double hi = x > y ? x : y, d = Kokkos::fabs(x - y);
  return m::exp(hi) * m::exprel(-d);
}
LAPLACE_HD double log_dd_exp(double x, double y) {
  const double hi = x > y ? x : y, d = Kokkos::fabs(x - y);
  return hi + m::log(m::exprel(-d));
}

// out = V diag(g(lam)) V^T, packed or full
template <class G>
LAPLACE_HD void spectral(int p, const double* V, const double* lam, G&& g, double* out, bool pack) {
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int k = 0; k < p; ++k) s += V[i * p + k] * g(lam[k]) * V[j * p + k];
      if (pack) out[packed(i, j)] = s;
      else out[i * p + j] = out[j * p + i] = s;
    }
}

LAPLACE_HD void unpack_matrix(int p, const double* v, double* M) {
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) M[i * p + j] = M[j * p + i] = v[packed(i, j)];
}

// C = A B over the arithmetic; elements are g.size blocks of p x p
LAPLACE_HD void mat_mul(int p, const taylor_index& g, const double* A, const double* B, double* C) {
  const int pp = p * p;
  for (int T = 0; T < g.size; ++T) {
    double* c = C + T * pp;
    for (int i = 0; i < pp; ++i) c[i] = 0.0;
    g.below(T, [&](int U) {
      const double* a = A + U * pp;
      const double* b = B + (T - U) * pp;
      for (int i = 0; i < p; ++i)
        for (int k = 0; k < p; ++k) {
          const double aik = a[i * p + k];
          if (aik == 0.0) continue;
          for (int j = 0; j < p; ++j) c[i * p + j] += aik * b[k * p + j];
        }
    });
  }
}

// E = exp(X); w: 3 g.size p^2 doubles. Taylor degree 12 + K after scaling the
// constant part to norm <= 1/4 (the coefficient of order K is then exact to
// about 0.25^13/13!)
LAPLACE_HD void mat_exp(int p, const taylor_index& g, int K, const double* X, double* E, double* w) {
  const int pp = p * p, N = g.size * pp;
  double* Y = w;
  double* P = w + N;
  double* T = w + 2 * N;
  double nrm = 0.0;
  for (int i = 0; i < p; ++i) {
    double r = 0.0;
    for (int j = 0; j < p; ++j) r += Kokkos::fabs(X[i * p + j]);
    nrm = r > nrm ? r : nrm;
  }
  int s = 0;
  while (nrm > 0.25) nrm *= 0.5, ++s;
  const double sc = Kokkos::ldexp(1.0, -s);
  for (int i = 0; i < N; ++i) Y[i] = X[i] * sc, P[i] = 0.0;
  for (int i = 0; i < p; ++i) P[i * p + i] = 1.0;
  for (int k = 12 + K; k >= 1; --k) {  // P = I + Y P / k
    mat_mul(p, g, Y, P, T);
    for (int i = 0; i < N; ++i) P[i] = T[i] / k;
    for (int i = 0; i < p; ++i) P[i * p + i] += 1.0;
  }
  for (int k = 0; k < s; ++k) {
    mat_mul(p, g, P, P, T);
    for (int i = 0; i < N; ++i) P[i] = T[i];
  }
  for (int i = 0; i < N; ++i) E[i] = P[i];
}

// scalar arithmetic (elements of g.size doubles)
LAPLACE_HD void sc_mul(const taylor_index& g, const double* a, const double* b, double* c) {
  for (int T = 0; T < g.size; ++T) {
    double s = 0.0;
    g.below(T, [&](int U) { s += a[U] * b[T - U]; });
    c[T] = s;
  }
}
LAPLACE_HD void sc_inv(const taylor_index& g, const double* a, double* c) {
  c[0] = 1.0 / a[0];
  for (int T = 1; T < g.size; ++T) {
    double s = 0.0;
    g.below(T, [&](int U) {
      if (U != T) s += c[U] * a[T - U];
    });
    c[T] = -s / a[0];
  }
}
// acc += log|a|: log|a_0| + sum_k (-1)^(k+1) z^k/k, z = a/a_0 - 1; w: 2 g.size
LAPLACE_HD void sc_log_add(const taylor_index& g, int K, const double* a, double* acc, double* w) {
  double* z = w;
  double* zk = w + g.size;
  double* t = w + 2 * g.size;
  for (int T = 0; T < g.size; ++T) z[T] = a[T] / a[0], zk[T] = z[T];
  z[0] = zk[0] = 0.0;
  acc[0] += m::log(Kokkos::fabs(a[0]));
  for (int k = 1; k <= K; ++k) {
    for (int T = 1; T < g.size; ++T) acc[T] += (k % 2 ? 1.0 : -1.0) * zk[T] / k;
    sc_mul(g, zk, z, t);
    for (int T = 0; T < g.size; ++T) zk[T] = t[T];
  }
}

// log det of the Jacobian of exp at X (an element over g): the top coefficient
// of log det J(X(e)); w as scratch. The determinant does not depend on the
// basis of the symmetric matrices, and J is written in the eigenbasis V of
// X_0 (columns along V (e_a e_b^T + e_b e_a^T) V^T, rows read as
// (V^T . V)_ij), where J(0) is diagonal, f[s_a, s_b]: in the packed
// coordinates the LU mixed the columns and lost up to 3e5 ulp at order 6.
LAPLACE_HD double log_det_jacobian(int p, const taylor_index& g, int K, const double* X, double* w) {
  const int n = p * (p + 1) / 2, D = g.size, pp = p * p;
  taylor_index h = g;
  h.add(-1, 1);  // d, the column direction
  const int dr = h.radix[h.nv - 1];
  double* J = w;                    // n n D
  double* Xh = J + n * n * D;       // h.size pp
  double* Eh = Xh + h.size * pp;    // h.size pp
  double* we = Eh + h.size * pp;    // 3 h.size pp
  double* acc = we + 3 * h.size * pp;
  double* inv = acc + D;
  double* t = inv + D;
  double* t2 = t + D;
  double* ws = t2 + D;  // 3 D
  double* V = ws + 3 * D;
  double* lam = V + pp;
  double* A = lam + p;
  double* Tm = A + pp;
  for (int i = 0; i < pp; ++i) A[i] = X[i];
  jacobi(p, A, V, lam);
  for (int f = 0; f < n; ++f) {
    for (int i = 0; i < h.size * pp; ++i) Xh[i] = 0.0;
    for (int T = 0; T < D; ++T)  // the same monomial with d^0 has the same index
      for (int i = 0; i < pp; ++i) Xh[T * pp + i] = X[T * pp + i];
    int a, b;
    detail_coupled::unpack(f, a, b);
    double* dir = Xh + dr * pp;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j < p; ++j)
        dir[i * p + j] = a == b ? V[i * p + a] * V[j * p + a] : V[i * p + a] * V[j * p + b] + V[i * p + b] * V[j * p + a];
    mat_exp(p, h, K + 1, Xh, Eh, we);
    for (int T = 0; T < D; ++T) {
      const double* E = Eh + (T + dr) * pp;
      for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) {
          double s = 0.0;
          for (int k = 0; k < p; ++k) s += E[i * p + k] * V[k * p + j];
          Tm[i * p + j] = s;
        }
      for (int e = 0; e < n; ++e) {
        int i, j;
        detail_coupled::unpack(e, i, j);
        double s = 0.0;
        for (int k = 0; k < p; ++k) s += V[k * p + i] * Tm[k * p + j];
        J[(e * n + f) * D + T] = s;
      }
    }
  }
  for (int T = 0; T < D; ++T) acc[T] = 0.0;
  for (int c = 0; c < n; ++c) {
    int piv = c;
    for (int r = c + 1; r < n; ++r)
      if (Kokkos::fabs(J[(r * n + c) * D]) > Kokkos::fabs(J[(piv * n + c) * D])) piv = r;
    if (piv != c)
      for (int k = 0; k < n; ++k)
        for (int T = 0; T < D; ++T) {
          const double x = J[(c * n + k) * D + T];
          J[(c * n + k) * D + T] = J[(piv * n + k) * D + T];
          J[(piv * n + k) * D + T] = x;
        }
    sc_log_add(g, K, J + (c * n + c) * D, acc, ws);
    sc_inv(g, J + (c * n + c) * D, inv);
    for (int r = c + 1; r < n; ++r) {
      sc_mul(g, J + (r * n + c) * D, inv, t);
      for (int k = c + 1; k < n; ++k) {
        sc_mul(g, t, J + (c * n + k) * D, t2);
        for (int T = 0; T < D; ++T) J[(r * n + k) * D + T] -= t2[T];
      }
    }
  }
  return acc[D - 1];
}

// doubles of scratch for log_det_jacobian at order K <= 6
LAPLACE_HD int log_det_scratch(int p) {
  const int n = p * (p + 1) / 2, pp = p * p;
  return n * n * detail_coupled::max_masks + 5 * max_terms * pp + 7 * detail_coupled::max_masks + 3 * pp + p;
}

}  // namespace detail_matrix_log

/**
 * \ingroup bijector_coupled
 * \brief The matrix logarithm bijector between the \f$p \times p\f$ positive definite matrices and \f$\mathbb{R}^{p(p+1)/2}\f$.
 *
 * Maps a symmetric positive definite matrix \f$\Sigma\f$ stored as a packed lower triangle to an unconstrained
 * vector \f$u\f$ of \f$p(p+1)/2\f$ free coordinates via the matrix logarithm \f$S = \log\Sigma\f$,
 * and back via the matrix exponential \f$\Sigma = \exp(S)\f$.
 *
 * Reference coordinates for forward derivatives and log-Jacobians are the lower triangle of \f$\Sigma\f$
 * with its diagonal.
 *
 * \see coupled_bijector_orders, bijector_coupled
 */
struct matrix_log_bijector : coupled_bijector_orders<matrix_log_bijector> {
  /**
   * \ingroup bijector_coupled
   * \brief Matrix dimension \f$p \ge 1\f$.
   */
  int dim = 1;  // p

  /**
   * \ingroup bijector_coupled
   * \brief Default constructor, initializing with dimension \f$p = 1\f$.
   */
  LAPLACE_HD matrix_log_bijector() = default;

  /**
   * \ingroup bijector_coupled
   * \brief Constructs a matrix logarithm bijector for \f$p \times p\f$ matrices.
   *
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit matrix_log_bijector(int p) : dim(p) {}

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
   * The scratch requirement grows as \f$O(p^4)\f$ (23360 doubles at \f$p = 4\f$).
   *
   * \return Number of doubles required by scratch buffer \c w.
   */
  // the forward log-Jacobian derivative needs the most: the Newton arithmetic
  // (6 elements of 64 p x p blocks) and the Jacobian's (log_det_scratch)
  LAPLACE_HD int scratch_size() const {
    const int pp = dim * dim;
    return 6 * detail_coupled::max_masks * pp + 8 * pp + detail_matrix_log::log_det_scratch(dim);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse map \f$\Sigma = h(u) = \exp(S)\f$.
   *
   * Computes the matrix exponential via cyclic Jacobi eigendecomposition \f$S = V \operatorname{diag}(s) V^\top\f$,
   * with \f$\Sigma = V \operatorname{diag}(\exp(s)) V^\top\f$.
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param v Output packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 12.0 ulpN on gcc (K=6) / 12.0 ulpN on CUDA (K=3).
   * \see forward(), inverse_log_jacobian(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* u, double* v, double* w) const {
    namespace d = detail_matrix_log;
    const int pp = dim * dim;
    double* A = w;
    double* V = w + pp;
    double* lam = w + 2 * pp;
    d::unpack_matrix(dim, u, A);
    d::jacobi(dim, A, V, lam);
    d::spectral(dim, V, lam, [](double x) { return d::m::exp(x); }, v, true);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward map \f$S = g(v) = \log\Sigma\f$.
   *
   * Computes the matrix logarithm via cyclic Jacobi eigendecomposition \f$\Sigma = V \operatorname{diag}(\sigma) V^\top\f$,
   * with \f$S = V \operatorname{diag}(\log \sigma) V^\top\f$.
   *
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param u Output free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Measured accuracy: within 199 ulpN on gcc (K=6) / 210 ulpN on CUDA (K=6).
   * \see inverse(), forward_log_jacobian(), forward_dk()
   */
  LAPLACE_HD void forward(const double* v, double* u, double* w) const {
    namespace d = detail_matrix_log;
    const int pp = dim * dim;
    double* A = w;
    double* V = w + pp;
    double* lam = w + 2 * pp;
    d::unpack_matrix(dim, v, A);
    d::jacobi(dim, A, V, lam);
    d::spectral(dim, V, lam, [](double x) { return d::m::log(x); }, u, true);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse log-Jacobian \f$\log|\det J_h(u)|\f$.
   *
   * Computes
   * \f[
   * \log\lvert\det J_h(u)\rvert = \sum_{i \le j} \log f[s_i, s_j]
   * \f]
   * where \f$f[x, y] = e^{\max(x, y)}\operatorname{exprel}(-|x - y|)\f$ and \f$s_i\f$ are the eigenvalues of \f$S\f$.
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of inverse log-Jacobian.
   *
   * Measured accuracy: within 102794 ulp on gcc (K=6) / 291832 ulp on CUDA (K=6).
   * \see inverse(), forward_log_jacobian(), inverse_log_jacobian_dk()
   */
  LAPLACE_HD double inverse_log_jacobian(const double* u, double* w) const {
    namespace d = detail_matrix_log;
    const int pp = dim * dim;
    double* lam = w + 2 * pp;
    d::unpack_matrix(dim, u, w);
    d::jacobi(dim, w, w + pp, lam);
    return log_jacobian_of(lam);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward log-Jacobian \f$\log|\det J_g(v)|\f$ in reference coordinates.
   *
   * Computes
   * \f[
   * \log\lvert\det J_g(v)\rvert = -\sum_{i \le j} \log f[\log \sigma_i, \log \sigma_j]
   * \f]
   * where \f$\sigma_i\f$ are the eigenvalues of \f$\Sigma\f$.
   *
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of forward log-Jacobian.
   *
   * Measured accuracy: within 3293 ulp on gcc (K=3) / 2364 ulp on CUDA (K=3).
   * \see forward(), inverse_log_jacobian(), forward_log_jacobian_dk()
   */
  LAPLACE_HD double forward_log_jacobian(const double* v, double* w) const {
    namespace d = detail_matrix_log;
    const int pp = dim * dim;
    double* lam = w + 2 * pp;
    d::unpack_matrix(dim, v, w);
    d::jacobi(dim, w, w + pp, lam);
    for (int i = 0; i < dim; ++i) lam[i] = d::m::log(lam[i]);
    return -log_jacobian_of(lam);
  }

  // dense: every entry of exp(S) depends on every entry of S
  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse partial derivatives of order \f$K\f$ along free coordinate tuple \c idx via truncated Taylor arithmetic.
   *
   * Evaluates \f$\exp(S + \sum_v \epsilon_v E_v)\f$ using scaling and squaring with a Taylor polynomial of degree \f$12 + K\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true (matrix logarithm derivatives are dense and never structurally zero).
   *
   * Measured accuracy: within 12.0 ulpN on gcc (K=6) / 12.0 ulpN on CUDA (K=3).
   * \see inverse(), inverse_log_jacobian_dk(), inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* u, const int (&idx)[K], double* out, double* w) const {
    namespace d = detail_matrix_log;
    d::taylor_index g;
    g.from_tuple(idx);
    const int pp = dim * dim, N = g.size * pp;
    double* X = w;
    double* E = w + N;
    seed(u, g, X);
    d::mat_exp(dim, g, K, X, E, w + 2 * N);
    write_top(g, E, out);
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx via \f$K\f$ Newton steps.
   *
   * Computes \f$X = \log(\Sigma + \sum_v \epsilon_v E_v)\f$ in truncated Taylor arithmetic using \f$K\f$ Newton steps
   * with the inverse Fréchet derivative in the eigenbasis.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true (matrix logarithm derivatives are dense and never structurally zero).
   *
   * Measured accuracy: within 199 ulpN on gcc (K=6) / 210 ulpN on CUDA (K=6).
   * \see forward(), forward_log_jacobian_dk(), forward_d1()
   */
  template <int K>
  LAPLACE_HD bool forward_dk(const double* v, const int (&idx)[K], double* out, double* w) const {
    namespace d = detail_matrix_log;
    d::taylor_index g;
    g.from_tuple(idx);
    log_newton<K>(v, g, w);
    write_top(g, w, out);
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse log-Jacobian partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Computes the log-determinant of the Jacobian \f$J(S(\epsilon))\f$ in the eigenbasis of \f$S_0\f$
   * via LU factorization with partial pivoting in scalar Taylor arithmetic.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 102794 ulp on gcc (K=6) / 291832 ulp on CUDA (K=6).
   * \see inverse_log_jacobian(), inverse_dk(), inverse_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double inverse_log_jacobian_dk(const double* u, const int (&idx)[K], double* w) const {
    namespace d = detail_matrix_log;
    d::taylor_index g;
    g.from_tuple(idx);
    const int N = g.size * dim * dim;
    seed(u, g, w);
    return d::log_det_jacobian(dim, g, K, w, w + N) * g.factorials();
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward log-Jacobian partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx.
   *
   * Evaluates the negative log-determinant of the Jacobian at \f$X(\epsilon) = \log\Sigma(\epsilon)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input packed lower triangle of \f$\Sigma\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 3293 ulp on gcc (K=3) / 2364 ulp on CUDA (K=3).
   * \see forward_log_jacobian(), forward_dk(), forward_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double forward_log_jacobian_dk(const double* v, const int (&idx)[K], double* w) const {
    namespace d = detail_matrix_log;
    d::taylor_index g;
    g.from_tuple(idx);
    const int pp = dim * dim;
    log_newton<K>(v, g, w);
    return -d::log_det_jacobian(dim, g, K, w, w + 6 * detail_coupled::max_masks * pp + 8 * pp) * g.factorials();
  }

 private:
  // sum_(i<=j) log f[l_i, l_j]
  LAPLACE_HD double log_jacobian_of(const double* lam) const {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
      s += lam[i];
      for (int j = i + 1; j < dim; ++j) s += detail_matrix_log::log_dd_exp(lam[i], lam[j]);
    }
    return s;
  }
  // X = S + sum_v e_v E_v over g
  LAPLACE_HD void seed(const double* u, const detail_matrix_log::taylor_index& g, double* X) const {
    const int pp = dim * dim;
    for (int i = 0; i < g.size * pp; ++i) X[i] = 0.0;
    detail_matrix_log::unpack_matrix(dim, u, X);
    for (int v = 0; v < g.nv; ++v) detail_matrix_log::add_direction(dim, g.coord[v], X + g.radix[v] * pp);
  }
  LAPLACE_HD void write_top(const detail_matrix_log::taylor_index& g, const double* E, double* out) const {
    const int pp = dim * dim;
    const double* G = E + (g.size - 1) * pp;
    const double f = g.factorials();
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) out[detail_matrix_log::packed(i, j)] = G[i * dim + j] * f;
  }
  // X(e) = log(Sigma + sum_v e_v E_v) over g, written at w[0, g.size p^2)
  template <int K>
  LAPLACE_HD void log_newton(const double* v, const detail_matrix_log::taylor_index& g, double* w) const {
    namespace d = detail_matrix_log;
    const int pp = dim * dim, N = g.size * pp;
    double* X = w;
    double* Sg = w + N;
    double* E = w + 2 * N;
    double* we = w + 3 * N;  // 3 N
    double* V = w + 6 * N;
    double* lam = V + pp;
    double* f = lam + dim;  // p p: 1/f[s_i, s_j]
    double* R = f + pp;
    double* T = R + pp;
    d::unpack_matrix(dim, v, R);
    d::jacobi(dim, R, V, lam);
    for (int i = 0; i < dim; ++i) lam[i] = d::m::log(lam[i]);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) f[i * dim + j] = 1.0 / d::dd_exp(lam[i], lam[j]);
    for (int i = 0; i < N; ++i) X[i] = 0.0;
    d::spectral(dim, V, lam, [](double x) { return x; }, X, false);
    seed(v, g, Sg);
    for (int it = 0; it < K; ++it) {
      d::mat_exp(dim, g, K, X, E, we);
      for (int Tm = 1; Tm < g.size; ++Tm) {
        // X[Tm] += V ((V^T (Sigma[Tm] - E[Tm]) V) .* f) V^T
        const double* sg = Sg + Tm * pp;
        const double* e = E + Tm * pp;
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j) {
            double s = 0.0;
            for (int k = 0; k < dim; ++k) s += (sg[i * dim + k] - e[i * dim + k]) * V[k * dim + j];
            T[i * dim + j] = s;
          }
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j) {
            double s = 0.0;
            for (int k = 0; k < dim; ++k) s += V[k * dim + i] * T[k * dim + j];
            R[i * dim + j] = s * f[i * dim + j];
          }
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j) {
            double s = 0.0;
            for (int k = 0; k < dim; ++k) s += R[i * dim + k] * V[j * dim + k];
            T[i * dim + j] = s;
          }
        double* x = X + Tm * pp;
        for (int i = 0; i < dim; ++i)
          for (int j = 0; j < dim; ++j) {
            double s = 0.0;
            for (int k = 0; k < dim; ++k) s += V[i * dim + k] * T[k * dim + j];
            x[i * dim + j] += s;
          }
      }
    }
  }
};

}  // namespace laplace::bijector
