// Coupled bijectors (laplace::bijector, kind C of note/bijectors.md): maps
// between an open set of R^n and a constrained set of vectors or matrices
// whose components are coupled (the simplex, the positive definite matrices,
// the correlation matrices). Decisions 17-19 of 2026-11-12 in
// note/bijectors.md; the maps are in alr.hpp, log_cholesky.hpp and
// partial_corr.hpp.
//
// A coupled bijector is a small struct copied by value into kernels, with a
// runtime dimension dim (K categories for alr, p for the matrices). Forward
// g goes from the constrained value v to the free vector u in R^n, inverse h
// back. Every function writes into the caller's buffers and may use a scratch
// buffer of scratch_size() doubles, which every function takes.
//
//   free_size()                         n, the length of u
//   value_size()                        m, the length of the stored value v
//   scratch_size()                      doubles of scratch any function needs
//   forward(v, u, w), inverse(u, v, w)  the two maps
//   forward_log_jacobian(v, w)          log |det J_g(v)| in the reference coordinates
//   inverse_log_jacobian(u, w)          log |det J_h(u)|
//   forward_dK(v, idx, out, w)          d^K g / dv_idx[0] ... dv_idx[K-1], n values, K = 1..6
//   inverse_dK(u, idx, out, w)          d^K h / du_idx[0] ... du_idx[K-1], m values
//   forward_log_jacobian_dK(v, idx, w), inverse_log_jacobian_dK(u, idx, w)
//
// idx is a sorted tuple of indices, idx[0] <= ... <= idx[K-1]: into u for the
// inverse direction, into the reference coordinates of v for the forward one
// (the coordinates in which the log-Jacobians are defined and in which a
// density on v must be written). The derivatives of g and h return false,
// and leave out untouched, when the derivative is identically zero for that
// tuple; a log-Jacobian derivative that is identically zero returns 0.
//
// Matrices are stored as the lower triangle by rows, entry (i, j), j <= i, at
// i (i + 1)/2 + j (the packing of laplace::pdmat); a strict lower triangle at
// i (i - 1)/2 + j.
#pragma once

#include <laplace/bijector/scalar.hpp>
#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::bijector {

namespace detail_coupled {

namespace ds = detail_scalar;
namespace m = laplace::math;

// the sub-tuple recursions index the subsets of the K positions of a tuple by
// bit masks; their arrays have 2^6 = 64 entries at order 6
constexpr int max_order = 6;
constexpr int max_masks = 1 << max_order;

LAPLACE_HD_FORCEINLINE int packed(int i, int j) { return i * (i + 1) / 2 + j; }
LAPLACE_HD_FORCEINLINE int strict(int i, int j) { return i * (i - 1) / 2 + j; }
// (i, j) of the packed position k
LAPLACE_HD_FORCEINLINE void unpack(int k, int& i, int& j) {
  i = 0;
  while ((i + 1) * (i + 2) / 2 <= k) ++i;
  j = k - i * (i + 1) / 2;
}
// (i, j), i > j, of the strict position k
LAPLACE_HD_FORCEINLINE void unstrict(int k, int& i, int& j) {
  i = 1;
  while ((i + 1) * i / 2 <= k) ++i;
  j = k - i * (i - 1) / 2;
}
LAPLACE_HD_FORCEINLINE int lowest_bit(int s) {
  int b = 0;
  while (!(s & (1 << b))) ++b;
  return b;
}

// d^k tanh u, k = 0..6 (k = 0 the value)
LAPLACE_HD double tanh_d(int k, double u) {
  switch (k) {
    case 0: return ds::tanh_of(u).T;
    case 1: return ds::tanh_dk<1>(u);
    case 2: return ds::tanh_dk<2>(u);
    case 3: return ds::tanh_dk<3>(u);
    case 4: return ds::tanh_dk<4>(u);
    case 5: return ds::tanh_dk<5>(u);
    default: return ds::tanh_dk<6>(u);
  }
}

// d^k sech u = sech(u) P_k, k = 0..6, with T = tanh u and S = sech^2 u = 1 - T^2
// (tools/reference/bijector_coupled_forms.py, SymPy 1.14.0): P_k in T^2 while
// T^2 < 1/2, in S beyond, where the T^2 form cancels (as tanh_dk)
LAPLACE_HD double sech_d(int k, double u) {
  const double e = m::exp(-Kokkos::fabs(u));
  const double sech = 2.0 * e / (1.0 + e * e);
  if (k == 0) return sech;
  const auto p = ds::tanh_of(u);
  const double T = p.T, T2 = T * T, S = p.S;
  const bool small = T2 < 0.5;
  switch (k) {
    case 1: return -sech * T;
    case 2: return sech * (small ? 2.0 * T2 - 1.0 : 1.0 - 2.0 * S);
    case 3: return sech * T * (small ? 5.0 - 6.0 * T2 : 6.0 * S - 1.0);
    case 4: {
      constexpr double a[] = {5.0, -28.0, 24.0}, b[] = {1.0, -20.0, 24.0};
      return sech * (small ? ds::poly(a, T2) : ds::poly(b, S));
    }
    case 5: {
      constexpr double a[] = {-61.0, 180.0, -120.0}, b[] = {-1.0, 60.0, -120.0};
      return sech * T * (small ? ds::poly(a, T2) : ds::poly(b, S));
    }
    default: {
      constexpr double a[] = {-61.0, 662.0, -1320.0, 720.0}, b[] = {1.0, -182.0, 840.0, -720.0};
      return sech * (small ? ds::poly(a, T2) : ds::poly(b, S));
    }
  }
}

// ------------------------------------------------ sub-tuple recursions
// For a tuple of K positions, x[S] is the derivative of a quantity x along
// the positions in the mask S (x[0] its value). The products below are
// Leibniz's rule over the subsets; each "solve" isolates the term that holds
// the unknown x[S].

// sum over T subset of S of f[T] g[S \ T]
LAPLACE_HD_FORCEINLINE double leibniz(const double* f, const double* g, int S) {
  double s = 0.0;
  for (int T = S;; T = (T - 1) & S) {
    s += f[T] * g[S ^ T];
    if (T == 0) break;
  }
  return s;
}
// y with y^2 = q: y[S] from q[S] and y on the proper submasks of S
LAPLACE_HD_FORCEINLINE void sqrt_solve(const double* q, double* y, int S) {
  if (S == 0) {
    y[0] = Kokkos::sqrt(q[0]);
    return;
  }
  double s = q[S];
  for (int T = (S - 1) & S; T != 0; T = (T - 1) & S) s -= y[T] * y[S ^ T];
  y[S] = s / (2.0 * y[0]);
}
// y with y d = r: y[S] from r[S], d and y on the proper submasks of S
LAPLACE_HD_FORCEINLINE void div_solve(const double* r, const double* d, double* y, int S) {
  double s = r[S];
  if (S != 0)
    for (int T = (S - 1) & S;; T = (T - 1) & S) {
      s -= y[T] * d[S ^ T];
      if (T == 0) break;
    }
  y[S] = s / d[0];
}
// y with c dy = dx along every variable, c > 0 a quantity of its own (y = log x
// is c = x): for S != 0, with j the lowest position of S and R = S \ {j},
// sum over T subset of R of c[T] y[S \ T] = x[S]; y[0] is set by the caller
LAPLACE_HD_FORCEINLINE void pivot_solve(const double* x, const double* c, double* y, int S) {
  const int R = S & (S - 1);  // S without its lowest bit
  double s = x[S];
  for (int T = R; T != 0; T = (T - 1) & R) s -= c[T] * y[S ^ T];
  y[S] = s / c[0];
}

// The Cholesky factor L of a p x p matrix X (packed lower triangle x) and its
// derivatives along a tuple of K entries of X: D[e * 64 + S] = d_S L_e, e the
// packed position. var[q] is the packed position of the entry of X at tuple
// position q (an off-diagonal entry moves X_ij and X_ji together). With
// unit_diag the diagonal of X is 1 and x's diagonal is not read. From
// L_jj^2 = X_jj - sum_(l<j) L_jl^2 and L_ij L_jj = X_ij - sum_(l<j) L_il L_jl,
// differentiated over the subsets and solved for the term of L_jj or L_ij.
template <int K>
LAPLACE_HD void cholesky_subtuple(int p, const double* x, const int* var, bool unit_diag, double* D) {
  constexpr int NM = 1 << K, M = max_masks;
  auto dx = [&](int e, int S) {
    if (S == 0) {
      int i, j;
      unpack(e, i, j);
      return unit_diag && i == j ? 1.0 : x[e];
    }
    if (S & (S - 1)) return 0.0;
    return var[lowest_bit(S)] == e ? 1.0 : 0.0;
  };
  for (int j = 0; j < p; ++j) {
    double* Ljj = D + packed(j, j) * M;
    for (int S = 0; S < NM; ++S) {
      double q = dx(packed(j, j), S);
      for (int l = 0; l < j; ++l) {
        const double* a = D + packed(j, l) * M;
        q -= leibniz(a, a, S);
      }
      if (S == 0) {
        Ljj[0] = Kokkos::sqrt(q);
        continue;
      }
      for (int T = (S - 1) & S; T != 0; T = (T - 1) & S) q -= Ljj[T] * Ljj[S ^ T];
      Ljj[S] = q / (2.0 * Ljj[0]);
    }
    for (int i = j + 1; i < p; ++i) {
      double* Lij = D + packed(i, j) * M;
      for (int S = 0; S < NM; ++S) {
        double r = dx(packed(i, j), S);
        for (int l = 0; l < j; ++l) r -= leibniz(D + packed(i, l) * M, D + packed(j, l) * M, S);
        if (S != 0)
          for (int T = (S - 1) & S;; T = (T - 1) & S) {
            r -= Lij[T] * Ljj[S ^ T];
            if (T == 0) break;
          }
        Lij[S] = r / Ljj[0];
      }
    }
  }
}

// y = log x over the subsets of a tuple of K positions (y[0] = log x[0])
template <int K>
LAPLACE_HD void log_subtuple(const double* x, double* y) {
  y[0] = m::log(x[0]);
  for (int S = 1; S < (1 << K); ++S) pivot_solve(x, x, y, S);
}

}  // namespace detail_coupled

/**
 * \ingroup bijector_coupled
 * \brief CRTP base providing named derivative member functions of orders 1 to 6 for a coupled bijector.
 *
 * A coupled bijector class \c B derives from `coupled_bijector_orders<B>` and implements
 * the member function templates `forward_dk<K>`, `inverse_dk<K>`,
 * `forward_log_jacobian_dk<K>`, and `inverse_log_jacobian_dk<K>` for \f$K \in \{1, \dots, 6\}\f$.
 *
 * \tparam B Derived coupled bijector class.
 */
template <class B>
struct coupled_bijector_orders {
  /**
   * \fn bool forward_d1(const double* v, const int (&i)[1], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 1 along reference coordinate index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial g(v)}{\partial v_{i_1}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 1 reference coordinate index.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d2(), inverse_d1(), forward_log_jacobian_d1()
   */
  /**
   * \fn bool forward_d2(const double* v, const int (&i)[2], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 2 along reference coordinate indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 g(v)}{\partial v_{i_1}\partial v_{i_2}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 2 reference coordinate indices \f$i_1 \le i_2\f$.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d1(), forward_d3(), inverse_d2(), forward_log_jacobian_d2()
   */
  /**
   * \fn bool forward_d3(const double* v, const int (&i)[3], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 3 along reference coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 g(v)}{\partial v_{i_1}\partial v_{i_2}\partial v_{i_3}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 3 reference coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d2(), forward_d4(), inverse_d3(), forward_log_jacobian_d3()
   */
  /**
   * \fn bool forward_d4(const double* v, const int (&i)[4], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 4 along reference coordinate indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 g(v)}{\partial v_{i_1}\cdots\partial v_{i_4}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 4 reference coordinate indices \f$i_1 \le \dots \le i_4\f$.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d3(), forward_d5(), inverse_d4(), forward_log_jacobian_d4()
   */
  /**
   * \fn bool forward_d5(const double* v, const int (&i)[5], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 5 along reference coordinate indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 g(v)}{\partial v_{i_1}\cdots\partial v_{i_5}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 5 reference coordinate indices \f$i_1 \le \dots \le i_5\f$.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d4(), forward_d6(), inverse_d5(), forward_log_jacobian_d5()
   */
  /**
   * \fn bool forward_d6(const double* v, const int (&i)[6], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the forward map \f$g(v)\f$ of order 6 along reference coordinate indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 g(v)}{\partial v_{i_1}\cdots\partial v_{i_6}}
   * \f]
   * into caller-provided buffer \c out of \c free_size() doubles.
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 6 reference coordinate indices \f$i_1 \le \dots \le i_6\f$.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see forward_d5(), inverse_d6(), forward_log_jacobian_d6()
   */
  /**
   * \fn bool inverse_d1(const double* u, const int (&i)[1], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 1 along free coordinate index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial h(u)}{\partial u_{i_1}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 1 free coordinate index.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d2(), forward_d1(), inverse_log_jacobian_d1()
   */
  /**
   * \fn bool inverse_d2(const double* u, const int (&i)[2], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 2 along free coordinate indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 h(u)}{\partial u_{i_1}\partial u_{i_2}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 2 free coordinate indices \f$i_1 \le i_2\f$.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d1(), inverse_d3(), forward_d2(), inverse_log_jacobian_d2()
   */
  /**
   * \fn bool inverse_d3(const double* u, const int (&i)[3], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 3 along free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 h(u)}{\partial u_{i_1}\partial u_{i_2}\partial u_{i_3}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 3 free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d2(), inverse_d4(), forward_d3(), inverse_log_jacobian_d3()
   */
  /**
   * \fn bool inverse_d4(const double* u, const int (&i)[4], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 4 along free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 h(u)}{\partial u_{i_1}\cdots\partial u_{i_4}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 4 free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d3(), inverse_d5(), forward_d4(), inverse_log_jacobian_d4()
   */
  /**
   * \fn bool inverse_d5(const double* u, const int (&i)[5], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 5 along free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 h(u)}{\partial u_{i_1}\cdots\partial u_{i_5}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 5 free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d4(), inverse_d6(), forward_d5(), inverse_log_jacobian_d5()
   */
  /**
   * \fn bool inverse_d6(const double* u, const int (&i)[6], double* out, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the inverse map \f$h(u)\f$ of order 6 along free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 h(u)}{\partial u_{i_1}\cdots\partial u_{i_6}}
   * \f]
   * into caller-provided buffer \c out of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 6 free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_d5(), forward_d6(), inverse_log_jacobian_d6()
   */
  /**
   * \fn double forward_log_jacobian_d1(const double* v, const int (&i)[1], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 1 along reference coordinate index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 1 reference coordinate index.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d2(), forward_d1(), inverse_log_jacobian_d1()
   */
  /**
   * \fn double forward_log_jacobian_d2(const double* v, const int (&i)[2], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 2 along reference coordinate indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\partial v_{i_2}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 2 reference coordinate indices \f$i_1 \le i_2\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d1(), forward_log_jacobian_d3(), forward_d2(), inverse_log_jacobian_d2()
   */
  /**
   * \fn double forward_log_jacobian_d3(const double* v, const int (&i)[3], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 3 along reference coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\partial v_{i_2}\partial v_{i_3}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 3 reference coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d2(), forward_log_jacobian_d4(), forward_d3(), inverse_log_jacobian_d3()
   */
  /**
   * \fn double forward_log_jacobian_d4(const double* v, const int (&i)[4], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 4 along reference coordinate indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\cdots\partial v_{i_4}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 4 reference coordinate indices \f$i_1 \le \dots \le i_4\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d3(), forward_log_jacobian_d5(), forward_d4(), inverse_log_jacobian_d4()
   */
  /**
   * \fn double forward_log_jacobian_d5(const double* v, const int (&i)[5], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 5 along reference coordinate indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\cdots\partial v_{i_5}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 5 reference coordinate indices \f$i_1 \le \dots \le i_5\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d4(), forward_log_jacobian_d6(), forward_d5(), inverse_log_jacobian_d5()
   */
  /**
   * \fn double forward_log_jacobian_d6(const double* v, const int (&i)[6], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of forward log-Jacobian of order 6 along reference coordinate indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 \log\lvert\det J_g(v)\rvert}{\partial v_{i_1}\cdots\partial v_{i_6}}
   * \f]
   *
   * \param v Pointer to constrained value \f$v\f$ of \c value_size() doubles.
   * \param i Sorted tuple of 6 reference coordinate indices \f$i_1 \le \dots \le i_6\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see forward_log_jacobian_d5(), forward_d6(), inverse_log_jacobian_d6()
   */
  /**
   * \fn double inverse_log_jacobian_d1(const double* u, const int (&i)[1], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 1 along free coordinate index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 1 free coordinate index.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d2(), inverse_d1(), forward_log_jacobian_d1()
   */
  /**
   * \fn double inverse_log_jacobian_d2(const double* u, const int (&i)[2], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 2 along free coordinate indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}\partial u_{i_2}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 2 free coordinate indices \f$i_1 \le i_2\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d1(), inverse_log_jacobian_d3(), inverse_d2(), forward_log_jacobian_d2()
   */
  /**
   * \fn double inverse_log_jacobian_d3(const double* u, const int (&i)[3], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 3 along free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}\partial u_{i_2}\partial u_{i_3}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 3 free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d2(), inverse_log_jacobian_d4(), inverse_d3(), forward_log_jacobian_d3()
   */
  /**
   * \fn double inverse_log_jacobian_d4(const double* u, const int (&i)[4], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 4 along free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}\cdots\partial u_{i_4}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 4 free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d3(), inverse_log_jacobian_d5(), inverse_d4(), forward_log_jacobian_d4()
   */
  /**
   * \fn double inverse_log_jacobian_d5(const double* u, const int (&i)[5], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 5 along free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}\cdots\partial u_{i_5}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 5 free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d4(), inverse_log_jacobian_d6(), inverse_d5(), forward_log_jacobian_d5()
   */
  /**
   * \fn double inverse_log_jacobian_d6(const double* u, const int (&i)[6], double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of inverse log-Jacobian of order 6 along free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 \log\lvert\det J_h(u)\rvert}{\partial u_{i_1}\cdots\partial u_{i_6}}
   * \f]
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 6 free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see inverse_log_jacobian_d5(), inverse_d6(), forward_log_jacobian_d6()
   */
// the named functions, written out (not by a macro) so that the documentation finds them
  LAPLACE_HD bool forward_d1(const double* v, const int (&i)[1], double* out, double* w) const {
    return b().template forward_dk<1>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d1(const double* u, const int (&i)[1], double* out, double* w) const {
    return b().template inverse_dk<1>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d1(const double* v, const int (&i)[1], double* w) const {
    return b().template forward_log_jacobian_dk<1>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d1(const double* u, const int (&i)[1], double* w) const {
    return b().template inverse_log_jacobian_dk<1>(u, i, w);
  }
  LAPLACE_HD bool forward_d2(const double* v, const int (&i)[2], double* out, double* w) const {
    return b().template forward_dk<2>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d2(const double* u, const int (&i)[2], double* out, double* w) const {
    return b().template inverse_dk<2>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d2(const double* v, const int (&i)[2], double* w) const {
    return b().template forward_log_jacobian_dk<2>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d2(const double* u, const int (&i)[2], double* w) const {
    return b().template inverse_log_jacobian_dk<2>(u, i, w);
  }
  LAPLACE_HD bool forward_d3(const double* v, const int (&i)[3], double* out, double* w) const {
    return b().template forward_dk<3>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d3(const double* u, const int (&i)[3], double* out, double* w) const {
    return b().template inverse_dk<3>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d3(const double* v, const int (&i)[3], double* w) const {
    return b().template forward_log_jacobian_dk<3>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d3(const double* u, const int (&i)[3], double* w) const {
    return b().template inverse_log_jacobian_dk<3>(u, i, w);
  }
  LAPLACE_HD bool forward_d4(const double* v, const int (&i)[4], double* out, double* w) const {
    return b().template forward_dk<4>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d4(const double* u, const int (&i)[4], double* out, double* w) const {
    return b().template inverse_dk<4>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d4(const double* v, const int (&i)[4], double* w) const {
    return b().template forward_log_jacobian_dk<4>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d4(const double* u, const int (&i)[4], double* w) const {
    return b().template inverse_log_jacobian_dk<4>(u, i, w);
  }
  LAPLACE_HD bool forward_d5(const double* v, const int (&i)[5], double* out, double* w) const {
    return b().template forward_dk<5>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d5(const double* u, const int (&i)[5], double* out, double* w) const {
    return b().template inverse_dk<5>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d5(const double* v, const int (&i)[5], double* w) const {
    return b().template forward_log_jacobian_dk<5>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d5(const double* u, const int (&i)[5], double* w) const {
    return b().template inverse_log_jacobian_dk<5>(u, i, w);
  }
  LAPLACE_HD bool forward_d6(const double* v, const int (&i)[6], double* out, double* w) const {
    return b().template forward_dk<6>(v, i, out, w);
  }
  LAPLACE_HD bool inverse_d6(const double* u, const int (&i)[6], double* out, double* w) const {
    return b().template inverse_dk<6>(u, i, out, w);
  }
  LAPLACE_HD double forward_log_jacobian_d6(const double* v, const int (&i)[6], double* w) const {
    return b().template forward_log_jacobian_dk<6>(v, i, w);
  }
  LAPLACE_HD double inverse_log_jacobian_d6(const double* u, const int (&i)[6], double* w) const {
    return b().template inverse_log_jacobian_dk<6>(u, i, w);
  }

 private:
  LAPLACE_HD const B& b() const { return static_cast<const B&>(*this); }
};

/**
 * \ingroup bijector_coupled
 * \brief CRTP base providing named derivative member functions of orders 1 to 6 for the Cholesky factor.
 *
 * A coupled bijector class \c B mapping onto positive definite or correlation matrices
 * derives from `cholesky_bijector_orders<B>` (which itself derives from `coupled_bijector_orders<B>`)
 * and implements the member function template `inverse_cholesky_dk<K>` for \f$K \in \{1, \dots, 6\}\f$,
 * in addition to the member functions of `coupled_bijector_orders<B>`.
 *
 * \tparam B Derived coupled bijector class.
 */
// One chain of empty bases, never two side by side: MSVC lays out a second empty
// base at a nonzero offset and the device compiler does not, so a bijector with
// two bases, copied from the host into a CUDA kernel, was read with the wrong
// dim (an illegal address in benchmarks/bijector_coupled.cpp, 2026-11-13).
template <class B>
struct cholesky_bijector_orders : coupled_bijector_orders<B> {
  /**
   * \fn bool inverse_cholesky_d1(const double* u, const int (&i)[1], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 1 along free coordinate index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial L(u)}{\partial u_{i_1}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 1 free coordinate index.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d2(), inverse_d1()
   */
  /**
   * \fn bool inverse_cholesky_d2(const double* u, const int (&i)[2], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 2 along free coordinate indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 L(u)}{\partial u_{i_1}\partial u_{i_2}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 2 free coordinate indices \f$i_1 \le i_2\f$.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d1(), inverse_cholesky_d3(), inverse_d2()
   */
  /**
   * \fn bool inverse_cholesky_d3(const double* u, const int (&i)[3], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 3 along free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 L(u)}{\partial u_{i_1}\partial u_{i_2}\partial u_{i_3}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 3 free coordinate indices \f$i_1 \le i_2 \le i_3\f$.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d2(), inverse_cholesky_d4(), inverse_d3()
   */
  /**
   * \fn bool inverse_cholesky_d4(const double* u, const int (&i)[4], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 4 along free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 L(u)}{\partial u_{i_1}\cdots\partial u_{i_4}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 4 free coordinate indices \f$i_1 \le \dots \le i_4\f$.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d3(), inverse_cholesky_d5(), inverse_d4()
   */
  /**
   * \fn bool inverse_cholesky_d5(const double* u, const int (&i)[5], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 5 along free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 L(u)}{\partial u_{i_1}\cdots\partial u_{i_5}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 5 free coordinate indices \f$i_1 \le \dots \le i_5\f$.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d4(), inverse_cholesky_d6(), inverse_d5()
   */
  /**
   * \fn bool inverse_cholesky_d6(const double* u, const int (&i)[6], double* l, double* w) const
   * \ingroup bijector_coupled
   * \brief Partial derivative of the Cholesky factor \f$L(u)\f$ of order 6 along free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 L(u)}{\partial u_{i_1}\cdots\partial u_{i_6}}
   * \f]
   * into caller-provided packed buffer \c l of \c value_size() doubles.
   *
   * \param u Pointer to free vector \f$u\f$ of \c free_size() doubles.
   * \param i Sorted tuple of 6 free coordinate indices \f$i_1 \le \dots \le i_6\f$.
   * \param l Output buffer of at least \c value_size() doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse_cholesky_d5(), inverse_d6()
   */
  LAPLACE_HD bool inverse_cholesky_d1(const double* u, const int (&i)[1], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<1>(u, i, l, w);
  }
  LAPLACE_HD bool inverse_cholesky_d2(const double* u, const int (&i)[2], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<2>(u, i, l, w);
  }
  LAPLACE_HD bool inverse_cholesky_d3(const double* u, const int (&i)[3], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<3>(u, i, l, w);
  }
  LAPLACE_HD bool inverse_cholesky_d4(const double* u, const int (&i)[4], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<4>(u, i, l, w);
  }
  LAPLACE_HD bool inverse_cholesky_d5(const double* u, const int (&i)[5], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<5>(u, i, l, w);
  }
  LAPLACE_HD bool inverse_cholesky_d6(const double* u, const int (&i)[6], double* l, double* w) const {
    return static_cast<const B&>(*this).template inverse_cholesky_dk<6>(u, i, l, w);
  }
};

}  // namespace laplace::bijector
