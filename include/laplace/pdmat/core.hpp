// The pieces shared by the structured matrices (laplace::pdmat); the design is
// in note/pdmat_structures.md, the interface was decided with Giovanni on
// 2026-11-12.
//
// A structure maps q unconstrained parameters theta to a symmetric p x p
// matrix M(theta). Each structure owns the constraint of its inputs (a fixed
// scalar bijector: rho = tanh(theta) for AR(1), ...) and its derivatives are
// written directly in theta, never composed through the bijector by Faa di
// Bruno (which loses up to 1.2e-5 relative on the AR(1) log-determinant at
// theta = 10, order 6; docs/numerics.md).
//
// A structure is a small struct copied by value into kernels. Its dimension p
// is a runtime field (dim); every matrix is written by the caller's buffer in
// packed lower-triangular storage by rows, entry (i, j), j <= i, at
// packed_index(i, j) = i (i + 1)/2 + j, packed_size(p) values in all.
//
//   dim, n_params()                p and q
//   unit_diagonal                  M has unit diagonal (a correlation matrix)
//   closed_inverse, closed_factor  inverse() and cholesky() are available
//   matrix(theta, m)               M
//   log_det(theta)                 log det M
//   inverse(theta, m)              M^-1
//   cholesky(theta, l)             L, lower triangular with positive diagonal, M = L L^T
//   matrix_dK(theta, idx, m)       d^K M / dtheta_idx[0] ... dtheta_idx[K-1], K = 1..6
//   log_det_dK(theta, idx)         likewise for log det M
//   inverse_dK, cholesky_dK        likewise for M^-1 and L
//
// idx is a sorted tuple of parameter indices, idx[0] <= ... <= idx[K-1]. The
// matrix derivatives return false, and leave the buffer untouched, when the
// derivative is identically zero for that tuple by the structure (for
// instance a mixed tuple of diagonal_pdmat); a log_det_dK that is identically
// zero returns 0.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/bijector/scalar.hpp>

#include <concepts>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Position of entry \f$(i, j)\f$, \f$j \le i\f$, in packed lower-triangular storage by rows.
 *
 * Computes the 0-based linear index:
 * \f[
 * k = \frac{i(i + 1)}{2} + j
 * \f]
 *
 * \param i Row index (\f$i \ge 0\f$).
 * \param j Column index (\f$0 \le j \le i\f$).
 * \return 0-based packed linear index.
 *
 * \see packed_size()
 */
LAPLACE_HD constexpr int packed_index(int i, int j) { return i * (i + 1) / 2 + j; }

/**
 * \ingroup pdmat
 * \brief Number of values of a packed symmetric or lower-triangular \f$p \times p\f$ matrix.
 *
 * Computes the total number of entries in packed lower-triangular storage:
 * \f[
 * N = \frac{p(p + 1)}{2}
 * \f]
 *
 * \param p Matrix dimension (\f$p \ge 0\f$).
 * \return Number of packed elements.
 *
 * \see packed_index()
 */
LAPLACE_HD constexpr int packed_size(int p) { return p * (p + 1) / 2; }

// The derivatives under their public names, one per order and quantity. A
// structure S derives from pdmat_orders<S> and implements the templates
// matrix_dk<K>, log_det_dk<K>, inverse_dk<K> and cholesky_dk<K>.
/**
 * \ingroup pdmat
 * \brief CRTP base providing the named derivative functions of orders 1 to 6.
 *
 * A structured matrix class \c S derives from `pdmat_orders<S>` and implements
 * the function templates `matrix_dk<K>`, `log_det_dk<K>`, `inverse_dk<K>`,
 * and `cholesky_dk<K>` for \f$K \in \{1, \dots, 6\}\f$.
 *
 * \tparam S Derived structured matrix class modeling \ref pdmat_structure.
 */
template <class S>
struct pdmat_orders {
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 1 along parameter index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial M(\theta)}{\partial\theta_{i_1}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 1 parameter index.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d1(), inverse_d1(), cholesky_d1()
   */
  LAPLACE_HD bool matrix_d1(const double* t, const int (&i)[1], double* m) const { return s().template matrix_dk<1>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 2 along parameter indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 M(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 2 parameter indices \f$i_1 \le i_2\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d2(), inverse_d2(), cholesky_d2()
   */
  LAPLACE_HD bool matrix_d2(const double* t, const int (&i)[2], double* m) const { return s().template matrix_dk<2>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 3 along parameter indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 M(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 3 parameter indices \f$i_1 \le i_2 \le i_3\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d3(), inverse_d3(), cholesky_d3()
   */
  LAPLACE_HD bool matrix_d3(const double* t, const int (&i)[3], double* m) const { return s().template matrix_dk<3>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 4 along parameter indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_4}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 4 parameter indices \f$i_1 \le \dots \le i_4\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d4(), inverse_d4(), cholesky_d4()
   */
  LAPLACE_HD bool matrix_d4(const double* t, const int (&i)[4], double* m) const { return s().template matrix_dk<4>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 5 along parameter indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_5}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 5 parameter indices \f$i_1 \le \dots \le i_5\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d5(), inverse_d5(), cholesky_d5()
   */
  LAPLACE_HD bool matrix_d5(const double* t, const int (&i)[5], double* m) const { return s().template matrix_dk<5>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order 6 along parameter indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_6}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 6 parameter indices \f$i_1 \le \dots \le i_6\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see matrix(), log_det_d6(), inverse_d6(), cholesky_d6()
   */
  LAPLACE_HD bool matrix_d6(const double* t, const int (&i)[6], double* m) const { return s().template matrix_dk<6>(t, i, m); }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 1 along parameter index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial \log \det M(\theta)}{\partial\theta_{i_1}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 1 parameter index.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d1(), inverse_d1(), cholesky_d1()
   */
  LAPLACE_HD double log_det_d1(const double* t, const int (&i)[1]) const { return s().template log_det_dk<1>(t, i); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 2 along parameter indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 \log \det M(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 2 parameter indices \f$i_1 \le i_2\f$.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d2(), inverse_d2(), cholesky_d2()
   */
  LAPLACE_HD double log_det_d2(const double* t, const int (&i)[2]) const { return s().template log_det_dk<2>(t, i); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 3 along parameter indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 \log \det M(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 3 parameter indices \f$i_1 \le i_2 \le i_3\f$.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d3(), inverse_d3(), cholesky_d3()
   */
  LAPLACE_HD double log_det_d3(const double* t, const int (&i)[3]) const { return s().template log_det_dk<3>(t, i); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 4 along parameter indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 \log \det M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_4}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 4 parameter indices \f$i_1 \le \dots \le i_4\f$.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d4(), inverse_d4(), cholesky_d4()
   */
  LAPLACE_HD double log_det_d4(const double* t, const int (&i)[4]) const { return s().template log_det_dk<4>(t, i); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 5 along parameter indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 \log \det M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_5}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 5 parameter indices \f$i_1 \le \dots \le i_5\f$.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d5(), inverse_d5(), cholesky_d5()
   */
  LAPLACE_HD double log_det_d5(const double* t, const int (&i)[5]) const { return s().template log_det_dk<5>(t, i); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order 6 along parameter indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 \log \det M(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_6}}
   * \f]
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 6 parameter indices \f$i_1 \le \dots \le i_6\f$.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \see log_det(), matrix_d6(), inverse_d6(), cholesky_d6()
   */
  LAPLACE_HD double log_det_d6(const double* t, const int (&i)[6]) const { return s().template log_det_dk<6>(t, i); }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 1 along parameter index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial M^{-1}(\theta)}{\partial\theta_{i_1}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 1 parameter index.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d1(), log_det_d1(), cholesky_d1()
   */
  LAPLACE_HD bool inverse_d1(const double* t, const int (&i)[1], double* m) const { return s().template inverse_dk<1>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 2 along parameter indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 M^{-1}(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 2 parameter indices \f$i_1 \le i_2\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d2(), log_det_d2(), cholesky_d2()
   */
  LAPLACE_HD bool inverse_d2(const double* t, const int (&i)[2], double* m) const { return s().template inverse_dk<2>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 3 along parameter indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 M^{-1}(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 3 parameter indices \f$i_1 \le i_2 \le i_3\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d3(), log_det_d3(), cholesky_d3()
   */
  LAPLACE_HD bool inverse_d3(const double* t, const int (&i)[3], double* m) const { return s().template inverse_dk<3>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 4 along parameter indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 M^{-1}(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_4}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 4 parameter indices \f$i_1 \le \dots \le i_4\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d4(), log_det_d4(), cholesky_d4()
   */
  LAPLACE_HD bool inverse_d4(const double* t, const int (&i)[4], double* m) const { return s().template inverse_dk<4>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 5 along parameter indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 M^{-1}(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_5}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 5 parameter indices \f$i_1 \le \dots \le i_5\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d5(), log_det_d5(), cholesky_d5()
   */
  LAPLACE_HD bool inverse_d5(const double* t, const int (&i)[5], double* m) const { return s().template inverse_dk<5>(t, i, m); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order 6 along parameter indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 M^{-1}(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_6}}
   * \f]
   * into the caller-provided packed buffer \c m.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 6 parameter indices \f$i_1 \le \dots \le i_6\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see inverse(), matrix_d6(), log_det_d6(), cholesky_d6()
   */
  LAPLACE_HD bool inverse_d6(const double* t, const int (&i)[6], double* m) const { return s().template inverse_dk<6>(t, i, m); }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 1 along parameter index \f$i_1\f$.
   *
   * Computes the first-order partial derivative
   * \f[
   * \frac{\partial L(\theta)}{\partial\theta_{i_1}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 1 parameter index.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d1(), log_det_d1(), inverse_d1()
   */
  LAPLACE_HD bool cholesky_d1(const double* t, const int (&i)[1], double* l) const { return s().template cholesky_dk<1>(t, i, l); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 2 along parameter indices \f$i_1 \le i_2\f$.
   *
   * Computes the second-order partial derivative
   * \f[
   * \frac{\partial^2 L(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 2 parameter indices \f$i_1 \le i_2\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d2(), log_det_d2(), inverse_d2()
   */
  LAPLACE_HD bool cholesky_d2(const double* t, const int (&i)[2], double* l) const { return s().template cholesky_dk<2>(t, i, l); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 3 along parameter indices \f$i_1 \le i_2 \le i_3\f$.
   *
   * Computes the third-order partial derivative
   * \f[
   * \frac{\partial^3 L(\theta)}{\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 3 parameter indices \f$i_1 \le i_2 \le i_3\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d3(), log_det_d3(), inverse_d3()
   */
  LAPLACE_HD bool cholesky_d3(const double* t, const int (&i)[3], double* l) const { return s().template cholesky_dk<3>(t, i, l); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 4 along parameter indices \f$i_1 \le \dots \le i_4\f$.
   *
   * Computes the fourth-order partial derivative
   * \f[
   * \frac{\partial^4 L(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_4}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 4 parameter indices \f$i_1 \le \dots \le i_4\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d4(), log_det_d4(), inverse_d4()
   */
  LAPLACE_HD bool cholesky_d4(const double* t, const int (&i)[4], double* l) const { return s().template cholesky_dk<4>(t, i, l); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 5 along parameter indices \f$i_1 \le \dots \le i_5\f$.
   *
   * Computes the fifth-order partial derivative
   * \f[
   * \frac{\partial^5 L(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_5}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 5 parameter indices \f$i_1 \le \dots \le i_5\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d5(), log_det_d5(), inverse_d5()
   */
  LAPLACE_HD bool cholesky_d5(const double* t, const int (&i)[5], double* l) const { return s().template cholesky_dk<5>(t, i, l); }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order 6 along parameter indices \f$i_1 \le \dots \le i_6\f$.
   *
   * Computes the sixth-order partial derivative
   * \f[
   * \frac{\partial^6 L(\theta)}{\partial\theta_{i_1}\cdots\partial\theta_{i_6}}
   * \f]
   * into the caller-provided packed buffer \c l.
   *
   * \param t Pointer to parameter vector \f$\theta\f$.
   * \param i Sorted tuple of 6 parameter indices \f$i_1 \le \dots \le i_6\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if computed; \c false and buffer left untouched if identically zero for that tuple.
   *
   * \see cholesky(), matrix_d6(), log_det_d6(), inverse_d6()
   */
  LAPLACE_HD bool cholesky_d6(const double* t, const int (&i)[6], double* l) const { return s().template cholesky_dk<6>(t, i, l); }

 private:
  LAPLACE_HD const S& s() const { return static_cast<const S&>(*this); }
};

/**
 * \ingroup pdmat
 * \brief Concept defining what every structured positive definite matrix supplies.
 *
 * A type \c S models `pdmat_structure` if it provides runtime dimension `dim`,
 * parameter count `n_params()`, compile-time descriptors `unit_diagonal`,
 * `closed_inverse`, and `closed_factor`, matrix evaluation `matrix()`,
 * log-determinant `log_det()`, and the named derivative methods
 * `matrix_d1()`, `matrix_d6()`, `log_det_d1()`, and `log_det_d6()`.
 */
template <class S>
concept pdmat_structure = requires(const S s, const double* t, double* m, const int (&i1)[1], const int (&i6)[6]) {
  { s.dim } -> std::convertible_to<int>;
  { s.n_params() } -> std::convertible_to<int>;
  { S::unit_diagonal } -> std::convertible_to<bool>;
  { S::closed_inverse } -> std::convertible_to<bool>;
  { S::closed_factor } -> std::convertible_to<bool>;
  s.matrix(t, m);
  { s.log_det(t) } -> std::same_as<double>;
  { s.matrix_d1(t, i1, m) } -> std::same_as<bool>;
  { s.matrix_d6(t, i6, m) } -> std::same_as<bool>;
  { s.log_det_d1(t, i1) } -> std::same_as<double>;
  { s.log_det_d6(t, i6) } -> std::same_as<double>;
};

namespace detail_pdmat {

namespace m = laplace::math;

LAPLACE_HD_FORCEINLINE double ipow(double x, int n) {  // n >= 0, by squaring
  double r = 1.0;
  while (n > 0) {
    if (n & 1) r *= x;
    x *= x;
    n >>= 1;
  }
  return r;
}

LAPLACE_HD_FORCEINLINE void fill_zero(double* m, int n) {
  for (int k = 0; k < n; ++k) m[k] = 0.0;
}

// expit(x) and its derivatives of orders 1..n (n <= 6) into d[0..n], from one
// exponential, with the formulas of the inverse of the logit bijector
// (bijector/scalar.hpp): with s = expit(x), q = 1 - s, U = s q, D = q - s,
// expit^(K) = U, U D, U (1 - 6U), U D (1 - 12U), U (1 - 30U + 120U^2),
// U D (1 - 60U + 360U^2).
LAPLACE_HD void expit_all(double x, int n, double* d) {
  const double a = x < 0.0 ? -x : x, e = m::exp(-a), r = 1.0 / (1.0 + e);
  const double big = r, small = e * r;  // expit(|x|), expit(-|x|)
  const double U = big * small, D = (x < 0.0 ? 1.0 : -1.0) * (-Kokkos::expm1(-a)) * r;
  d[0] = x < 0.0 ? small : big;
  if (n >= 1) d[1] = U;
  if (n >= 2) d[2] = U * D;
  if (n >= 3) d[3] = U * (1.0 - 6.0 * U);
  if (n >= 4) d[4] = U * D * (1.0 - 12.0 * U);
  if (n >= 5) d[5] = U * (1.0 - 30.0 * U + 120.0 * U * U);
  if (n >= 6) d[6] = U * D * (1.0 - 60.0 * U + 360.0 * U * U);
}
// one order alone, from the logit bijector itself (up to 3 ulp more accurate
// than expit_all on the compound symmetry derivatives, measured 2026-11-12)
LAPLACE_HD double expit_d(int n, double x) {
  const bijector::scalar_bijector b{bijector::scalar_bijector_map::logit};
  switch (n) {
    case 0: return b.inverse(x);
    case 1: return b.inverse_d1(x);
    case 2: return b.inverse_d2(x);
    case 3: return b.inverse_d3(x);
    case 4: return b.inverse_d4(x);
    case 5: return b.inverse_d5(x);
    case 6: return b.inverse_d6(x);
  }
  return m::detail::quiet_nan();
}

// tanh(theta) = T, sech^2 = S = 1 - T^2 and sech, from e = exp(-2|theta|), so
// that S does not cancel as |theta| grows.
struct tanh_parts {
  double T, S, sech;
};
LAPLACE_HD tanh_parts tanh_of(double theta) {
  const double a = theta < 0.0 ? -theta : theta;
  const double e = m::exp(-2.0 * a), d = 1.0 + e;
  const double T = -Kokkos::expm1(-2.0 * a) / d;
  return {theta < 0.0 ? -T : T, 4.0 * e / (d * d), 2.0 * m::exp(-a) / d};
}

// d^K/dtheta^K of T^k sech^h, h = 0 or 1. With D T = S and D S^b = -2b T S^b,
// D^K T^k sech^h = sum_n c_n T^(k + K - 2n) S^n sech^h, n = 0..K, and one more
// derivative maps c_n to c'_(n+1) += (k + K - 2n) c_n, c'_n -= (2n + h) c_n.
// Every term keeps the factor S, so nothing cancels as T -> +-1.
LAPLACE_HD double tanh_power_d(int K, int k, int h, const tanh_parts& t) {
  double c[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int s = 0; s < K; ++s)
    for (int n = s; n >= 0; --n) {
      const double cn = c[n];
      c[n + 1] += (k + s - 2 * n) * cn;
      c[n] = -(2 * n + h) * cn;
    }
  double r = 0.0;
  for (int n = K; n >= 0; --n) {
    const int a = k + K - 2 * n;
    if (c[n] == 0.0 || a < 0) continue;
    r += c[n] * ipow(t.T, a) * ipow(t.S, n);
  }
  return h ? r * t.sech : r;
}

// log(A + B e^theta), A >= 0, B > 0, into d[0], and its derivatives of orders
// 1..n into d[1..n]: the j-th is expit^(j-1)(t), t = theta + log(B/A), or
// 1, 0, 0, ... when A = 0.
LAPLACE_HD void log_lin_exp_all(double A, double B, double theta, int n, double* d) {
  if (A == 0.0) {
    d[0] = m::log(B) + theta;
    for (int j = 1; j <= n; ++j) d[j] = j == 1 ? 1.0 : 0.0;
    return;
  }
  const double t = theta + m::log(B / A);
  double e[7];
  expit_all(t, n > 0 ? n - 1 : 0, e);
  // log(1 + e^t) = -log expit(-t)
  d[0] = m::log(A) + (t > 0.0 ? t + Kokkos::log1p(m::exp(-t)) : Kokkos::log1p(m::exp(t)));
  for (int j = 1; j <= n; ++j) d[j] = e[j - 1];
}

// d^n/dtheta^n exp(g(theta)) from eg = exp(g) and g[1..n]: eg Y_n(g1, ..., gn),
// the complete Bell polynomial.
LAPLACE_HD double exp_chain(int n, double eg, const double* g) {
  if (n == 0) return eg;
  double s = 0.0;
  for (int k = n; k >= 1; --k) s += m::bell_partial(n, k, g);
  return eg * s;
}

}  // namespace detail_pdmat
}  // namespace laplace::pdmat
