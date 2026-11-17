// mvstudent_t_state<G>: the state of mvstudent_t1_distrib and mvstudent_t2_distrib
// (note/distrib_multivariate.md section 15), in a header of its own so that the
// manual documents it (detail/ is not documented).
#pragma once

namespace laplace::distrib {

/**
 * \ingroup distrib_mvstudent_t1
 * \brief Precomputed state of a multivariate Student t distribution at fixed \f$(\theta, \nu)\f$.
 *
 * Holds the Gaussian state `gauss` for the same matrix parameter (the matrix of
 * the quadratic form, the Cholesky factor, the log determinant or log
 * pseudo-determinant, and any attached \f$\theta\f$-derivative table), the
 * degrees of freedom \f$\nu\f$, the dimension \f$\rho\f$ of the law (\f$d\f$,
 * or the rank in the degenerate class), and the normalizing constant
 * \f$c_\rho(\nu)\f$ with its derivatives up to order six. Caller-owned storage
 * backing `gauss` and `theta` must remain alive while this state is in use.
 *
 * \tparam G State type of the multivariate Gaussian with the same matrix parameter.
 * \see mvstudent_t1_distrib, mvstudent_t2_distrib, distrib_mvstudent_t1, distrib_mvstudent_t2
 */
template <class G>
struct mvstudent_t_state {
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Precomputed state of the Gaussian with the same matrix parameter \f$M(\theta)\f$.
   */
  G gauss;
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Degrees of freedom \f$\nu > 0\f$.
   */
  double nu;
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Dimension of the law: \f$d\f$ in the proper class, or the rank \f$\rho\f$ in the degenerate class.
   */
  int rho;
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Normalizing constant \f$c_\rho(\nu)\f$ and its \f$\nu\f$-derivatives \f$c_\rho^{(k)}(\nu)\f$ for \f$k = 0, \dots, 6\f$.
   */
  double cnu[7];
};

}  // namespace laplace::distrib
