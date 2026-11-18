// mvpowerexp_state<G>: the state of mvpowerexp1_distrib and mvpowerexp2_distrib
// (note/distrib_multivariate.md section 7), in a header of its own so that the
// manual documents it (detail/ is not documented).
#pragma once

namespace laplace::distrib {

/**
 * \ingroup distrib_mvpowerexp1
 * \brief Precomputed state of a multivariate power exponential distribution at fixed \f$(\theta, \beta)\f$.
 *
 * Holds the Gaussian state `gauss` for the same matrix parameter (the matrix of
 * the quadratic form, the Cholesky factor, the log determinant, and any
 * attached \f$\theta\f$-derivative table), the shape \f$\beta > 0\f$, the
 * dimension \f$d\f$, and \f$\log Z(d, \beta)\f$ with its \f$\beta\f$-derivatives
 * up to order six. Caller-owned storage backing `gauss` and `theta` must remain
 * alive while this state is in use.
 *
 * \tparam G State type of the multivariate Gaussian with the same matrix parameter.
 * \see mvpowerexp1_distrib, mvpowerexp2_distrib, distrib_mvpowerexp1, distrib_mvpowerexp2
 */
template <class G>
struct mvpowerexp_state {
  /**
   * \ingroup distrib_mvpowerexp1
   * \brief Precomputed state of the Gaussian with the same matrix parameter \f$M(\theta)\f$.
   */
  G gauss;
  /**
   * \ingroup distrib_mvpowerexp1
   * \brief Shape \f$\beta > 0\f$.
   */
  double beta;
  /**
   * \ingroup distrib_mvpowerexp1
   * \brief Dimension \f$d\f$ of the law.
   */
  int dim;
  /**
   * \ingroup distrib_mvpowerexp1
   * \brief Log normalizing constant \f$\log Z(d, \beta)\f$ and its \f$\beta\f$-derivatives for \f$k = 0, \dots, 6\f$.
   */
  double lz[7];
};

}  // namespace laplace::distrib
