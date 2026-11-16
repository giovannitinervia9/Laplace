// Special functions that the rest of Laplace calls (laplace::math::...).
//
// Each one is the implementation chosen by measurement (docs/numerics.md):
//   lgamma          Boost.Math (accurate near the zeros at 1 and 2 on every
//                   platform; the Rtools libm loses all relative accuracy
//                   there). On CUDA the platform's is 4.5 times faster with
//                   the same accuracy: a per-backend choice is pending.
//   digamma         Boost.Math (2.1 ulp; the own version is worse near the root)
//   trigamma        Boost.Math (1.8-2.0 ulp, faster than the own version)
//   polygamma n>=2  Laplace's own (3-9 ulp; Boost has no GPU version, and on
//                   the host it reaches 72-830 ulp and is 6 times slower)
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/polygamma.hpp>

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

namespace laplace::math {

namespace detail {
namespace bp = boost::math::policies;
// Never throw (the code runs in kernels), never promote double to long double
// (the device has none, and host and device should follow the same path).
using boost_policy = bp::policy<bp::domain_error<bp::ignore_error>, bp::pole_error<bp::ignore_error>,
                                bp::overflow_error<bp::ignore_error>, bp::evaluation_error<bp::ignore_error>,
                                bp::promote_double<false>>;
}  // namespace detail

/**
 * \ingroup gamma
 * \brief Natural logarithm of the gamma function, \f$\log \Gamma(x)\f$.
 *
 * Evaluates \f$\operatorname{lgamma}(x) = \log \Gamma(x)\f$ for \f$x > 0\f$ via
 * Boost.Math (`boost::math::lgamma`) with a non-throwing policy that disables
 * `double` to `long double` promotion.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\log \Gamma(x)\f$.
 *
 * \par Domain and special values
 * Defined for \f$x > 0\f$, with zeros at \f$x = 1.0\f$ and \f$x = 2.0\f$.
 *
 * \par Measured accuracy
 * Maximum error is 2.42 ulp on gcc, MSVC host, and CUDA (test bound 4.0 ulp);
 * the correctly rounded share is 0.72 on gcc and 0.74 on CUDA.
 *
 * \see gamma, digamma(), trigamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double lgamma(double x) { return boost::math::lgamma(x, detail::boost_policy()); }

/**
 * \ingroup gamma
 * \brief Digamma function \f$\psi(x) = \frac{\mathrm{d}}{\mathrm{d}x} \log \Gamma(x)\f$.
 *
 * Evaluates \f$\psi(x) = \Gamma'(x)/\Gamma(x)\f$ for \f$x > 0\f$ via Boost.Math
 * (`boost::math::digamma`) with a non-throwing policy that disables `double` to
 * `long double` promotion.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi(x)\f$.
 *
 * \par Domain and special values
 * Defined for \f$x > 0\f$; vanishes at \f$x_0 \approx 1.4616321449683623\f$, where
 * relative error is unbounded while absolute error remains small.
 *
 * \par Measured accuracy
 * Maximum error is 2.14 ulp and 2.14 ulp1 on gcc and CUDA (test bound 3.5 ulp1);
 * the correctly rounded share is 0.82 on gcc and 0.84 on CUDA.
 *
 * \see gamma, lgamma(), trigamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double digamma(double x) { return boost::math::digamma(x, detail::boost_policy()); }

/**
 * \ingroup gamma
 * \brief Trigamma function \f$\psi'(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \log \Gamma(x)\f$.
 *
 * Evaluates \f$\psi_1(x) = \psi'(x) = \frac{\mathrm{d}}{\mathrm{d}x}\psi(x)\f$ for
 * \f$x > 0\f$ via Boost.Math (`boost::math::trigamma`) with a non-throwing
 * policy that disables `double` to `long double` promotion.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi'(x)\f$.
 *
 * \par Domain and special values
 * Defined for \f$x > 0\f$; strictly positive and diverges to \f$+\infty\f$ as
 * \f$x \to 0^+\f$.
 *
 * \par Measured accuracy
 * Maximum error is 1.82 ulp (1.57 ulp1) on gcc and 2.03 ulp (2.03 ulp1) on CUDA
 * (test bound 3.0 ulp); the correctly rounded share is 0.82.
 *
 * \see gamma, lgamma(), digamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double trigamma(double x) { return boost::math::trigamma(x, detail::boost_policy()); }

// psi^(n)(x), n = 0..6, x > 0
/**
 * \ingroup gamma
 * \brief Polygamma function \f$\psi^{(n)}(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)\f$ for \f$n \in \{0, \dots, 6\}\f$.
 *
 * Evaluates \f$\psi^{(n)}(x) = \frac{\mathrm{d}^n}{\mathrm{d}x^n}\psi(x)\f$ by
 * dispatching \f$n = 0\f$ to digamma(), \f$n = 1\f$ to trigamma(), and
 * \f$n \in \{2, \dots, 6\}\f$ to own::polygamma().
 *
 * \param n Derivative order \f$n \in \{0, \dots, 6\}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi^{(n)}(x)\f$.
 *
 * \par Domain and special values
 * Returns quiet NaN for \f$n < 0\f$ or \f$n > 6\f$, and for \f$x \le 0\f$ or NaN
 * when \f$n \ge 2\f$. Overflows to \f$\pm\infty\f$ as \f$x \to 0^+\f$ for higher orders.
 *
 * \par Measured accuracy
 * Maximum error is 2.14 ulp1 for \f$n = 0\f$ (bound 3.5 ulp1), 1.82 ulp on gcc and
 * 2.03 ulp on CUDA for \f$n = 1\f$ (bound 3.0 ulp), and 3.36 to 9.44 ulp for
 * \f$n \in \{2, \dots, 6\}\f$ (bounds 5.0 to 14.0 ulp); see \ref gamma for per-order figures.
 *
 * \see gamma, lgamma(), digamma(), trigamma(), own::polygamma()
 */
LAPLACE_HD_FORCEINLINE double polygamma(int n, double x) {
  if (n == 0) return digamma(x);
  if (n == 1) return trigamma(x);
  return own::polygamma(n, x);
}

/**
 * \ingroup gamma
 * \brief Polygamma minus its leading asymptotic term, \f$\psi^{(n)}(x) - \mu_n(x)\f$.
 *
 * Forwards to own::polygamma_monomial_remainder() at every order, which defines
 * \f$\mu_n\f$, gives the thresholds of the two routes and states the measured
 * accuracy. Unlike polygamma(), there is no dispatch to
 * Boost at \f$n = 0\f$ and \f$n = 1\f$: Boost offers no remainder, and the expansion is
 * Laplace's own at every order.
 *
 * \param n Derivative order \f$n \in \{0, \dots, 6\}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi^{(n)}(x) - \mu_n(x)\f$.
 *
 * \see gamma, polygamma(), own::polygamma_monomial_remainder()
 */
LAPLACE_HD_FORCEINLINE double polygamma_monomial_remainder(int n, double x) {
  return own::polygamma_monomial_remainder(n, x);
}

}  // namespace laplace::math
