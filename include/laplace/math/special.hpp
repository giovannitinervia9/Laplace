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
 * Computes the logarithm of the gamma function:
 * \f[
 * \operatorname{lgamma}(x) = \log \Gamma(x)
 * \f]
 * where \f$x \in (0, \infty)\f$ is the positive real argument, evaluated via Boost.Math
 * (`boost::math::lgamma`) with `laplace::math::detail::boost_policy`.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\log \Gamma(x)\f$.
 *
 * \par Domain and special values
 * - Defined for \f$x > 0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.42 ulp on gcc, MSVC, and CUDA (test bound 4.0 ulp); correctly rounded share 0.72 (gcc) / 0.74 (CUDA).
 *
 * \see gamma, digamma(), trigamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double lgamma(double x) { return boost::math::lgamma(x, detail::boost_policy()); }

/**
 * \ingroup gamma
 * \brief Digamma function \f$\psi(x) = \frac{\mathrm{d}}{\mathrm{d}x} \log \Gamma(x)\f$.
 *
 * Computes the logarithmic derivative of the gamma function:
 * \f[
 * \psi(x) = \frac{\mathrm{d}}{\mathrm{d}x} \log \Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}
 * \f]
 * where \f$x \in (0, \infty)\f$ is the positive real argument, evaluated via Boost.Math
 * (`boost::math::digamma`) with `laplace::math::detail::boost_policy`.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi(x)\f$.
 *
 * \par Domain and special values
 * - Defined for \f$x > 0\f$.
 * - At the positive root \f$x_0 \approx 1.4616321449683623\f$: \f$\psi(x_0) = 0\f$; relative error is unbounded near \f$x_0\f$ while absolute error remains small.
 *
 * \par Measured accuracy
 * Maximum error 2.14 ulp (2.14 ulp1) on gcc and CUDA (test bound 3.5 ulp1); correctly rounded share 0.82 (gcc) / 0.84 (CUDA).
 *
 * \see gamma, lgamma(), trigamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double digamma(double x) { return boost::math::digamma(x, detail::boost_policy()); }

/**
 * \ingroup gamma
 * \brief Trigamma function \f$\psi'(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \log \Gamma(x)\f$.
 *
 * Computes the first derivative of the digamma function:
 * \f[
 * \psi_1(x) = \psi'(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \log \Gamma(x)
 * \f]
 * where \f$x \in (0, \infty)\f$ is the positive real argument, evaluated via Boost.Math
 * (`boost::math::trigamma`) with `laplace::math::detail::boost_policy`.
 *
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi'(x)\f$.
 *
 * \par Domain and special values
 * - Defined for \f$x > 0\f$.
 * - As \f$x \to 0^+\f$: \f$\psi'(x) \to +\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 1.82 ulp (gcc) / 2.03 ulp (CUDA), max error 1.57 ulp1 (gcc) / 2.03 ulp1 (CUDA), test bound 3.0 ulp; correctly rounded share 0.82.
 *
 * \see gamma, lgamma(), digamma(), polygamma()
 */
LAPLACE_HD_FORCEINLINE double trigamma(double x) { return boost::math::trigamma(x, detail::boost_policy()); }

// psi^(n)(x), n = 0..6, x > 0
/**
 * \ingroup gamma
 * \brief Polygamma function \f$\psi^{(n)}(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)\f$ of order \f$n \in \{0, \dots, 6\}\f$.
 *
 * Computes the \f$n\f$-th derivative of the digamma function:
 * \f[
 * \psi^{(n)}(x) = \frac{\mathrm{d}^n}{\mathrm{d}x^n} \psi(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)
 * \f]
 * where \f$n \in \{0, \dots, 6\}\f$ is the derivative order and \f$x \in (0, \infty)\f$ is the positive real argument.
 * Dispatches to digamma() for \f$n = 0\f$, to trigamma() for \f$n = 1\f$, and to own::polygamma() for \f$n \in \{2, \dots, 6\}\f$.
 *
 * \param n Derivative order \f$n \in \{0, \dots, 6\}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\psi^{(n)}(x)\f$.
 *
 * \par Domain and special values
 * - Defined for \f$x > 0\f$ with \f$n \in \{0, \dots, 6\}\f$.
 * - Returns quiet NaN for \f$n < 0\f$ or \f$n > 6\f$, and for \f$x \le 0\f$ when \f$n \ge 2\f$.
 * - For small \f$x\f$ and higher orders \f$n\f$, overflows to \f$\pm\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error by order \f$n\f$:
 * - \f$n = 0\f$: 2.14 ulp1 (test bound 3.5 ulp1).
 * - \f$n = 1\f$: 1.82 ulp gcc / 2.03 ulp CUDA (test bound 3.0 ulp).
 * - \f$n = 2\f$: 3.36 ulp (test bound 5.0 ulp).
 * - \f$n = 3\f$: 5.26 ulp (test bound 8.0 ulp).
 * - \f$n = 4\f$: 6.34 ulp (test bound 9.5 ulp).
 * - \f$n = 5\f$: 7.90 ulp (test bound 12.0 ulp).
 * - \f$n = 6\f$: 9.44 ulp (test bound 14.0 ulp).
 *
 * \see gamma, lgamma(), digamma(), trigamma()
 */
LAPLACE_HD_FORCEINLINE double polygamma(int n, double x) {
  if (n == 0) return digamma(x);
  if (n == 1) return trigamma(x);
  return own::polygamma(n, x);
}

}  // namespace laplace::math
