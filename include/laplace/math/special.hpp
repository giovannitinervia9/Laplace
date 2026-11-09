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

LAPLACE_HD_FORCEINLINE double lgamma(double x) { return boost::math::lgamma(x, detail::boost_policy()); }
LAPLACE_HD_FORCEINLINE double digamma(double x) { return boost::math::digamma(x, detail::boost_policy()); }
LAPLACE_HD_FORCEINLINE double trigamma(double x) { return boost::math::trigamma(x, detail::boost_policy()); }

// psi^(n)(x), n = 0..6, x > 0
LAPLACE_HD_FORCEINLINE double polygamma(int n, double x) {
  if (n == 0) return digamma(x);
  if (n == 1) return trigamma(x);
  return own::polygamma(n, x);
}

}  // namespace laplace::math
