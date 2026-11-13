// cauchy_distrib (laplace::distrib): the Cauchy distribution in its location mu and scale
// sigma > 0. With z = (y - mu)/sigma,
//
//   l(mu, sigma; y) = -log(pi sigma) - log(1 + z^2),       F(y) = 1/2 + atan(z)/pi.
//
// l, log F and log(1 - F) are C^inf in both parameters and in y: no kinks.
//
// The z-derivatives are closed forms in u_j(z) = Re((z + i)^(-j)) = (1 + z^2)^(-j/2) cos(j theta),
// theta = atan2(1, z) (theta is the argument of z + i). u_j is computed by the recurrence
//   u_0 = 1,  u_1 = z w,  u_(j+1) = w (2 z u_j - u_(j-1)),   w = 1/(1 + z^2),
// and then
//   d^j/dz^j [-log(1 + z^2)] = 2 (-1)^j (j - 1)! u_j(z).
// For the cumulative functions, Lambda(z) = log(1 - F(z)) = log(theta(z)/pi) has first
// derivative q = -w/theta, which obeys the Riccati equation
//   q' = -q^2 - p q,   p(z) = 2 z w = 1/(z + i) + 1/(z - i),
// and p^(i) = 2 (-1)^i i! u_(i+1). Leibniz on the Riccati equation gives
//   q^(j+1) = -sum_(i=0)^j C(j, i) (q^(i) + p^(i)) q^(j-i),
// and d^j/dz^j Lambda = q^(j-1). log F(z) = Lambda(-z), so its derivatives carry the sign (-1)^j.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_cauchy {

// pi to the double precision of the platform constant
inline constexpr double pi_value = 3.14159265358979323846;

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// u[j] = Re((z + i)^(-j)) = (1 + z^2)^(-j/2) cos(j theta), j = 0..K, by the recurrence in z and w
template <int K>
LAPLACE_HD void u_derivs(double z, double* u) {
  const double w = 1.0 / (1.0 + z * z);
  u[0] = 1.0;
  u[1] = z * w;
  for (int j = 1; j < K; ++j) u[j + 1] = w * (2.0 * z * u[j] - u[j - 1]);
}

// L[j] = d^j/dz^j [-log(1 + z^2)], j = 1..K
template <int K>
LAPLACE_HD void logf_derivs(double z, double* L) {
  double u[K + 1];
  u_derivs<K>(z, u);
  for (int j = 1; j <= K; ++j) L[j] = 2.0 * sign_of_power(j) * fact(j - 1) * u[j];
}

// L[j] = d^j/dx^j log(1 - F(x)) = d^j/dx^j log(theta(x)/pi), j = 1..K, by the Riccati
// recurrence of the header comment (q = L[1], q^(i) = L[i + 1]).
template <int K>
LAPLACE_HD void log_ccdf0_derivs(double x, double* L) {
  double u[K + 1];
  u_derivs<K>(x, u);
  const double w = 1.0 / (1.0 + x * x);
  double p[K], q[K];
  for (int i = 0; i < K; ++i) p[i] = 2.0 * sign_of_power(i) * fact(i) * u[i + 1];
  q[0] = -w / Kokkos::atan2(1.0, x);
  for (int j = 0; j + 1 < K; ++j) {
    double s = 0.0;
    for (int i = 0; i <= j; ++i) s += binomial_coef(j, i) * (q[i] + p[i]) * q[j - i];
    q[j + 1] = -s;
  }
  for (int j = 1; j <= K; ++j) L[j] = q[j - 1];
}

// Far tail (|z| >= far_split on the side where log F or log(1 - F) tends to 0, the side
// where the z-polynomials of detail_location_scale::mixed_dk cancel): with s = 1/|z| the
// function is Lambda = log(1 - atan(s)/pi) = sum_(n >= 1) d_n s^n, and the carry to (mu, sigma)
// is a sum of monomials. With kappa = sign(z), a = K - b derivatives in mu, b in sigma:
//   d^a/dmu^a d^b/dsigma^b Lambda = sigma^-K kappa^a sum_n d_n rise(n, a) fall(n, b) |z|^(-n-a),
// rise(n, a) = n (n + 1) ... (n + a - 1), fall(n, b) = n (n - 1) ... (n - b + 1) (0 if b > n).
// The coefficients d_n are those of log(1 - atan(s)/pi) (tools/reference/cauchy_logistic_forms.py,
// which checks them against SymPy's series); the sum is cut at far_terms.
// The coefficients below are the Taylor coefficients of log(1 - atan(s)/pi), computed with mpmath
// at 80 digits from the recurrence for log of a series; tools/reference/cauchy_logistic_forms.py
// checks the first twelve against SymPy's own series.
constexpr int far_terms = 100;
constexpr double far_split = 2.0;
#define LAPLACE_CAUCHY_FAR_COEF -0.3183098861837906715377675, -0.05066059182116888572193973, 0.09535278391686372745111516, 0.03120723231710817335067163, \
-0.05356501848763564834060514, -0.02264455753591862746639991, 0.03648098562136395317256342, 0.01779855358096177898071911, \
-0.02733511667641241239426563, -0.01467206806028395606950955, 0.02168999525747323033848866, 0.0124844036344296403699575, \
-0.0178819652514310491765912, -0.01086638901680432399673947, 0.01515174593211998421493645, 0.009620415527471095354479002, \
-0.01310521844687044385949121, -0.008631002952570432845978851, 0.01151821590715596625331238, 0.007826085047723933709591547, \
-0.01025418181261025132693381, -0.00715833060079228315579782, 0.009225358795152644954108691, 0.006595354519185231447594084, \
-0.008372876196993066416205333, -0.006114241609654603608144089, 0.007655824769147759493949018, 0.005698316235174236682974714, \
-0.007044926306823545774329198, -0.005335151179171830007683051, 0.006518695853094571710823863, 0.005015294005973010285075395, \
-0.00606102441306355279078015, -0.004731426137728389431156954, 0.005659605779747773763185141, 0.004477792193067395360214014, \
-0.005304883297071235377942777, -0.004249803329585387283725019, 0.00498932703000789348781416, 0.004043755639729362374595007, \
-0.004706926725124856546218408, -0.003856626434011753442143766, 0.004452829157984524073051513, 0.003685924369035326014453673, \
-0.004223074197536569366538566, -0.003529577506057163711095463, 0.004014399681797493396368485, 0.00338584854651348474411315, \
-0.003824095103560265978695488, -0.003253269841408790405528452, 0.003649890471105510812328882, 0.003130592991062168128248011, \
-0.003489870886123194002533892, -0.003016749349351541703042773, 0.003342410173866226053471672, 0.002910818774207631061541236, \
-0.003206118800047952280529448, -0.002812004682131664457711418, 0.003079802621378364823639444, 0.002719613970536842403787722, \
-0.0029624299365985746923766, -0.002633040734038723865853204, 0.002853104958440752682617832, 0.002551752963414940079654361, \
-0.002751046297044138718090596, -0.002475281608425781219809112, 0.002655569387415699909523777, 0.002403211528248322569022532, \
-0.00256607204509980141829027, -0.002335173959910056942405163, 0.002482022521114607202379301, 0.002270840215603090328980204, \
-0.002402949567364337024340542, -0.002209916381049249395990399, 0.002328434129766365576997172, 0.002152138834131574135277632, \
-0.002258102367215143537028257, -0.002097270439395422034203943, 0.002191619756688513858811972, 0.002045097302370227955975121, \
-0.002128686092962717036690049, -0.001995425989897929087729443, 0.00206903122896306170170829, 0.001948081140206934793261421, \
-0.002012411432264616818105736, -0.001902903400412021464690357, 0.001958606256551737579120326, 0.001859747640257999233415157, \
-0.001907415845356899415355042, -0.001818481399871529753478741, 0.00185865860019338611530187, 0.001778983536509833384061465, \
-0.00181216915708308743446041, -0.001741143041157933886039632, 0.00176779662508036294234942, 0.001704858000606847364269517, \
-0.001725403048183584911334443, -0.001670034684561253921807274, 0.001684862058378046735193724, 0.001636586740547059858537432
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double far_dev[far_terms] = {LAPLACE_CAUCHY_FAR_COEF};
#endif
inline constexpr double far_host[far_terms] = {LAPLACE_CAUCHY_FAR_COEF};
LAPLACE_HD_FORCEINLINE double far_coef(int n) {
  KOKKOS_IF_ON_DEVICE((return far_dev[n - 1];))
  KOKKOS_IF_ON_HOST((return far_host[n - 1];))
}

template <int K>
LAPLACE_HD sym_tensor<2, K> far_tail_dk(double z, double sigma) {
  const double az = Kokkos::fabs(z), inv = 1.0 / az, kap = z < 0.0 ? -1.0 : 1.0;
  double ia[K + 1];  // inv^a
  ia[0] = 1.0;
  for (int a = 1; a <= K; ++a) ia[a] = ia[a - 1] * inv;
  double acc[K + 1] = {};  // acc[b] = sum_n d_n rise(n, a) fall(n, b) inv^(n + a)
  double pn = 1.0;         // inv^n
  for (int n = 1; n <= far_terms; ++n) {
    pn *= inv;
    const double dn = far_coef(n);
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      double c = 1.0;
      for (int k = 0; k < a; ++k) c *= (n + k);
      for (int k = 0; k < b; ++k) c *= (n - k);
      acc[b] += dn * c * pn * ia[a];
    }
  }
  sym_tensor<2, K> t;
  double isk = 1.0;
  for (int k = 0; k < K; ++k) isk /= sigma;
  for (int b = 0; b <= K; ++b) {
    const int a = K - b;
    t[b] = (a % 2 == 0 ? 1.0 : kap) * acc[b] * isk;
  }
  return t;
}

}  // namespace detail_cauchy

/**
 * \ingroup distrib_cauchy
 * \brief Univariate Cauchy distribution parameterized by location \f$\mu\f$ and scale \f$\sigma > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \sigma; y) = -\log(\pi\sigma) - \log(1 + z^2)\f$ with
 * \f$z = (y - \mu)/\sigma\f$, its parameter derivatives in closed form, the Fisher information
 * \f$\operatorname{diag}(1/(2\sigma^2), 1/(2\sigma^2))\f$, and the cumulative distribution function
 * \f$F(y) = 1/2 + \arctan(z)/\pi\f$ with both tails written without cancellation.
 * \see distrib_core
 */
struct cauchy_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Scale parameter \f$\sigma \in (0, \infty)\f$.
  double sigma;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_cauchy
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth, no kinks).
   */
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c sigma (both \c C^inf, neither moves the support).
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma) = -\log(\pi\sigma) - \log(1 + z^2)\f$.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    return -math::log(detail_cauchy::pi_value * sigma) - Kokkos::log1p(z * z);
  }

  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \sigma)\f$ in \f$(\mu, \sigma)\f$.
   *
   * The density is \f$\log f = \Lambda(z) - \log\sigma - \log\pi\f$ with \f$\Lambda(z) = -\log(1 + z^2)\f$; the
   * derivatives of \f$\Lambda\f$ are carried to \f$(\mu, \sigma)\f$ by \c detail_location_scale::mixed_dk, and
   * \f$-\log\sigma\f$ adds \f$(-1)^K (K - 1)!\,\sigma^{-K}\f$ to the component with all \f$K\f$ indices in \f$\sigma\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double L[K + 1];
    detail_cauchy::logf_derivs<K>(z, L);
    sym_tensor<2, K> t = detail_location_scale::mixed_dk<K>(L, z, sigma);
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[K] += detail_cauchy::sign_of_power(K) * detail_cauchy::fact(K - 1) * isk;
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \sigma)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  // l depends on y and mu only through y - mu, so d/dy = -d/dmu and
  // d^J/dy^J d^a/dmu^a d^b/dsigma^b l = (-1)^J d^(a+J)/dmu^(a+J) d^b/dsigma^b l. The component at
  // position b of logpdf_dk<J + K> is the one with b derivatives in sigma and K - b + J in mu.
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \sigma)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * Evaluates \f$\frac{\partial^{J+K}}{\partial y^J \partial\theta^K} \log f(y; \mu, \sigma)\f$ as a symmetric tensor of order
   * \f$K\f$ in \f$(\mu, \sigma)\f$. The density depends on \f$y - \mu\f$ and \f$\sigma\f$ only, so
   * \f$\partial_y = -\partial_\mu\f$ and the component with \f$b\f$ derivatives in \f$\sigma\f$ is
   * \f$(-1)^J\f$ times that of \c logpdf_dk<J + K>. For \f$J = 0\f$ the result is \c logpdf_dk<K> (or the value for \f$K = 0\f$).
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \sigma)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \see logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<2, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<2, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<2, K> r;
      const double s = detail_cauchy::sign_of_power(J);
      for (int b = 0; b <= K; ++b) r[b] = s * t[b];
      return r;
    }
  }

  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \sigma) = \operatorname{diag}(1/(2\sigma^2), 1/(2\sigma^2))\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\sigma\sigma}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double i2 = 1.0 / (2.0 * sigma * sigma);
    sym_tensor<2, 2> t;
    t[0] = i2;
    t[2] = i2;
    return t;
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma)\f$.
   *
   * Both diagonal entries are \f$1/(2\sigma^2)\f$, so their \f$K\f$-th derivatives in \f$\sigma\f$ are
   * \f$(-1)^K (K+1)!/(2\sigma^{K+2})\f$ and the derivatives in \f$\mu\f$ vanish.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    double d = detail_cauchy::fact(K + 1) / (2.0 * sigma * sigma);
    for (int k = 0; k < K; ++k) d /= -sigma;
    expected_info_tensor<2, K> t;
    t.pair[0][K] = d;
    t.pair[2][K] = d;
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \sigma)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the cumulative distribution function \f$F(y) = \operatorname{atan2}(1, -z)/\pi\f$.
   *
   * \f$\operatorname{atan2}(1, -z) = \pi/2 + \arctan z\f$ for every real \f$z\f$, so each tail is
   * a single arctangent of the same sign of argument.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double z = (y - mu) / sigma;
    return Kokkos::atan2(1.0, -z) / detail_cauchy::pi_value;
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \operatorname{atan2}(1, z)/\pi\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return Kokkos::atan2(1.0, z) / detail_cauchy::pi_value;
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   *
   * For \f$z \ge 0\f$, \f$\log F = \operatorname{log1p}(-(1 - F))\f$, which keeps the digits of \f$F\f$ near 1.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double z = (y - mu) / sigma;
    return z < 0.0 ? math::log(Kokkos::atan2(1.0, -z) / detail_cauchy::pi_value)
                   : Kokkos::log1p(-Kokkos::atan2(1.0, z) / detail_cauchy::pi_value);
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return z > 0.0 ? math::log(Kokkos::atan2(1.0, z) / detail_cauchy::pi_value)
                   : Kokkos::log1p(-Kokkos::atan2(1.0, -z) / detail_cauchy::pi_value);
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu + \sigma\tan(\pi(p - 1/2))\f$.
   *
   * For \f$p \le 1/2\f$ the form \f$\mu - \sigma\cot(\pi p)\f$ is used, and for \f$p > 1/2\f$ the form
   * \f$\mu + \sigma\cot(\pi(1 - p))\f$, where \f$1 - p\f$ is exact (Sterbenz), so neither end loses digits.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const {
    if (p <= 0.5) return mu - sigma / Kokkos::tan(detail_cauchy::pi_value * p);
    return mu + sigma / Kokkos::tan(detail_cauchy::pi_value * (1.0 - p));
  }

  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \sigma)\f$.
   *
   * \f$\log F(z) = \Lambda_{\mathrm{up}}(-z)\f$, so its \f$j\f$-th derivative in \f$z\f$ is \f$(-1)^j\Lambda_{\mathrm{up}}^{(j)}(-z)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z >= detail_cauchy::far_split) return detail_cauchy::far_tail_dk<K>(z, sigma);
    double L[K + 1];
    detail_cauchy::log_ccdf0_derivs<K>(-z, L);
    for (int j = 1; j <= K; j += 2) L[j] = -L[j];
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \ingroup distrib_cauchy
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z <= -detail_cauchy::far_split) return detail_cauchy::far_tail_dk<K>(z, sigma);
    double L[K + 1];
    detail_cauchy::log_ccdf0_derivs<K>(z, L);
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }
  /**
   * \ingroup distrib_cauchy
   * \brief Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$: none, for both parameters.
   * \return Kinks descriptor \c distrib_kinks with \c count = 0.
   */
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /**
   * \ingroup distrib_cauchy
   * \brief Non-smooth points in the observation \f$y\f$: none.
   * \return Kinks descriptor \c distrib_kinks with \c count = 0.
   */
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
};

}  // namespace laplace::distrib
