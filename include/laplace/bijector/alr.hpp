// The additive log-ratio bijector between the simplex of K categories and
// R^(K-1) (note/bijectors.md section 4, decisions 17-19; interface in
// coupled.hpp).
//
//   forward   u_a = log(pi_a / pi_K), a = 1..K-1
//   inverse   pi_a = exp(u_a + l), pi_K = exp(l), l = -log(1 + sum_b e^(u_b))
//   log |det J_h(u)| = sum_(a=1..K) log pi_a = sum_a u_a + K l
//
// The value v is the whole vector pi (K entries), so that pi_K, when small,
// is used as given rather than as 1 - sum of the others. The reference
// coordinates of the forward derivatives and of the log-Jacobians are
// pi_1..pi_(K-1), with pi_K = 1 - sum_b pi_b. inverse_log gives log pi
// without underflow.
//
// Derivatives (0-based indices below, category K is index K-1):
//   forward, closed:  d^k u_a / dpi_t = [t = (a, .., a)] (-1)^(k-1) (k-1)!/pi_a^k + (k-1)!/pi_K^k
//   forward log-Jacobian: [t = (a, .., a)] (-1)^k (k-1)!/pi_a^k + (k-1)!/pi_K^k
//   inverse: from log pi_c = u_c + l, d_b l = -pi_b, so d_b pi_c = pi_c (delta_bc - pi_b):
//     d_S l = -d_(S \ j) pi_(cat j),
//     d_S pi_c = sum_(T in S \ j) d_T pi_c (d_(S \ T) l + [S \ T = {j}, cat j = c]),
//   with j the lowest position of S
//   over the subsets S of the tuple's positions (coupled.hpp); a category c
//   outside the tuple has d_S pi_c = pi_c G[S], G = d_S exp(l)/exp(l), the
//   same for all of them.
//   inverse log-Jacobian: 1 - K pi_b at order 1, K d_t l beyond.
#pragma once

#include <laplace/bijector/coupled.hpp>

namespace laplace::bijector {

/**
 * \ingroup bijector_coupled
 * \brief The additive log-ratio bijector between the simplex of \f$K\f$ categories and \f$\mathbb{R}^{K-1}\f$.
 *
 * Maps a probability vector \f$\pi \in \mathbb{R}^K\f$ on the open simplex (\f$\pi_a > 0\f$, \f$\sum_a \pi_a = 1\f$)
 * to an unconstrained vector \f$u \in \mathbb{R}^{K-1}\f$.
 *
 * Reference coordinates for forward derivatives and log-Jacobians are \f$\pi_0, \dots, \pi_{K-2}\f$,
 * with \f$\pi_{K-1} = 1 - \sum_{b<K-1} \pi_b\f$.
 *
 * \see coupled_bijector_orders, bijector_coupled
 */
struct alr_bijector : coupled_bijector_orders<alr_bijector> {
  /**
   * \ingroup bijector_coupled
   * \brief Number of categories \f$K \ge 2\f$.
   */
  int dim = 2;  // K, the number of categories, K >= 2

  /**
   * \ingroup bijector_coupled
   * \brief Default constructor, initializing with \f$K = 2\f$ categories.
   */
  LAPLACE_HD alr_bijector() = default;

  /**
   * \ingroup bijector_coupled
   * \brief Constructs an ALR bijector with the given number of categories \f$K\f$.
   *
   * \param categories Number of categories (\f$K \ge 2\f$).
   */
  LAPLACE_HD explicit alr_bijector(int categories) : dim(categories) {}

  /**
   * \ingroup bijector_coupled
   * \brief Number of free unconstrained coordinates, \f$n = K - 1\f$.
   *
   * \return Length of free vector \f$u\f$.
   */
  LAPLACE_HD int free_size() const { return dim - 1; }

  /**
   * \ingroup bijector_coupled
   * \brief Number of stored constrained coordinates, \f$m = K\f$.
   *
   * \return Length of probability vector \f$\pi\f$.
   */
  LAPLACE_HD int value_size() const { return dim; }

  /**
   * \ingroup bijector_coupled
   * \brief Required capacity of caller scratch buffer in doubles.
   *
   * \return Number of doubles required by scratch buffer \c w.
   */
  // l, G and one array per distinct category of a tuple of order 6
  LAPLACE_HD int scratch_size() const { return (2 + detail_coupled::max_order) * detail_coupled::max_masks; }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward map \f$u = g(v)\f$: \f$u_a = \log(\pi_a / \pi_K)\f$.
   *
   * Computes
   * \f[
   * u_a = \log \pi_a - \log \pi_{K-1}, \quad a = 0, \dots, K - 2
   * \f]
   * The reference category \f$\pi_{K-1}\f$ is used as given in \c v rather than as \f$1 - \sum_{b<K-1} \pi_b\f$.
   *
   * \param v Input probability vector \f$\pi\f$ of \c value_size() doubles.
   * \param u Output free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer (unused).
   *
   * Measured accuracy: within 3.8 ulpN on gcc and CUDA.
   * \see inverse(), forward_log_jacobian(), forward_dk()
   */
  LAPLACE_HD void forward(const double* v, double* u, [[maybe_unused]] double* w) const {
    const double lk = detail_coupled::m::log(v[dim - 1]);
    for (int a = 0; a + 1 < dim; ++a) u[a] = detail_coupled::m::log(v[a]) - lk;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the logarithm of the inverse map, \f$\log \pi\f$, without underflow.
   *
   * Computes \f$\log \pi_a = u_a + l\f$ for \f$a = 0, \dots, K - 2\f$ and \f$\log \pi_{K-1} = l\f$,
   * where \f$l = -\log(1 + \sum_{b=0}^{K-2} e^{u_b})\f$.
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param lv Output vector of \c value_size() doubles receiving \f$\log \pi\f$.
   * \param w Caller scratch buffer (unused).
   *
   * \see inverse(), inverse_log_jacobian()
   */
  LAPLACE_HD void inverse_log(const double* u, double* lv, [[maybe_unused]] double* w) const {
    const double l = log_last(u);
    for (int a = 0; a + 1 < dim; ++a) lv[a] = u[a] + l;
    lv[dim - 1] = l;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse map \f$\pi = h(u)\f$: \f$\pi_a = \exp(u_a + l)\f$, \f$\pi_K = \exp(l)\f$.
   *
   * Computes
   * \f[
   * \pi_a = \exp(u_a + l) \quad (a = 0, \dots, K - 2), \quad \pi_{K-1} = \exp(l)
   * \f]
   * where \f$l = -\log(1 + \sum_{b=0}^{K-2} e^{u_b})\f$.
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param v Output probability vector \f$\pi\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   *
   * Measured accuracy: within 90.5 ulpN on gcc / 55.2 ulpN on CUDA.
   * \see forward(), inverse_log(), inverse_log_jacobian(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* u, double* v, [[maybe_unused]] double* w) const {
    const double l = log_last(u);
    for (int a = 0; a + 1 < dim; ++a) v[a] = detail_coupled::m::exp(u[a] + l);
    v[dim - 1] = detail_coupled::m::exp(l);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the forward log-Jacobian \f$\log|\det J_g(v)|\f$ in reference coordinates.
   *
   * Computes
   * \f[
   * \log\lvert\det J_g(v)\rvert = -\sum_{a=0}^{K-1} \log \pi_a
   * \f]
   *
   * \param v Input probability vector \f$\pi\f$ of \c value_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return Value of forward log-Jacobian.
   *
   * Measured accuracy: within 3.4 ulp on gcc and CUDA.
   * \see forward(), inverse_log_jacobian(), forward_log_jacobian_dk()
   */
  LAPLACE_HD double forward_log_jacobian(const double* v, [[maybe_unused]] double* w) const {
    double s = 0.0;
    for (int a = 0; a < dim; ++a) s -= detail_coupled::m::log(v[a]);
    return s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates the inverse log-Jacobian \f$\log|\det J_h(u)|\f$.
   *
   * Computes
   * \f[
   * \log\lvert\det J_h(u)\rvert = \sum_{a=0}^{K-1} \log \pi_a = \sum_{a=0}^{K-2} u_a + K l
   * \f]
   *
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return Value of inverse log-Jacobian.
   *
   * Measured accuracy: within 90.5 ulp on gcc / 93.9 ulp on CUDA.
   * \see inverse(), forward_log_jacobian(), inverse_log_jacobian_dk()
   */
  LAPLACE_HD double inverse_log_jacobian(const double* u, [[maybe_unused]] double* w) const {
    double s = dim * log_last(u);
    for (int a = 0; a + 1 < dim; ++a) s += u[a];
    return s;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx.
   *
   * Computes the closed-form derivative
   * \f[
   * \frac{\partial^K u_a}{\partial\pi_{t_1}\cdots\partial\pi_{t_K}} = [t = (a, \dots, a)] (-1)^{K-1} \frac{(K-1)!}{\pi_a^K} + \frac{(K-1)!}{\pi_{K-1}^K}
   * \f]
   * using difference-of-powers factoring for even \f$K\f$ when \f$t = (a, \dots, a)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input probability vector \f$\pi\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param out Output buffer of at least \c free_size() doubles.
   * \param w Caller scratch buffer (unused).
   * \return \c true (forward derivatives are never structurally zero).
   *
   * Measured accuracy: within 3.8 ulpN on gcc and CUDA.
   * \see forward(), forward_log_jacobian_dk(), forward_d1()
   */
  template <int K>
  LAPLACE_HD bool forward_dk(const double* v, const int (&idx)[K], double* out, [[maybe_unused]] double* w) const {
    namespace ds = detail_scalar;
    const double last = ds::cpow<K>(ds::fact<K - 1>(), 1.0 / v[dim - 1]);
    for (int a = 0; a + 1 < dim; ++a) out[a] = last;
    if (idx[0] == idx[K - 1]) {
      const int a = idx[0];
      // (K-1)! ((-1)^(K-1) x^K + y^K), x = 1/pi_a, y = 1/pi_K; the even orders as a difference of powers
      if constexpr (K % 2 == 1) out[a] += ds::cpow<K>(ds::fact<K - 1>(), 1.0 / v[a]);
      else out[a] = ds::fact<K - 1>() * pow_difference<K>(v[a], v[dim - 1]);
    }
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates forward log-Jacobian partial derivatives of order \f$K\f$ along reference coordinate tuple \c idx.
   *
   * Computes
   * \f[
   * \frac{\partial^K \log\lvert\det J_g(v)\rvert}{\partial\pi_{t_1}\cdots\partial\pi_{t_K}} = [t = (a, \dots, a)] (-1)^K \frac{(K-1)!}{\pi_a^K} + \frac{(K-1)!}{\pi_{K-1}^K}
   * \f]
   * using difference-of-powers factoring for odd \f$K\f$ when \f$t = (a, \dots, a)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param v Input probability vector \f$\pi\f$ of \c value_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ reference coordinate indices.
   * \param w Caller scratch buffer (unused).
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 3.4 ulp on gcc and CUDA.
   * \see forward_log_jacobian(), forward_dk(), forward_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double forward_log_jacobian_dk(const double* v, const int (&idx)[K], [[maybe_unused]] double* w) const {
    namespace ds = detail_scalar;
    const double last = ds::cpow<K>(ds::fact<K - 1>(), 1.0 / v[dim - 1]);
    if (idx[0] != idx[K - 1]) return last;
    const int a = idx[0];
    // (K-1)! ((-1)^K x^K + y^K); the odd orders as a difference of powers
    if constexpr (K % 2 == 0) return last + ds::cpow<K>(ds::fact<K - 1>(), 1.0 / v[a]);
    else return ds::fact<K - 1>() * pow_difference<K>(v[a], v[dim - 1]);
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse partial derivatives of order \f$K\f$ along free coordinate tuple \c idx via sub-tuple recursion.
   *
   * Differentiates \f$\pi_c = \exp(u_c + l)\f$ using Leibniz's rule over the subsets of the tuple's positions,
   * evaluating \f$1 - \pi_c\f$ as the sum of all other probabilities.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param out Output buffer of at least \c value_size() doubles.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true (inverse derivatives of ALR are dense).
   *
   * Measured accuracy: within 90.5 ulpN on gcc / 55.2 ulpN on CUDA.
   * \see inverse(), inverse_log_jacobian_dk(), inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* u, const int (&idx)[K], double* out, double* w) const {
    constexpr int full = (1 << K) - 1;
    const double l = log_last(u);
    int cats[K], nc = 0;
    const double* G = recursion<K>(u, idx, l, w, cats, nc);
    const double g = G[full];
    for (int a = 0; a + 1 < dim; ++a) out[a] = detail_coupled::m::exp(u[a] + l) * g;
    out[dim - 1] = detail_coupled::m::exp(l) * g;
    for (int c = 0; c < nc; ++c) out[cats[c]] = w[(2 + c) * detail_coupled::max_masks + full];
    return true;
  }

  /**
   * \ingroup bijector_coupled
   * \brief Evaluates inverse log-Jacobian partial derivatives of order \f$K\f$ along free coordinate tuple \c idx.
   *
   * Returns \f$1 - K \pi_b\f$ at order 1, and \f$K \partial_t l\f$ from the sub-tuple recursion at orders \f$K \ge 2\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param u Input free vector \f$u\f$ of \c free_size() doubles.
   * \param idx Sorted tuple of \f$K\f$ free coordinate indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the partial derivative.
   *
   * Measured accuracy: within 90.5 ulp on gcc / 93.9 ulp on CUDA.
   * \see inverse_log_jacobian(), inverse_dk(), inverse_log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double inverse_log_jacobian_dk(const double* u, const int (&idx)[K], double* w) const {
    const double l = log_last(u);
    if constexpr (K == 1) return 1.0 - dim * detail_coupled::m::exp(u[idx[0]] + l);
    int cats[K], nc = 0;
    recursion<K>(u, idx, l, w, cats, nc);
    return dim * w[(1 << K) - 1];
  }

 private:
  // 1/pk^K - 1/pa^K = (pa - pk)/(pa pk) S_K(1/pa, 1/pk), no difference of two large powers
  template <int K>
  LAPLACE_HD static double pow_difference(double pa, double pk) {
    const double x = 1.0 / pa, y = 1.0 / pk;
    return (pa - pk) * x * y * detail_scalar::geo_sum<K>(x, y);
  }

  // l = log pi_K = -log(1 + sum_b e^(u_b)), with the largest term factored out
  LAPLACE_HD double log_last(const double* u) const {
    int am = 0;
    for (int a = 1; a + 1 < dim; ++a)
      if (u[a] > u[am]) am = a;
    const double mx = u[am];
    double s = 0.0;
    if (mx <= 0.0) {
      for (int a = 0; a + 1 < dim; ++a) s += detail_coupled::m::exp(u[a]);
      return -Kokkos::log1p(s);
    }
    s = detail_coupled::m::exp(-mx);
    for (int a = 0; a + 1 < dim; ++a)
      if (a != am) s += detail_coupled::m::exp(u[a] - mx);
    return -(mx + Kokkos::log1p(s));
  }

  // fills w: [0, 64) d_S l, [64, 128) G[S], then d_S pi_c for each distinct
  // category c of the tuple (cats[0..nc)); returns G
  template <int K>
  LAPLACE_HD const double* recursion(const double* u, const int (&idx)[K], double l, double* w, int (&cats)[K],
                                     int& nc) const {
    constexpr int NM = 1 << K, M = detail_coupled::max_masks;
    int slot[K];
    nc = 0;
    for (int q = 0; q < K; ++q) {
      if (q == 0 || idx[q] != idx[q - 1]) cats[nc++] = idx[q];
      slot[q] = nc - 1;
    }
    double* L = w;
    double* G = w + M;
    double* D = w + 2 * M;
    L[0] = l;
    G[0] = 1.0;
    // 1 - pi_c as the sum of the other probabilities, exact where pi_c is near 1
    double om[K];
    for (int c = 0; c < nc; ++c) {
      D[c * M] = detail_coupled::m::exp(u[cats[c]] + l);
      double s = detail_coupled::m::exp(l);
      for (int a = 0; a + 1 < dim; ++a)
        if (a != cats[c]) s += detail_coupled::m::exp(u[a] + l);
      om[c] = s;
    }
    for (int S = 1; S < NM; ++S) {
      const int j = detail_coupled::lowest_bit(S), R = S ^ (1 << j);
      L[S] = -D[slot[j] * M + R];
      double g = 0.0;
      for (int T = R;; T = (T - 1) & R) {
        g += G[T] * L[S ^ T];
        if (T == 0) break;
      }
      G[S] = g;
      for (int c = 0; c < nc; ++c) {
        double d = 0.0;
        for (int T = R;; T = (T - 1) & R) {
          const int Q = S ^ T;  // contains j
          // d_Q log pi_c: 1 - pi_c on the single position j of category c, d_Q l otherwise
          const double dl = (Q & (Q - 1)) == 0 && slot[j] == c ? om[c] : L[Q];
          d += D[c * M + T] * dl;
          if (T == 0) break;
        }
        D[c * M + S] = d;
      }
    }
    return G;
  }
};

}  // namespace laplace::bijector
