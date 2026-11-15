// Starting values of a fit from a sample (laplace::distrib; note/optimizers.md,
// section 8).
//
//   summarize_sample(y)                    the distrib_sample_summary of y (core.hpp)
//   distrib_start_value(prototype, s)      the distrib's own start_value(s) member,
//                                          a copy of prototype with theta set
//
// Every distrib carries its own member (a moment estimate, or the closed-form
// MLE where one exists); a distrib without one does not compile here. The
// member returns theta strictly inside its domain (start_inside), so that the
// default bijectors map it to a finite free coordinate even when the MLE lies
// on the boundary (a sample of zeros under a bernoulli). Host code: the summary
// copies y to the host once and reads the order statistics by nth_element.
#pragma once

#include <laplace/distrib/core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace laplace::distrib {

namespace detail_start {

// R's quantile type 7 of the sorted-by-selection v at probability p (v is reordered)
inline double quantile_type7(std::vector<double>& v, double p) {
  const double h = (static_cast<double>(v.size()) - 1.0) * p;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(lo), v.end());
  const double a = v[lo];
  if (lo + 1 >= v.size()) return a;
  const double b = *std::min_element(v.begin() + static_cast<std::ptrdiff_t>(lo) + 1, v.end());
  return a + (h - static_cast<double>(lo)) * (b - a);
}

}  // namespace detail_start

/**
 * \ingroup distrib_start
 * \brief Computes summary statistics of a sample held on the host.
 *
 * Host-only routine that evaluates sample moments, extreme values, positive-observation
 * log and reciprocal moments, and type-7 quartiles (obtained by partial sorting via
 * selection in time linear in the sample size). Returns zero in all fields when \p y is empty,
 * and zero in the four positive-only fields when no observation is positive.
 * \param y Sample vector (taken by value and reordered internally for selection).
 * \return Populated \c distrib_sample_summary struct.
 * \see distrib_sample_summary, distrib_start_value(), distrib_start
 */
// The summary of a sample held on the host.
inline distrib_sample_summary summarize_sample(std::vector<double> y) {
  distrib_sample_summary s;
  const double n = static_cast<double>(y.size());
  s.count = n;
  if (y.empty()) return s;
  for (double v : y) s.mean += v;
  s.mean /= n;
  for (double v : y) s.variance += (v - s.mean) * (v - s.mean);
  s.variance /= n;
  // the standardized third central moment, which the one family whose third
  // parameter is the skewness starts from; zero where the sample has no spread
  if (s.variance > 0.0) {
    for (double v : y) s.skewness += (v - s.mean) * (v - s.mean) * (v - s.mean);
    s.skewness /= n * s.variance * std::sqrt(s.variance);
  }
  s.minimum = *std::min_element(y.begin(), y.end());
  s.maximum = *std::max_element(y.begin(), y.end());
  for (double v : y)
    if (v > 0.0) {
      s.positive_count += 1.0;
      s.mean_log += std::log(v);
      s.mean_inverse += 1.0 / v;
    }
  if (s.positive_count > 0.0) {
    s.mean_log /= s.positive_count;
    s.mean_inverse /= s.positive_count;
    for (double v : y)
      if (v > 0.0) s.variance_log += (std::log(v) - s.mean_log) * (std::log(v) - s.mean_log);
    s.variance_log /= s.positive_count;
  }
  s.median = detail_start::quantile_type7(y, 0.5);
  for (double v : y) s.mean_abs_dev_median += std::abs(v - s.median);
  s.mean_abs_dev_median /= n;
  s.lower_quartile = detail_start::quantile_type7(y, 0.25);
  s.upper_quartile = detail_start::quantile_type7(y, 0.75);
  return s;
}

/**
 * \ingroup distrib_start
 * \brief Computes summary statistics of a sample stored in a \c Kokkos::View.
 *
 * Host-only routine that copies \p y from its memory space to the host once and delegates
 * to summarize_sample(std::vector<double>).
 * \tparam Props View properties deduced from \p y.
 * \param y Sample view in any Kokkos memory space.
 * \return Populated \c distrib_sample_summary struct.
 * \see distrib_sample_summary, distrib_start_value(), distrib_start
 */
// The summary of a sample in any memory space (copied to the host once).
template <class... Props>
distrib_sample_summary summarize_sample(const Kokkos::View<const double*, Props...>& y) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  return summarize_sample(std::vector<double>(h.data(), h.data() + h.extent(0)));
}

/**
 * \ingroup distrib_start
 * \brief Computes starting parameters for a distribution from a sample summary.
 *
 * Host-only routine returning \c prototype.start_value(s), preserving any non-parameter
 * fields of \p prototype (such as \c size on \c binomial_distrib). Fails to compile with
 * a \c static_assert if \p D has no \c start_value member.
 * \tparam D Distribution type.
 * \param prototype Distribution instance supplying any non-parameter fields.
 * \param s Sample summary computed by summarize_sample().
 * \return Copy of \p prototype with starting parameters strictly inside their open domains.
 * \see summarize_sample(), distrib_sample_summary, start_inside(), distrib_start
 */
// The distrib's starting value: its start_value member, applied to the prototype
// (which carries the fields that are not parameters, such as binomial's size).
template <class D>
D distrib_start_value(const D& prototype, const distrib_sample_summary& s) {
  static_assert(requires { prototype.start_value(s); },
                "distrib_start_value: this distrib has no start_value member; supply a start");
  return prototype.start_value(s);
}

}  // namespace laplace::distrib
