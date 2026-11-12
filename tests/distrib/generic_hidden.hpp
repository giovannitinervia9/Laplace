// A pilot distribution with its own quantile and expected information hidden
// (and, for a discrete one, its cdf family): the generic routes of
// laplace::distrib take over, and the hidden members are their twins.
#pragma once

#include <laplace/distrib/core.hpp>

namespace laplace_test {

template <class D>
struct hidden_distrib {
  D d;
  static constexpr laplace::distrib::distrib_kind kind = D::kind;
  static constexpr int n_params = D::n_params;
  static constexpr const laplace::distrib::distrib_param_info* params = D::params;
  static constexpr bool cont = kind == laplace::distrib::distrib_kind::continuous;

  LAPLACE_HD double logpdf(double y) const requires cont { return d.logpdf(y); }
  LAPLACE_HD auto logpdf_d1(double y) const requires cont { return d.logpdf_d1(y); }
  LAPLACE_HD auto logpdf_d2(double y) const requires cont { return d.logpdf_d2(y); }
  LAPLACE_HD auto logpdf_d3(double y) const requires cont { return d.logpdf_d3(y); }
  LAPLACE_HD auto logpdf_d4(double y) const requires cont { return d.logpdf_d4(y); }
  LAPLACE_HD auto logpdf_d5(double y) const requires cont { return d.logpdf_d5(y); }
  LAPLACE_HD auto logpdf_d6(double y) const requires cont { return d.logpdf_d6(y); }
  LAPLACE_HD double logpmf(double y) const requires(!cont) { return d.logpmf(y); }
  LAPLACE_HD auto logpmf_d1(double y) const requires(!cont) { return d.logpmf_d1(y); }
  LAPLACE_HD auto logpmf_d2(double y) const requires(!cont) { return d.logpmf_d2(y); }
  LAPLACE_HD auto logpmf_d3(double y) const requires(!cont) { return d.logpmf_d3(y); }
  LAPLACE_HD auto logpmf_d4(double y) const requires(!cont) { return d.logpmf_d4(y); }
  LAPLACE_HD auto logpmf_d5(double y) const requires(!cont) { return d.logpmf_d5(y); }
  LAPLACE_HD auto logpmf_d6(double y) const requires(!cont) { return d.logpmf_d6(y); }

  LAPLACE_HD double cdf(double y) const requires cont { return d.cdf(y); }
  LAPLACE_HD double ccdf(double y) const requires cont { return d.ccdf(y); }
  LAPLACE_HD double log_cdf(double y) const requires cont { return d.log_cdf(y); }
  LAPLACE_HD double log_ccdf(double y) const requires cont { return d.log_ccdf(y); }
  LAPLACE_HD auto y_kinks() const requires cont { return d.y_kinks(); }

  LAPLACE_HD double support_lower() const { return d.support_lower(); }
  LAPLACE_HD double support_upper() const { return d.support_upper(); }
  LAPLACE_HD double center() const { return d.center(); }
  LAPLACE_HD double scale() const { return d.scale(); }
};

template <class D>
LAPLACE_HD hidden_distrib<D> hide(const D& d) {
  return hidden_distrib<D>{d};
}

}  // namespace laplace_test
