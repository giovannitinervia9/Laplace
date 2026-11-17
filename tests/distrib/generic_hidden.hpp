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

// A law that carries atoms with every closed route hidden: the log density and
// its derivatives, the descriptors, the support and the atoms are forwarded, and
// the cdf family, the quantile and the starting value deliberately are not, so
// distrib_cdf and the rest fall to the generic routes. The atoms reach them
// through the three descriptors of core.hpp and through nothing else, which is
// what the generic routes have to be right with.
template <class D>
struct atom_hidden_distrib {
  D d;
  static constexpr laplace::distrib::distrib_kind kind = D::kind;
  static constexpr int n_params = D::n_params;
  static constexpr const laplace::distrib::distrib_param_info* params = D::params;
  static constexpr int n_atoms = laplace::distrib::distrib_n_atoms<D>();
  static_assert(kind == laplace::distrib::distrib_kind::continuous,
                "atom_hidden_distrib is for a continuous law with atoms");

  LAPLACE_HD double logpdf(double y) const { return d.logpdf(y); }
  LAPLACE_HD auto logpdf_d1(double y) const { return d.template logpdf_dk<1>(y); }
  LAPLACE_HD auto logpdf_d2(double y) const { return d.template logpdf_dk<2>(y); }
  LAPLACE_HD auto logpdf_d3(double y) const { return d.template logpdf_dk<3>(y); }
  LAPLACE_HD auto logpdf_d4(double y) const { return d.template logpdf_dk<4>(y); }
  LAPLACE_HD auto logpdf_d5(double y) const { return d.template logpdf_dk<5>(y); }
  LAPLACE_HD auto logpdf_d6(double y) const { return d.template logpdf_dk<6>(y); }

  LAPLACE_HD double atom_point(int i) const { return laplace::distrib::distrib_atom_point(d, i); }
  LAPLACE_HD double atom_log_mass(int i) const { return laplace::distrib::distrib_atom_log_mass(d, i); }
  LAPLACE_HD auto y_kinks() const { return d.y_kinks(); }
  LAPLACE_HD auto param_kinks(int j, double y) const { return d.param_kinks(j, y); }

  LAPLACE_HD double support_lower() const { return d.support_lower(); }
  LAPLACE_HD double support_upper() const { return d.support_upper(); }
  LAPLACE_HD double center() const { return d.center(); }
  LAPLACE_HD double scale() const { return d.scale(); }
};

template <class D>
LAPLACE_HD atom_hidden_distrib<D> hide_atom_routes(const D& d) {
  return atom_hidden_distrib<D>{d};
}

}  // namespace laplace_test
