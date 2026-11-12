// The distrib pilot (gaussian1, laplace, poisson, gamma1; note/distrib.md): packing of the
// symmetric tensors, every derivative order against a 5-point difference of the order
// below, the expected information against the variance of the score (quadrature with
// a knot at the kink, or a sum over the support) and its derivatives against
// differences, the cdf family against identities, and a kernel launch of every order.
#include <doctest/doctest.h>

#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/laplace.hpp>
#include <laplace/distrib/poisson.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ld = laplace::distrib;

namespace {

template <class D>
D make(const double* t) {
  if constexpr (D::n_params == 1) return D{t[0]};
  else return D{t[0], t[1]};
}

template <int P, int K, class T>
std::vector<double> as_vec(const T& t) {
  return std::vector<double>(t.v, t.v + ld::sym_tensor<P, K>::size);
}

// the components of d^K l, K = 0 the value
template <class D>
std::vector<double> lderiv(const D& d, double y, int K) {
  constexpr int P = D::n_params;
  constexpr bool cont = D::kind == ld::distrib_kind::continuous;
  switch (K) {
    case 0: if constexpr (cont) return {d.logpdf(y)}; else return {d.logpmf(y)};
    case 1: if constexpr (cont) return as_vec<P, 1>(d.logpdf_d1(y)); else return as_vec<P, 1>(d.logpmf_d1(y));
    case 2: if constexpr (cont) return as_vec<P, 2>(d.logpdf_d2(y)); else return as_vec<P, 2>(d.logpmf_d2(y));
    case 3: if constexpr (cont) return as_vec<P, 3>(d.logpdf_d3(y)); else return as_vec<P, 3>(d.logpmf_d3(y));
    case 4: if constexpr (cont) return as_vec<P, 4>(d.logpdf_d4(y)); else return as_vec<P, 4>(d.logpmf_d4(y));
    case 5: if constexpr (cont) return as_vec<P, 5>(d.logpdf_d5(y)); else return as_vec<P, 5>(d.logpmf_d5(y));
    default: if constexpr (cont) return as_vec<P, 6>(d.logpdf_d6(y)); else return as_vec<P, 6>(d.logpmf_d6(y));
  }
}

// the components of d^K I, pair-major (pair a <= b, then the sorted K-index); K = 0 is I
template <int P, int K, class T>
std::vector<double> info_vec(const T& t) {
  std::vector<double> r;
  for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
    for (int i = 0; i < ld::sym_tensor_size(P, K); ++i) r.push_back(t.pair[p][i]);
  return r;
}
template <class D>
std::vector<double> ideriv(const D& d, int K) {
  constexpr int P = D::n_params;
  switch (K) {
    case 0: return as_vec<P, 2>(d.expected_info());
    case 1: return info_vec<P, 1>(d.expected_info_d1());
    case 2: return info_vec<P, 2>(d.expected_info_d2());
    case 3: return info_vec<P, 3>(d.expected_info_d3());
    default: return info_vec<P, 4>(d.expected_info_d4());
  }
}

// sorted multi-indices of length K over 0..P-1, in lexicographic order
void sorted_indices(int P, int K, int lo, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
  if ((int)cur.size() == K) {
    out.push_back(cur);
    return;
  }
  for (int v = lo; v < P; ++v) {
    cur.push_back(v);
    sorted_indices(P, K, v, cur, out);
    cur.pop_back();
  }
}
std::vector<std::vector<int>> sorted_indices(int P, int K) {
  std::vector<std::vector<int>> out;
  std::vector<int> cur;
  sorted_indices(P, K, 0, cur, out);
  return out;
}
int position_of(int P, const std::vector<int>& idx) {
  const auto all = sorted_indices(P, (int)idx.size());
  for (int i = 0; i < (int)all.size(); ++i)
    if (all[i] == idx) return i;
  return -1;
}

// 5-point difference of g(theta) in theta_j, step h (error O(h^4))
template <class G>
std::vector<double> diff5(const G& g, std::vector<double> th, int j, double h) {
  const double x = th[j];
  std::vector<std::vector<double>> f;
  for (double s : {-2.0, -1.0, 1.0, 2.0}) {
    th[j] = x + s * h;
    f.push_back(g(th.data()));
  }
  std::vector<double> r(f[0].size());
  for (size_t i = 0; i < r.size(); ++i) r[i] = (f[0][i] - 8.0 * f[1][i] + 8.0 * f[2][i] - f[3][i]) / (12.0 * h);
  return r;
}

// d^K l against the difference of d^(K-1) l in the last index, every component
template <class D>
void check_derivs(const std::vector<double>& th, double y, double rel) {
  constexpr int P = D::n_params;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = lderiv(make<D>(th.data()), y, K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return lderiv(make<D>(t), y, K - 1); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      const double e = exact[position_of(P, idx)], a = fd[position_of(P, lower)];
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y);
      CHECK(std::fabs(e - a) <= rel * scale);
    }
  }
}

// d^K log F (upper: log(1 - F)) of a location-scale distribution, K = 0 the value
template <class D>
std::vector<double> cderiv(const D& d, double y, int K, bool upper) {
  constexpr int P = D::n_params;
  switch (K) {
    case 0: return {upper ? d.log_ccdf(y) : d.log_cdf(y)};
    case 1: return as_vec<P, 1>(upper ? d.log_ccdf_d1(y) : d.log_cdf_d1(y));
    case 2: return as_vec<P, 2>(upper ? d.log_ccdf_d2(y) : d.log_cdf_d2(y));
    case 3: return as_vec<P, 3>(upper ? d.log_ccdf_d3(y) : d.log_cdf_d3(y));
    case 4: return as_vec<P, 4>(upper ? d.log_ccdf_d4(y) : d.log_cdf_d4(y));
    case 5: return as_vec<P, 5>(upper ? d.log_ccdf_d5(y) : d.log_cdf_d5(y));
    default: return as_vec<P, 6>(upper ? d.log_ccdf_d6(y) : d.log_cdf_d6(y));
  }
}
// zscale: the scale on which log F varies relative to the step (|z| for location-scale)
template <class D>
void check_cdf_derivs(const std::vector<double>& th, double y, bool upper, double rel, double zscale = 0.0) {
  constexpr int P = D::n_params;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = cderiv(make<D>(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(P, K)) {
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const int j = idx.back();
      const auto fd = diff5([&](const double* t) { return cderiv(make<D>(t), y, K - 1, upper); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])) / (1.0 + zscale));
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y << ", upper " << upper);
      // the absolute floor: where a derivative is exactly 0 (Poisson log F at y = 0 is -mu)
      CHECK(std::fabs(exact[position_of(P, idx)] - fd[position_of(P, lower)]) <= rel * scale + 1e-12);
    }
  }
}

// d^(J+K) l / dy^J dtheta^K of a two-parameter continuous distribution, J + K <= 6
template <class D>
std::vector<double> yderiv(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    case 0: return as_vec<2, 0>(d.template logpdf_dy_dk<0, 0>(y));
    case 1: return as_vec<2, 1>(d.template logpdf_dy_dk<0, 1>(y));
    case 2: return as_vec<2, 2>(d.template logpdf_dy_dk<0, 2>(y));
    case 3: return as_vec<2, 3>(d.template logpdf_dy_dk<0, 3>(y));
    case 4: return as_vec<2, 4>(d.template logpdf_dy_dk<0, 4>(y));
    case 5: return as_vec<2, 5>(d.template logpdf_dy_dk<0, 5>(y));
    case 6: return as_vec<2, 6>(d.template logpdf_dy_dk<0, 6>(y));
    case 10: return as_vec<2, 0>(d.template logpdf_dy_dk<1, 0>(y));
    case 11: return as_vec<2, 1>(d.template logpdf_dy_dk<1, 1>(y));
    case 12: return as_vec<2, 2>(d.template logpdf_dy_dk<1, 2>(y));
    case 13: return as_vec<2, 3>(d.template logpdf_dy_dk<1, 3>(y));
    case 14: return as_vec<2, 4>(d.template logpdf_dy_dk<1, 4>(y));
    case 15: return as_vec<2, 5>(d.template logpdf_dy_dk<1, 5>(y));
    case 20: return as_vec<2, 0>(d.template logpdf_dy_dk<2, 0>(y));
    case 21: return as_vec<2, 1>(d.template logpdf_dy_dk<2, 1>(y));
    case 22: return as_vec<2, 2>(d.template logpdf_dy_dk<2, 2>(y));
    case 23: return as_vec<2, 3>(d.template logpdf_dy_dk<2, 3>(y));
    case 24: return as_vec<2, 4>(d.template logpdf_dy_dk<2, 4>(y));
    case 30: return as_vec<2, 0>(d.template logpdf_dy_dk<3, 0>(y));
    case 31: return as_vec<2, 1>(d.template logpdf_dy_dk<3, 1>(y));
    case 32: return as_vec<2, 2>(d.template logpdf_dy_dk<3, 2>(y));
    case 33: return as_vec<2, 3>(d.template logpdf_dy_dk<3, 3>(y));
    case 40: return as_vec<2, 0>(d.template logpdf_dy_dk<4, 0>(y));
    case 41: return as_vec<2, 1>(d.template logpdf_dy_dk<4, 1>(y));
    case 42: return as_vec<2, 2>(d.template logpdf_dy_dk<4, 2>(y));
    case 50: return as_vec<2, 0>(d.template logpdf_dy_dk<5, 0>(y));
    case 51: return as_vec<2, 1>(d.template logpdf_dy_dk<5, 1>(y));
    case 60: return as_vec<2, 0>(d.template logpdf_dy_dk<6, 0>(y));
    default: return {};
  }
}

// every (J, K): against the difference in y of (J - 1, K) and in theta of (J, K - 1)
template <class D>
void check_y_derivs(const std::vector<double>& th, double y, double hy, double rel) {
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv(make<D>(th.data()), y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      // in y
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yderiv(make<D>(th.data()), y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12);
      }
      if (K == 0) continue;
      // in theta
      for (const auto& idx : sorted_indices(2, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const auto fd = diff5([&](const double* t) { return yderiv(make<D>(t), y, J, K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(2, idx) << ", y = " << y);
        CHECK(std::fabs(exact[position_of(2, idx)] - fd[position_of(2, lower)]) <= rel * scale + 1e-12);
      }
    }
}

// d^K I against the difference of d^(K-1) I
template <class D>
void check_info_derivs(const std::vector<double>& th, double rel) {
  constexpr int P = D::n_params;
  const int npair = ld::sym_tensor_size(P, 2);
  for (int K = 1; K <= 4; ++K) {
    const auto exact = ideriv(make<D>(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (int p = 0; p < npair; ++p)
      for (const auto& idx : sorted_indices(P, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ideriv(make<D>(t), K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(P, K - 1);
        const double e = exact[p * ld::sym_tensor_size(P, K) + position_of(P, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + position_of(P, lower)];
        INFO("K = " << K << ", pair " << p);
        CHECK(std::fabs(e - a) <= rel * scale);
      }
  }
}

// E[l_a l_b] by quadrature over the support, with knots at the y-kinks
template <class D>
std::vector<double> score_variance_continuous(const D& d) {
  constexpr int P = D::n_params;
  const auto yk = d.y_kinks();
  std::vector<double> r;
  for (int a = 0; a < P; ++a)
    for (int b = a; b < P; ++b) {
      auto f = [&](double y) {
        const auto s = d.logpdf_d1(y);
        return s[a] * s[b] * std::exp(d.logpdf(y));
      };
      r.push_back(laplace::math::quad::panels_quad(f, d.support_lower(), d.support_upper(), d.center(), d.scale(),
                                             yk.point, yk.count).value);
    }
  return r;
}

}  // namespace

TEST_CASE("distrib: packing of symmetric tensors") {
  for (int P = 1; P <= 4; ++P)
    for (int K = 1; K <= 6; ++K) {
      const auto all = sorted_indices(P, K);
      CHECK((int)all.size() == ld::sym_tensor_size(P, K));
    }
  // positions of every sorted index of (P, K) = (3, 3) and (4, 2) agree with the enumeration
  const auto a33 = sorted_indices(3, 3);
  for (int i = 0; i < (int)a33.size(); ++i) {
    const int idx[3] = {a33[i][0], a33[i][1], a33[i][2]};
    CHECK(ld::sym_tensor_position<3, 3>(idx) == i);
  }
  const auto a42 = sorted_indices(4, 2);
  for (int i = 0; i < (int)a42.size(); ++i) {
    const int idx[2] = {a42[i][0], a42[i][1]};
    CHECK(ld::sym_tensor_position<4, 2>(idx) == i);
  }
  // two parameters: position = number of indices equal to 1
  const int i2[4] = {0, 1, 1, 1};
  CHECK(ld::sym_tensor_position<2, 4>(i2) == 3);
}

TEST_CASE("distrib: descriptors") {
  CHECK(ld::gaussian1_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::poisson_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::gaussian1_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::laplace_distrib::params[0].regularity == 0);
  CHECK(ld::laplace_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::laplace_distrib::params[0].cdf_regularity == 1);
  CHECK(ld::gaussian1_distrib::params[1].cdf_regularity == ld::infinitely_smooth);
  const ld::laplace_distrib d{0.5, 2.0};
  CHECK(d.param_kinks(0, 1.25).count == 1);
  CHECK(d.param_kinks(0, 1.25).point[0] == 1.25);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().point[0] == 0.5);
}

TEST_CASE("distrib gaussian1: derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, 1e-8);
  check_derivs<ld::gaussian1_distrib>({-20.0, 0.7}, -18.5, 1e-8);
}

TEST_CASE("distrib laplace: derivatives against differences, away from the kink") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::laplace_distrib>({0.4, 1.3}, y, 1e-8);
  // at the kink: the mean of the one-sided derivatives in mu
  const ld::laplace_distrib d{0.4, 1.3};
  CHECK(d.logpdf_d1(0.4)[0] == 0.0);
}

TEST_CASE("distrib poisson: derivatives against differences") {
  for (double y : {0.0, 1.0, 4.0, 30.0}) check_derivs<ld::poisson_distrib>({3.5}, y, 1e-8);
  check_derivs<ld::poisson_distrib>({0.2}, 2.0, 1e-8);
}

TEST_CASE("distrib gamma1: derivatives against differences") {
  for (double y : {0.05, 0.9, 2.0, 2.1, 7.0}) check_derivs<ld::gamma1_distrib>({2.0, 0.3}, y, 1e-8);
  check_derivs<ld::gamma1_distrib>({1.5, 0.02}, 1.4, 1e-8);  // shape 50: asymptotic series
  check_derivs<ld::gamma1_distrib>({1.5, 4.0}, 0.3, 1e-8);   // shape 1/4: direct route
}

TEST_CASE("distrib: expected information against the variance of the score") {
  for (double phi : {0.02, 0.3, 4.0}) {
    const ld::gamma1_distrib d{2.0, phi};
    // tanh-sinh on [0, mu] (the integrable singularity y^(s-1) at 0), exp-sinh beyond
    std::vector<double> q;
    for (int a = 0; a < 2; ++a)
      for (int b = a; b < 2; ++b) {
        auto f = [&](double y) {
          const auto sc = d.logpdf_d1(y);
          return sc[a] * sc[b] * std::exp(d.logpdf(y));
        };
        q.push_back(laplace::math::quad::tanh_sinh_quad(f, 0.0, 2.0).value +
                    laplace::math::quad::exp_sinh_quad(f, 2.0, 2.0 * (1.0 + phi)).value);
      }
    const auto e = d.expected_info();
    INFO("gamma1, phi = " << phi);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-12 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-12 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-10 * e[2]);
  }
  {
    const ld::gaussian1_distrib d{0.4, 1.3};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-12 * e[2]);
  }
  {
    // the knot at the kink y = mu; -E[l_mumu] would be 0 here
    const ld::laplace_distrib d{0.4, 1.3};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-12 * e[2]);
  }
  for (double mu : {0.05, 3.5, 40.0}) {
    const ld::poisson_distrib d{mu};
    double s = 0.0;
    for (int y = 0; y < 400; ++y) {
      const double sc = d.logpmf_d1(y)[0];
      s += sc * sc * std::exp(d.logpmf(y));
    }
    CHECK(std::fabs(s - d.expected_info()[0]) <= 1e-13 * d.expected_info()[0]);
  }
}

TEST_CASE("distrib: derivatives of the expected information against differences") {
  check_info_derivs<ld::gaussian1_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::laplace_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::poisson_distrib>({3.5}, 1e-8);
  for (double phi : {0.02, 0.3, 4.0}) check_info_derivs<ld::gamma1_distrib>({2.0, phi}, 1e-8);
}

TEST_CASE("distrib: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {-12.0, -3.0, 0.2, 1.7, 9.0, 20.0})
      check_cdf_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
    for (double y : {-6.0, -0.5, 0.3, 1.7, 9.0})
      check_cdf_derivs<ld::laplace_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
    for (double y : {0.0, 2.0, 9.0, 10.0, 11.0, 30.0}) check_cdf_derivs<ld::poisson_distrib>({10.0}, y, upper, 1e-7, 1.0);
    for (double phi : {0.05, 0.3, 4.0})
      for (double y : {0.05, 0.8, 2.0, 9.0}) check_cdf_derivs<ld::gamma1_distrib>({2.0, phi}, y, upper, 1e-7, 1.0);
  }
}

TEST_CASE("distrib: derivatives in y and mixed (y, theta) against differences") {
  // J = 0 is logpdf_dk, the order K = 0 the value
  const ld::gamma1_distrib g{2.0, 0.3};
  CHECK(g.logpdf_dy_dk<0, 0>(1.3)[0] == g.logpdf(1.3));
  CHECK(g.logpdf_dy_dk<0, 3>(1.3)[2] == g.logpdf_d3(1.3)[2]);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, 1e-3, 1e-8);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::laplace_distrib>({0.4, 1.3}, y, 1e-3, 1e-8);
  for (double phi : {0.02, 0.3, 4.0})
    for (double y : {0.3, 1.4, 2.0, 5.0}) check_y_derivs<ld::gamma1_distrib>({2.0, phi}, y, 1e-4 * y, 1e-7);
  CHECK(ld::gaussian1_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::laplace_distrib::y_regularity == 0);
}

TEST_CASE("distrib: cdf, ccdf, logarithms, quantile") {
  const ld::gaussian1_distrib g{0.4, 1.3};
  const ld::laplace_distrib l{0.4, 1.3};
  for (double y : {-30.0, -2.0, 0.4, 1.0, 5.0, 30.0}) {
    CHECK(g.cdf(y) == doctest::Approx(0.5 * std::erfc(-(y - 0.4) / (1.3 * std::sqrt(2.0)))).epsilon(1e-14));
    CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-14));
    CHECK(l.cdf(y) + l.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(l.log_cdf(y) == doctest::Approx(std::log(l.cdf(y))).epsilon(1e-14));
    CHECK(l.log_ccdf(y) == doctest::Approx(std::log(l.ccdf(y))).epsilon(1e-14));
    if (std::fabs(y) < 10.0) {
      CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
      CHECK(l.quantile(l.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
    }
  }
  const ld::gamma1_distrib ga{2.0, 0.3};
  for (double y : {0.01, 0.5, 2.0, 6.0, 20.0}) {
    CHECK(ga.cdf(y) + ga.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(ga.log_cdf(y) == doctest::Approx(std::log(ga.cdf(y))).epsilon(1e-13));
    if (y < 10.0) CHECK(ga.quantile(ga.cdf(y)) == doctest::Approx(y).epsilon(1e-11));
  }
  // logpdf matches the textbook form, also where (y - mu)/mu rounds to -1
  const ld::gamma1_distrib gs{2.0, 4.0};
  for (double y : {1e-30, 1e-5, 9.0})
    CHECK(gs.logpdf(y) == doctest::Approx(-std::lgamma(0.25) - 0.25 * std::log(8.0) - 0.75 * std::log(y) - y / 8.0).epsilon(1e-14));
  CHECK(ga.logpdf(1.3) == doctest::Approx(-std::lgamma(1 / 0.3) - (1 / 0.3) * std::log(2.0 * 0.3) +
                                           (1 / 0.3 - 1) * std::log(1.3) - 1.3 / (2.0 * 0.3)).epsilon(1e-14));
  // the saddle-point log-mass against the textbook form, where the latter does not cancel
  for (double y : {0.0, 1.0, 3.0, 12.0})
    CHECK(ld::poisson_distrib{3.5}.logpmf(y) ==
          doctest::Approx(y * std::log(3.5) - 3.5 - std::lgamma(y + 1.0)).epsilon(1e-14));
  // mu = y = 1000 (mpmath, 200 bits: -4.3728995060262968242), where the textbook form cancels
  CHECK(ld::poisson_distrib{1000.0}.logpmf(1000.0) == doctest::Approx(-4.3728995060262968242).epsilon(4e-16));
  const ld::poisson_distrib p{3.5};
  double F = 0.0;
  for (int k = 0; k < 20; ++k) {
    F += std::exp(p.logpmf(k));
    CHECK(p.cdf(k) == doctest::Approx(F).epsilon(1e-13));
    CHECK(p.cdf(k + 0.5) == p.cdf(k));
    CHECK(p.cdf(k) + p.ccdf(k) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(p.log_cdf(k) == doctest::Approx(std::log(p.cdf(k))).epsilon(1e-13));
  }
  CHECK(p.cdf(-1.0) == 0.0);
}

TEST_CASE("distrib: every order inside a kernel") {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_pilot", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::gaussian1_distrib g{0.4, 1.3};
        const ld::laplace_distrib l{0.4, 1.3};
        const ld::poisson_distrib p{3.5};
        const ld::gamma1_distrib ga{2.0, 0.3};
        out(i) = ga.logpdf_d6(0.5 + 0.1 * i)[2] + ga.logpdf_dy_dk<2, 3>(0.5 + 0.1 * i)[1] + g.logpdf_dy_dk<1, 5>(y)[2] + l.logpdf_dy_dk<1, 2>(y)[1] + ga.expected_info_d4().pair[2][4] + g.logpdf_d6(y)[3] + l.logpdf_d6(y)[6] + p.logpmf_d6(double(i % 7))[0] + g.expected_info_d4().pair[2][4] +
                 l.expected_info_d4().pair[0][4] + p.expected_info_d4().pair[0][0] + g.log_cdf(y) + l.log_ccdf(y) +
                 p.log_cdf(double(i % 7));
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::gaussian1_distrib g{0.4, 1.3};
    const ld::laplace_distrib l{0.4, 1.3};
    const ld::poisson_distrib p{3.5};
    const ld::gamma1_distrib ga{2.0, 0.3};
    const double r = ga.logpdf_d6(0.5 + 0.1 * i)[2] + ga.logpdf_dy_dk<2, 3>(0.5 + 0.1 * i)[1] + g.logpdf_dy_dk<1, 5>(y)[2] + l.logpdf_dy_dk<1, 2>(y)[1] + ga.expected_info_d4().pair[2][4] + g.logpdf_d6(y)[3] + l.logpdf_d6(y)[6] + p.logpmf_d6(double(i % 7))[0] + g.expected_info_d4().pair[2][4] +
                     l.expected_info_d4().pair[0][4] + p.expected_info_d4().pair[0][0] + g.log_cdf(y) + l.log_ccdf(y) +
                     p.log_cdf(double(i % 7));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}


TEST_CASE("distrib: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_log_cdf.py; the error of each component in ulp of the largest
  // component of its order at that y (the components change sign, so a plain relative error
  // is undefined at their zeros), and in ulp of the component itself
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_log_cdf.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo, mu, phi; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64];
    char mus[64], phis[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b, his, los, mus,
                    phis) != 9)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.phi = std::strtod(phis, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 900);
  double worst_scaled[4][2] = {}, worst_own[4][2] = {};
  for (const auto& r : rows) {
    std::vector<double> v;
    if (r.dist == 0) v = cderiv(ld::gaussian1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else if (r.dist == 1) v = cderiv(ld::laplace_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else if (r.dist == 2) v = cderiv(ld::poisson_distrib{r.mu}, r.y, r.K, r.upper);
    else v = cderiv(ld::gamma1_distrib{r.mu, r.phi}, r.y, r.K, r.upper);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.dist == r.dist && q.upper == r.upper && q.y == r.y && q.K == r.K && q.mu == r.mu && q.phi == r.phi) scale = std::fmax(scale, std::fabs(q.hi));
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(scale) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.dist][r.upper] = std::fmax(worst_scaled[r.dist][r.upper], ulp_scaled);
    worst_own[r.dist][r.upper] = std::fmax(worst_own[r.dist][r.upper], ulp_own);
    INFO("dist " << r.dist << " upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " ulp " << ulp_scaled);
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_own > 200)
      std::printf("dist %d upper %d y %g K %d b %d: %.3g ulp own, %.3g scaled, value %.3g\n", r.dist, r.upper, r.y, r.K, r.b, ulp_own, ulp_scaled, r.hi);
    // measured 2026-11-12 (gcc): 30.9, 29.7 (gaussian1), 11.2, 9.86 (laplace; 57.2, 391 before
    // the series route for log(1 - e^(-|z|)/2)); bounds half again;
    // poisson (2026-11-12, evening): 3.84e3 (k = 39 at mu = 40, order 6, value 4.8e-7 next to
    // 2e-6), 867 (log(1 - F), after Temme's repair near c = 1, 2026-11-12 night); gamma1, composed
    // from the partials at fixed ratio (incomplete_gamma_ratio.hpp): 379, 356
    const double bound[4][2] = {{46.4, 44.6}, {16.8, 14.8}, {5.8e3, 1.3e3}, {570.0, 540.0}};
    CHECK(ulp_scaled <= bound[r.dist][r.upper]);
  }
  for (int d = 0; d < 4; ++d)
    for (int u = 0; u < 2; ++u)
      if (std::getenv("LAPLACE_PRINT_ERRORS"))
        std::printf("log_cdf derivs dist %d upper %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n", d, u,
                    worst_scaled[d][u], worst_own[d][u]);
}

TEST_CASE("distrib: derivatives of log F and log(1 - F) in the tails against mpmath") {
  // tools/reference/distrib_log_cdf_tails.py: gaussian1 where log Phi is close to -z^2/2,
  // laplace where log(1 - F) or log F is log(1 - e^(-|z|)/2); errors as in the case above,
  // in ulp of the largest component of the order at that y and in ulp of the component
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_log_cdf_tails.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],", &r.dist, &r.upper, ys, &r.K, &r.b, his, los) != 7) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 1500);
  // worst errors per distribution, side, order and component (b), and the y of the worst own error
  double w_scaled[2][2][7][7] = {}, w_own[2][2][7][7] = {}, w_y[2][2][7][7] = {};
  for (const auto& r : rows) {
    std::vector<double> v;
    if (r.dist == 0) v = cderiv(ld::gaussian1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else v = cderiv(ld::laplace_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.dist == r.dist && q.upper == r.upper && q.y == r.y && q.K == r.K) scale = std::fmax(scale, std::fabs(q.hi));
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(scale) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    w_scaled[r.dist][r.upper][r.K][r.b] = std::fmax(w_scaled[r.dist][r.upper][r.K][r.b], ulp_scaled);
    if (ulp_own > w_own[r.dist][r.upper][r.K][r.b]) w_own[r.dist][r.upper][r.K][r.b] = ulp_own, w_y[r.dist][r.upper][r.K][r.b] = r.y;
    INFO("dist " << r.dist << " upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " ulp " << ulp_scaled << " own " << ulp_own);
    // measured 2026-11-12 (gcc), scaled: 4.81, 4.38 (gaussian1), 16.7, 6.05 (laplace); own: 1.46e3,
    // 1.48e3 (gaussian1, order 6 at z = -7 and 7, where the component has a zero next to it at
    // |z| = 7.3), 53.9, 35.2 (laplace); before the tail routes the own errors were 1.57e5, 1.57e5,
    // 986, 921; bounds half again
    const double bound_scaled[2][2] = {{7.3, 6.6}, {25.0, 9.1}};
    const double bound_own[2][2] = {{2.2e3, 2.2e3}, {81.0, 53.0}};
    CHECK(ulp_scaled <= bound_scaled[r.dist][r.upper]);
    CHECK(ulp_own <= bound_own[r.dist][r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int d = 0; d < 2; ++d)
      for (int u = 0; u < 2; ++u)
        for (int K = 1; K <= 6; ++K) {
          double ms = 0.0, mo = 0.0;
          std::printf("tails dist %d upper %d K %d, own ulp (scaled ulp) per b:", d, u, K);
          for (int b = 0; b <= K; ++b) {
            std::printf(" %.3g (%.3g) [y %.4g]", w_own[d][u][K][b], w_scaled[d][u][K][b], w_y[d][u][K][b]);
            ms = std::fmax(ms, w_scaled[d][u][K][b]), mo = std::fmax(mo, w_own[d][u][K][b]);
          }
          std::printf("; order: %.3g own, %.3g scaled\n", mo, ms);
        }
}
