// The checks shared by the tests of the univariate distributions
// (tests/distrib/test_*.cpp): packing of the symmetric tensors, every derivative
// order against a 5-point difference of the order below (in the parameters, in y
// and mixed), the derivatives of the expected information against differences,
// the derivatives of log F and log(1 - F) against differences, and the variance
// of the score by quadrature. Written for the pilot (note/distrib.md) and pulled
// out of tests/distrib/test_pilot.cpp when the second batch was added.
#pragma once

#include <doctest/doctest.h>

#include <laplace/distrib/core.hpp>
#include <laplace/math/quadrature.hpp>

#include <cmath>
#include <vector>

namespace laplace_distrib_check {

namespace ld = laplace::distrib;


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
inline void sorted_indices(int P, int K, int lo, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
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
inline std::vector<std::vector<int>> sorted_indices(int P, int K) {
  std::vector<std::vector<int>> out;
  std::vector<int> cur;
  sorted_indices(P, K, 0, cur, out);
  return out;
}
inline int position_of(int P, const std::vector<int>& idx) {
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


}  // namespace laplace_distrib_check
