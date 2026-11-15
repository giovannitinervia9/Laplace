// The location-scale assemblies of laplace::distrib
// (include/laplace/distrib/detail/location_scale.hpp). mixed_dk and
// mixed_power_dk are measured through the families that use them (gaussian1,
// laplace1, cauchy, gaussian2, gaussian3); what is checked here is
// mixed_shape_dk, which adds a smooth shape parameter and is pure index work
// over mixed_dk: the components it writes must agree with mixed_dk where there
// is no derivative in the shape, must be the shape derivative itself where
// there is no other, and must agree with central differences of the value
// everywhere else.
//
// The test function is Lambda(z, s) = sin(z) exp(-s), whose mixed derivatives
// are closed, so that every entry of the table is non-zero and an error in the
// packing or in the order of the two directions cannot cancel.
#include <doctest/doctest.h>

#include <laplace/distrib/detail/location_scale.hpp>

#include <cmath>
#include <initializer_list>

namespace {

namespace ls = laplace::distrib::detail_location_scale;

// d^j/dz^j sin(z): sin, cos, -sin, -cos, ...
double sin_dj(int j, double z) {
  switch (j % 4) {
    case 0: return std::sin(z);
    case 1: return std::cos(z);
    case 2: return -std::sin(z);
    default: return -std::cos(z);
  }
}

// Lambda(z, s) = sin(z) exp(-s), so d^c/ds^c d^j/dz^j Lambda = (-1)^c e^-s d^j sin / dz^j
template <int K>
void fill_table(double z, double s, double* D) {
  const double e = std::exp(-s);
  for (int c = 0; c <= K; ++c) {
    const double sc = (c % 2 == 0) ? e : -e;
    for (int j = 0; j <= K; ++j) D[c * (K + 1) + j] = (c + j <= K) ? sc * sin_dj(j, z) : 0.0;
  }
}

double value(double mu, double sigma, double s, double y) {
  return std::sin((y - mu) / sigma) * std::exp(-s);
}

// Central difference of order n in the parameters named by idx (0 = mu,
// 1 = sigma, 2 = s), each step relative to the parameter's own size.
double fd(const int* idx, int n, double mu, double sigma, double s, double y) {
  if (n == 0) return value(mu, sigma, s, y);
  const double base[3] = {std::fabs(mu) + 1.0, sigma, s};
  const double h = 1e-3 * base[idx[0]];
  double up[3] = {mu, sigma, s}, dn[3] = {mu, sigma, s};
  up[idx[0]] += h;
  dn[idx[0]] -= h;
  return (fd(idx + 1, n - 1, up[0], up[1], up[2], y) - fd(idx + 1, n - 1, dn[0], dn[1], dn[2], y)) /
         (2.0 * h);
}

template <int K>
void check_order(double mu, double sigma, double s, double y, double tol) {
  const double z = (y - mu) / sigma;
  double D[(K + 1) * (K + 1)];
  fill_table<K>(z, s, D);
  const laplace::distrib::sym_tensor<3, K> t = ls::mixed_shape_dk<K>(D, z, sigma);

  // no derivative in the shape: the two-parameter assembly, component for component
  const laplace::distrib::sym_tensor<2, K> m = ls::mixed_dk<K>(D, z, sigma);
  for (int b = 0; b <= K; ++b) {
    int idx[K];
    int p = 0;
    for (int i = 0; i < K - b; ++i) idx[p++] = 0;
    for (int i = 0; i < b; ++i) idx[p++] = 1;
    CHECK(t[laplace::distrib::sym_tensor_position<3, K>(idx)] == m[b]);
  }
  // every derivative in the shape
  {
    int idx[K];
    for (int i = 0; i < K; ++i) idx[i] = 2;
    CHECK(t[laplace::distrib::sym_tensor_position<3, K>(idx)] == D[K * (K + 1)]);
  }
  // every component against the central differences
  for (int c = 0; c <= K; ++c)
    for (int b = 0; b + c <= K; ++b) {
      const int a = K - b - c;
      int idx[K];
      int p = 0;
      for (int i = 0; i < a; ++i) idx[p++] = 0;
      for (int i = 0; i < b; ++i) idx[p++] = 1;
      for (int i = 0; i < c; ++i) idx[p++] = 2;
      const double got = t[laplace::distrib::sym_tensor_position<3, K>(idx)];
      const double want = fd(idx, K, mu, sigma, s, y);
      INFO("K = ", K, ", a = ", a, ", b = ", b, ", c = ", c, ": got ", got, ", fd ", want);
      CHECK(std::fabs(got - want) <= tol * (std::fabs(want) + 1e-3));
    }
}

}  // namespace

TEST_CASE("mixed_shape_dk: a location, a scale and one shape") {
  // the tolerance is that of the nested central differences, h = 1e-3 relative
  // per level: about h^2 at order one and h^2 compounded by the nesting above
  for (double sigma : {0.7, 1.0, 2.5})
    for (double s : {0.4, 1.3}) {
      check_order<1>(0.3, sigma, s, 1.1, 1e-5);
      check_order<2>(0.3, sigma, s, 1.1, 1e-4);
      check_order<3>(0.3, sigma, s, 1.1, 1e-3);
    }
  // the exact identities alone at the orders where a central difference of
  // order four and above is noise
  check_order<4>(0.3, 1.0, 0.4, 1.1, 1e300);
  check_order<5>(0.3, 1.0, 0.4, 1.1, 1e300);
  check_order<6>(0.3, 1.0, 0.4, 1.1, 1e300);
}
