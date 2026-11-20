// The multivariate normal rectangle probability, its derivatives in the shift
// and the moments of the truncated law (include/laplace/math/mvnormal.hpp)
// against the mpmath references of tools/reference/mvnormal.py, evaluated in a
// Kokkos kernel on the default execution space, one thread per row:
//   bvn_rect           2654 bivariate rectangles, r to +-0.9999, log P to -1.1e7;
//   mvn_factor         one-factor correlations, k = 3..10, orthants and rectangles;
//   mvn_laml           the orthants of the piecewise LAML of the lasso (one
//                      factor, k = 2..6, t = lam sigma to 20);
//   tvn_general        trivariate rectangles with a general correlation (Plackett);
//   mvn_trunc_moments  the cumulants of the truncated law (orders 1-2, and 3-4
//                      for k <= 3).
// The error of log P is absolute (the relative error of P); in the bins of log P
// below -2300 it is read relative to |log P|. The cumulants are read in units
// of the product of the truncated law's standard deviations. The derivatives
// of the recursion are checked on the host against central differences of the
// order below, and the cumulant route against log_Phi_dk at k = 1.
// Bounds: the errors measured on the gcc/OpenMP and CUDA builds (docs/numerics.md)
// with a margin of about half again.
#include <doctest/doctest.h>

#include <laplace/math/mvnormal.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;
namespace dm = laplace::math::detail_mvnormal;

using Rows = Kokkos::View<double**, Kokkos::LayoutRight>;

// the rows of a reference file (hexadecimal and inf parsed by strtod), header skipped
std::vector<std::vector<double>> read_rows(const std::string& name) {
  std::ifstream f(refdir + "/" + name + ".csv");
  REQUIRE(f.good());
  std::string line;
  std::getline(f, line);
  std::vector<std::vector<double>> v;
  while (std::getline(f, line)) {
    std::vector<double> x;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) x.push_back(std::strtod(tok.c_str(), nullptr));
    if (!x.empty()) v.push_back(x);
  }
  return v;
}

Rows to_device(const std::vector<std::vector<double>>& v) {
  Rows d("rows", v.size(), v[0].size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t j = 0; j < v[i].size(); ++j) h(i, j) = v[i][j];
  Kokkos::deep_copy(d, h);
  return d;
}

// the bins of log P
int bin_of(double lp) { return lp > -2.3 ? 0 : lp > -23 ? 1 : lp > -230 ? 2 : lp > -2300 ? 3 : 4; }

struct Bvn {
  Rows x, y;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    y(i, 0) = lm::log_bvn_rect(x(i, 0), x(i, 1), x(i, 2), x(i, 3), x(i, 4));
  }
};

// a one-factor row: k, a[KF], b[KF], lam[KF], then log P
template <int KF>
struct Factor {
  Rows x, y, w;
  int n_lattice, tensor_max_dim;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int k = int(x(i, 0));
    double a[KF], b[KF], S[KF * (KF + 1) / 2];
    for (int p = 0; p < k; ++p) {
      a[p] = x(i, 1 + p);
      b[p] = x(i, 1 + KF + p);
      for (int q = 0; q <= p; ++q) S[p * (p + 1) / 2 + q] = p == q ? 1.0 : x(i, 1 + 2 * KF + p) * x(i, 1 + 2 * KF + q);
    }
    y(i, 0) = lm::log_mvn_rect(k, a, b, S, &w(i, 0), n_lattice, tensor_max_dim);
  }
};

struct Tvn {
  Rows x, y, w;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    double a[3], b[3];
    for (int p = 0; p < 3; ++p) a[p] = x(i, p), b[p] = x(i, 3 + p);
    const double S[6] = {1.0, x(i, 6), 1.0, x(i, 7), x(i, 8), 1.0};
    y(i, 0) = lm::log_mvn_rect(3, a, b, S, &w(i, 0));
  }
};

// a moment row: k, a[6], b[6], lam[6], log P, cumulants; y: log P then the cumulants
struct Trunc {
  Rows x, y, w;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    constexpr int KM = 6;
    const int k = int(x(i, 0)), N = k <= 3 ? 4 : 2;
    double a[KM], b[KM], S[KM * (KM + 1) / 2];
    for (int p = 0; p < k; ++p) {
      a[p] = x(i, 1 + p);
      b[p] = x(i, 1 + KM + p);
      for (int q = 0; q <= p; ++q) S[p * (p + 1) / 2 + q] = p == q ? 1.0 : x(i, 1 + 2 * KM + p) * x(i, 1 + 2 * KM + q);
    }
    int m, keep[KM];
    y(i, 0) = lm::log_mvn_trunc_cumulants(k, N, a, b, S, m, keep, &y(i, 1), &w(i, 0));
  }
};

}  // namespace

// The kernels sit in functions with external linkage (note/brief_agenti.md).
std::vector<double> mvnormal_run_bvn(const Rows& x) {
  Rows y("y", x.extent(0), 1);
  Kokkos::parallel_for("bvn", int(x.extent(0)), Bvn{x, y});
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  std::vector<double> o(x.extent(0));
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = h(i, 0);
  return o;
}
template <int KF>
std::vector<double> mvnormal_run_factor(const Rows& x, int n_lattice, int tensor_max_dim = lm::mvn_tensor_max_dim) {
  Rows y("y", x.extent(0), 1), w("w", x.extent(0), lm::mvn_rect_scratch(KF));
  Kokkos::parallel_for("factor", int(x.extent(0)), Factor<KF>{x, y, w, n_lattice, tensor_max_dim});
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  std::vector<double> o(x.extent(0));
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = h(i, 0);
  return o;
}
std::vector<double> mvnormal_run_tvn(const Rows& x) {
  Rows y("y", x.extent(0), 1), w("w", x.extent(0), lm::mvn_rect_scratch(3));
  Kokkos::parallel_for("tvn", int(x.extent(0)), Tvn{x, y, w});
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  std::vector<double> o(x.extent(0));
  for (std::size_t i = 0; i < o.size(); ++i) o[i] = h(i, 0);
  return o;
}
Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> mvnormal_run_trunc(const Rows& x) {
  const int nk = dm::tuples_upto(6, 4);
  Rows y("y", x.extent(0), 1 + nk), w("w", x.extent(0), lm::mvn_trunc_scratch(6, 4));
  Kokkos::parallel_for("trunc", int(x.extent(0)), Trunc{x, y, w});
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
}

namespace {

// the largest error of log P by bin, against bounds by bin
void check_logp(const char* name, const std::vector<std::vector<double>>& rows, const std::vector<double>& y,
                int col_hi, int k_col, int k_want, const double (&bound)[5]) {
  double worst[5] = {0, 0, 0, 0, 0};
  int nan_rows = 0, n = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (k_col >= 0 && int(rows[i][k_col]) != k_want) continue;
    ++n;
    const double hi = rows[i][col_hi], lo = rows[i][col_hi + 1];
    if (!std::isfinite(y[i])) {
      ++nan_rows;
      continue;
    }
    const int b = bin_of(hi);
    double e = std::fabs((y[i] - hi) - lo);
    if (b == 4) e /= std::fabs(hi);
    worst[b] = std::fmax(worst[b], e);
  }
  INFO(name, " k=", k_want, " (", n, " rows): max |error of log P| by bin (> -2.3, > -23, > -230, > -2300, relative below) ",
       worst[0], " ", worst[1], " ", worst[2], " ", worst[3], " ", worst[4]);
  CHECK(nan_rows == 0);
  for (int b = 0; b < 5; ++b) CHECK(worst[b] <= bound[b]);
}

}  // namespace

TEST_CASE("mvnormal: bivariate rectangles against mpmath") {
  const auto rows = read_rows("bvn_rect");
  const auto y = mvnormal_run_bvn(to_device(rows));
  check_logp("bvn_rect", rows, y, 5, -1, 2, {4.8e-16, 1.5e-14, 2.1e-13, 1.4e-12, 1.7e-14});
}

TEST_CASE("mvnormal: one-factor rectangles and orthants against mpmath, k = 3..8") {
  const auto rows = read_rows("mvn_factor");
  std::vector<std::vector<double>> sub;
  for (const auto& r : rows)
    if (int(r[0]) <= 8) sub.push_back(r);
  const auto y = mvnormal_run_factor<10>(to_device(sub), lm::mvn_default_lattice);
  // the tensor rule to k = 4, the lattice (N = 16381) beyond
  check_logp("mvn_factor", sub, y, 31, 0, 3, {1.2e-15, 2.6e-14, 1.5e-13, 3.2e-12, 1e-13});
  check_logp("mvn_factor", sub, y, 31, 0, 4, {1.1e-14, 7.2e-14, 9.2e-13, 3.2e-12, 1e-13});
  for (int k = 5; k <= 8; ++k) check_logp("mvn_factor", sub, y, 31, 0, k, {1.8e-5, 7.1e-5, 4.3e-5, 1.4e-5, 1.9e-9});
}

TEST_CASE("mvnormal: the orthants of the lasso LAML against mpmath, k = 2..6" * doctest::test_suite("slow")) {
  const auto rows = read_rows("mvn_laml");
  const auto y = mvnormal_run_factor<10>(to_device(rows), lm::mvn_default_lattice);
  for (int k = 2; k <= 4; ++k) check_logp("mvn_laml", rows, y, 31, 0, k, {7.8e-15, 1.7e-14, 6.6e-14, 3.7e-13, 1e-14});
  for (int k = 5; k <= 6; ++k) check_logp("mvn_laml", rows, y, 31, 0, k, {2.5e-6, 1.1e-4, 7.3e-5, 2.3e-5, 3.9e-9});
}

TEST_CASE("mvnormal: the lattice at four coordinates (tensor_max_dim = 3) on the LAML orthants") {
  const auto rows = read_rows("mvn_laml");
  std::vector<std::vector<double>> sub;
  for (const auto& r : rows)
    if (int(r[0]) == 4) sub.push_back(r);
  const auto y = mvnormal_run_factor<10>(to_device(sub), lm::mvn_default_lattice, 3);
  check_logp("mvn_laml lattice", sub, y, 31, 0, 4, {6.5e-6, 2e-5, 2e-5, 1.5e-5, 1e-7});
}

TEST_CASE("mvnormal: trivariate rectangles with a general correlation against mpmath") {
  const auto rows = read_rows("tvn_general");
  const auto y = mvnormal_run_tvn(to_device(rows));
  check_logp("tvn_general", rows, y, 9, -1, 3, {1.5e-15, 8.3e-14, 1.8e-12, 3.4e-11, 1e-13});
}

TEST_CASE("mvnormal: the truncated law's cumulants against mpmath") {
  const auto rows = read_rows("mvn_trunc_moments");
  const auto y = mvnormal_run_trunc(to_device(rows));
  // worst error by k and order, in units of the product of standard deviations
  double worst[7][5] = {};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    const int k = int(r[0]), N = k <= 3 ? 4 : 2;
    worst[k][0] = std::fmax(worst[k][0], std::fabs(y(i, 0) - r[19]));
    const double* ref = &r[20];
    double sd[6];
    for (int p = 0; p < k; ++p) {
      const int t[2] = {p, p};
      sd[p] = std::sqrt(ref[k + dm::tuple_rank(k, 2, t)]);
    }
    int off = 0;
    for (int n = 1; n <= N; ++n) {
      int t[6] = {0, 0, 0, 0, 0, 0}, q = 0;
      do {
        double sc = 1.0;
        for (int p = 0; p < n; ++p) sc *= sd[t[p]];
        const double e = std::fabs(y(i, 1 + dm::tuples_upto(k, n - 1) + q) - ref[off + q]) / sc;
        worst[k][n] = std::fmax(worst[k][n], e);
        ++q;
      } while (dm::next_tuple(k, n, t));
      off += q;
    }
  }
  // log P, mean, covariance, kappa_3, kappa_4
  const double bound[7][5] = {{0}, {0},
                              {1.8e-13, 6.1e-13, 4.4e-14, 2.2e-12, 9.6e-12},
                              {3.5e-12, 1.7e-10, 5.5e-11, 1.6e-9, 3.5e-8},
                              {8.4e-10, 2.3e-8, 4.5e-7, 0, 0},
                              {4.6e-6, 4.4e-5, 6.5e-4, 0, 0},
                              {1.3e-5, 9.3e-5, 1.1e-3, 0, 0}};
  for (int k = 2; k <= 6; ++k) {
    INFO("k = ", k, ": log P ", worst[k][0], ", mean ", worst[k][1], ", cov ", worst[k][2], ", kappa3 ", worst[k][3],
         ", kappa4 ", worst[k][4]);
    for (int n = 0; n <= (k <= 3 ? 4 : 2); ++n) CHECK(worst[k][n] <= bound[k][n]);
  }
}

TEST_CASE("mvnormal: derivatives in the shift against central differences of the order below") {
  const double I = lm::detail_mvnormal::inf();
  struct Case {
    int k;
    double lo[4], hi[4], S[10];
  };
  const Case cases[] = {
      {2, {-I, -I}, {0.3, -0.7}, {1.0, 0.5, 1.0}},
      {2, {-1.0, 0.2}, {0.5, 1.4}, {2.0, -0.6, 1.5}},
      {3, {-I, -I, -I}, {0.3, -0.7, 1.1}, {1.0, 0.5, 1.0, 0.2, -0.3, 1.2}},
      {3, {-1.0, -I, 0.1}, {0.5, 0.4, I}, {1.0, 0.5, 1.0, 0.2, -0.3, 1.2}},
      {4, {-I, -I, -I, -I}, {0.3, -0.7, 1.1, 0.0}, {1.0, 0.5, 1.0, 0.2, -0.3, 1.2, 0.1, 0.2, 0.3, 1.0}},
  };
  const int N = 6;
  for (const auto& c : cases) {
    const int k = c.k;
    std::vector<double> f(lm::mvn_rect_derivs_size(k, N)), fp(f.size()), fm(f.size()), w(lm::mvn_rect_derivs_scratch(k, N));
    const double lF = lm::log_mvn_rect_derivs(k, N, c.lo, c.hi, c.S, f.data(), w.data());
    const double h = 1e-5;
    double worst = 0.0;
    for (int d = 0; d < k; ++d) {
      double lp[4], hp[4], lmn[4], hmn[4];
      for (int i = 0; i < k; ++i) lp[i] = lmn[i] = c.lo[i], hp[i] = hmn[i] = c.hi[i];
      lp[d] += h, hp[d] += h, lmn[d] -= h, hmn[d] -= h;
      const double lFp = lm::log_mvn_rect_derivs(k, N - 1, lp, hp, c.S, fp.data(), w.data());
      const double lFm = lm::log_mvn_rect_derivs(k, N - 1, lmn, hmn, c.S, fm.data(), w.data());
      for (int n = 0; n < N; ++n) {
        int t[8] = {0}, t2[8];
        do {
          const int pn = dm::tuples_upto(k, n - 1) + dm::tuple_rank(k, n, t);
          const double fd = (fp[pn] * std::exp(lFp - lF) - fm[pn] * std::exp(lFm - lF)) / (2 * h);
          int m = 0;
          bool ins = false;
          for (int i = 0; i < n; ++i) {
            if (!ins && d < t[i]) t2[m++] = d, ins = true;
            t2[m++] = t[i];
          }
          if (!ins) t2[m++] = d;
          const double an = f[dm::tuples_upto(k, n) + dm::tuple_rank(k, n + 1, t2)];
          worst = std::fmax(worst, std::fabs(fd - an) / std::fmax(1.0, std::fabs(an)));
        } while (n > 0 && dm::next_tuple(k, n, t));
      }
    }
    INFO("k = ", k, ": max relative difference ", worst);
    CHECK(worst <= 1e-8);
  }
}

TEST_CASE("mvnormal: derivatives in the limits, summed over the two sides, are those in the shift") {
  // d_delta_i = d_lo_i + d_hi_i, so every shift tuple is the sum of the limit
  // tuples over the 2^n choices of a side for each position; both come from the
  // same table, so the sums agree to rounding
  const double I = lm::detail_mvnormal::inf();
  const double S[10] = {1.0, 0.5, 1.0, 0.2, -0.3, 1.2, 0.1, 0.2, 0.3, 1.0};
  const double lo[4] = {-0.5, -I, -1.2, 0.1}, hi[4] = {0.7, 0.4, 0.3, I};
  for (int k = 1; k <= 4; ++k) {
    const int N = 5;
    std::vector<double> f(lm::mvn_rect_limit_derivs_size(k, N)), fs(lm::mvn_rect_derivs_size(k, N)),
        w(lm::mvn_rect_derivs_scratch(k, N));
    const double lF = lm::log_mvn_rect_limit_derivs(k, N, lo, hi, S, f.data(), w.data());
    CHECK(lm::log_mvn_rect_derivs(k, N, lo, hi, S, fs.data(), w.data()) == lF);
    double worst = 0.0;
    for (int n = 1; n <= N; ++n) {
      int a[8] = {0};
      int q = dm::tuples_upto(k, n - 1);
      do {
        double s = 0.0;
        for (unsigned c = 0; c < (1u << n); ++c) {
          int z[8];
          for (int r = 0; r < n; ++r) z[r] = a[r] + ((c >> r) & 1u ? k : 0);
          for (int x = 1; x < n; ++x)
            for (int y = x; y > 0 && z[y] < z[y - 1]; --y) std::swap(z[y], z[y - 1]);
          s += f[dm::tuples_upto(2 * k, n - 1) + dm::tuple_rank(2 * k, n, z)];
        }
        worst = std::fmax(worst, std::fabs(s - fs[q]) / std::fmax(1.0, std::fabs(fs[q])));
        ++q;
      } while (dm::next_tuple(k, n, a));
    }
    INFO("k = ", k, ": max relative difference ", worst);
    CHECK(worst <= 1e-14);
  }
}

TEST_CASE("mvnormal: the cumulant route to order 12 against central differences of the order below") {
  // bivariate and trivariate, bulk and tail: d^n log F against the difference of
  // d^(n-1) log F, n = 2..12 (the orders beyond six through Miller's recurrence)
  const double I = lm::detail_mvnormal::inf();
  struct Case {
    int k;
    double hi[3], S[6];
  };
  const Case cases[] = {{2, {0.3, -0.7}, {1.0, 0.5, 1.0}}, {2, {-8.0, -7.5}, {1.0, 0.5, 1.0}},
                        {3, {-6.0, -5.0, -7.0}, {1.0, 0.4, 1.0, 0.2, -0.3, 1.0}}};
  const int N = 12;
  double worst = 0.0;
  for (const auto& c : cases) {
    const int k = c.k;
    double lo[3] = {-I, -I, -I};
    std::vector<double> l(lm::mvn_rect_derivs_size(k, N)), lp(l.size()), lmn(l.size()), w(lm::mvn_rect_log_derivs_scratch(k, N));
    lm::log_mvn_rect_log_derivs(k, N, lo, c.hi, c.S, l.data(), w.data());
    const double h = 1e-4;
    for (int d = 0; d < k; ++d) {
      double hp[3], hm[3];
      for (int i = 0; i < k; ++i) hp[i] = hm[i] = c.hi[i];
      hp[d] += h, hm[d] -= h;
      lm::log_mvn_rect_log_derivs(k, N - 1, lo, hp, c.S, lp.data(), w.data());
      lm::log_mvn_rect_log_derivs(k, N - 1, lo, hm, c.S, lmn.data(), w.data());
      for (int n = 1; n < N; ++n) {
        int t[16] = {0}, t2[16];
        do {
          const int pn = dm::tuples_upto(k, n - 1) + dm::tuple_rank(k, n, t);
          const double fd = (lp[pn] - lmn[pn]) / (2 * h);
          int m = 0;
          bool ins = false;
          for (int i = 0; i < n; ++i) {
            if (!ins && d < t[i]) t2[m++] = d, ins = true;
            t2[m++] = t[i];
          }
          if (!ins) t2[m++] = d;
          const double an = l[dm::tuples_upto(k, n) + dm::tuple_rank(k, n + 1, t2)];
          worst = std::fmax(worst, std::fabs(fd - an) / std::fmax(1.0, std::fabs(an)));
        } while (dm::next_tuple(k, n, t));
      }
    }
  }
  INFO("max relative difference ", worst);
  CHECK(worst <= 1e-6);
}

TEST_CASE("mvnormal: the cumulant route of the log-derivatives against log_Phi_dk in the tail") {
  const double I = lm::detail_mvnormal::inf();
  double worst = 0.0;
  for (double t : {3.0, 0.0, -2.0, -5.0, -10.0, -20.0, -40.0}) {
    double lo = -I, hi = t, S = 1.0, f[7];
    std::vector<double> w(lm::mvn_rect_log_derivs_scratch(1, 6));
    lm::log_mvn_rect_log_derivs(1, 6, &lo, &hi, &S, f, w.data());
    const double ref[7] = {0, lm::log_Phi_d1(t), lm::log_Phi_d2(t), lm::log_Phi_d3(t),
                           lm::log_Phi_d4(t), lm::log_Phi_d5(t), lm::log_Phi_d6(t)};
    for (int n = 1; n <= 6; ++n) worst = std::fmax(worst, std::fabs(f[n] - ref[n]) / std::fabs(ref[n]));
  }
  INFO("max relative error ", worst);
  CHECK(worst <= 5e-13);
}
