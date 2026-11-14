// weighted_sum_psdmat (include/laplace/pdmat/weighted_sum.hpp), on the host:
// log|M|_+ and its derivatives to order 6 against the mpmath references
// (tools/reference/pdmat_weighted_sum.py), in ulps of max(|ref|, 1); the rank
// identity sum_j tr(B_j) = r; central differences of the derivative one order
// below (h = 1e-4, within 1e-6 of max(1, |value|)) at the points with
// |theta_j| <= 8; M and its derivatives against the sum; the rank and the
// null-space basis; the rejected inputs. Set LAPLACE_PRINT_ERRORS=1 to print
// the table of errors.
#include <doctest/doctest.h>

#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

namespace lp = laplace::pdmat;
using host_matrix = lp::weighted_sum_psdmat::host_matrix;

struct ws_case {
  int p = 0, m = 0, rank = 0;
  std::vector<host_matrix> S;
};

struct ws_ref {
  std::string name;
  std::vector<double> theta;
  std::vector<int> tup;
  double hi, lo;
};

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string x;
  while (std::getline(ss, x, ',')) f.push_back(x);
  return f;
}

std::vector<double> split_numbers(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, ';')) v.push_back(std::strtod(f.c_str(), nullptr));
  return v;
}

std::map<std::string, ws_case> read_cases() {
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_weighted_sum_cases.csv");
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::string, ws_case> out;
  while (std::getline(in, line)) {
    const auto f = fields(line);
    ws_case& c = out[f[0]];
    if (c.S.empty()) {
      c.p = std::stoi(f[1]);
      c.m = std::stoi(f[2]);
      c.rank = std::stoi(f[3]);
      for (int j = 0; j < c.m; ++j) c.S.emplace_back("S", c.p, c.p);
    }
    const int j = std::stoi(f[4]), i = std::stoi(f[5]), k = std::stoi(f[6]);
    c.S[j](i, k) = c.S[j](k, i) = std::strtod(f[7].c_str(), nullptr);
  }
  return out;
}

std::vector<ws_ref> read_refs() {
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_weighted_sum.csv");
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::vector<ws_ref> out;
  while (std::getline(in, line)) {
    const auto f = fields(line);
    ws_ref r{f[0], split_numbers(f[1]), {}, std::strtod(f[4].c_str(), nullptr), std::strtod(f[5].c_str(), nullptr)};
    if (f[3] != "-")
      for (double t : split_numbers(f[3])) r.tup.push_back(static_cast<int>(t));
    out.push_back(r);
  }
  return out;
}

double log_pdet_any(const lp::weighted_sum_psdmat& s, const std::vector<int>& tup, const double* t, double* w) {
  auto pick = [&](auto k) {
    constexpr int K = decltype(k)::value;
    int idx[K];
    for (int q = 0; q < K; ++q) idx[q] = tup[q];
    return s.template log_pdet_dk<K>(t, idx, w);
  };
  switch (tup.size()) {
    case 0: return s.log_pdet(t, w);
    case 1: return pick(std::integral_constant<int, 1>{});
    case 2: return pick(std::integral_constant<int, 2>{});
    case 3: return pick(std::integral_constant<int, 3>{});
    case 4: return pick(std::integral_constant<int, 4>{});
    case 5: return pick(std::integral_constant<int, 5>{});
    default: return pick(std::integral_constant<int, 6>{});
  }
}

// bounds in ulps of max(|ref|, 1), per case and order 0..6: the measured
// worst (gcc, 2026-11-14) times about 4
double ulp_bound(const std::string& name, int order) {
  static const std::map<std::string, std::vector<double>> b = {
      {"lowrank", {4, 20, 40, 64, 128, 512, 5e3}},
      {"lowrank_ill", {5.5e4, 4.4e4, 9e4, 9e4, 9e4, 9e4, 9e4}},
      {"ridge", {24, 8, 48, 144, 1.2e3, 4e3, 8.5e3}},
      {"single", {64, 4, 16, 32, 128, 1e3, 8.5e3}},
      {"te2", {8, 4, 200, 400, 720, 7.5e3, 7e4}},
      {"te3", {8, 4, 150, 300, 1e3, 8.5e3, 4.2e4}}};
  return b.at(name).at(order);
}

}  // namespace

TEST_CASE("weighted_sum_psdmat: log pseudo-determinant against mpmath") {
  const auto cases = read_cases();
  const auto refs = read_refs();
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  std::map<std::string, lp::weighted_sum_psdmat> structs;
  for (const auto& [name, c] : cases) structs.emplace(name, lp::weighted_sum_psdmat(c.S));
  std::map<std::pair<std::string, int>, double> worst;
  std::map<std::string, double> worst_identity;
  std::vector<double> w;
  for (const auto& r : refs) {
    const auto& s = structs.at(r.name);
    w.assign(s.scratch_size(), 0.0);
    const double y = log_pdet_any(s, r.tup, r.theta.data(), w.data());
    const double ref = r.hi + r.lo;
    const double err = std::abs(y - ref) / (std::max(std::abs(ref), 1.0) * 2.220446049250313e-16);
    double& wk = worst[{r.name, static_cast<int>(r.tup.size())}];
    wk = std::max(wk, std::isfinite(err) ? err : 1e300);
    CHECK_MESSAGE(err <= ulp_bound(r.name, static_cast<int>(r.tup.size())), r.name, " order ", r.tup.size());
    if (r.tup.empty()) {
      const double ri = std::abs(s.rank_identity(r.theta.data(), w.data()) - s.rank());
      worst_identity[r.name] = std::max(worst_identity[r.name], ri);
      CHECK(ri <= 1e-9 * s.rank());
    }
  }
  if (print) {
    std::printf("weighted_sum_psdmat, log|M|_+: worst error in ulps of max(|ref|, 1), by order\n");
    std::string last;
    for (const auto& [key, e] : worst) {
      if (key.first != last) std::printf("%s%-12s", last.empty() ? "" : "\n", key.first.c_str());
      last = key.first;
      std::printf(" %d: %8.3g", key.second, e);
    }
    std::printf("\nrank identity |sum_j tr B_j - r|:");
    for (const auto& [n, e] : worst_identity) std::printf(" %s %.2g", n.c_str(), e);
    std::printf("\n");
  }
}

TEST_CASE("weighted_sum_psdmat: derivatives against central differences") {
  const auto cases = read_cases();
  const double h = 1e-4;
  for (const auto& [name, c] : cases) {
    const lp::weighted_sum_psdmat s(c.S);
    std::vector<double> w(s.scratch_size());
    const std::vector<std::vector<double>> points = {
        std::vector<double>{0.3, -1.2, 2.1}, std::vector<double>{6.0, 0.0, -4.5}, std::vector<double>{-7.0, 3.0, 8.0}};
    for (auto t : points) {
      t.resize(c.m);
      // every sorted tuple of orders 1..6
      std::vector<std::vector<int>> tuples = {{}};
      for (int K = 1; K <= 6; ++K) {
        std::vector<std::vector<int>> next;
        for (const auto& u : tuples)
          if (static_cast<int>(u.size()) == K - 1)
            for (int j = u.empty() ? 0 : u.back(); j < c.m; ++j) {
              auto v = u;
              v.push_back(j);
              next.push_back(v);
            }
        for (const auto& u : next) {
          const int v = u.back();
          std::vector<int> below(u.begin(), u.end() - 1);
          auto tp = t, tm = t;
          tp[v] += h;
          tm[v] -= h;
          const double fd = (log_pdet_any(s, below, tp.data(), w.data()) - log_pdet_any(s, below, tm.data(), w.data())) / (2 * h);
          const double y = log_pdet_any(s, u, t.data(), w.data());
          CHECK_MESSAGE(std::abs(y - fd) <= 1e-6 * std::max(1.0, std::abs(y)), name, " order ", K);
        }
        tuples.insert(tuples.end(), next.begin(), next.end());
      }
    }
  }
}

TEST_CASE("weighted_sum_psdmat: M, its derivatives, rank and null space") {
  const auto cases = read_cases();
  for (const auto& [name, c] : cases) {
    const lp::weighted_sum_psdmat s(c.S);
    CHECK(s.dim == c.p);
    CHECK(s.n_params() == c.m);
    CHECK(s.rank() == c.rank);
    const auto& N = s.null_space_basis();
    REQUIRE(static_cast<int>(N.extent(1)) == c.p - c.rank);
    // orthonormal, and annihilated by every S_j
    for (int a = 0; a < c.p - c.rank; ++a)
      for (int b = 0; b < c.p - c.rank; ++b) {
        double d = 0.0;
        for (int i = 0; i < c.p; ++i) d += N(i, a) * N(i, b);
        CHECK(std::abs(d - (a == b)) <= 64 * 2.2e-16 * c.p);
      }
    for (int j = 0; j < c.m; ++j) {
      double mx = 0.0, sn = 0.0;
      for (int i = 0; i < c.p; ++i)
        for (int k = 0; k < c.p; ++k) mx = std::max(mx, std::abs(c.S[j](i, k)));
      for (int a = 0; a < c.p - c.rank; ++a)
        for (int i = 0; i < c.p; ++i) {
          double v = 0.0;
          for (int k = 0; k < c.p; ++k) v += c.S[j](i, k) * N(k, a);
          sn = std::max(sn, std::abs(v));
        }
      CHECK(sn <= 64 * 2.2e-16 * c.p * mx);
    }
    std::vector<double> t = {0.7, -2.0, 1.5}, w(s.scratch_size());
    t.resize(c.m);
    std::vector<double> M(lp::packed_size(c.p)), D(lp::packed_size(c.p));
    s.matrix(t.data(), M.data(), w.data());
    for (int i = 0; i < c.p; ++i)
      for (int k = 0; k <= i; ++k) {
        double ref = 0.0;
        for (int j = 0; j < c.m; ++j) ref += std::exp(t[j]) * c.S[j](i, k);
        CHECK(M[lp::packed_index(i, k)] == doctest::Approx(ref).epsilon(1e-15));
      }
    for (int j = 0; j < c.m; ++j) {
      const int i3[3] = {j, j, j};
      REQUIRE(s.matrix_d3(t.data(), i3, D.data(), w.data()));
      for (int i = 0; i < c.p; ++i)
        for (int k = 0; k <= i; ++k) CHECK(D[lp::packed_index(i, k)] == std::exp(t[j]) * c.S[j](i, k));
    }
    if (c.m > 1) {
      const int i2[2] = {0, 1};
      CHECK_FALSE(s.matrix_d2(t.data(), i2, D.data(), w.data()));
    }
  }
}

TEST_CASE("weighted_sum_psdmat: rejected inputs") {
  host_matrix a("a", 3, 3), b("b", 4, 4), c("c", 3, 3);
  a(0, 0) = 1.0;
  b(0, 0) = 1.0;
  c(0, 0) = 1.0;
  c(1, 0) = 2.0;  // [[1, 2], [2, 0]] is indefinite
  CHECK_THROWS_AS(lp::weighted_sum_psdmat(std::vector<host_matrix>{}), std::invalid_argument);
  CHECK_THROWS_AS(lp::weighted_sum_psdmat(std::vector<host_matrix>{a, b}), std::invalid_argument);
  CHECK_THROWS_AS(lp::weighted_sum_psdmat(std::vector<host_matrix>{a, c}), std::invalid_argument);
}
