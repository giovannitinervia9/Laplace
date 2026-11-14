// Accuracy and speed of weighted_sum_psdmat (include/laplace/pdmat/weighted_sum.hpp).
// Accuracy: every reference of tests/data/reference/pdmat_weighted_sum.csv
// (tools/reference/pdmat_weighted_sum.py), the worst error per case and order
// in ulps of max(|ref|, 1), and the worst |sum_j tr B_j - r|, for the split
// thresholds eps^a, a in 0 0.05 ... 0.50 (zero test 8 p eps), and
// the zero tests 1, 64, 1024 p eps at eps^0.05, and for the naive route
// (eigenvalues of the sum in double, the r largest kept, M^+ from them,
// B_j = lambda_j M^+ S_j, the same partition formula). Speed: microseconds per
// call on one thread of log_pdet and log_pdet_dK, K = 1, 2, 6, on the cases.
//
// Usage: pdmat_weighted_sum

#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
namespace ld = laplace::dense;
using host_matrix = lp::weighted_sum_psdmat::host_matrix;
constexpr double eps = 2.220446049250313e-16;

struct ws_case {
  int p = 0, m = 0, rank = 0;
  std::vector<host_matrix> S;
};
struct ws_ref {
  std::string name;
  std::vector<double> theta;
  std::vector<int> tup;
  double ref;
};

std::vector<std::string> fields(const std::string& line, char sep) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string x;
  while (std::getline(ss, x, sep)) f.push_back(x);
  return f;
}

std::map<std::string, ws_case> read_cases() {
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_weighted_sum_cases.csv");
  std::string line;
  std::getline(in, line);
  std::map<std::string, ws_case> out;
  while (std::getline(in, line)) {
    const auto f = fields(line, ',');
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
  std::string line;
  std::getline(in, line);
  std::vector<ws_ref> out;
  while (std::getline(in, line)) {
    const auto f = fields(line, ',');
    ws_ref r{f[0], {}, {}, std::strtod(f[4].c_str(), nullptr) + std::strtod(f[5].c_str(), nullptr)};
    for (const auto& x : fields(f[1], ';')) r.theta.push_back(std::strtod(x.c_str(), nullptr));
    if (f[3] != "-")
      for (const auto& x : fields(f[3], ';')) r.tup.push_back(std::stoi(x));
    out.push_back(r);
  }
  return out;
}

template <class F>
double by_order(const std::vector<int>& tup, F&& f) {
  switch (tup.size()) {
    case 1: return f(std::integral_constant<int, 1>{});
    case 2: return f(std::integral_constant<int, 2>{});
    case 3: return f(std::integral_constant<int, 3>{});
    case 4: return f(std::integral_constant<int, 4>{});
    case 5: return f(std::integral_constant<int, 5>{});
    default: return f(std::integral_constant<int, 6>{});
  }
}

double eval(const lp::weighted_sum_psdmat& s, const std::vector<int>& tup, const double* t, double* w) {
  if (tup.empty()) return s.log_pdet(t, w);
  return by_order(tup, [&](auto k) {
    constexpr int K = decltype(k)::value;
    int idx[K];
    for (int q = 0; q < K; ++q) idx[q] = tup[q];
    return s.template log_pdet_dk<K>(t, idx, w);
  });
}

// the naive route
double naive(const ws_case& c, const std::vector<int>& tup, const double* t) {
  const int p = c.p, r = c.rank;
  host_matrix M("M", p, p), V("V", p, p);
  Kokkos::View<double*, Kokkos::HostSpace> e("e", p);
  for (int j = 0; j < c.m; ++j)
    for (int b = 0; b < p; ++b)
      for (int a = 0; a < p; ++a) M(a, b) += std::exp(t[j]) * c.S[j](a, b);
  ld::symmetric_eigen_decomposition(M, e, V);
  if (tup.empty()) {
    double v = 0.0;
    for (int a = p - r; a < p; ++a) v += std::log(e(a));
    return v;
  }
  // B_j = lambda_j M^+ S_j in the eigenbasis of the kept eigenvalues: r x r,
  // (B_j)_ab = lambda_j v_a^T S_j v_b / e_a (similar to lambda_j M^+ S_j)
  std::vector<std::vector<double>> B(c.m, std::vector<double>(r * r));
  for (int j = 0; j < c.m; ++j)
    for (int a = 0; a < r; ++a)
      for (int b = 0; b < r; ++b) {
        double x = 0.0;
        for (int i = 0; i < p; ++i)
          for (int k = 0; k < p; ++k) x += V(i, p - r + a) * c.S[j](i, k) * V(k, p - r + b);
        B[j][a + b * r] = std::exp(t[j]) * x / e(p - r + a);
      }
  std::vector<const double*> bp(c.m);
  for (int j = 0; j < c.m; ++j) bp[j] = B[j].data();
  std::vector<double> P(r * r), P2(r * r);
  return by_order(tup, [&](auto k) {
    constexpr int K = decltype(k)::value;
    int idx[K];
    for (int q = 0; q < K; ++q) idx[q] = tup[q];
    return lp::detail_weighted_sum::log_det_derivative<K>(idx, r, bp.data(), P.data(), P2.data());
  });
}

double err_ulps(double y, double ref) {
  const double e = std::abs(y - ref) / (std::max(std::abs(ref), 1.0) * eps);
  return std::isfinite(e) ? e : 1e300;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const auto cases = read_cases();
  const auto refs = read_refs();
  std::printf("worst error in ulps of max(|ref|, 1), orders 0..6, and worst |sum tr B_j - r|\n");
  const double exps[] = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50};
  struct setting { double a, zero; };
  std::vector<setting> settings;
  for (double a : exps) settings.push_back({a, 8.0});
  for (double z : {1.0, 64.0, 1024.0}) settings.push_back({0.05, z});
  for (const auto& [a, zf] : settings) {
    std::map<std::string, std::vector<double>> worst;
    std::map<std::string, double> ident;
    for (const auto& [n, c] : cases) worst[n].assign(7, 0.0);
    std::map<std::string, lp::weighted_sum_psdmat> st;
    for (const auto& [n, c] : cases) {
      lp::weighted_sum_psdmat s(c.S);
      s.split_tolerance = std::pow(eps, a);
      s.zero_tolerance = zf * c.p * eps;
      st.emplace(n, s);
    }
    for (const auto& r : refs) {
      const auto& s = st.at(r.name);
      std::vector<double> w(s.scratch_size());
      auto& wk = worst[r.name][r.tup.size()];
      wk = std::max(wk, err_ulps(eval(s, r.tup, r.theta.data(), w.data()), r.ref));
      if (r.tup.empty()) {
        const double d = std::abs(s.rank_identity(r.theta.data(), w.data()) - s.rank());
        ident[r.name] = std::max(ident[r.name], std::isfinite(d) ? d : 1e300);
      }
    }
    std::printf("\nsplit eps^%.2f, zero test %g p eps\n", a, zf);
    for (const auto& [n, v] : worst) {
      std::printf("  %-12s", n.c_str());
      for (double e : v) std::printf(" %9.3g", e);
      std::printf("   identity %.2g\n", ident[n]);
    }
  }
  {
    std::map<std::string, std::vector<double>> worst;
    for (const auto& [n, c] : cases) worst[n].assign(7, 0.0);
    for (const auto& r : refs) {
      auto& wk = worst[r.name][r.tup.size()];
      wk = std::max(wk, err_ulps(naive(cases.at(r.name), r.tup, r.theta.data()), r.ref));
    }
    std::printf("\nnaive route (eigenvalues of the sum in double)\n");
    for (const auto& [n, v] : worst) {
      std::printf("  %-12s", n.c_str());
      for (double e : v) std::printf(" %9.3g", e);
      std::printf("\n");
    }
  }
  // the errors by spread for one case, graded (default threshold) and naive
  for (const char* name : {"te2", "lowrank_ill"}) {
    const auto& c = cases.at(name);
    const lp::weighted_sum_psdmat s(c.S);
    std::vector<double> w(s.scratch_size());
    std::map<double, std::pair<double, double>> by_spread;  // theta_0 -> worst graded, naive
    for (const auto& r : refs) {
      if (r.name != name) continue;
      auto& b = by_spread[r.theta[0]];
      b.first = std::max(b.first, err_ulps(eval(s, r.tup, r.theta.data(), w.data()), r.ref));
      b.second = std::max(b.second, err_ulps(naive(c, r.tup, r.theta.data()), r.ref));
    }
    std::printf("\n%s by theta_0 / ln 10: worst over orders 0..6, graded and naive\n", name);
    for (const auto& [t0, b] : by_spread) std::printf("  %7.2f %9.3g %9.3g\n", t0 / std::log(10.0), b.first, b.second);
  }
  std::printf("\nmicroseconds per call, one thread\n");
  for (const auto& [n, c] : cases) {
    const lp::weighted_sum_psdmat s(c.S);
    std::vector<double> w(s.scratch_size());
    std::vector<double> t = {9.2, 0.0, 4.6};
    t.resize(c.m);
    double sink = 0.0;
    auto time = [&](const std::vector<int>& tup) {
      const int reps = 50;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) sink += eval(s, tup, t.data(), w.data());
      return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    std::vector<int> six(6);
    for (int q = 0; q < 6; ++q) six[q] = q * c.m / 6;
    std::printf("  %-12s p %3d r %3d m %d: value %8.1f  d1 %8.1f  d2 %8.1f  d6 %8.1f  (%g)\n", n.c_str(), c.p, c.rank,
                c.m, time({}), time({0}), time({0, c.m - 1}), time(six), sink * 0.0);
  }
  return 0;
}
