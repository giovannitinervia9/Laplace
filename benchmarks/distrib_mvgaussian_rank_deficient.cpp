// Speed of the rank-deficient mvgaussian1 (degenerate) and mvgaussian2
// (improper) over weighted_sum_psdmat on the host, one thread, at d = 500 and
// 2000 (the penalty scale): S = (D1'D1, D2'D2, D3'D3), D_k the k-th
// differences, null space the constants (rank d - 1).
// Usage: distrib_mvgaussian_rank_deficient [d ...]   (default 500 2000)
// Prints the seconds of each operation (the median of reps calls) and the
// doubles of the state and the scratch.
#include <laplace/distrib/mvgaussian1.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using ws_t = lp::weighted_sum_psdmat;
using host_matrix = ws_t::host_matrix;

host_matrix diff_gram(int d, int k) {
  std::vector<std::vector<double>> r(d, std::vector<double>(d, 0.0));
  for (int i = 0; i < d; ++i) r[i][i] = 1.0;
  for (int s = 0; s < k; ++s) {
    std::vector<std::vector<double>> n(r.size() - 1, std::vector<double>(d));
    for (std::size_t i = 0; i + 1 < r.size(); ++i)
      for (int j = 0; j < d; ++j) n[i][j] = r[i + 1][j] - r[i][j];
    r = n;
  }
  host_matrix m("S", d, d);
  for (std::size_t q = 0; q < r.size(); ++q)
    for (int i = 0; i < d; ++i) {
      if (r[q][i] == 0.0) continue;
      for (int j = 0; j < d; ++j) m(i, j) += r[q][i] * r[q][j];
    }
  return m;
}

template <class F>
double seconds(int reps, F f) {
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    const auto a = std::chrono::steady_clock::now();
    f();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

template <class D>
void run(const char* name, const D& g, const ws_t& s, bool improper) {
  const int d = s.dim, np = lp::packed_size(d);
  const double th[3] = {1.0, -2.0, 0.5};
  const int sc = std::max(g.scratch_size(), g.info_scratch_size());
  std::vector<double> w(std::size_t(g.state_size()) + sc + g.theta_table_size(2)), y(d), mu(d, 0.0), out(np);
  for (int i = 0; i < d; ++i) y[i] = 0.01 * (i % 7) - 0.03;
  double m = 0.0;
  for (double v : y) m += v / d;
  for (double& v : y) v -= m;  // in the range (sum zero)
  double* ws = w.data() + g.state_size();
  const int reps = d <= 500 ? 5 : 1;
  typename D::state_type st;
  std::printf("%s d = %d: state %d doubles, scratch %d, info scratch %d\n", name, d, g.state_size(), g.scratch_size(),
              g.info_scratch_size());
  std::printf("  theta_state            %.4g s\n", seconds(reps, [&] { st = g.theta_state(th, w.data()); }));
  if constexpr (requires { g.constrained_state(th, w.data()); })
    std::printf("  constrained_state      %.4g s\n", seconds(reps, [&] { st = g.constrained_state(th, w.data()); }));
  const int i0[1] = {0}, i01[2] = {0, 1}, i00[2] = {0, 0};
  std::printf("  logpdf                 %.4g s\n", seconds(reps, [&] { out[0] = g.logpdf(st, y.data(), mu.data()); }));
  std::printf("  A=1 K=0 (W r)          %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<1, 0>(st, y.data(), mu.data(), nullptr, out.data(), ws); }));
  std::printf("  A=0 K=1                %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<0, 1>(st, y.data(), mu.data(), i0, out.data(), ws); }));
  std::printf("  A=0 K=2 (0,0)          %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<0, 2>(st, y.data(), mu.data(), i00, out.data(), ws); }));
  std::printf("  A=0 K=2 (0,1)          %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<0, 2>(st, y.data(), mu.data(), i01, out.data(), ws); }));
  std::printf("  A=1 K=1                %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<1, 1>(st, y.data(), mu.data(), i0, out.data(), ws); }));
  double* tab = ws + sc;
  std::printf("  table orders 1-2 (%d)  %.4g s\n", g.theta_table_entries(2), seconds(1, [&] {
                for (int e = 0; e < g.theta_table_entries(2); ++e) g.fill_theta_table_entry(st, 2, e, tab, ws);
              }));
  const auto stt = g.with_theta_table(st, 2, tab);
  std::printf("  A=0 K=2 (0,1) table    %.4g s\n",
              seconds(reps, [&] { g.template logpdf_dmu_dk<0, 2>(stt, y.data(), mu.data(), i01, out.data(), ws); }));
  std::printf("  log_pdet (structure)   %.4g s\n", seconds(reps, [&] { out[0] = s.log_pdet(th, ws); }));
  (void)improper;
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<int> dims;
  for (int a = 1; a < argc; ++a)
    if (argv[a][0] != '-') dims.push_back(std::atoi(argv[a]));
  if (dims.empty()) dims = {500, 2000};
  for (int d : dims) {
    const auto a = std::chrono::steady_clock::now();
    const ws_t s(std::vector<host_matrix>{diff_gram(d, 1), diff_gram(d, 2), diff_gram(d, 3)});
    std::printf("weighted_sum_psdmat d = %d, rank %d: construction %.4g s\n", d, s.rank(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
    run("mvgaussian2 (improper)", ld::mvgaussian2_distrib<ws_t>(s), s, true);
    run("mvgaussian1 (degenerate)", ld::mvgaussian1_distrib<ws_t>(s), s, false);
  }
}
