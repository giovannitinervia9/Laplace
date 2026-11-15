// Repeated work in weighted_sum_psdmat: the time of a full table of log pseudo-determinant
// derivatives at one theta (every sorted tuple of orders 0..K, one call each, as the API is
// used today) against the time of the shared part alone (the transform, the Cholesky factor
// and the B_j, measured by the order-1 call, whose trace costs nothing). Cases: m penalties
// that split the second-difference penalty of p columns by rows.
//
// Usage: pdmat_ws_state

#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>
#include <functional>
#include <type_traits>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
using host_matrix = lp::weighted_sum_psdmat::host_matrix;

// S = D'D for the d-th difference operator on n columns (rank n - d), as a p x p matrix
// acting on a Kronecker block: block j of the m blocks of an m-dimensional tensor grid.
host_matrix diff_penalty(int n, int d) {
  host_matrix S("S", n, n);
  std::vector<double> c = {1.0};
  for (int k = 0; k < d; ++k) {
    std::vector<double> e(c.size() + 1, 0.0);
    for (size_t i = 0; i < c.size(); ++i) {
      e[i] += c[i];
      e[i + 1] -= c[i];
    }
    c = e;
  }
  for (int r = 0; r + d < n; ++r)
    for (size_t a = 0; a < c.size(); ++a)
      for (size_t b = 0; b < c.size(); ++b) S(r + a, r + b) += c[a] * c[b];
  return S;
}

// S_j = sum over the second-difference rows r = j (mod m) of d_r d_r': the m penalties add up to
// the full second-difference penalty of p columns (rank p - 2), each one has its own rank
host_matrix row_penalty(int p, int m, int j) {
  host_matrix S("S", p, p);
  const double c[3] = {1.0, -2.0, 1.0};
  for (int r = j; r + 2 < p; r += m)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) S(r + a, r + b) += c[a] * c[b];
  return S;
}

template <int K>
double call(const lp::weighted_sum_psdmat& s, const double* t, const std::vector<int>& tup, double* w) {
  int idx[K];
  for (int q = 0; q < K; ++q) idx[q] = tup[q];
  return s.template log_pdet_dk<K>(t, idx, w);
}

double eval(const lp::weighted_sum_psdmat& s, const double* t, const std::vector<int>& tup, double* w) {
  switch (tup.size()) {
    case 0: return s.log_pdet(t, w);
    case 1: return call<1>(s, t, tup, w);
    case 2: return call<2>(s, t, tup, w);
    case 3: return call<3>(s, t, tup, w);
    case 4: return call<4>(s, t, tup, w);
    case 5: return call<5>(s, t, tup, w);
    default: return call<6>(s, t, tup, w);
  }
}

void tuples(int m, int K, std::vector<int>& cur, int from, std::vector<std::vector<int>>& out) {
  if (static_cast<int>(cur.size()) == K) {
    out.push_back(cur);
    return;
  }
  for (int j = from; j < m; ++j) {
    cur.push_back(j);
    tuples(m, K, cur, j, out);
    cur.pop_back();
  }
}

double micros(const std::function<void()>& f, int reps) {
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) f();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    struct cfg {
      int m, n;
    };
    const std::vector<cfg> cfgs = {{2, 10}, {2, 20}, {2, 50}, {3, 10}, {3, 20}, {3, 50}, {4, 10}, {4, 20}, {4, 50}};
    double sink = 0.0;
    std::printf("%-8s %5s %4s | %9s | %10s %10s %6s | %10s %10s %6s\n", "case", "p", "N6", "transform", "table K<=2",
                "of which", "share", "table K<=6", "of which", "share");
    for (const auto& c : cfgs) {
      std::vector<host_matrix> S;
      for (int j = 0; j < c.m; ++j) S.push_back(row_penalty(c.n, c.m, j));
      const lp::weighted_sum_psdmat s(S);
      std::vector<double> w(s.scratch_size());
      std::vector<double> t(c.m);
      for (int j = 0; j < c.m; ++j) t[j] = 1.5 * j - 1.0;
      auto table = [&](int Kmax) {
        std::vector<std::vector<int>> all;
        for (int K = 0; K <= Kmax; ++K) {
          std::vector<int> cur;
          tuples(c.m, K, cur, 0, all);
        }
        return all;
      };
      const auto t2 = table(2), t6 = table(6);
      const std::vector<int> one = {0};
      const int reps = c.n > 30 ? 5 : 30;
      const double T = micros([&] { sink += eval(s, t.data(), one, w.data()); }, reps * 3);
      const double F2 = micros([&] { for (const auto& u : t2) sink += eval(s, t.data(), u, w.data()); }, reps);
      const double F6 = micros([&] { for (const auto& u : t6) sink += eval(s, t.data(), u, w.data()); }, reps);
      // the same tables from one evaluated state
      auto from_state = [&](const std::vector<std::vector<int>>& all) {
        sink += s.log_pdet_state(t.data(), w.data());
        for (const auto& u : all) {
          if (u.empty()) continue;
          switch (u.size()) {
            case 1: { int i[1] = {u[0]}; sink += s.log_pdet_dk_at<1>(i, w.data()); break; }
            case 2: { int i[2] = {u[0], u[1]}; sink += s.log_pdet_dk_at<2>(i, w.data()); break; }
            case 3: { int i[3] = {u[0], u[1], u[2]}; sink += s.log_pdet_dk_at<3>(i, w.data()); break; }
            case 4: { int i[4] = {u[0], u[1], u[2], u[3]}; sink += s.log_pdet_dk_at<4>(i, w.data()); break; }
            case 5: { int i[5] = {u[0], u[1], u[2], u[3], u[4]}; sink += s.log_pdet_dk_at<5>(i, w.data()); break; }
            default: { int i[6] = {u[0], u[1], u[2], u[3], u[4], u[5]}; sink += s.log_pdet_dk_at<6>(i, w.data()); }
          }
        }
      };
      const double S2 = micros([&] { from_state(t2); }, reps);
      const double S6 = micros([&] { from_state(t6); }, reps);
      std::printf("%-8s state: K<=2 %10.1f (x%.1f)   K<=6 %10.1f (x%.1f)\n", "", S2, F2 / S2, S6, F6 / S6);
      const double sh2 = t2.size() * T, sh6 = t6.size() * T;
      char name[32];
      std::snprintf(name, sizeof name, "m%d n%d", c.m, c.n);
      const int p = c.n;
      std::printf("%-8s %5d %4zu | %9.1f | %10.1f %10.1f %5.0f%% | %10.1f %10.1f %5.0f%%\n", name, p, t6.size(), T, F2,
                  sh2, 100.0 * sh2 / F2, F6, sh6, 100.0 * sh6 / F6);
    }
    std::printf("(%g)\n", sink * 0.0);
  }
  Kokkos::finalize();
  return 0;
}
