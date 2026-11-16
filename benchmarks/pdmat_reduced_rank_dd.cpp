// Measurement of a candidate for the derivatives of log|M|_+ of
// reduced_rank_psdmat at d < p: the same recursion on C = Lambda^T Lambda as
// the header (X[S] = -X[0] sum_T C[T] X[S \ T], d^F log det C = sum_T
// tr(X[T] C[f + rest])), with C, the C[T], the X[T] of nonempty masks and the
// traces in double-double; Lambda, the loading derivatives and X[0] = C^-1 (a
// Cholesky factorization of the rounded C) stay in double. Errors of both
// routes on the reference points (tests/data/reference/pdmat_reduced_rank.csv,
// log|M|_+ in ulp of max(|ref|, 1), orders 1 to 6, d < p) on the default
// execution space, then ns per call at the sizes of the bench
// pdmat_reduced_rank, one host thread and the default space. Since
// 2026-11-16 the header itself runs the last route (C^-1 in double-double as
// well), so the "header" column repeats it; the route in double that it
// replaced is measured in docs/numerics.md.
#include "measure.hpp"

#include <laplace/math/detail/double_double.hpp>
#include <laplace/pdmat/reduced_rank.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
namespace dp = laplace::pdmat::detail_pdmat;
using laplace::math::detail::ddm::dd;
using laplace::math::detail::ddm::two_prod;
using rr_psdmat = lp::reduced_rank_psdmat;

// doubles of scratch for the twin at order K (d < p)
KOKKOS_INLINE_FUNCTION int dd_scratch(int p, int d, int K) {
  const int nm = 1 << K, dd2 = d * d;
  return p * d + 2 * dd2 + 2 * (2 * nm * dd2) + 2 * dd2 + 2 * d + 6 * dd2;
}

template <int K, bool DDINV>
KOKKOS_INLINE_FUNCTION double log_pdet_dk_dd(const rr_psdmat& s, const double* theta, const int (&idx)[K], double* w) {
  const int p = s.dim, d = s.n_factors, dd2 = d * d;
  constexpr int NM = 1 << K;
  const dp::trapezoid_loadings z{p, d};
  double* lam = w;
  double* R = lam + p * d;
  double* X0d = R + dd2;
  dd* C = reinterpret_cast<dd*>(X0d + dd2);  // C[T], NM d x d
  dd* X = C + NM * dd2;                     // X[T], NM d x d (X[0] unused)
  dd* acc = X + NM * dd2;
  double* u = reinterpret_cast<double*>(acc + dd2);
  double* v = u + d;
  dd* X0 = reinterpret_cast<dd*>(v + d);  // C^-1
  dd* Rd = X0 + dd2;                      // dd Cholesky factor
  dd* Yd = Rd + dd2;
  bool cnz[NM], xnz[NM];
  z.load_lambda(theta, lam);
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      dd t(0.0);
      for (int i = 0; i < p; ++i) t = t + two_prod(lam[i * d + a], lam[i * d + b]);
      C[a * d + b] = t;
    }
  cnz[0] = true;
  for (int T = 1; T < NM; ++T) {
    int r = -1, nsl = 0, sl[K];
    bool one_row = true;
    for (int q = 0; q < K; ++q)
      if (T & (1 << q)) {
        int rq, cq;
        z.loading_position(idx[q], rq, cq);
        if (r < 0) r = rq;
        one_row = one_row && rq == r;
        sl[nsl++] = idx[q];
      }
    cnz[T] = false;
    if (!one_row) continue;
    dd* ct = C + T * dd2;
    for (int e = 0; e < dd2; ++e) ct[e] = dd(0.0);
    for (int A = 0; A < (1 << nsl); ++A) {
      int sa[K], sb[K], na = 0, nb = 0;
      for (int k = 0; k < nsl; ++k) (A & (1 << k) ? sa[na++] : sb[nb++]) = sl[k];
      if (!z.lambda_row_d(theta, lam, r, sa, na, u) || !z.lambda_row_d(theta, lam, r, sb, nb, v)) continue;
      cnz[T] = true;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) ct[a * d + b] = ct[a * d + b] + two_prod(u[a], v[b]);
    }
  }
  if constexpr (!DDINV) {  // X[0] = C^-1 in double from the Cholesky factor of the rounded C
    for (int e = 0; e < dd2; ++e) R[e] = C[e].hi;
    dp::fa_cholesky(d, R);
    double* Y = X0d;
    for (int j = 0; j < d; ++j) {
      for (int i = 0; i < j; ++i) Y[i * d + j] = 0.0;
      Y[j * d + j] = 1.0 / R[j * d + j];
      for (int i = j + 1; i < d; ++i) {
        double t = 0.0;
        for (int k = j; k < i; ++k) t += R[i * d + k] * Y[k * d + j];
        Y[i * d + j] = -t / R[i * d + i];
      }
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int k = a > b ? a : b; k < d; ++k) t += Y[k * d + a] * Y[k * d + b];
        X0[a * d + b] = dd(t);
      }
  } else {  // the same in double-double
    for (int j = 0; j < d; ++j) {
      dd s0 = C[j * d + j];
      for (int k = 0; k < j; ++k) s0 = s0 - Rd[j * d + k] * Rd[j * d + k];
      Rd[j * d + j] = laplace::math::detail::ddm::sqrt(s0);
      for (int i = j + 1; i < d; ++i) {
        dd t = C[i * d + j];
        for (int k = 0; k < j; ++k) t = t - Rd[i * d + k] * Rd[j * d + k];
        Rd[i * d + j] = t / Rd[j * d + j];
      }
    }
    for (int j = 0; j < d; ++j) {
      for (int i = 0; i < j; ++i) Yd[i * d + j] = dd(0.0);
      Yd[j * d + j] = dd(1.0) / Rd[j * d + j];
      for (int i = j + 1; i < d; ++i) {
        dd t(0.0);
        for (int k = j; k < i; ++k) t = t + Rd[i * d + k] * Yd[k * d + j];
        Yd[i * d + j] = -t / Rd[i * d + i];
      }
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        dd t(0.0);
        for (int k = a > b ? a : b; k < d; ++k) t = t + Yd[k * d + a] * Yd[k * d + b];
        X0[a * d + b] = t;
      }
  }
  xnz[0] = true;
  const int rest = (NM - 1) ^ 1;
  for (int Q = 2; Q < NM; Q += 2) {  // masks within rest only
    xnz[Q] = false;
    for (int e = 0; e < dd2; ++e) acc[e] = dd(0.0);
    for (int T = Q; T != 0; T = (T - 1) & Q) {
      const int Rm = Q ^ T;
      if (!cnz[T] || !xnz[Rm]) continue;
      xnz[Q] = true;
      const dd* ct = C + T * dd2;
      for (int a = 0; a < d; ++a)
        for (int k = 0; k < d; ++k) {
          const dd c = ct[a * d + k];
          if (Rm == 0)
            for (int b = 0; b < d; ++b) acc[a * d + b] = acc[a * d + b] + c * X0[k * d + b];
          else
            for (int b = 0; b < d; ++b) acc[a * d + b] = acc[a * d + b] + c * X[Rm * dd2 + k * d + b];
        }
    }
    if (!xnz[Q]) continue;
    dd* xq = X + Q * dd2;
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        dd t(0.0);
        for (int k = 0; k < d; ++k) t = t + acc[k * d + b] * X0[a * d + k];
        xq[a * d + b] = -t;
      }
  }
  dd r(0.0);
  for (int T = rest;; T = (T - 1) & rest) {
    const int U = 1 | (rest ^ T);
    if (xnz[T] && cnz[U]) {
      const dd* c = C + U * dd2;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b)
          r = r + (T == 0 ? c[b * d + a] * X0[a * d + b] : c[b * d + a] * X[T * dd2 + a * d + b]);
    }
    if (T == 0) break;
  }
  return r.hi + r.lo;
}

// --- accuracy -------------------------------------------------------------

constexpr int max_q = 16;
struct row {
  int p, d, K;
  double theta[max_q];
  int idx[6];
  double ref;
};

std::vector<row> read_rows(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);
  std::vector<row> out;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) f.push_back(c);
    row r{};
    r.p = std::stoi(f[0]);
    r.d = std::stoi(f[1]);
    r.K = std::stoi(f[4]);
    if (std::stoi(f[3]) != 2 || r.K == 0 || r.d == r.p) continue;
    std::stringstream st(f[2]);
    int n = 0;
    while (std::getline(st, c, ';')) r.theta[n++] = std::strtod(c.c_str(), nullptr);
    std::stringstream si(f[5]);
    n = 0;
    while (std::getline(si, c, ';')) r.idx[n++] = std::stoi(c);
    r.ref = std::strtod(f[8].c_str(), nullptr) + std::strtod(f[9].c_str(), nullptr);
    out.push_back(r);
  }
  return out;
}

template <int K>
KOKKOS_INLINE_FUNCTION void eval_both(const row& r, double* w, double& a, double& b, double& c) {
  const rr_psdmat s(r.p, r.d);
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = r.idx[k];
  a = s.template log_pdet_dk<K>(r.theta, idx, w);
  b = log_pdet_dk_dd<K, false>(s, r.theta, idx, w);
  c = log_pdet_dk_dd<K, true>(s, r.theta, idx, w);
}

void accuracy() {
  const auto rows = read_rows(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_reduced_rank.csv");
  const int n = int(rows.size());
  using Mem = bench::Exec::memory_space;
  Kokkos::View<row*, Mem> dr("rows", n);
  Kokkos::View<double* [3], Mem> res("res", n);
  const int sw = std::max(rr_psdmat(5, 4).scratch_size(), dd_scratch(5, 4, 6));
  Kokkos::View<double**, Kokkos::LayoutRight, Mem> w("w", n, sw);
  auto hr = Kokkos::create_mirror_view(dr);
  for (int i = 0; i < n; ++i) hr(i) = rows[i];
  Kokkos::deep_copy(dr, hr);
  Kokkos::parallel_for(
      "acc", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(int i) {
        const row& r = dr(i);
        double a = 0.0, b = 0.0, c = 0.0;
        double* ww = &w(i, 0);
        switch (r.K) {
          case 1: eval_both<1>(r, ww, a, b, c); break;
          case 2: eval_both<2>(r, ww, a, b, c); break;
          case 3: eval_both<3>(r, ww, a, b, c); break;
          case 4: eval_both<4>(r, ww, a, b, c); break;
          case 5: eval_both<5>(r, ww, a, b, c); break;
          default: eval_both<6>(r, ww, a, b, c); break;
        }
        res(i, 0) = a;
        res(i, 1) = b;
        res(i, 2) = c;
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  // category 0: every tuple; 1: tuples with a diagonal loading repeated; 2: the others
  double worst[3][7][3] = {};
  for (int i = 0; i < n; ++i) {
    const row& r = rows[i];
    const dp::trapezoid_loadings z{r.p, r.d};
    bool rep = false;
    for (int k = 1; k < r.K; ++k) {
      int rr, cc;
      z.loading_position(r.idx[k], rr, cc);
      rep = rep || (r.idx[k] == r.idx[k - 1] && rr == cc);
    }
    const double m1 = std::max(std::fabs(r.ref), 1.0), u = std::nextafter(m1, 1e300) - m1;
    for (int m = 0; m < 3; ++m) {
      const double e = std::fabs(h(i, m) - r.ref) / u;
      for (int cat : {0, rep ? 1 : 2}) worst[cat][r.K][m] = std::max(worst[cat][r.K][m], e);
    }
  }
  const char* names[3] = {"all tuples", "a diagonal loading repeated", "no diagonal loading repeated"};
  for (int cat = 0; cat < 3; ++cat) {
    std::printf("log|M|_+, d < p, %s, %s: worst ulp of max(|ref|, 1)  [header | dd, C^-1 double | dd, C^-1 dd]\n",
                names[cat], bench::Exec::name());
    for (int K = 1; K <= 6; ++K)
      std::printf("  K=%d  %9.3g  %9.3g  %9.3g\n", K, worst[cat][K][0], worst[cat][K][1], worst[cat][K][2]);
  }
}

// --- speed ----------------------------------------------------------------

constexpr int max_n = 16 * 4;

template <int K>
struct Bench {
  rr_psdmat s;
  int route;  // 0 header, 1 dd with C^-1 in double, 2 dd
  KOKKOS_INLINE_FUNCTION double operator()(double x, double* w) const {
    double u[max_n];
    int idx[K] = {};
    const int n = s.n_params();
    for (int k = 0; k < n; ++k) u[k] = 0.1 * x + 0.05 * Kokkos::sin(double(k));
    for (int k = 0; k < K; ++k) idx[k] = k * (n - 1) / K;  // as in pdmat_reduced_rank
    return route == 0 ? s.template log_pdet_dk<K>(u, idx, w)
           : route == 1 ? log_pdet_dk_dd<K, false>(s, u, idx, w) : log_pdet_dk_dd<K, true>(s, u, idx, w);
  }
};

template <class Space, class F>
double time_calls(const F& f, const std::vector<double>& xs, int scratch, int reps) {
  using Mem = typename Space::memory_space;
  Kokkos::View<double*, Mem> x("x", xs.size());
  Kokkos::View<double**, Kokkos::LayoutRight, Mem> w("w", xs.size(), scratch);
  auto hx = Kokkos::create_mirror_view(x);
  for (std::size_t i = 0; i < xs.size(); ++i) hx(i) = xs[i];
  Kokkos::deep_copy(x, hx);
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double sum = 0.0;
    Kokkos::fence();
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(
        "time", Kokkos::RangePolicy<Space>(0, xs.size()),
        KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += f(x(i), &w(i, 0)); }, sum);
    Kokkos::fence();
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (sum == 12345.678) std::printf(" ");
  }
  return best * 1e9 / double(xs.size());
}

template <int K>
void speed(int p, int d, const std::vector<double>& all) {
  const rr_psdmat s(p, d);
  const int sw = std::max(s.scratch_size(), dd_scratch(p, d, K));
  const std::size_t n = std::min<std::size_t>(all.size(), std::size_t(2.5e7 / sw) + 1);
  const std::vector<double> xs(all.begin(), all.begin() + n);
  double t[3][2];
  for (int route = 0; route < 3; ++route) {
    const Bench<K> f{s, route};
    t[route][0] = time_calls<bench::HostOne>(f, xs, sw, 5);
    t[route][1] = time_calls<bench::Exec>(f, xs, sw, 7);
  }
  std::printf("p=%2d d=%d K=%d  header %8.1f / %7.2f   dd %8.1f / %7.2f   dd+inv %8.1f / %7.2f ns  (1 thread / %s)\n",
              p, d, K, t[0][0], t[0][1], t[1][0], t[1][1], t[2][0], t[2][1], bench::Exec::name());
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    accuracy();
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<double> xs(20000);
    for (double& x : xs) x = u(rng);
    for (auto [p, d] : {std::pair{4, 1}, std::pair{8, 2}, std::pair{16, 4}}) {
      speed<1>(p, d, xs);
      speed<3>(p, d, xs);
      speed<6>(p, d, xs);
    }
  }
  Kokkos::finalize();
}
