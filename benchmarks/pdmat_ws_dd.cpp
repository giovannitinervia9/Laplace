// Measurement twin of weighted_sum_psdmat: the derivatives of log|M|_+ with a growing part
// of the partition sum in double-double, read from the state of log_pdet_state (the matrices
// G_j and B_j in the scratch w, r x r, leading dimension r). Routes:
//   double : the production route (log_pdet_dk_at)
//   (a)    : the products of the words in double, their traces and the weighted sum in
//            double-double
//   (b)    : (a) with the matrix products of the words in double-double (B_j in double)
//   (c)    : (b) with B_j = H^-1 G_j recomputed in double-double from G_j (H = sum G_j,
//            Cholesky factor and the two triangular solves in double-double)
// Accuracy: the 3045 references of tests/data/reference/pdmat_weighted_sum.csv, worst error
// per case and order in ulps of max(|ref|, 1). Speed: microseconds per call (state included)
// and from the state, one thread and 8 threads running the same call each.
//
// Usage: pdmat_ws_dd

#include <laplace/math/detail/double_double.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace lp = laplace::pdmat;
using host_matrix = lp::weighted_sum_psdmat::host_matrix;
using dd = laplace::math::detail::ddm::dd;
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

double err_ulps(double y, double ref) {
  const double e = std::abs(y - ref) / (std::max(std::abs(ref), 1.0) * eps);
  return std::isfinite(e) ? e : 1e300;
}

// ---- the words of the partition sum (the enumeration of weighted_sum.hpp) ----
std::vector<int> least_rotation(const std::vector<int>& w) {
  std::vector<int> best = w, r = w;
  for (size_t s = 1; s < w.size(); ++s) {
    std::rotate(r.begin(), r.begin() + 1, r.end());
    if (r < best) best = r;
  }
  return best;
}

std::map<std::vector<int>, double> words_of(const std::vector<int>& idx) {
  const int K = static_cast<int>(idx.size());
  std::map<std::vector<int>, double> coef;
  std::vector<int> a(K, 0);
  while (true) {
    int nb = 0;
    for (int q = 0; q < K; ++q) nb = std::max(nb, a[q] + 1);
    std::vector<int> label(nb, -1);
    bool uniform = true;
    for (int q = 0; q < K && uniform; ++q) {
      if (label[a[q]] < 0) label[a[q]] = idx[q];
      uniform = label[a[q]] == idx[q];
    }
    if (uniform) {
      const double sign = nb % 2 == 1 ? 1.0 : -1.0;
      std::vector<int> perm(nb);
      for (int b = 0; b < nb; ++b) perm[b] = b;
      do {
        std::vector<int> word(nb);
        for (int b = 0; b < nb; ++b) word[b] = label[perm[b]];
        coef[least_rotation(word)] += sign;
      } while (std::next_permutation(perm.begin() + 1, perm.end()));
    }
    int q = K - 1;
    for (; q > 0; --q) {
      int mxp = 0;
      for (int s = 0; s < q; ++s) mxp = std::max(mxp, a[s]);
      if (a[q] <= mxp) {
        ++a[q];
        for (int s = q + 1; s < K; ++s) a[s] = 0;
        break;
      }
    }
    if (q == 0) break;
  }
  return coef;
}

// ---- double matrices (column-major, r x r) ----
void mul_d(int r, const double* A, const double* B, double* C) {
  for (int j = 0; j < r; ++j) {
    for (int i = 0; i < r; ++i) C[i + j * r] = 0.0;
    for (int l = 0; l < r; ++l) {
      const double b = B[l + j * r];
      for (int i = 0; i < r; ++i) C[i + j * r] += A[i + l * r] * b;
    }
  }
}

// ---- double-double matrices ----
using dmat = std::vector<dd>;
void mul_dd(int r, const dmat& A, const dmat& B, dmat& C) {
  for (int j = 0; j < r; ++j)
    for (int i = 0; i < r; ++i) {
      dd s(0.0);
      for (int l = 0; l < r; ++l) s = s + A[i + l * r] * B[l + j * r];
      C[i + j * r] = s;
    }
}

double hi_lo(const dd& x) { return x.hi + x.lo; }

enum class route { dbl, a, b, c, d };

// B_j = H^-1 G_j in double-double, from the G_j of the state
std::vector<dmat> b_in_dd(int m, int r, const double* w, int p) {
  const size_t pp = static_cast<size_t>(p) * p;
  dmat L(r * r, dd(0.0));
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < r * r; ++i) L[i] = L[i] + w[j * pp + i];
  // Cholesky of H (lower), in place
  for (int j = 0; j < r; ++j) {
    dd d = L[j + j * r];
    for (int k = 0; k < j; ++k) d = d - L[j + k * r] * L[j + k * r];
    d = laplace::math::detail::ddm::sqrt(d);
    L[j + j * r] = d;
    for (int i = j + 1; i < r; ++i) {
      dd s = L[i + j * r];
      for (int k = 0; k < j; ++k) s = s - L[i + k * r] * L[j + k * r];
      L[i + j * r] = s / d;
    }
  }
  std::vector<dmat> B(m, dmat(r * r));
  for (int jm = 0; jm < m; ++jm) {
    dmat& X = B[jm];
    for (int i = 0; i < r * r; ++i) X[i] = dd(w[jm * pp + i]);
    for (int c = 0; c < r; ++c) {  // L Y = G, then L^T X = Y, column by column
      for (int i = 0; i < r; ++i) {
        dd s = X[i + c * r];
        for (int k = 0; k < i; ++k) s = s - L[i + k * r] * X[k + c * r];
        X[i + c * r] = s / L[i + i * r];
      }
      for (int i = r - 1; i >= 0; --i) {
        dd s = X[i + c * r];
        for (int k = i + 1; k < r; ++k) s = s - L[k + i * r] * X[k + c * r];
        X[i + c * r] = s / L[i + i * r];
      }
    }
  }
  return B;
}

// the derivative along idx from the state in w (G_j at j p^2, B_j at (m + j) p^2, r x r)
double derivative(route rt, int m, int r, int p, const double* w, const std::vector<int>& idx,
                  const std::vector<dmat>* Bdd /* routes b, c, d: the B_j in double-double */,
                  std::map<std::vector<int>, dd>* memo = nullptr /* route d: traces shared between tuples */) {
  const size_t pp = static_cast<size_t>(p) * p;
  const auto coef = words_of(idx);
  if (rt == route::d) {
    // the words are visited in lexicographic order: the products of the first k matrices of the
    // previous word that are also a prefix of this one are kept (stk[k] = product of the first
    // k + 1 matrices, k >= 1), and the last factor only enters the trace
    const auto& B = *Bdd;
    std::vector<dmat> stk(8, dmat(r * r));
    std::vector<int> prev;
    int have = 0;  // stk[1..have] are valid for the prefix of `prev`
    dd tot(0.0);
    for (const auto& [word, c] : coef) {
      if (c == 0.0) continue;
      const int n = static_cast<int>(word.size());
      dd tr(0.0);
      auto hit = memo ? memo->find(word) : std::map<std::vector<int>, dd>::iterator();
      if (memo && hit != memo->end()) {
        tr = hit->second;
      } else if (n == 1) {
        for (int a = 0; a < r; ++a) tr = tr + B[word[0]][a + a * r];
      } else {
        const dmat* left = &B[word[0]];
        if (n > 2) {
          int cp = 0;
          while (cp < static_cast<int>(prev.size()) && cp < n && prev[cp] == word[cp]) ++cp;
          have = std::min(have, cp - 1);
          if (have < 0) have = 0;
          for (int k = have + 1; k <= n - 2; ++k) {
            if (k == 1) mul_dd(r, B[word[0]], B[word[1]], stk[1]);
            else mul_dd(r, stk[k - 1], B[word[k]], stk[k]);
          }
          have = n - 2;
          prev = word;
          left = &stk[n - 2];
        }
        const dmat& last = B[word[n - 1]];
        for (int a = 0; a < r; ++a)
          for (int b = 0; b < r; ++b) tr = tr + (*left)[a + b * r] * last[b + a * r];
        if (memo) (*memo)[word] = tr;
      }
      tot = tot + tr * c;
    }
    return hi_lo(tot);
  }
  dd total(0.0);
  std::vector<double> P(r * r), P2(r * r);
  dmat Q(r * r), Q2(r * r);
  for (const auto& [word, c] : coef) {
    if (c == 0.0) continue;
    const int n = static_cast<int>(word.size());
    dd tr(0.0);
    if (rt == route::dbl || rt == route::a) {
      auto Bj = [&](int j) { return w + (m + j) * pp; };
      const double* left = Bj(word[0]);
      if (n == 1) {
        for (int a = 0; a < r; ++a) tr = tr + left[a + a * r];
      } else {
        if (n > 2) {
          mul_d(r, left, Bj(word[1]), P.data());
          for (int s = 2; s < n - 1; ++s) {
            mul_d(r, P.data(), Bj(word[s]), P2.data());
            std::swap(P, P2);
          }
          left = P.data();
        }
        const double* last = Bj(word[n - 1]);
        if (rt == route::dbl) {
          double t = 0.0;
          for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b) t += left[a + b * r] * last[b + a * r];
          tr = dd(t);
        } else {
          for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b) tr = tr + left[a + b * r] * last[b + a * r];
        }
      }
      if (rt == route::dbl) total = dd(total.hi + c * tr.hi);
      else total = total + tr * c;
    } else {
      const auto& B = *Bdd;
      if (n == 1) {
        for (int a = 0; a < r; ++a) tr = tr + B[word[0]][a + a * r];
      } else {
        const dmat* left = &B[word[0]];
        if (n > 2) {
          mul_dd(r, *left, B[word[1]], Q);
          for (int s = 2; s < n - 1; ++s) {
            mul_dd(r, Q, B[word[s]], Q2);
            std::swap(Q, Q2);
          }
          left = &Q;
        }
        const dmat& last = B[word[n - 1]];
        for (int a = 0; a < r; ++a)
          for (int b = 0; b < r; ++b) tr = tr + (*left)[a + b * r] * last[b + a * r];
      }
      total = total + tr * c;
    }
  }
  return hi_lo(total);
}

// B_j in double-double: routes b and c, from the double B_j (b) or recomputed from G_j (c)
std::vector<dmat> make_bdd(route rt, int m, int r, int p, const double* w) {
  if (rt >= route::c) return b_in_dd(m, r, w, p);
  const size_t pp = static_cast<size_t>(p) * p;
  std::vector<dmat> B(m, dmat(r * r));
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < r * r; ++i) B[j][i] = dd(w[(m + j) * pp + i]);
  return B;
}

double eval(route rt, const lp::weighted_sum_psdmat& s, int m, const std::vector<int>& tup, const double* t, double* w) {
  const int p = s.dim, r = s.rank();
  const double v = s.log_pdet_state(t, w);
  if (tup.empty()) return v;
  std::vector<dmat> Bdd;
  if (rt >= route::b) Bdd = make_bdd(rt, m, r, p, w);
  return derivative(rt, m, r, p, w, tup, rt >= route::b ? &Bdd : nullptr);
}

const char* name_of(route rt) {
  switch (rt) {
    case route::dbl: return "double";
    case route::a: return "(a) traces and sum in dd";
    case route::b: return "(b) + products in dd";
    case route::c: return "(c) + B_j in dd";
    default: return "(d) = (c), prefix products shared";
  }
}

template <class F>
double micros(F&& f, int reps) {
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) f();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const auto cases = read_cases();
  const auto refs = read_refs();
  std::map<std::string, lp::weighted_sum_psdmat> st;
  for (const auto& [n, c] : cases) st.emplace(n, lp::weighted_sum_psdmat(c.S));
  const route routes[] = {route::dbl, route::a, route::b, route::c, route::d};

  std::printf("worst error in ulps of max(|ref|, 1) by case and order 1..6 (order 0 is the same for every route)\n");
  for (route rt : routes) {
    std::map<std::string, std::vector<double>> worst;
    for (const auto& [n, c] : cases) worst[n].assign(7, 0.0);
    for (const auto& r : refs) {
      if (r.tup.empty()) continue;
      const auto& s = st.at(r.name);
      std::vector<double> w(s.scratch_size());
      const double y = eval(rt, s, cases.at(r.name).m, r.tup, r.theta.data(), w.data());
      auto& wk = worst[r.name][r.tup.size()];
      wk = std::max(wk, err_ulps(y, r.ref));
    }
    std::printf("\n%s\n", name_of(rt));
    for (const auto& [n, v] : worst) {
      std::printf("  %-12s", n.c_str());
      for (int k = 1; k <= 6; ++k) std::printf(" %9.3g", v[k]);
      std::printf("\n");
    }
    std::fflush(stdout);
  }

  std::printf("\nmicroseconds per call, theta = (9.2, 0, 4.6): per call (state included) / from the state;"
              " one thread, then 8 threads each running the call\n");
  const int nthreads = 8;
  for (const auto& [n, c] : cases) {
    const auto& s = st.at(n);
    std::vector<double> t = {9.2, 0.0, 4.6};
    t.resize(c.m);
    std::vector<int> six(6);
    for (int q = 0; q < 6; ++q) six[q] = q * c.m / 6;
    const std::vector<std::vector<int>> tups = {{0}, {0, c.m - 1}, six};
    const char* tn[] = {"K=1", "K=2", "K=6"};
    std::printf("  %-12s p %3d r %3d m %d\n", n.c_str(), c.p, c.rank, c.m);
    for (route rt : routes) {
      std::printf("    %-26s", name_of(rt));
      for (size_t k = 0; k < tups.size(); ++k) {
        const int reps = k == 2 ? (rt == route::dbl ? 20 : 3) : (rt == route::dbl ? 200 : 20);
        std::vector<double> w(s.scratch_size());
        double sink = 0.0;
        const double full = micros([&] { sink += eval(rt, s, c.m, tups[k], t.data(), w.data()); }, reps);
        // from the state: the state is built once, only the derivative is repeated
        s.log_pdet_state(t.data(), w.data());
        const double from = micros(
            [&] {
              std::vector<dmat> Bdd;
              if (rt >= route::b) Bdd = make_bdd(rt, c.m, s.rank(), s.dim, w.data());
              sink += derivative(rt, c.m, s.rank(), s.dim, w.data(), tups[k], rt >= route::b ? &Bdd : nullptr);
            },
            reps);
        // 8 threads, each its own scratch, the same call
        std::vector<double> per(nthreads, 0.0);
        std::vector<std::thread> th;
        for (int q = 0; q < nthreads; ++q)
          th.emplace_back([&, q] {
            std::vector<double> wq(s.scratch_size());
            double sk = 0.0;
            per[q] = micros([&] { sk += eval(rt, s, c.m, tups[k], t.data(), wq.data()); }, reps);
            if (sk == 1.2345) std::printf(" ");
          });
        for (auto& x : th) x.join();
        const double par = *std::max_element(per.begin(), per.end());
        std::printf("  %s %9.1f / %9.1f / %9.1f", tn[k], full, from, par);
        if (sink == 1.2345) std::printf(" ");
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  }
  // a full table of every sorted tuple of orders 1 to 6 at one theta
  std::printf("\nfull table, all sorted tuples of orders 1..6 at theta = (9.2, 0, 4.6), microseconds\n");
  for (const char* name : {"te2", "te3"}) {
    const auto& c = cases.at(name);
    const auto& s = st.at(name);
    std::vector<double> t = {9.2, 0.0, 4.6};
    t.resize(c.m);
    std::vector<std::vector<int>> all;
    std::function<void(std::vector<int>&, int, int)> gen = [&](std::vector<int>& cur, int K, int from) {
      if (static_cast<int>(cur.size()) == K) {
        all.push_back(cur);
        return;
      }
      for (int j = from; j < c.m; ++j) {
        cur.push_back(j);
        gen(cur, K, j);
        cur.pop_back();
      }
    };
    for (int K = 1; K <= 6; ++K) {
      std::vector<int> cur;
      gen(cur, K, 0);
    }
    size_t total_words = 0;
    std::map<std::vector<int>, int> distinct;
    for (const auto& u : all)
      for (const auto& [word, cf] : words_of(u))
        if (cf != 0.0) {
          ++total_words;
          distinct[word] = 1;
        }
    std::vector<double> w(s.scratch_size());
    double sink = 0.0;
    auto prod = [&](const std::vector<int>& u) {
      switch (u.size()) {
        case 1: { int i[1] = {u[0]}; return s.log_pdet_dk_at<1>(i, w.data()); }
        case 2: { int i[2] = {u[0], u[1]}; return s.log_pdet_dk_at<2>(i, w.data()); }
        case 3: { int i[3] = {u[0], u[1], u[2]}; return s.log_pdet_dk_at<3>(i, w.data()); }
        case 4: { int i[4] = {u[0], u[1], u[2], u[3]}; return s.log_pdet_dk_at<4>(i, w.data()); }
        case 5: { int i[5] = {u[0], u[1], u[2], u[3], u[4]}; return s.log_pdet_dk_at<5>(i, w.data()); }
        default: { int i[6] = {u[0], u[1], u[2], u[3], u[4], u[5]}; return s.log_pdet_dk_at<6>(i, w.data()); }
      }
    };
    const int reps = 2;
    const double t_state = micros([&] { sink += s.log_pdet_state(t.data(), w.data()); }, 5);
    s.log_pdet_state(t.data(), w.data());
    const double t_prod = micros([&] { for (const auto& u : all) sink += prod(u); }, reps);
    const int r = s.rank(), p = s.dim;
    std::vector<dmat> Bdd = make_bdd(route::c, c.m, r, p, w.data());
    const double t_d = micros([&] { for (const auto& u : all) sink += derivative(route::d, c.m, r, p, w.data(), u, &Bdd); }, reps);
    const double t_memo = micros(
        [&] {
          std::map<std::vector<int>, dd> memo;
          for (const auto& u : all) sink += derivative(route::d, c.m, r, p, w.data(), u, &Bdd, &memo);
        },
        reps);
    double dmax = 0.0, dmax_memo = 0.0;
    std::map<std::vector<int>, dd> memo;
    for (const auto& u : all) {
      const double a = prod(u), b = derivative(route::d, c.m, r, p, w.data(), u, &Bdd);
      const double e = derivative(route::d, c.m, r, p, w.data(), u, &Bdd, &memo);
      dmax = std::max(dmax, std::abs(a - b) / std::max(std::abs(a), 1.0));
      dmax_memo = std::max(dmax_memo, std::abs(e - b) / std::max(std::abs(b), 1.0));
    }
    std::printf("  %s: %zu tuples, %zu words in all, %zu distinct\n", name, all.size(), total_words, distinct.size());
    std::printf("    state (with B_j in dd) %9.1f\n", t_state);
    std::printf("    derivatives, production (c)        %10.1f\n", t_prod);
    std::printf("    derivatives, (d) prefix sharing    %10.1f   max rel. difference to (c) %.2g\n", t_d, dmax);
    std::printf("    derivatives, (d) + trace memo      %10.1f   max rel. difference to (d) %.2g\n", t_memo, dmax_memo);
    if (sink == 1.2345) std::printf(" ");
  }
  return 0;
}
