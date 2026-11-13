// The coupled bijectors (alr_bijector, log_cholesky_bijector,
// partial_corr_bijector, matrix_log_bijector) against the mpmath references
// (tools/reference/bijector_coupled.py) and against central differences of
// the derivative one order below, evaluated in a Kokkos kernel on the default
// execution space. For each map, direction, quantity (the map, the
// log-Jacobian, the Cholesky factor) and order 0..6 the error is
//   ulpN  max_e |y_e - ref_e| in ulps of max_e |ref_e| (normwise over the
//         outputs of that derivative; one output for a log-Jacobian), with
//         max(|ref|, 1) at order 0 for a log-Jacobian (it crosses 0). For a
//         mixed tuple the scale of output e is the larger of |ref_e| and the
//         geometric mean prod_v |ref of the pure tuple (v, .., v), e|^(m_v/K),
//         m_v the multiplicity of v in the tuple, as for the mixed partials of
//         laplace::math. A reference below 1e-40 of the largest reference of
//         the same order at the same point is a zero up to the noise of the
//         differences and is measured against that largest one, or, when the
//         whole order is below 1e-40 of the order below it, against the
//         largest reference of the nearest lower order that does not vanish.
// The difference check: (f^(K-1)(x + h e_j) - f^(K-1)(x - h e_j))/(2h) along
// the last variable j of the tuple, h = 1e-6 max(1, |x_j|), times the
// smallest L_ii^2 on the forward side of log_cholesky and partial_corr,
// within 1e-5 of the largest entry of order K plus 1e-5 of that of order
// K - 1 (not for the alr forward, whose references are exact). A derivative
// that a map reports as structurally zero must have an exact zero reference.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/bijector/alr.hpp>
#include <laplace/bijector/log_cholesky.hpp>
#include <laplace/bijector/matrix_log.hpp>
#include <laplace/bijector/partial_corr.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

namespace lb = laplace::bijector;
// a bijector copied from the host into a kernel must have the same layout on
// both sides: its empty bases add nothing (see cholesky_bijector_orders)
static_assert(sizeof(lb::alr_bijector) == sizeof(int));
static_assert(sizeof(lb::log_cholesky_bijector) == sizeof(int));
static_assert(sizeof(lb::partial_corr_bijector) == sizeof(int));
static_assert(sizeof(lb::matrix_log_bijector) == sizeof(int));
// the maps with a Cholesky factor
template <class B>
constexpr bool has_cholesky = std::is_base_of_v<lb::cholesky_bijector_orders<B>, B>;
constexpr int max_out = 16;      // the largest value size: p = 4 or 5 packed
constexpr long scratch_doubles = 160000000;  // scratch per launch: at most 1.28 GB

// one quantity of order K along idx: false if structurally zero (out zeroed)
template <int K, class B>
KOKKOS_INLINE_FUNCTION bool eval_k(const B& b, int dir, int q, const int* t, const double* x, double* out, double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = t[k];
  if (q == 1) {
    out[0] = dir == 0 ? b.template inverse_log_jacobian_dk<K>(x, idx, w) : b.template forward_log_jacobian_dk<K>(x, idx, w);
    return true;
  }
  if (dir == 1) return b.template forward_dk<K>(x, idx, out, w);
  if (q == 0) return b.template inverse_dk<K>(x, idx, out, w);
  if constexpr (has_cholesky<B>) return b.template inverse_cholesky_dk<K>(x, idx, out, w);
  return false;
}

template <class B>
KOKKOS_INLINE_FUNCTION bool eval(const B& b, int dir, int q, int K, const int* t, const double* x, double* out,
                                 double* w) {
  const int n = q == 1 ? 1 : (dir == 0 ? b.value_size() : b.free_size());
  bool nz = true;
  switch (K) {
    case 0:
      if (q == 1) out[0] = dir == 0 ? b.inverse_log_jacobian(x, w) : b.forward_log_jacobian(x, w);
      else if (dir == 1) b.forward(x, out, w);
      else if (q == 0) b.inverse(x, out, w);
      else if constexpr (has_cholesky<B>) b.inverse_cholesky(x, out, w);
      break;
    case 1: nz = eval_k<1>(b, dir, q, t, x, out, w); break;
    case 2: nz = eval_k<2>(b, dir, q, t, x, out, w); break;
    case 3: nz = eval_k<3>(b, dir, q, t, x, out, w); break;
    case 4: nz = eval_k<4>(b, dir, q, t, x, out, w); break;
    case 5: nz = eval_k<5>(b, dir, q, t, x, out, w); break;
    case 6: nz = eval_k<6>(b, dir, q, t, x, out, w); break;
  }
  if (!nz)
    for (int k = 0; k < n; ++k) out[k] = 0.0;
  return nz;
}

KOKKOS_INLINE_FUNCTION bool eval_any(int map, int dim, int dir, int q, int K, const int* t, const double* x,
                                     double* out, double* w) {
  if (map == 0) return eval(lb::alr_bijector(dim), dir, q, K, t, x, out, w);
  if (map == 1) return eval(lb::log_cholesky_bijector(dim), dir, q, K, t, x, out, w);
  if (map == 2) return eval(lb::partial_corr_bijector(dim), dir, q, K, t, x, out, w);
  return eval(lb::matrix_log_bijector(dim), dir, q, K, t, x, out, w);
}

// the storage position of reference coordinate k on the forward side
KOKKOS_INLINE_FUNCTION int value_pos(int map, int k) {
  if (map != 2) return k;
  int a, b;
  lb::detail_coupled::unstrict(k, a, b);
  return lb::detail_coupled::packed(a, b);
}

struct Group {
  int map, dim, point, dir, q, order;
  std::array<int, 6> t;
  std::vector<int> out;
  std::vector<double> hi, lo;
};

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string s;
  while (std::getline(ss, s, ',')) f.push_back(s);
  return f;
}

// The four evaluations of one group: part 0 the derivative of order K at x
// (y, and whether it is structurally zero), part 1 order K - 1 at x (lower),
// parts 2 and 3 order K - 1 at x + h and x - h along the last variable of the
// tuple (fp, fm; step in hs).
struct group_eval {
  Kokkos::View<int* [11]> meta;
  Kokkos::View<double* [max_out]> x, y, lower, fp, fm;
  Kokkos::View<double*> hs;
  Kokkos::View<int*> nonzero;

  KOKKOS_INLINE_FUNCTION void operator()(int g, int part, double* w) const {
    const int mp = meta(g, 0), dim = meta(g, 1), d = meta(g, 2), q = meta(g, 3), K = meta(g, 4);
    if (K == 0 && part > 0) return;
    int t[6];
    for (int k = 0; k < 6; ++k) t[k] = meta(g, 5 + k);
    double xx[max_out], a[max_out];
    for (int k = 0; k < max_out; ++k) xx[k] = x(g, k), a[k] = 0.0;
    if (part == 0) {
      nonzero(g) = eval_any(mp, dim, d, q, K, t, xx, a, w) ? 1 : 0;
      for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
      return;
    }
    if (part == 1) {
      eval_any(mp, dim, d, q, K - 1, t, xx, a, w);
      for (int k = 0; k < max_out; ++k) lower(g, k) = a[k];
      return;
    }
    // shift along the last variable of the tuple; alr forward moves pi_K the other way
    const int j = d == 1 ? value_pos(mp, t[K - 1]) : t[K - 1];
    const bool last = mp == 0 && d == 1;
    const double xj = xx[j], ax = xj < 0 ? -xj : xj;
    double h = 1e-6 * (ax < 1.0 ? 1.0 : ax);
    // the matrix maps forward: a step below the smallest L_ii^2, so that the
    // matrix stays positive definite and the truncation stays small
    if (d == 1 && (mp == 1 || mp == 2)) {
      eval_any(mp, dim, 1, 0, 0, t, xx, a, w);
      double lam = 1.0;
      for (int i = 0; i < dim; ++i) {
        double l2;
        if (mp == 1) l2 = Kokkos::exp(2.0 * a[lb::detail_coupled::packed(i, i)]);
        else {
          l2 = 1.0;
          for (int c = 0; c < i; ++c) {
            const double ch = Kokkos::cosh(a[lb::detail_coupled::strict(i, c)]);
            l2 /= ch * ch;
          }
        }
        lam = l2 < lam ? l2 : lam;
      }
      h *= lam;
    }
    const double s = part == 2 ? 1.0 : -1.0;
    xx[j] = xj + s * h;
    if (last) xx[dim - 1] -= s * h;
    for (int k = 0; k < max_out; ++k) a[k] = 0.0;
    eval_any(mp, dim, d, q, K - 1, t, xx, a, w);
    const auto& out = part == 2 ? fp : fm;
    for (int k = 0; k < max_out; ++k) out(g, k) = a[k];
    if (part == 2) hs(g) = h;
  }
};

const std::string mname[] = {"alr", "log_cholesky", "partial_corr", "matrix_log"};
const std::string qname[] = {"map", "log_jacobian", "cholesky"};
const std::string dname[] = {"inverse", "forward"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE: doctest's case function has internal linkage, and the extended device lambda
// below takes the address of its enclosing function as a tag, which collides between
// translation units (note/brief_agenti.md, found on 2026-11-13).
void coupled_mpmath_case() {
  const std::string dir = LAPLACE_REFERENCE_DIR;
  // points: (map, dim, point, side) -> values
  std::map<std::array<int, 4>, std::vector<double>> pts;
  {
    std::ifstream in(dir + "/bijector_coupled_points.csv");
    REQUIRE(in.good());
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = split(line);
      auto& v = pts[{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]), std::stoi(f[3])}];
      const int k = std::stoi(f[4]);
      if (int(v.size()) <= k) v.resize(k + 1);
      v[k] = std::strtod(f[5].c_str(), nullptr);
    }
  }
  std::vector<Group> groups;
  {
    std::ifstream in(dir + "/bijector_coupled.csv");
    REQUIRE(in.good());
    std::string line;
    std::getline(in, line);
    std::map<std::vector<int>, int> where;
    while (std::getline(in, line)) {
      const auto f = split(line);
      std::vector<int> key;
      for (int k = 0; k < 12; ++k) key.push_back(std::stoi(f[k]));
      auto it = where.find(key);
      if (it == where.end()) {
        it = where.emplace(key, int(groups.size())).first;
        Group g{key[0], key[1], key[2], key[3], key[4], key[5], {}, {}, {}, {}};
        for (int k = 0; k < 6; ++k) g.t[k] = key[6 + k];
        groups.push_back(g);
      }
      Group& g = groups[it->second];
      g.out.push_back(std::stoi(f[12]));
      g.hi.push_back(std::strtod(f[13].c_str(), nullptr));
      g.lo.push_back(std::strtod(f[14].c_str(), nullptr));
    }
  }
  const int ng = int(groups.size());
  REQUIRE(ng > 0);

  // per group: meta (map, dim, dir, q, order, t0..t5) and the input point
  Kokkos::View<int* [11]> meta("meta", ng);
  Kokkos::View<double* [max_out]> x("x", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto hx = Kokkos::create_mirror_view(x);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int m[5] = {G.map, G.dim, G.dir, G.q, G.order};
    for (int k = 0; k < 5; ++k) hm(g, k) = m[k];
    for (int k = 0; k < 6; ++k) hm(g, 5 + k) = G.t[k];
    const auto& p = pts.at({G.map, G.dim, G.point, G.dir});
    for (int k = 0; k < max_out; ++k) hx(g, k) = k < int(p.size()) ? p[k] : 0.0;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(x, hx);

  // the four evaluations of each group, in one thread (group_eval)
  const group_eval ev{meta, x, Kokkos::View<double* [max_out]>("y", ng), Kokkos::View<double* [max_out]>("lower", ng),
                      Kokkos::View<double* [max_out]>("fp", ng), Kokkos::View<double* [max_out]>("fm", ng),
                      Kokkos::View<double*>("hs", ng), Kokkos::View<int*>("nonzero", ng)};
  // The groups come in runs of one map; each run gets the scratch of its largest
  // dimension (matrix_log p = 4: 23412 doubles per thread, the other maps at most
  // 896) and as many threads per launch as scratch_doubles allows, at most 64 per
  // unit of concurrency (a few launches on the CPU). On CUDA all 6520 matrix_log
  // groups run in one launch: the time is that of the slowest thread, the order-6
  // tuples of p = 4. With 512 groups per launch, as before 2026-11-13, the test
  // took 44 s on CUDA (RTX 4070 SUPER); now 8 s.
  const int max_chunk = 64 * Kokkos::DefaultExecutionSpace().concurrency();
  const bool print_time = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  int r0 = 0;
  while (r0 < ng) {
    const int mp0 = groups[r0].map;
    int r1 = r0, ws = 0;
    for (; r1 < ng && groups[r1].map == mp0; ++r1) {
      const int dm = groups[r1].dim;
      ws = std::max(ws, mp0 == 0   ? lb::alr_bijector(dm).scratch_size()
                        : mp0 == 1 ? lb::log_cholesky_bijector(dm).scratch_size()
                        : mp0 == 2 ? lb::partial_corr_bijector(dm).scratch_size()
                                   : lb::matrix_log_bijector(dm).scratch_size());
    }
    const int chunk = std::max(1, std::min({r1 - r0, max_chunk, int(scratch_doubles / ws)}));
    // LayoutRight: each kernel thread takes a contiguous row as its scratch buffer
    Kokkos::View<double**, Kokkos::LayoutRight> scratch("scratch", chunk, ws);
    Kokkos::Timer timer;
    for (int g0 = r0; g0 < r1; g0 += chunk) {
      const int g1 = std::min(r1, g0 + chunk);
      Kokkos::parallel_for(
          "bijector_coupled", Kokkos::RangePolicy<>(g0, g1), KOKKOS_LAMBDA(int g) {
            for (int part = 0; part < 4; ++part) ev(g, part, &scratch(g - g0, 0));
          });
      Kokkos::fence();
    }
    if (print_time)
      std::printf("%-13s %6d groups, chunk %6d, scratch %5d: %.2f s\n", mname[mp0].c_str(), r1 - r0, chunk, ws,
                  timer.seconds());
    r0 = r1;
  }
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.y);
  auto hlo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.lower);
  auto hfp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.fp);
  auto hfm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.fm);
  auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.hs);
  auto hnz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ev.nonzero);
  // the central difference of order K - 1
  auto hfd = [&](int g, int k) { return (hfp(g, k) - hfm(g, k)) / (2.0 * hh(g)); };

  // the pure derivatives of each order, per variable and output, for the scale of the mixed ones
  std::map<std::vector<int>, std::vector<double>> pure;
  for (const Group& G : groups)
    if (G.order > 0 && G.t[0] == G.t[G.order - 1]) {
      auto& v = pure[{G.map, G.dim, G.point, G.dir, G.q, G.order, G.t[0]}];
      v.assign(max_out, 0.0);
      for (std::size_t r = 0; r < G.out.size(); ++r) v[G.out[r]] = std::fabs(G.hi[r]);
    }
  // the largest reference of each order at each point: a derivative that is zero
  // at that point (odd orders at S = 0 for matrix_log), whose difference reference
  // is noise below 1e-40 of it, is measured against it
  std::map<std::vector<int>, double> order_scale;
  for (const Group& G : groups) {
    double& s = order_scale[{G.map, G.dim, G.point, G.dir, G.q, G.order}];
    for (double h : G.hi) s = std::fmax(s, std::fabs(h));
  }

  double worst[4][2][3][7] = {};
  std::map<std::array<int, 4>, double> per_point;  // forward matrix maps: (map, dim, point, order)
  std::string where[4][2][3][7];
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    double scale = 0.0, err = 0.0;
    for (std::size_t r = 0; r < G.out.size(); ++r) scale = std::fmax(scale, std::fabs(G.hi[r]));
    if (!hnz(g) && scale != 0.0)
      FAIL_CHECK(mname[G.map] << " " << dname[G.dir] << " " << qname[G.q] << " order " << G.order << " dim "
                              << G.dim << " point " << G.point << ": reported zero, reference " << scale);
    if (scale == 0.0) continue;  // a structural zero, checked above
    // zero at this point up to the noise of the differences (below 1e-40 of the
    // largest reference of its order there, or a whole order below 1e-40 of the
    // order below it: odd orders of the matrix_log log-Jacobian at S = 0):
    // measured against the largest reference of the nearest order that does not vanish
    {
      int k = G.order;
      auto os = [&](int o) { return order_scale.at({G.map, G.dim, G.point, G.dir, G.q, o}); };
      while (k > 0 && os(k) <= 1e-40 * os(k - 1)) --k;
      if (k < G.order || scale <= 1e-40 * os(k)) scale = std::fmax(scale, os(k));
    }
    scale = std::fmax(scale, G.q == 1 && G.order == 0 ? 1.0 : 1e-300);
    // a mixed tuple: each output against the geometric mean of its pure derivatives,
    // prod_v |f_(v..v),e|^(m_v/K), m_v the multiplicity of variable v (the manual's ulpN of mixed partials)
    if (G.order > 0 && G.t[0] != G.t[G.order - 1])
      for (std::size_t r = 0; r < G.out.size(); ++r) {
        double nm = 1.0;
        for (int k = 0; k < G.order;) {
          int m = 1;
          while (k + m < G.order && G.t[k + m] == G.t[k]) ++m;
          const auto& pv = pure.at({G.map, G.dim, G.point, G.dir, G.q, G.order, G.t[k]});
          nm *= std::pow(pv[G.out[r]], double(m) / G.order);
          k += m;
        }
        scale = std::fmax(scale, nm);
      }
    for (std::size_t r = 0; r < G.out.size(); ++r) {
      const double e = std::fabs((hy(g, G.out[r]) - G.hi[r]) - G.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double u = err / (scale * eps);
    if (G.dir == 1 && G.map != 0 && G.q == 0) {
      double& pw = per_point[{G.map, G.dim, G.point, G.order}];
      pw = std::fmax(pw, u);
    }
    double& w = worst[G.map][G.dir][G.q][G.order];
    if (!(u <= w)) {
      w = u;
      std::ostringstream s;
      s << "dim " << G.dim << " point " << G.point << " tuple";
      for (int k = 0; k < G.order; ++k) s << " " << G.t[k];
      where[G.map][G.dir][G.q][G.order] = s.str();
    }
    // alr forward: no difference check, since a step below pi_j vanishes in pi_K when
    // pi_j << pi_K (the spread points); its references are exact SymPy derivatives
    if (G.order > 0 && !(G.map == 0 && G.dir == 1)) {
      double lscale = 0.0;
      const int n = int(G.out.size());
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double d = std::fabs(hfd(g, k) - hy(g, k)) / (1e-5 * (scale + lscale));
        worst_fd = std::fmax(worst_fd, d);
        if (!(d <= 1.0))
          FAIL_CHECK(mname[G.map] << " " << dname[G.dir] << " " << qname[G.q] << " order " << G.order << " dim "
                                  << G.dim << " point " << G.point << ": difference check " << d);
      }
    }
  }

  // bounds on ulpN / ulp, [map][direction][quantity][order]: the larger of the
  // errors measured on 2026-11-12 on gcc/OpenMP and CUDA, times 1.5, rounded up.
  // The large ones: tanh^(k) near a root of its polynomial (partial_corr inverse,
  // orders 5-6) and the conditioning of Sigma and R on the forward side, which
  // the forward derivatives of order K multiply by K, and the log-determinant
  // behind the matrix_log log-Jacobians at orders 3-6, whose pivots' logarithms
  // cancel to values 1/70 of the largest of their order (docs/numerics.md).
  // matrix_log measured on 2026-11-13
  const double bound[4][2][3][7] = {
      {{{12, 41, 107, 58, 136, 75, 92}, {2, 141, 41, 107, 58, 136, 75}, {0, 0, 0, 0, 0, 0, 0}},
       {{2, 2, 3, 3, 5, 5, 6}, {1, 3, 2, 5, 4, 6, 5}, {0, 0, 0, 0, 0, 0, 0}}},
      {{{2, 2, 2, 2, 2, 2, 2}, {2, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}},
       {{62, 498, 992, 1488, 1985, 2481, 2976}, {46, 498, 992, 1488, 1985, 2482, 2976}, {0, 0, 0, 0, 0, 0, 0}}},
      {{{3, 4, 6, 90, 5, 4258, 35}, {2, 2, 3, 4, 90, 2939, 4258}, {1, 4, 4, 5, 6, 8, 10}},
       {{366, 1826, 3649, 5475, 7302, 9126, 10959}, {127, 1838, 3651, 5470, 7300, 9150, 11041}, {0, 0, 0, 0, 0, 0, 0}}},
      {{{18, 17, 17, 19, 17, 18, 19}, {12, 87, 63, 3096, 5540, 112236, 437749}, {0, 0, 0, 0, 0, 0, 0}},
       {{15, 12, 54, 93, 156, 217, 316}, {20, 836, 43, 4939, 161, 3753, 350}, {0, 0, 0, 0, 0, 0, 0}}}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int mp = 0; mp < 4; ++mp)
    for (int d = 0; d < 2; ++d)
      for (int q = 0; q < 3; ++q)
        for (int K = 0; K <= 6; ++K) {
          if (bound[mp][d][q][K] == 0) continue;  // no such quantity
          if (print)
            std::printf("%-13s %-8s %-12s K=%d  %10.2f  (%s)\n", mname[mp].c_str(), dname[d].c_str(), qname[q].c_str(), K, worst[mp][d][q][K],
                        where[mp][d][q][K].c_str());
          CHECK_MESSAGE(worst[mp][d][q][K] <= bound[mp][d][q][K], mname[mp] << " " << dname[d] << " " << qname[q]
                                                                          << " order " << K << ": "
                                                                          << worst[mp][d][q][K] << " ulp at "
                                                                          << where[mp][d][q][K]);
        }
  if (print) {
    std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
    for (const auto& [k, v] : per_point)
      std::printf("forward %-13s dim %d point %d order %d: %.2f\n", mname[k[0]].c_str(), k[1], k[2], k[3], v);
  }
}

TEST_CASE("coupled bijectors: alr, log_cholesky, partial_corr against mpmath and differences") { coupled_mpmath_case(); }

// the nonzero tuples at order 6 against the SymPy count of 2026-11-12
// (log_cholesky p = 4: 10 of 5005; partial_corr p = 4: 251 of 462; alr dense)
TEST_CASE("coupled bijectors: structural zeros at order 6") {
  auto count = [](auto b, int n) {
    std::vector<double> w(b.scratch_size()), out(b.value_size()), u(n);
    for (int k = 0; k < n; ++k) u[k] = 0.1 + 0.05 * k;
    int t[6] = {0, 0, 0, 0, 0, 0}, nz = 0;
    for (;;) {
      if (b.inverse_d6(u.data(), t, out.data(), w.data())) ++nz;
      int k = 5;
      while (k >= 0 && t[k] == n - 1) --k;
      if (k < 0) break;
      ++t[k];
      for (int r = k + 1; r < 6; ++r) t[r] = t[k];
    }
    return nz;
  };
  CHECK(count(lb::log_cholesky_bijector(4), 10) == 10);
  CHECK(count(lb::partial_corr_bijector(4), 6) == 251);
  CHECK(count(lb::alr_bijector(4), 3) == 28);
}
