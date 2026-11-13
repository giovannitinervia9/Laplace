// The host part of the shared check of the scalar bijectors with parameters
// (param_common.hpp): reading the references, the comparison, the bounds. The
// kernels are in param_order.cpp, one translation unit per order, compiled
// once for all the test files instead of once in each of them.
#include "param_common.hpp"

namespace bijector_param_test {

struct Row {
  double a[7], hi, lo;
};

inline std::vector<Row> read(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream in(path);
  REQUIRE_MESSAGE(static_cast<bool>(in), "cannot open ", path);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Row r;
    char* p = line.data();
    char* end = nullptr;
    for (double& v : r.a) v = std::strtod(p, &end), p = end + 1;
    r.hi = std::strtod(p, &end);
    r.lo = std::strtod(end + 1, &end);
    rows.push_back(r);
  }
  return rows;
}

inline const char* which_name(int w) {
  static const char* n[] = {"forward", "inverse", "fwd log-jac", "inv log-jac", "to upper", "to lower"};
  return n[w];
}

// bound[w][k]: the largest ulpC accepted for quantity w at total order k (< 0: no rows expected)
void check_map(const char* name, map_t m, const double (&bound)[6][7]) {
  const auto rows = read(std::string(LAPLACE_REFERENCE_DIR) + "/bijector_" + name + ".csv");
  REQUIRE(!rows.empty());
  using Key = std::tuple<int, int, int, int, double, double, double>;
  std::map<Key, double> ref;
  for (const auto& r : rows)
    ref[Key{int(r.a[3]), int(r.a[4]), int(r.a[5]), int(r.a[6]), r.a[0], r.a[1], r.a[2]}] = r.hi;
  std::vector<const Row*> tested;
  for (const auto& r : rows)
    if (r.a[4] + r.a[5] + r.a[6] <= 6) tested.push_back(&r);
  const std::size_t n = tested.size();
  Rows dv("rows", n);
  auto hv = Kokkos::create_mirror_view(dv);
  for (std::size_t i = 0; i < n; ++i)
    for (int j = 0; j < 7; ++j) hv(i, j) = tested[i]->a[j];
  Kokkos::deep_copy(dv, hv);
  Out dy("y", n);
  eval_order<0>(m, dv, dy);
  eval_order<1>(m, dv, dy);
  eval_order<2>(m, dv, dy);
  eval_order<3>(m, dv, dy);
  eval_order<4>(m, dv, dy);
  eval_order<5>(m, dv, dy);
  eval_order<6>(m, dv, dy);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);

  struct Acc {
    double ulpc = 0, ulp = 0, x = 0, p1 = 0, p2 = 0, c0 = 0;
    std::size_t rows = 0, bad = 0;
    int n1 = 0, n2 = 0;
  };
  std::map<int, Acc> acc;  // key w * 8 + k
  auto get = [&](int w, int n0, int n1, int n2, const Row& r) {
    const auto it = ref.find(Key{w, n0, n1, n2, r.a[0], r.a[1], r.a[2]});
    return it == ref.end() ? 0.0 : it->second;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const Row& r = *tested[i];
    const int w = int(r.a[3]), n0 = int(r.a[4]), n1 = int(r.a[5]), n2 = int(r.a[6]), k = n0 + n1 + n2;
    auto& e = acc[w * 8 + k];
    ++e.rows;
    const double y = hy(i, 0);
    if (!std::isfinite(y)) {
      ++e.bad;
      e.x = r.a[0], e.p1 = r.a[1], e.p2 = r.a[2];
      continue;
    }
    const int wb = w >= 4 ? 1 : w;  // the distances scale with the derivatives of h
    const int b0 = w >= 4 ? 0 : n0;
    const double scale = std::fabs(r.hi) + std::fabs(r.a[0] * get(wb, b0 + 1, n1, n2, r)) +
                         std::fabs(r.a[1] * get(wb, b0, n1 + 1, n2, r)) +
                         std::fabs(r.a[2] * get(wb, b0, n1, n2 + 1, r));
    const double d = std::fabs((y - r.hi) - r.lo);
    const double uc = d / bench::ulp_of(std::max(scale, 2.2250738585072014e-308));
    const double u = d / bench::ulp_of(std::max(std::fabs(r.hi), 2.2250738585072014e-308));
    if (uc > e.ulpc) e.ulpc = uc, e.x = r.a[0], e.p1 = r.a[1], e.p2 = r.a[2], e.n1 = n1, e.n2 = n2;
    e.ulp = std::max(e.ulp, u);
    if (w < 4 && k >= 1 && n1 == 0 && n2 == 0) {
      const double y1 = hy(i, 1);
      const double u1 = std::isfinite(y1) ? std::fabs((y1 - r.hi) - r.lo) / bench::ulp_of(std::max(scale, 2.2250738585072014e-308))
                                          : 1e300;
      e.c0 = std::max(e.c0, u1);
    }
  }
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (const auto& [key, e] : acc) {
    const int w = key / 8, k = key % 8;
    if (print)
      std::printf("%-14s %-11s k=%d  rows %5zu  max %9.3g ulpC (%9.3g ulp)  c[0] %8.3g  at x=%.6g p=(%.6g, %.6g) n1=%d n2=%d%s\n",
                  name, which_name(w), k, e.rows, e.ulpc, e.ulp, e.c0, e.x, e.p1, e.p2, e.n1, e.n2,
                  e.bad ? "  NOT FINITE" : "");
    INFO(name, " ", which_name(w), " order ", k, ": max ", e.ulpc, " ulpC at x = ", e.x, ", p = (", e.p1, ", ",
         e.p2, "), n1 = ", e.n1, ", n2 = ", e.n2, "; c[0] ", e.c0, "; ", e.bad, " not finite");
    CHECK(e.bad == 0);
    CHECK(bound[w][k] >= 0.0);
    CHECK(e.ulpc <= bound[w][k]);
    CHECK(e.c0 <= bound[w][k]);
  }
}

}  // namespace bijector_param_test
