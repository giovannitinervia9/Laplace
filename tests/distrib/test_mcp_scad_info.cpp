// The expected information of mcp_distrib and scad_distrib in closed form
// (include/laplace/distrib/mcp.hpp, scad.hpp, properizing_tail.hpp, and
// detail/properized_nonconvex.hpp for the moments of the inner piece), over the three
// properizing tails: the references of tools/reference/distrib_mcp_scad_info.py, the
// generic route as a second opinion, the components that symmetry makes zero, the two
// branches of the moments of the inner piece where they meet, and a kernel launch.
#include <doctest/doctest.h>

#include <laplace/distrib/detail/properized_nonconvex.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/mcp.hpp>
#include <laplace/distrib/scad.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "generic_hidden.hpp"

namespace ld = laplace::distrib;
using laplace_test::hide;

namespace {

// the law from (mu, lambda, p, t1, t2); brace elision carries the last ones into the
// tail. The make<> of check_common.hpp is not used here, because its specializations
// for these six types belong to the other two test files.
template <class D>
LAPLACE_HD D info_make(const double* t) {
  if constexpr (D::n_params == 4) return D{t[0], t[1], t[2], t[3]};
  else return D{t[0], t[1], t[2], t[3], t[4]};
}

using mcp_g = ld::mcp_distrib<ld::gaussian_tail>;
using mcp_l = ld::mcp_distrib<ld::logistic_tail>;
using mcp_s = ld::mcp_distrib<ld::student_tail>;
using scad_g = ld::scad_distrib<ld::gaussian_tail>;
using scad_l = ld::scad_distrib<ld::logistic_tail>;
using scad_s = ld::scad_distrib<ld::student_tail>;

constexpr double EPS = std::numeric_limits<double>::epsilon();

// the bounds, each the measurement of 2026-11-20 plus a margin; every case prints what
// it measured. Measured: 58.7 ulp of the largest component against the references, at
// scad with the logistic tail, where the assembly's own condition number is 7.4; the
// generic quadrature on the same grid 3.84 ulp, so what the closed form buys is speed and
// not accuracy; 1.3e-14 relative between the two routes; 2.46e-15 across the branch of
// the inner moments; the host against the device 0 ulp on the OpenMP build and
// 1.65 ulp on the CUDA one, where the generic route measures 4.14 ulp instead of 3.84.
constexpr double BOUND_REF = 90.0;
constexpr double BOUND_GENERIC = 5.0e-14;
constexpr double BOUND_DEVICE = 16.0;
constexpr double BOUND_BRANCH = 1.0e-14;

constexpr int INFO_FIELDS = 10;

std::vector<std::string> split_csv(const char* line) {
  std::vector<std::string> out;
  std::string cur;
  for (const char* p = line; *p && *p != '\n' && *p != '\r'; ++p) {
    if (*p == ',') out.push_back(cur), cur.clear();
    else cur.push_back(*p);
  }
  out.push_back(cur);
  return out;
}

struct info_row {
  std::string law, tail;
  int i, j;
  double value, lam, p, t1, t2;
};

std::vector<info_row> read_info_reference() {
  std::vector<info_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/distrib_mcp_scad_info.csv";
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if ((int)s.size() != INFO_FIELDS) continue;
    auto num = [&](int k) { return std::strtod(s[k].c_str(), nullptr); };
    info_row r;
    r.law = s[0], r.tail = s[1];
    r.i = std::atoi(s[2].c_str()), r.j = std::atoi(s[3].c_str());
    r.value = num(4) + num(5);
    r.lam = num(6), r.p = num(7), r.t1 = num(8), r.t2 = num(9);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// the state the checks accumulate, so that one line reports the whole file
struct info_state {
  double worst_ref = 0.0, worst_generic = 0.0, worst_generic_ref = 0.0;
  int checked = 0, settings = 0;
  std::tuple<std::string, std::string, double, double, double, double, int, int> where{};
};

// One setting: the closed matrix against the reference rows that belong to it, the
// components symmetry makes zero, and the generic score-form route as a second opinion.
// A named function, not a generic lambda: MSVC ends nvcc's host pass with an internal
// compiler error on a generic lambda instantiated over several of these laws inside a
// doctest TEST_CASE (2026-11-20), the same cure as the extended-lambda trap of
// 2026-11-13.
template <class D>
void info_one(const double* th, const std::vector<const info_row*>& rows, info_state& st) {
  constexpr int P = D::n_params;
  const D d = info_make<D>(th);
  const auto closed = ld::distrib_expected_info<0>(d);
  const auto generic = ld::expected_info_by_integral<0, ld::info_form::score>(hide(d));
  double scale = 0.0;
  for (const auto* r : rows) scale = std::fmax(scale, std::fabs(r->value));
  REQUIRE(scale > 0.0);
  ++st.settings;
  for (const auto* r : rows) {
    const int idx[2] = {r->i, r->j};
    const double got = closed[ld::sym_tensor_position<P, 2>(idx)];
    const double e = std::fabs(got - r->value) / (EPS * scale);
    if (r->i == 0 && r->j > 0) CHECK(got == 0.0);  // exactly zero, by symmetry
    if (e > st.worst_ref) {
      st.worst_ref = e;
      st.where = {r->law, r->tail, r->lam, r->p, r->t1, r->t2, r->i, r->j};
    }
    ++st.checked;
    INFO("law " << r->law << " tail " << r->tail << " lambda " << r->lam << " p " << r->p
                << " t1 " << r->t1 << " t2 " << r->t2 << " component (" << r->i << ", "
                << r->j << ")");
    CHECK(e <= BOUND_REF);
    // the generic route against the same reference, so that the two can be compared on
    // what they are both trying to compute; it is not bounded here, the generic page
    // carries its own figures
    const double g = generic[ld::sym_tensor_position<P, 2>(idx)];
    if (std::isfinite(g)) {
      const double eg = std::fabs(g - r->value) / (EPS * scale);
      st.worst_generic_ref = std::fmax(st.worst_generic_ref, eg);
    }
  }
  for (int pair = 0; pair < ld::sym_tensor<P, 2>::size; ++pair) {
    const double a = closed[pair], b = generic[pair];
    if (!std::isfinite(b)) continue;
    const double e = std::fabs(a - b) / scale;
    st.worst_generic = std::fmax(st.worst_generic, e);
    INFO("against the generic route, pair " << pair);
    CHECK(e <= BOUND_GENERIC);
  }
}

// the law and the tail are read from the reference, so the type is chosen at run time
void info_setting(const std::string& law, const std::string& tail, const double* th,
                  const std::vector<const info_row*>& rows, info_state& st) {
  if (law == "mcp") {
    if (tail == "gaussian") info_one<mcp_g>(th, rows, st);
    else if (tail == "logistic") info_one<mcp_l>(th, rows, st);
    else info_one<mcp_s>(th, rows, st);
  } else {
    if (tail == "gaussian") info_one<scad_g>(th, rows, st);
    else if (tail == "logistic") info_one<scad_l>(th, rows, st);
    else info_one<scad_s>(th, rows, st);
  }
}

TEST_CASE("mcp and scad: the expected information against the references") {
  const auto rows = read_info_reference();
  REQUIRE(rows.size() > 500);
  info_state st;
  std::size_t i = 0;
  while (i < rows.size()) {
    std::size_t j = i;
    while (j < rows.size() && rows[j].law == rows[i].law && rows[j].tail == rows[i].tail &&
           rows[j].lam == rows[i].lam && rows[j].p == rows[i].p && rows[j].t1 == rows[i].t1 &&
           rows[j].t2 == rows[i].t2)
      ++j;
    std::vector<const info_row*> group;
    for (std::size_t k = i; k < j; ++k) group.push_back(&rows[k]);
    const double th[5] = {0.0, rows[i].lam, rows[i].p, rows[i].t1, rows[i].t2};
    info_setting(rows[i].law, rows[i].tail, th, group, st);
    i = j;
  }
  std::printf("mcp/scad expected information: worst %.3g ulp of the largest component at "
              "%s %s lambda = %g p = %g tail (%g, %g) component (%d, %d); "
              "the generic route against the same reference %.3g ulp; the two routes "
              "against each other %.3g relative; %d components over %d settings\n",
              st.worst_ref, std::get<0>(st.where).c_str(), std::get<1>(st.where).c_str(),
              std::get<2>(st.where), std::get<3>(st.where), std::get<4>(st.where),
              std::get<5>(st.where), std::get<6>(st.where), std::get<7>(st.where),
              st.worst_generic_ref, st.worst_generic, st.checked, st.settings);
}

// The moments of the inner piece switch from the double series in w = p lambda^2 to the
// closed form through Dawson's derivatives at w = 1, so the two must agree where they
// meet. Walked over consecutive doubles of lambda across the crossing: one step of
// lambda moves the true moments by a few eps, so a branch that disagreed with the other
// by more than that would show here.
TEST_CASE("mcp and scad: the two branches of the inner moments agree where they meet") {
  double worst = 0.0;
  int crossings = 0;
  for (double p : {0.3, 1.0, 4.0, 20.0, 300.0}) {
    double lam = 1.0 / std::sqrt(p);
    for (int k = 0; k < 3; ++k) lam = std::nextafter(lam, 0.0);
    double prev[3];
    ld::detail_properized::dawson_moments(lam, p, prev);
    bool seen = p * lam * lam >= 1.0;
    for (int k = 0; k < 6; ++k) {
      lam = std::nextafter(lam, 2.0);
      double cur[3];
      ld::detail_properized::dawson_moments(lam, p, cur);
      const bool now = p * lam * lam >= 1.0;
      if (now != seen) ++crossings;
      seen = now;
      for (int j = 1; j < 3; ++j) {
        const double e = std::fabs(cur[j] - prev[j]) / std::fabs(cur[j]);
        worst = std::fmax(worst, e);
        INFO("p = " << p << ", lambda = " << lam << ", moment " << j);
        CHECK(e <= BOUND_BRANCH);
      }
      for (int j = 0; j < 3; ++j) prev[j] = cur[j];
    }
  }
  CHECK(crossings == 5);  // one per p, or the walk missed the branch
  std::printf("mcp/scad expected information: the moment branches agree to %.3g relative "
              "across w = 1, over %d crossings\n", worst, crossings);
}

// the kernel: one law per launch, the matrix written to a view
template <class D>
void info_kernel_one(const double* th, double* out) {
  constexpr int n = ld::sym_tensor<D::n_params, 2>::size;
  Kokkos::View<double*, Kokkos::LayoutRight> dev("info", n);
  const D d = info_make<D>(th);
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    const auto t = ld::distrib_expected_info<0>(d);
    for (int k = 0; k < n; ++k) dev(k) = t[k];
  });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host, dev);
  for (int k = 0; k < n; ++k) out[k] = host(k);
}

template <class D>
void info_device_one(const double* th, double& worst) {
  constexpr int n = ld::sym_tensor<D::n_params, 2>::size;
  const auto host = ld::distrib_expected_info<0>(info_make<D>(th));
  double dev[n];
  info_kernel_one<D>(th, dev);
  double scale = 0.0;
  for (int k = 0; k < n; ++k) scale = std::fmax(scale, std::fabs(host[k]));
  for (int k = 0; k < n; ++k) {
    const double e = std::fabs(host[k] - dev[k]) / (EPS * scale);
    worst = std::fmax(worst, e);
    INFO("component " << k);
    CHECK(e <= BOUND_DEVICE);
  }
}

TEST_CASE("mcp and scad: the expected information inside a kernel") {
  const double th[5] = {0.3, 1.3, 3.0, 0.9, 3.5};
  double worst = 0.0;
  info_device_one<mcp_s>(th, worst);
  info_device_one<mcp_g>(th, worst);
  info_device_one<mcp_l>(th, worst);
  info_device_one<scad_s>(th, worst);
  info_device_one<scad_g>(th, worst);
  info_device_one<scad_l>(th, worst);
  std::printf("mcp/scad expected information: host against device %.3g ulp\n", worst);
}

}  // namespace
