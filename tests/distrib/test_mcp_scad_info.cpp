// The expected information of mcp_distrib and scad_distrib in closed form
// (include/laplace/distrib/mcp.hpp, scad.hpp, properizing_tail.hpp, and
// detail/properized_nonconvex.hpp for the moments of the inner piece), over the three
// properizing tails: the references of tools/reference/distrib_mcp_scad_info.py, the
// generic route as a second opinion, the components that symmetry makes zero, the two
// branches of the moments of the inner piece where they meet, and a kernel launch.
#include <doctest/doctest.h>

#include <laplace/distrib/detail/properized_info.hpp>
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

#include "check_common.hpp"
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
constexpr double BOUND_DEVICE_DK = 24.0;  // measured 6.69 ulp on CUDA, 0 on OpenMP
constexpr double BOUND_BRANCH = 1.0e-14;
constexpr double BOUND_GENERIC_DK = 5.0e-15;  // measured 1.05e-15 over 474 components

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

// the one parameter of a tail replaced, since the three keep their own under their own
// names
ld::gaussian_tail with_param(const ld::gaussian_tail&, int, double v) { return {v}; }
ld::logistic_tail with_param(const ld::logistic_tail&, int, double v) { return {v}; }
ld::student_tail with_param(const ld::student_tail& t, int which, double v) {
  return which == 0 ? ld::student_tail{v, t.tail_df} : ld::student_tail{t.tail_scale, v};
}

// The tails' own tables: their value at order zero against the generic assembly of
// info_xx, info_xt and info_tt, which the references of order zero already check, and
// their higher orders against five-point differences in the tail's parameters.
template <class Tail>
void tail_table_one(const Tail& t, double rel, double& worst0, double& worstd) {
  constexpr int NT = Tail::n_tail_params;
  auto rel_err = [](double a, double b) {
    return std::fabs(a - b) / std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1e-300);
  };
  worst0 = std::fmax(worst0, rel_err(t.info_xx_mixed(0, 0), ld::detail_tail::info_xx(t)));
  for (int j = 0; j < NT; ++j)
    worst0 = std::fmax(worst0, rel_err(t.info_xt_mixed(j, 0, 0), ld::detail_tail::info_xt(t, j)));
  for (int i = 0; i < NT; ++i)
    for (int j = i; j < NT; ++j)
      worst0 = std::fmax(worst0,
                         rel_err(t.info_tt_mixed(i, j, 0, 0), ld::detail_tail::info_tt(t, i, j)));
  CHECK(worst0 <= 5e-14);  // 4.43e-15 on gcc, 1.36e-14 on CUDA
  // the derivatives against differences, one parameter at a time
  for (int which = 0; which < NT; ++which)
    for (int n = 1; n <= 3; ++n) {
      const double x0 = t.param(which), h = 1e-3 * std::fmax(1e-3, x0);
      const double s[4] = {-2.0, -1.0, 1.0, 2.0};
      // the quantity index: 0 is xx, 1 + j is xt(j), 1 + NT + pair is tt
      const int n_q = 1 + NT + NT * (NT + 1) / 2;
      for (int q = 0; q < n_q; ++q) {
        auto value = [&](const Tail& u, int na, int nb) {
          if (q == 0) return u.info_xx_mixed(na, nb);
          if (q < 1 + NT) return u.info_xt_mixed(q - 1, na, nb);
          const int r = q - 1 - NT;
          int i = 0, j = 0, c = 0;
          for (int a = 0; a < NT; ++a)
            for (int b = a; b < NT; ++b, ++c)
              if (c == r) i = a, j = b;
          return u.info_tt_mixed(i, j, na, nb);
        };
        double f[4];
        for (int k = 0; k < 4; ++k) {
          const Tail u = with_param(t, which, x0 + s[k] * h);
          f[k] = value(u, which == 0 ? n - 1 : 0, which == 0 ? 0 : n - 1);
        }
        const double fd = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
        const double exact = value(t, which == 0 ? n : 0, which == 0 ? 0 : n);
        const double e = std::fabs(exact - fd) / std::fmax(std::fabs(exact), 1e-6);
        worstd = std::fmax(worstd, e);
        INFO("tail parameter " << which << ", order " << n << ", quantity " << q);
        CHECK(e <= rel);
      }
    }
}

// The tables over (lambda, p) of the moments of the quadratic piece
// (detail/properized_info.hpp) against five-point differences of their own lower orders,
// which is the twin of check_common.hpp for derivatives: the table holds every multi-index
// of total order at most K, so the component (i, m) is checked against the difference of
// (i - 1, m) in lambda or of (i, m - 1) in p.
template <int K>
void moment_table_one(double lambda, double p, double rel, double& worst) {
  using namespace ld::detail_properized;
  using namespace ld::detail_properized_info;
  double N[3][tab2_size(K)];
  moment_tables<K>(lambda, p, N[0], N[1], N[2]);
  double scale[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
    for (int q = 0; q < tab2_size(K); ++q) scale[k] = std::fmax(scale[k], std::fabs(N[k][q]));
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K && i + m >= 1; ++m) {
      const bool in_lambda = i >= 1;
      const double h = 1e-3 * (in_lambda ? std::fmax(1e-3, lambda) : std::fmax(1e-3, p));
      double f[4][3];  // the lower order at the four stencil points, per moment
      const double s[4] = {-2.0, -1.0, 1.0, 2.0};
      for (int t = 0; t < 4; ++t) {
        double M[3][tab2_size(K)];
        const double lam = in_lambda ? lambda + s[t] * h : lambda;
        const double pp = in_lambda ? p : p + s[t] * h;
        moment_tables<K>(lam, pp, M[0], M[1], M[2]);
        for (int k = 0; k < 3; ++k)
          f[t][k] = M[k][idx2<K>(in_lambda ? i - 1 : i, in_lambda ? m : m - 1)];
      }
      for (int k = 0; k < 3; ++k) {
        const double fd = (f[0][k] - 8.0 * f[1][k] + 8.0 * f[2][k] - f[3][k]) / (12.0 * h);
        const double e = std::fabs(N[k][idx2<K>(i, m)] - fd) / std::fmax(scale[k], 1e-300);
        worst = std::fmax(worst, e);
        INFO("K = " << K << ", moment " << k << ", (" << i << ", " << m << "), lambda = "
                    << lambda << ", p = " << p);
        CHECK(e <= rel);
      }
    }
}

TEST_CASE("the properizing tails: their three integrals and their derivatives") {
  double w0 = 0.0, wd = 0.0;
  tail_table_one(ld::gaussian_tail{0.8}, 2e-6, w0, wd);
  tail_table_one(ld::gaussian_tail{3.5}, 2e-6, w0, wd);
  tail_table_one(ld::logistic_tail{1.3}, 2e-6, w0, wd);
  tail_table_one(ld::logistic_tail{6.0}, 2e-6, w0, wd);
  tail_table_one(ld::student_tail{0.9, 3.5}, 2e-6, w0, wd);
  tail_table_one(ld::student_tail{2.0, 17.0}, 2e-6, w0, wd);
  tail_table_one(ld::student_tail{0.5, 0.7}, 2e-6, w0, wd);
  std::printf("properizing tails: the three integrals at order zero agree to %.3g with the "
              "generic assembly, their derivatives to %.3g with the differences\n", w0, wd);
}

TEST_CASE("mcp and scad: the tables of the inner moments against differences") {
  double worst = 0.0;
  // both branches of dawson_moments: w = p lambda^2 below and above one
  for (double lambda : {0.2, 0.7, 1.3, 3.0})
    for (double p : {0.4, 1.0, 4.0}) {
      moment_table_one<1>(lambda, p, 1e-7, worst);
      moment_table_one<2>(lambda, p, 1e-6, worst);
      moment_table_one<3>(lambda, p, 1e-4, worst);
      moment_table_one<4>(lambda, p, 1e-2, worst);
    }
  std::printf("mcp/scad inner moment tables: worst %.3g relative against the differences\n", worst);
}

// The derivatives of the expected information against five-point differences of the order
// below, the twin check_common.hpp applies to every law that has them. Its own
// check_info_derivs builds the law through the make<> of that header, whose
// specializations for these six types belong to the other two test files, so the loop is
// repeated here over info_make.
template <class D>
void info_derivs_one(const std::vector<double>& th, int kmax, const double* rel, double* worst) {
  constexpr int P = D::n_params;
  const int npair = ld::sym_tensor_size(P, 2);
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = laplace_distrib_check::ideriv(info_make<D>(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    const int n_this = ld::sym_tensor_size(P, K), n_lower = K == 1 ? 1 : ld::sym_tensor_size(P, K - 1);
    for (const auto& idx : laplace_distrib_check::sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = laplace_distrib_check::diff5(
          [&](const double* t) { return laplace_distrib_check::ideriv(info_make<D>(t), K - 1); },
          th, j, 1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      for (int p = 0; p < npair; ++p) {
        const double e = exact[p * n_this + laplace_distrib_check::position_of(P, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + laplace_distrib_check::position_of(P, lower)];
        const double err = std::fabs(e - a) / std::fmax(scale, 1e-300);
        worst[K] = std::fmax(worst[K], err);
        INFO("K = " << K << ", pair " << p << ", last index " << j);
        CHECK(err <= rel[K]);
      }
    }
  }
}

TEST_CASE("mcp and scad: the derivatives of the expected information against differences") {
  const double rel[5] = {0.0, 2e-7, 2e-6, 2e-4, 2e-2};
  double worst[5] = {0, 0, 0, 0, 0};
  info_derivs_one<mcp_s>({0.3, 1.3, 3.0, 0.9, 3.5}, 4, rel, worst);
  info_derivs_one<mcp_g>({0.3, 1.3, 3.0, 0.8}, 4, rel, worst);
  info_derivs_one<mcp_l>({0.3, 0.7, 0.9, 1.3}, 4, rel, worst);
  info_derivs_one<scad_s>({0.3, 1.1, 3.7, 0.9, 3.5}, 4, rel, worst);
  info_derivs_one<scad_g>({0.3, 0.6, 2.0, 0.8}, 4, rel, worst);
  info_derivs_one<scad_l>({0.3, 1.8, 6.0, 1.3}, 4, rel, worst);
  std::printf("mcp/scad expected information derivatives against differences: %.3g, %.3g, %.3g, %.3g "
              "relative at orders one to four\n", worst[1], worst[2], worst[3], worst[4]);
}

// The derivative orders against the generic quadrature, which is a second opinion on the
// values and not only on their consistency: the differences above check each order against
// the one below, and order zero against the 200-bit references, so a value that the whole
// chain shares would pass both. The generic route is asked at orders one and two only,
// since at order four it integrates 1050 components of a five-parameter law at once and
// costs seconds a call. Its own regularity rule returns NaN for every component whose
// multi-index carries the location, which is exactly where the closed form returns the
// exact zeros, so those are skipped here and checked as zeros above.
//
// Two derivatives among lambda and p are skipped as well. This cross-check is how the
// reason was found: the generic route's rule asked for the counts of one parameter at a
// time, so it allowed two derivatives in a parameter of regularity one, while the knots of
// these laws move with mu, with lambda and with p together. The first derivative of the
// integral carries no boundary term, the integrand being continuous at the knot; the
// second one does, the first derivative of the integrand jumping there. Measured on mcp
// with the Student tail at (1.3, 3, 0.9, 3.5): the generic second derivative of
// I_lambda_lambda in gamma was 2.32 times the closed one, while the first derivatives of
// the two routes agree, the five-point differences confirm the closed second, and mpmath
// differentiating the whole 40-digit quadrature agrees with it to 0.0. The rule now sums
// the counts over every parameter that may move the same kink and returns NaN there, so
// the components this filter names are refused by the route itself; the filter stays as the
// record of what it is for.
template <class D, int K>
void info_generic_one(const double* th, double rel, double& worst, int& checked) {
  constexpr int P = D::n_params;
  const D d = info_make<D>(th);
  const auto closed = ld::distrib_expected_info<K>(d);
  const auto generic = ld::expected_info_by_integral<K, ld::info_form::score>(hide(d));
  double scale = 0.0;
  for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
    for (int q = 0; q < ld::sym_tensor_size(P, K); ++q)
      scale = std::fmax(scale, std::fabs(closed.pair[p][q]));
  // the multi-index of each component, to leave out the ones where the generic route's
  // own rule lets it compute a value it cannot have: see the comment above the case
  std::vector<int> moving(ld::sym_tensor_size(P, K), 0);
  {
    int c[4] = {0, 0, 0, 0};
    for (int q = 0; q < ld::sym_tensor_size(P, K); ++q) {
      for (int t = 0; t < K; ++t) moving[q] += (c[t] == 1 || c[t] == 2) ? 1 : 0;
      int i = K - 1;  // the next sorted multi-index
      while (i >= 0 && c[i] == P - 1) --i;
      if (i >= 0) {
        ++c[i];
        for (int j = i + 1; j < K; ++j) c[j] = c[i];
      }
    }
  }
  for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
    for (int q = 0; q < ld::sym_tensor_size(P, K); ++q) {
      const double g = generic.pair[p][q];
      if (!std::isfinite(g) || moving[q] >= 2) continue;
      const double e = std::fabs(closed.pair[p][q] - g) / std::fmax(scale, 1e-300);
      worst = std::fmax(worst, e);
      ++checked;
      INFO("K = " << K << ", pair " << p << ", component " << q);
      CHECK(e <= rel);
    }
}

TEST_CASE("mcp and scad: the derivative orders against the generic quadrature") {
  double worst = 0.0;
  int checked = 0;
  const double th_m[5] = {0.3, 1.3, 3.0, 0.9, 3.5}, th_g[5] = {0.3, 1.3, 3.0, 0.8, 0.0};
  const double th_s[5] = {0.3, 1.1, 3.7, 0.9, 3.5}, th_sl[5] = {0.3, 1.8, 6.0, 1.3, 0.0};
  info_generic_one<mcp_s, 1>(th_m, BOUND_GENERIC_DK, worst, checked);
  info_generic_one<mcp_g, 1>(th_g, BOUND_GENERIC_DK, worst, checked);
  info_generic_one<mcp_s, 2>(th_m, BOUND_GENERIC_DK, worst, checked);
  info_generic_one<scad_s, 1>(th_s, BOUND_GENERIC_DK, worst, checked);
  info_generic_one<scad_l, 1>(th_sl, BOUND_GENERIC_DK, worst, checked);
  info_generic_one<scad_s, 2>(th_s, BOUND_GENERIC_DK, worst, checked);
  std::printf("mcp/scad information derivatives against the generic quadrature: worst %.3g "
              "relative over %d components\n", worst, checked);
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

// the derivative orders inside a kernel, where the tables live on a thread's stack: at
// order four a smooth table of the Student tail holds 70 doubles and the log Z table of
// order six 210, which is what this case is really testing on the device
template <class D, int K>
void info_dk_kernel_one(const double* th, double& worst) {
  constexpr int P = D::n_params, n = ld::sym_tensor_size(P, 2) * ld::sym_tensor_size(P, K);
  Kokkos::View<double*, Kokkos::LayoutRight> dev("info_dk", n);
  const D d = info_make<D>(th);
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    const auto t = ld::distrib_expected_info<K>(d);
    int q = 0;
    for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
      for (int i = 0; i < ld::sym_tensor_size(P, K); ++i) dev(q++) = t.pair[p][i];
  });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host, dev);
  const auto h = ld::distrib_expected_info<K>(d);
  double scale = 0.0;
  for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
    for (int i = 0; i < ld::sym_tensor_size(P, K); ++i)
      scale = std::fmax(scale, std::fabs(h.pair[p][i]));
  int q = 0;
  for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
    for (int i = 0; i < ld::sym_tensor_size(P, K); ++i, ++q) {
      const double e = std::fabs(h.pair[p][i] - host(q)) / (EPS * scale);
      worst = std::fmax(worst, e);
      INFO("K = " << K << ", pair " << p << ", component " << i);
      CHECK(e <= BOUND_DEVICE_DK);
    }
}

TEST_CASE("mcp and scad: the derivatives of the expected information inside a kernel") {
  const double th[5] = {0.3, 1.3, 3.0, 0.9, 3.5};
  const double ths[5] = {0.3, 1.1, 3.7, 0.9, 3.5};
  double worst = 0.0;
  info_dk_kernel_one<mcp_s, 1>(th, worst);
  info_dk_kernel_one<mcp_s, 4>(th, worst);
  info_dk_kernel_one<scad_s, 4>(ths, worst);
  info_dk_kernel_one<mcp_g, 4>(th, worst);
  std::printf("mcp/scad information derivatives: host against device %.3g ulp\n", worst);
}

}  // namespace
