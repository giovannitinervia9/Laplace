// fixed_distrib: a distribution with some of its parameters held at given
// values and removed from the parameter vector
// (include/laplace/distrib/fixed.hpp, note/piano_wrapper.md section 4).
//
// Holding is a restriction of the domain and not a reparametrization, so every
// derivative of the wrapper is a SELECTION of the parent's components and the
// comparisons here are EXACT: the parent is built with the held value in place,
// and each surviving component must be bit for bit the parent's at the mapped
// multi-index, with the count right and the dropped components absent rather
// than zero. The cases are a univariate law (student_t2 with sigma held, so the
// free set {mu, nu} is not a prefix and a naive copy of the first components
// fails), the re-indexing of param_kinks on enet, the starting value, fixed over
// truncated, over atom_mixture and over fixed, and on the multivariate side
// mvgaussian2 with mu held at zero (the penalty's case) and with one coordinate
// of Q held, where the index tuple is mapped back.
#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/fixed.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <laplace/distrib/truncated.hpp>
#include <laplace/pdmat/diagonal.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "check_common.hpp"
#include <doctest/doctest.h>

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace lp = laplace::pdmat;

// a named namespace, not an anonymous one: the kernels below must sit in
// functions with external linkage (note/brief_agenti.md)
namespace fixed_test {

using st2 = ld::student_t2_distrib;
using gau = ld::gaussian2_distrib;
using zag = ld::atom_mixture_distrib<ld::gamma2_distrib, ld::adjusted>;
using tgau = ld::truncated_distrib<gau>;
using mvg = ld::mvgaussian2_distrib<lp::diagonal_pdmat>;

// student_t2 with sigma held: the free set is {mu, nu}
constexpr int hold_sigma = ld::coord_of<st2>("sigma");
using fst = ld::fixed_distrib<st2, hold_sigma>;
static_assert(fst::n_params == 2);
static_assert(fst::n_held == 1);

// the parent's index of a free index, on the host
std::vector<int> free_list(int P, int n, void (*write)(int*)) {
  std::vector<int> fr(static_cast<std::size_t>(n));
  write(fr.data());
  (void)P;
  return fr;
}

// the components of the wrapper's order-K tensor, each paired with the parent's
// component it must equal: for every sorted K-tuple over the free indices, the
// parent's component at the mapped tuple
template <class W, class P>
void check_selection(const W& w, const P& p, double y, int K, const std::vector<int>& fr, const char* what) {
  const auto wv = lderiv(w, y, K);
  const auto pv = lderiv(p, y, K);
  const auto tuples = sorted_indices(W::n_params, K);
  REQUIRE_MESSAGE(wv.size() == tuples.size(), what);
  for (std::size_t t = 0; t < tuples.size(); ++t) {
    std::vector<int> mapped;
    for (int k : tuples[t]) mapped.push_back(fr[static_cast<std::size_t>(k)]);
    const int pos = position_of(P::n_params, mapped);
    REQUIRE(pos >= 0);
    CHECK_MESSAGE(wv[t] == pv[static_cast<std::size_t>(pos)], what);  // exact: a selection, not a computation
  }
}

// the same for the derivatives of log F or log(1 - F)
template <class W, class P>
void check_cdf_selection(const W& w, const P& p, double y, int K, bool upper, const std::vector<int>& fr,
                         const char* what) {
  const auto wv = cderiv(w, y, K, upper);
  const auto pv = cderiv(p, y, K, upper);
  const auto tuples = sorted_indices(W::n_params, K);
  REQUIRE(wv.size() == tuples.size());
  for (std::size_t t = 0; t < tuples.size(); ++t) {
    std::vector<int> mapped;
    for (int k : tuples[t]) mapped.push_back(fr[static_cast<std::size_t>(k)]);
    CHECK_MESSAGE(wv[t] == pv[static_cast<std::size_t>(position_of(P::n_params, mapped))], what);
  }
}

// the same for the expected information and its derivatives: the pair and the
// differentiating indices are both restricted
template <class W, class P>
void check_info_selection(const W& w, const P& p, int K, const std::vector<int>& fr, const char* what) {
  const auto wv = ideriv(w, K);
  const auto pv = ideriv(p, K);
  const auto pairs = sorted_indices(W::n_params, 2);
  const auto tuples = sorted_indices(W::n_params, K == 0 ? 2 : K);
  const int wn = K == 0 ? 1 : static_cast<int>(tuples.size());
  REQUIRE(wv.size() == (K == 0 ? pairs.size() : pairs.size() * tuples.size()));
  for (std::size_t q = 0; q < pairs.size(); ++q) {
    std::vector<int> mpair{fr[static_cast<std::size_t>(pairs[q][0])], fr[static_cast<std::size_t>(pairs[q][1])]};
    const int pq = position_of(P::n_params, mpair);
    if (K == 0) {
      CHECK_MESSAGE(wv[q] == pv[static_cast<std::size_t>(pq)], what);
      continue;
    }
    for (int t = 0; t < wn; ++t) {
      std::vector<int> mapped;
      for (int k : tuples[static_cast<std::size_t>(t)]) mapped.push_back(fr[static_cast<std::size_t>(k)]);
      const int pt = position_of(P::n_params, mapped);
      const std::size_t wi = q * static_cast<std::size_t>(wn) + static_cast<std::size_t>(t);
      const std::size_t pi = static_cast<std::size_t>(pq) * sorted_indices(P::n_params, K).size() +
                             static_cast<std::size_t>(pt);
      CHECK_MESSAGE(wv[wi] == pv[pi], what);
    }
  }
}

// ---------------------------------------------------------------------------
// the kernels, in named functions with external linkage
// ---------------------------------------------------------------------------

// a wrapper built inside the kernel, over a parent that carries the held value
LAPLACE_HD double univariate_kernel_value(int i) {
  const fst d{.parent = st2{0.4, 1.3, 7.0}, .held = {1.3}};
  const double y = -1.5 + 0.21 * i;
  return d.logpdf(y) + d.logpdf_d1(y)[0] + d.logpdf_d2(y)[2] + d.log_cdf_d1(y)[1] + d.log_ccdf(y) +
         d.expected_info()[0] + d.quantile(0.02 + 0.03 * i) + d.logpdf_dy_dk<1, 1>(y)[1] +
         static_cast<double>(d.param_kinks(0, y).count);
}

void univariate_kernel_case() {
  const int n = 24;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for("distrib_fixed_univariate", n,
                       KOKKOS_LAMBDA(const int i) { out(i) = univariate_kernel_value(i); });
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  for (int i = 0; i < n; ++i) {
    const double want = univariate_kernel_value(i);
    const double got = h(i);
    const double s = std::fmax(std::fabs(want), 1e-300);
    CHECK(std::fabs(got - want) <= 1e-12 * s);
  }
}

// mu of a three-dimensional mvgaussian2 held at zero, the penalty's case: the
// value and the order-two block in theta against the parent at mu = 0
void multivariate_kernel_case() {
  constexpr int d = 3, n = 16;
  const mvg parent{lp::diagonal_pdmat(d)};
  const std::vector<double> zeros(d, 0.0);
  Kokkos::View<double*> zero_dev("zero", d);
  Kokkos::deep_copy(zero_dev, 0.0);
  const double* zero_ptr = zero_dev.data();
  using fmv = ld::fixed_distrib<mvg, ld::coord_of<mvg>("mu")>;
  const fmv w{.parent = parent, .held = {{0.0, zero_ptr}}};

  const int ss = parent.state_size(), sc = w.scratch_size();
  Kokkos::View<double*> work("work", static_cast<std::size_t>(n) * (ss + sc));
  Kokkos::View<double*> theta("theta", d), y("y", static_cast<std::size_t>(n) * d);
  Kokkos::View<double*> vw("vw", n), vp("vp", n), gw("gw", n), gp("gp", n);
  auto th = Kokkos::create_mirror_view(theta);
  auto yh = Kokkos::create_mirror_view(y);
  for (int k = 0; k < d; ++k) th(k) = -0.3 + 0.4 * k;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) yh(static_cast<std::size_t>(i) * d + k) = -1.0 + 0.17 * i + 0.3 * k;
  Kokkos::deep_copy(theta, th);
  Kokkos::deep_copy(y, yh);

  const mvg p2 = parent;
  Kokkos::parallel_for(
      "distrib_fixed_multivariate", n, KOKKOS_LAMBDA(const int i) {
        double* wk = work.data() + static_cast<std::size_t>(i) * (ss + sc);
        const double* yi = y.data() + static_cast<std::size_t>(i) * d;
        const auto stw = w.theta_state(theta.data(), wk);
        vw(i) = w.logpdf(stw, yi, nullptr);
        const int idx[2] = {0, 1};
        double o1 = 0.0;
        w.template logpdf_dk<2>(stw, yi, nullptr, idx, &o1, wk + ss);
        gw(i) = o1;
        const auto stp = p2.theta_state(theta.data(), wk);
        vp(i) = p2.logpdf(stp, yi, zero_dev.data());
        double o2 = 0.0;
        p2.template logpdf_dk<2>(stp, yi, zero_dev.data(), idx, &o2, wk + ss);
        gp(i) = o2;
      });
  auto vwh = Kokkos::create_mirror_view(vw);
  auto vph = Kokkos::create_mirror_view(vp);
  auto gwh = Kokkos::create_mirror_view(gw);
  auto gph = Kokkos::create_mirror_view(gp);
  Kokkos::deep_copy(vwh, vw);
  Kokkos::deep_copy(vph, vp);
  Kokkos::deep_copy(gwh, gw);
  Kokkos::deep_copy(gph, gp);
  for (int i = 0; i < n; ++i) {
    CHECK(vwh(i) == vph(i));  // exact: the same call at the same arguments
    CHECK(gwh(i) == gph(i));
  }
}

}  // namespace fixed_test

using namespace fixed_test;

TEST_CASE("fixed: the descriptors of the free parameters") {
  CHECK(fst::n_params == 2);
  CHECK(std::string(fst::params[0].name) == "mu");
  CHECK(std::string(fst::params[1].name) == "nu");
  CHECK(fst::params[1].lower == st2::params[2].lower);
  CHECK(fst::params[1].upper == st2::params[2].upper);
  CHECK(fst::kind == st2::kind);
  CHECK(fst::y_regularity == st2::y_regularity);
  int fr[2];
  fst::free_slots(fr);
  CHECK(fr[0] == 0);
  CHECK(fr[1] == 2);
  CHECK(fst::parent_slot(1) == 2);
}

TEST_CASE("fixed: every surviving component is bit for bit the parent's") {
  const st2 p{0.4, 1.3, 7.0};
  const auto w = ld::fixed<hold_sigma>(p, 1.3);
  CHECK(w.parent.mu == p.mu);
  CHECK(w.parent.sigma == p.sigma);
  CHECK(w.parent.nu == p.nu);
  const std::vector<int> fr{0, 2};
  for (double y : {-2.3, -0.4, 0.4, 1.9, 5.0}) {
    CHECK(w.logpdf(y) == p.logpdf(y));
    CHECK(w.cdf(y) == ld::distrib_cdf(p, y));
    CHECK(w.log_ccdf(y) == ld::distrib_log_ccdf(p, y));
    for (int K = 1; K <= 6; ++K) check_selection(w, p, y, K, fr, "logpdf_dk");
    for (int K = 1; K <= 6; ++K) {
      check_cdf_selection(w, p, y, K, false, fr, "log_cdf_dk");
      check_cdf_selection(w, p, y, K, true, fr, "log_ccdf_dk");
    }
  }
  CHECK(w.support_lower() == p.support_lower());
  CHECK(w.support_upper() == p.support_upper());
  CHECK(w.center() == p.center());
  CHECK(w.scale() == p.scale());
  CHECK(w.quantile(0.3) == ld::distrib_quantile(p, 0.3));
  for (int K = 0; K <= 4; ++K) check_info_selection(w, p, K, fr, "expected_info");
}

TEST_CASE("fixed: the mixed derivatives in y are selected on the theta part only") {
  const st2 p{0.4, 1.3, 7.0};
  const auto w = ld::fixed<hold_sigma>(p, 1.3);
  const std::vector<int> fr{0, 2};
  for (double y : {-1.1, 0.7, 2.5}) {
    CHECK(w.logpdf_dy_dk<1, 0>(y)[0] == p.logpdf_dy_dk<1, 0>(y)[0]);
    CHECK(w.logpdf_dy_dk<3, 0>(y)[0] == p.logpdf_dy_dk<3, 0>(y)[0]);
    const auto w11 = w.logpdf_dy_dk<1, 1>(y);
    const auto p11 = p.logpdf_dy_dk<1, 1>(y);
    CHECK(w11.size == 2);
    CHECK(w11[0] == p11[0]);
    CHECK(w11[1] == p11[2]);  // the free index 1 is the parent's nu, not its sigma
    const auto w22 = w.logpdf_dy_dk<2, 2>(y);
    const auto p22 = p.logpdf_dy_dk<2, 2>(y);
    CHECK(w22.size == 3);
    const auto tuples = sorted_indices(2, 2);
    for (std::size_t t = 0; t < tuples.size(); ++t) {
      std::vector<int> mapped{fr[static_cast<std::size_t>(tuples[t][0])], fr[static_cast<std::size_t>(tuples[t][1])]};
      CHECK(w22[static_cast<int>(t)] == p22[position_of(3, mapped)]);
    }
  }
}

TEST_CASE("fixed: a free index is mapped back before param_kinks") {
  // enet declares a kink of mu placed at y and none for lambda or alpha. With mu
  // held the wrapper's free index 0 is lambda, so it must report no kink; an
  // identity map would read the parent's mu and report one.
  const ld::enet_distrib p{0.2, 1.4, 0.6};
  const auto w = ld::fixed<ld::coord_of<ld::enet_distrib>("mu")>(p, 0.2);
  CHECK(w.n_params == 2);
  CHECK(std::string(w.params[0].name) == "lambda");
  CHECK(p.param_kinks(0, 1.0).count == 1);
  CHECK(w.param_kinks(0, 1.0).count == 0);
  CHECK(w.param_kinks(1, 1.0).count == 0);
  // and with lambda held the free index 0 is mu, whose kink must come through
  const auto w2 = ld::fixed<ld::coord_of<ld::enet_distrib>("lambda")>(p, 1.4);
  CHECK(std::string(w2.params[0].name) == "mu");
  CHECK(w2.param_kinks(0, 1.0).count == 1);
  CHECK(w2.param_kinks(0, 1.0).point[0] == 1.0);
  CHECK(w2.param_kinks(1, 1.0).count == 0);
  CHECK(w2.y_kinks().count == p.y_kinks().count);
}

TEST_CASE("fixed: a start never moves a held parameter") {
  const auto w = ld::fixed<ld::coord_of<gau>("v")>(gau{0.0, 4.0}, 4.0);
  std::vector<double> y;
  for (int i = 0; i < 40; ++i) y.push_back(1.5 + 0.3 * std::sin(0.7 * i));
  const auto s = ld::distrib_start_value(w, y);
  CHECK(s.parent.v == 4.0);         // exactly the held value, not the sample variance
  CHECK(s.parent.mu != w.parent.mu);  // the free parameter did move
  const gau free_start = ld::distrib_start_value(gau{0.0, 1.0}, y);
  CHECK(s.parent.mu == free_start.mu);
}

TEST_CASE("fixed: the values are refused when they are not inside the domain") {
  CHECK_THROWS_AS(ld::fixed<ld::coord_of<gau>("v")>(gau{0.0, 1.0}, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(ld::fixed<ld::coord_of<st2>("nu")>(st2{0.0, 1.0, 5.0}, 1.0), std::invalid_argument);
  CHECK_NOTHROW(ld::fixed<ld::coord_of<st2>("nu")>(st2{0.0, 1.0, 5.0}, 5.0));
}

TEST_CASE("fixed over truncated") {
  const tgau p = ld::truncated(gau{0.3, 2.0}, -1.0, 2.5);
  const auto w = ld::fixed<ld::coord_of<tgau>("v")>(p, 2.0);
  CHECK(w.n_params == 1);
  CHECK(std::string(w.params[0].name) == "mu");
  const std::vector<int> fr{0};
  for (double y : {-0.5, 0.3, 1.7}) {
    CHECK(w.logpdf(y) == p.logpdf(y));
    CHECK(w.log_cdf(y) == p.log_cdf(y));
    for (int K = 1; K <= 6; ++K) check_selection(w, p, y, K, fr, "fixed over truncated, logpdf_dk");
    for (int K = 1; K <= 6; ++K) check_cdf_selection(w, p, y, K, false, fr, "fixed over truncated, log_cdf_dk");
  }
  CHECK(w.support_lower() == p.support_lower());
  CHECK(w.support_upper() == p.support_upper());
}

TEST_CASE("fixed over atom_mixture") {
  const zag p{.parent = ld::gamma2_distrib{1.4, 0.8}, .atom_alr = {-1.1}, .point = {0.0}};
  // the free parameters are the gamma's mu and the atom's coordinate
  const auto w = ld::fixed<ld::coord_of<zag>("sigma2")>(p, 0.8);
  CHECK(zag::n_params == 3);
  CHECK(w.n_params == 2);
  CHECK(std::string(w.params[1].name) == std::string(zag::params[2].name));
  CHECK(w.n_atoms == zag::n_atoms);
  CHECK(w.atom_point(0) == p.atom_point(0));
  CHECK(w.atom_log_mass(0) == p.atom_log_mass(0));
  const std::vector<int> fr{0, 2};
  for (double y : {0.0, 0.4, 2.2}) {
    CHECK(w.logpdf(y) == p.logpdf(y));
    for (int K = 1; K <= 4; ++K) check_selection(w, p, y, K, fr, "fixed over atom_mixture, logpdf_dk");
  }
}

TEST_CASE("fixed over fixed") {
  const st2 g{0.4, 1.3, 7.0};
  const auto inner = ld::fixed<hold_sigma>(g, 1.3);
  using inner_t = decltype(inner);
  const auto outer = ld::fixed<ld::coord_of<inner_t>("nu")>(inner, 7.0);
  CHECK(outer.n_params == 1);
  CHECK(std::string(outer.params[0].name) == "mu");
  const std::vector<int> fr{0};
  for (double y : {-1.0, 0.5, 3.0}) {
    CHECK(outer.logpdf(y) == g.logpdf(y));
    for (int K = 1; K <= 6; ++K) check_selection(outer, inner, y, K, fr, "fixed over fixed, logpdf_dk");
  }
  // a start moves only mu: both held parameters come back at their values, and
  // write_param walked down the nest to reach the law
  std::vector<double> y;
  for (int i = 0; i < 40; ++i) y.push_back(0.2 + 0.4 * std::sin(0.9 * i));
  const auto s = ld::distrib_start_value(outer, y);
  CHECK(s.parent.parent.sigma == 1.3);
  CHECK(s.parent.parent.nu == 7.0);
  CHECK(s.parent.parent.mu != g.mu);
}

TEST_CASE("fixed: a univariate wrapper inside a kernel") { univariate_kernel_case(); }

TEST_CASE("fixed over mvgaussian2 with mu held at zero") {
  constexpr int d = 3;
  const mvg p{lp::diagonal_pdmat(d)};
  const std::vector<double> zeros(static_cast<std::size_t>(d), 0.0);
  using fmv = ld::fixed_distrib<mvg, ld::coord_of<mvg>("mu")>;
  const auto w = ld::fixed<ld::coord_of<mvg>("mu")>(p, zeros.data());
  CHECK(fmv::n_params == 1);
  CHECK(std::string(fmv::params[0].name) == "Q");
  CHECK(fmv::normalization == mvg::normalization);
  CHECK(w.dim() == d);
  CHECK(w.theta_size() == p.theta_size());
  CHECK(w.dense_size() == 0);
  CHECK(w.param_size(0) == p.param_size(1));
  CHECK(w.param_offset(0) == 0);
  CHECK(w.scratch_size() == p.scratch_size());  // nothing indexed is held, so theta is not completed
  CHECK(w.state_size() == p.state_size());
  CHECK(fmv::dense_rank == 0);

  std::vector<double> theta{-0.3, 0.1, 0.5};
  std::vector<double> work(static_cast<std::size_t>(w.state_size() + w.scratch_size()), 0.0);
  std::vector<double> wp(static_cast<std::size_t>(p.state_size() + p.scratch_size()), 0.0);
  const auto stw = w.theta_state(theta.data(), work.data());
  const auto stp = p.theta_state(theta.data(), wp.data());
  const std::vector<double> y{0.7, -1.2, 0.4};
  CHECK(w.logpdf(stw, y.data(), nullptr) == p.logpdf(stp, y.data(), zeros.data()));
  for (int a = 0; a < d; ++a)
    for (int b = a; b < d; ++b) {
      const int idx[2] = {a, b};
      double ow = 0.0, op = 0.0;
      const bool fw = w.logpdf_dk<2>(stw, y.data(), nullptr, idx, &ow, work.data() + w.state_size());
      const bool fp = p.logpdf_dk<2>(stp, y.data(), zeros.data(), idx, &op, wp.data() + p.state_size());
      CHECK(fw == fp);
      if (fw) CHECK(ow == op);
    }
  // the start writes every coordinate of theta, none being held
  const ld::distrib_mv_sample_summary sum = ld::summarize_mv_sample(y.data(), 1, d);
  std::vector<double> t0(static_cast<std::size_t>(w.theta_size()), 0.0), t1(static_cast<std::size_t>(d), 0.0);
  w.start_value(sum, t0.data());
  p.start_value(sum, t1.data());
  for (int k = 0; k < d; ++k) CHECK(t0[static_cast<std::size_t>(k)] == t1[static_cast<std::size_t>(k)]);
}

TEST_CASE("fixed over mvgaussian2 with one coordinate of Q held") {
  constexpr int d = 3;
  const mvg p{lp::diagonal_pdmat(d)};
  constexpr int hold_q1 = ld::coord_of<mvg>("Q", 1);
  using fmv = ld::fixed_distrib<mvg, hold_q1>;
  const double held = 0.1;
  const auto w = ld::fixed<hold_q1>(p, held);
  CHECK(fmv::n_params == 2);          // both parameters survive, Q with one coordinate fewer
  CHECK(fmv::completes_theta == true);
  CHECK(fmv::dense_rank == -1);
  CHECK(w.param_size(0) == d);        // mu untouched
  CHECK(w.param_size(1) == d - 1);
  CHECK(w.theta_size() == d - 1);
  CHECK(w.dense_size() == d);
  CHECK(w.scratch_size() == p.scratch_size() + p.theta_size());
  CHECK(w.parent_element(1, 0) == 0);
  CHECK(w.parent_element(1, 1) == 2);  // the free coordinate 1 of Q is the parent's 2

  const std::vector<double> mu{0.2, -0.4, 0.1}, y{0.7, -1.2, 0.4};
  const std::vector<double> free_theta{-0.3, 0.5};
  const std::vector<double> full_theta{-0.3, held, 0.5};
  std::vector<double> work(static_cast<std::size_t>(w.state_size() + w.scratch_size()), 0.0);
  std::vector<double> wp(static_cast<std::size_t>(p.state_size() + p.scratch_size()), 0.0);
  const auto stw = w.theta_state(free_theta.data(), work.data());
  const auto stp = p.theta_state(full_theta.data(), wp.data());
  CHECK(w.logpdf(stw, y.data(), mu.data()) == p.logpdf(stp, y.data(), mu.data()));

  // the index tuple is mapped back: the free coordinate 1 is the parent's 2
  const int fidx[2] = {1, 1};
  const int pidx[2] = {2, 2};
  int got[2] = {-1, -1};
  w.map_theta_idx(fidx, 2, got);
  CHECK(got[0] == 2);
  CHECK(got[1] == 2);
  const double before = w.logpdf(stw, y.data(), mu.data());
  double ow = 0.0, op = 0.0;
  const bool fw = w.logpdf_dk<2>(stw, y.data(), mu.data(), fidx, &ow, work.data() + w.state_size());
  const bool fp = p.logpdf_dk<2>(stp, y.data(), mu.data(), pidx, &op, wp.data() + p.state_size());
  CHECK(fw == fp);
  if (fw) CHECK(ow == op);
  // the completed theta survives the parent's use of its own scratch: it is
  // written past it, and the state points at it for as long as the state lives
  CHECK(w.logpdf(stw, y.data(), mu.data()) == before);
  CHECK(w.logpdf(stw, y.data(), mu.data()) == p.logpdf(stp, y.data(), mu.data()));

  // and the start drops the held coordinate
  const ld::distrib_mv_sample_summary sum = ld::summarize_mv_sample(y.data(), 1, d);
  std::vector<double> t0(static_cast<std::size_t>(w.theta_size()), 0.0), t1(static_cast<std::size_t>(d), 0.0);
  w.start_value(sum, t0.data());
  p.start_value(sum, t1.data());
  CHECK(t0[0] == t1[0]);
  CHECK(t0[1] == t1[2]);
}

TEST_CASE("fixed: a multivariate wrapper inside a kernel") { multivariate_kernel_case(); }
