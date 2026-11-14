// chain_rule_pair_dk (include/laplace/math/chain_pair.hpp): the two-into-two chain
// rule to order 6, against the truncated bivariate Taylor composition of
// tools/reference/chain_pair.py at 300 bits (random derivatives with cancelling
// signs, and the maps of gamma2 and lognormal2), plus the packing of the output and
// a kernel launch of every order.
#include <doctest/doctest.h>

#include <laplace/math/chain_pair.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace cp = laplace::math::detail_chain_pair;

constexpr int tsize(int s) { return ((s + 1) * (s + 2)) / 2; }

struct chain_case {
  int id, K;  // id below 100: random derivatives; 100 and above: the maps of gamma2 and lognormal2
  std::vector<double> G, u0, u1, hi, lo;
};

// one row: case, K, G[tsize(K)], u0[1..], u1[1..], then (hi, lo) for b = 0..K
std::vector<chain_case> read_cases() {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/chain_pair.csv", "r");
  REQUIRE(f != nullptr);
  char line[4096];
  std::fgets(line, sizeof line, f);
  std::vector<chain_case> out;
  while (std::fgets(line, sizeof line, f)) {
    std::vector<double> v;
    for (char* tok = std::strtok(line, ",\n"); tok; tok = std::strtok(nullptr, ",\n"))
      v.push_back(std::strtod(tok, nullptr));
    if (v.size() < 4) continue;
    chain_case c;
    c.id = (int)v[0];
    c.K = (int)v[1];
    const int T = tsize(c.K);
    size_t p = 2;
    c.G.assign(v.begin() + p, v.begin() + p + T);
    p += T;
    c.u0.push_back(0.0);
    c.u0.insert(c.u0.end(), v.begin() + p, v.begin() + p + T - 1);
    p += T - 1;
    c.u1.push_back(0.0);
    c.u1.insert(c.u1.end(), v.begin() + p, v.begin() + p + T - 1);
    p += T - 1;
    for (int b = 0; b <= c.K; ++b, p += 2) c.hi.push_back(v[p]), c.lo.push_back(v[p + 1]);
    out.push_back(std::move(c));
  }
  std::fclose(f);
  return out;
}

// the worst error of one order, and the case it came from, per group
struct group_worst {
  double ulp[2] = {0.0, 0.0};
  int id[2] = {-1, -1};
};

template <int K>
void run_case(const chain_case& c, group_worst& w) {
  const int grp = c.id >= 100 ? 1 : 0;
  double got[K + 1];
  laplace::math::chain_rule_pair_dk<K>(c.G.data(), c.u0.data(), c.u1.data(), got);
  double scale = 0.0;
  for (int b = 0; b <= K; ++b) scale = std::fmax(scale, std::fabs(c.hi[b]));
  for (int b = 0; b <= K; ++b) {
    const double err = std::fabs((got[b] - c.hi[b]) - c.lo[b]);
    const double ulp = err / std::ldexp(1.0, std::ilogb(scale) - 52);
    INFO("K = " << K << ", b = " << b << ", ulp " << ulp);
    // bounds: the measured maxima of 2026-11-13 (gcc) times about 1.5, by group;
    // the random cases are built to cancel, the maps are what the distributions use
    CHECK(ulp <= (grp == 0 ? 40.0 : 40.0));
    if (ulp > w.ulp[grp]) w.ulp[grp] = ulp, w.id[grp] = c.id;
  }
}

}  // namespace

TEST_CASE("math chain_pair: packing and the sizes of the state") {
  CHECK(cp::pair_index(0, 0) == 0);
  CHECK(cp::pair_index(1, 0) == 1);
  CHECK(cp::pair_index(0, 1) == 2);
  CHECK(cp::pair_index(2, 0) == 3);
  CHECK(cp::pair_index(1, 1) == 4);
  CHECK(cp::pair_index(0, 2) == 5);
  CHECK(cp::pair_count(6) == 28);
  CHECK(cp::state_size(1) == 3 + 2 * 1);  // order 0: 3 entries; order 1: two multi-indices, 1 each
  CHECK(cp::state_size(6) == 210);
}

TEST_CASE("math chain_pair: the first orders against the chain rule written out") {
  // G(u0, u1), u0 and u1 of (x0, x1); only the first derivatives are non-zero
  double G[tsize(2)] = {}, u0[tsize(2)] = {}, u1[tsize(2)] = {};
  G[cp::pair_index(1, 0)] = 2.0;
  G[cp::pair_index(0, 1)] = -3.0;
  G[cp::pair_index(2, 0)] = 5.0;
  G[cp::pair_index(1, 1)] = 7.0;
  G[cp::pair_index(0, 2)] = -11.0;
  u0[cp::pair_index(1, 0)] = 0.5;
  u0[cp::pair_index(0, 1)] = 1.5;
  u1[cp::pair_index(1, 0)] = -0.25;
  u1[cp::pair_index(0, 1)] = 2.0;
  double d1[2], d2[3];
  laplace::math::chain_rule_pair_dk<1>(G, u0, u1, d1);
  CHECK(d1[0] == doctest::Approx(2.0 * 0.5 + (-3.0) * (-0.25)));
  CHECK(d1[1] == doctest::Approx(2.0 * 1.5 + (-3.0) * 2.0));
  laplace::math::chain_rule_pair_dk<2>(G, u0, u1, d2);
  // d^2/dx0^2 = G_00' a^2 + 2 G_01' a b + G_11' b^2 with a = du0/dx0, b = du1/dx0
  CHECK(d2[0] == doctest::Approx(5.0 * 0.25 + 2.0 * 7.0 * 0.5 * (-0.25) + (-11.0) * 0.0625));
  CHECK(d2[1] == doctest::Approx(5.0 * 0.5 * 1.5 + 7.0 * (0.5 * 2.0 + 1.5 * (-0.25)) + (-11.0) * (-0.25) * 2.0));
  CHECK(d2[2] == doctest::Approx(5.0 * 2.25 + 2.0 * 7.0 * 1.5 * 2.0 + (-11.0) * 4.0));
}

TEST_CASE("math chain_pair: against the bivariate Taylor composition of mpmath") {
  const auto cases = read_cases();
  CHECK(cases.size() > 50);
  // one entry per order; group 0 the random derivatives, group 1 the two maps
  group_worst worst[7];
  for (const auto& c : cases) {
    switch (c.K) {
      case 1: run_case<1>(c, worst[1]); break;
      case 2: run_case<2>(c, worst[2]); break;
      case 3: run_case<3>(c, worst[3]); break;
      case 4: run_case<4>(c, worst[4]); break;
      case 5: run_case<5>(c, worst[5]); break;
      default: run_case<6>(c, worst[6]); break;
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int K = 1; K <= 6; ++K)
      std::printf("chain_pair K %d: max %.3g ulp (random derivatives, case %d), %.3g ulp (the maps, case %d)\n", K,
                  worst[K].ulp[0], worst[K].id[0], worst[K].ulp[1], worst[K].id[1]);
}

// The kernel is a named function with external linkage, never the body of a TEST_CASE
// (note/brief_agenti.md: nvcc tags an extended device lambda with its enclosing
// function, whose doctest counter restarts in every translation unit).
void chain_pair_kernel_case() {
  const int n = 32;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "math_chain_pair", n, KOKKOS_LAMBDA(const int i) {
        double G[28], u0[28], u1[28];
        for (int k = 0; k < 28; ++k) {
          G[k] = 1.0 / (1.0 + k) + 0.01 * i;
          u0[k] = 0.5 - 0.1 * (k % 3);
          u1[k] = 0.25 + 0.05 * (k % 5);
        }
        double d[7];
        laplace::math::chain_rule_pair_dk<6>(G, u0, u1, d);
        out(i) = d[0] + d[3] + d[6];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    double G[28], u0[28], u1[28];
    for (int k = 0; k < 28; ++k) {
      G[k] = 1.0 / (1.0 + k) + 0.01 * i;
      u0[k] = 0.5 - 0.1 * (k % 3);
      u1[k] = 0.25 + 0.05 * (k % 5);
    }
    double d[7];
    laplace::math::chain_rule_pair_dk<6>(G, u0, u1, d);
    CHECK(h(i) == doctest::Approx(d[0] + d[3] + d[6]).epsilon(1e-13));
  }
}

TEST_CASE("math chain_pair: order 6 inside a kernel") { chain_pair_kernel_case(); }
