// The double-double sqrt, exp and log (include/laplace/math/detail/double_double.hpp)
// against the mpmath references of tools/reference/double_double.py, evaluated
// in a Kokkos kernel on the default execution space, and against exact
// identities. The bounds are the errors measured on 2026-11-11 (note/numerics.md)
// with a margin of about half again, on the CPU and on the GPU build.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/detail/double_double.hpp>

#include <cmath>
#include <string>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

namespace ddm = laplace::math::detail::ddm;
using ddm::dd;

const std::string refdir = LAPLACE_REFERENCE_DIR;

constexpr double kTwoM104 = 0x1p-104;
constexpr double kTwoM1022 = 0x1p-1022;

struct DdSqrt {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::sqrt(dd(h, l)); }
};
struct DdExp {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::exp(dd(h, l)); }
};
struct DdLog {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::log(dd(h, l)); }
};

// exp(log(x)) for the argument x = h + l (its reference is x itself)
struct DdExpLog {
  KOKKOS_INLINE_FUNCTION dd operator()(double h, double l) const { return ddm::exp(ddm::log(dd(h, l))); }
};

// the double-double result of f on every argument, from a Kokkos kernel
template <class F>
std::vector<dd> eval_dd(const std::vector<bench::Args>& x, F f) {
  const std::size_t n = x.size();
  Kokkos::View<double* [2], Kokkos::LayoutRight, bench::Exec::memory_space> dx("x", n), dy("y", n);
  auto hx = Kokkos::create_mirror_view(dx);
  for (std::size_t i = 0; i < n; ++i) {
    hx(i, 0) = x[i][0];
    hx(i, 1) = x[i][1];
  }
  Kokkos::deep_copy(dx, hx);
  Kokkos::parallel_for(
      "test_dd", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t i) {
        const dd r = f(dx(i, 0), dx(i, 1));
        dy(i, 0) = r.hi;
        dy(i, 1) = r.lo;
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  std::vector<dd> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = dd(hy(i, 0), hy(i, 1));
  return y;
}

// Error of y = yh + yl against the reference hi + lo, in units of 2^-104 |hi|
// (ddulp) and of 2^-104 max(|hi|, 1) (ddulp1). Returns false when a wrong
// infinity or NaN occurs. Subnormal or zero references are not measured in
// ddulp: their error is returned in units of 2^-1074 through `sub`.
struct Errors {
  double max_rel = 0, max_rel1 = 0, max_sub = 0;
  std::size_t n_bad = 0;
  std::size_t n_checked = 0;
  bench::Args worst{0, 0, 0}, worst1{0, 0, 0};
};

Errors measure(const bench::Ref& r, const std::vector<dd>& y) {
  Errors e;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double hi = r.hi[i], lo = r.lo[i];
    if (std::isnan(hi) || std::isinf(hi)) {
      if (!(y[i].hi == hi || (std::isnan(hi) && std::isnan(y[i].hi)))) ++e.n_bad;
      continue;
    }
    if (std::isnan(y[i].hi) || std::isinf(y[i].hi)) {
      ++e.n_bad;
      continue;
    }
    ++e.n_checked;
    const double diff = ((y[i].hi - hi) - lo) + y[i].lo;
    const double ah = std::fabs(hi);
    if (ah < kTwoM1022) {
      e.max_sub = std::max(e.max_sub, std::fabs(diff) / 0x1p-1074);
      continue;
    }
    const double rel = std::fabs(diff) / (kTwoM104 * ah);
    const double rel1 = std::fabs(diff) / (kTwoM104 * std::max(ah, 1.0));
    if (rel > e.max_rel) {
      e.max_rel = rel;
      e.worst = r.x[i];
    }
    if (rel1 > e.max_rel1) {
      e.max_rel1 = rel1;
      e.worst1 = r.x[i];
    }
  }
  return e;
}

}  // namespace

TEST_CASE("double-double sqrt against mpmath") {
  const auto r = bench::read_ref(refdir + "/dd_sqrt.csv");
  const auto e = measure(r, eval_dd(r.x, DdSqrt{}));
  INFO("sqrt: max ", e.max_rel, " ddulp, worst x_hi ", bench::show(e.worst, 2));
  CHECK(e.n_bad == 0);
  CHECK(e.max_rel <= 2.0);
  CHECK(e.max_sub <= 1.0);
}

TEST_CASE("double-double exp against mpmath") {
  const auto r = bench::read_ref(refdir + "/dd_exp.csv");
  const auto e = measure(r, eval_dd(r.x, DdExp{}));
  INFO("exp: max ", e.max_rel, " ddulp, worst x_hi ", bench::show(e.worst, 2));
  CHECK(e.n_bad == 0);
  CHECK(e.max_rel <= 1.0);
  CHECK(e.max_sub <= 1.0);
}

TEST_CASE("double-double log against mpmath") {
  const auto r = bench::read_ref(refdir + "/dd_log.csv");
  const auto e = measure(r, eval_dd(r.x, DdLog{}));
  INFO("log: max ", e.max_rel, " ddulp, worst x_hi ", bench::show(e.worst, 2), "; max ", e.max_rel1,
       " ddulp1, worst x_hi ", bench::show(e.worst1, 2));
  CHECK(e.n_bad == 0);
  CHECK(e.max_rel <= 3.0);  // relative, also near 1 (the log1p branch)
  CHECK(e.max_sub <= 1.0);
}

struct DdExpm1 {
  KOKKOS_INLINE_FUNCTION dd operator()(double hi, double lo) const { return ddm::expm1(dd(hi, lo)); }
};

TEST_CASE("double-double expm1 against mpmath") {
  const auto r = bench::read_ref(refdir + "/dd_expm1.csv");
  const auto e = measure(r, eval_dd(r.x, DdExpm1{}));
  INFO("expm1: max ", e.max_rel, " ddulp, worst x_hi ", bench::show(e.worst, 2));
  CHECK(e.n_bad == 0);
  CHECK(e.max_rel <= 3.5);
}

TEST_CASE("double-double identities") {
  // sqrt(4) = 2 exactly, and sqrt of squares of doubles is exact
  const dd s4 = ddm::sqrt(dd(4.0));
  CHECK(s4.hi == 2.0);
  CHECK(s4.lo == 0.0);
  CHECK(ddm::sqrt(dd(0.0)).hi == 0.0);
  CHECK(std::isnan(ddm::sqrt(dd(-1.0)).hi));
  CHECK(std::isinf(ddm::sqrt(dd(INFINITY)).hi));

  // exp(0) = 1 and log(1) = 0 exactly
  const dd e0 = ddm::exp(dd(0.0));
  CHECK(e0.hi == 1.0);
  CHECK(e0.lo == 0.0);
  const dd l1 = ddm::log(dd(1.0));
  CHECK(l1.hi == 0.0);
  CHECK(l1.lo == 0.0);
  // log of 2 and of 2^k is k ln 2 (the double-double ln 2 is 0x1.62e42fefa39efp-1 + 0x1.abc9e3b39803fp-56)
  const dd l2 = ddm::log(dd(2.0));
  CHECK(l2.hi == 0x1.62e42fefa39efp-1);
  CHECK(l2.lo == 0x1.abc9e3b39803fp-56);

  // special values of exp and log
  CHECK(std::isinf(ddm::exp(dd(710.0)).hi));
  CHECK(ddm::exp(dd(-746.0)).hi == 0.0);
  CHECK(std::isnan(ddm::exp(dd(NAN)).hi));
  CHECK(ddm::log(dd(0.0)).hi == -INFINITY);
  CHECK(std::isnan(ddm::log(dd(-1.0)).hi));
  CHECK(ddm::log(dd(INFINITY)).hi == INFINITY);
}

TEST_CASE("double-double exp(log(x)) returns x") {
  // log x has an absolute error of about 2^-104 max(|log x|, 1), which exp turns
  // into a relative error of the same size in x: the bound scales with max(|log x|, 1).
  const auto r = bench::read_ref(refdir + "/dd_log.csv");
  std::vector<bench::Args> x;
  for (std::size_t i = 0; i < r.x.size(); i += 7) x.push_back(r.x[i]);  // arguments of the log table
  const auto y = eval_dd(x, DdExpLog{});
  double worst = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const dd ref(x[i][0], x[i][1]);
    const double diff = ((y[i].hi - ref.hi) - ref.lo) + y[i].lo;
    const double rel = std::fabs(diff) / (kTwoM104 * std::fabs(ref.hi));
    const double scaled = rel / std::max(1.0, std::fabs(std::log(ref.hi)));
    worst = std::max(worst, scaled);
  }
  INFO("exp(log(x)): max ", worst, " ddulp per max(1, |log x|)");
  CHECK(worst <= 1.2);
}
