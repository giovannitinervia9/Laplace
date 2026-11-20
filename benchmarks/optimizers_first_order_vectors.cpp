// What the host vectors of the first-order optimizers cost against a Kokkos
// gradient (note/optimizers.md, section 13, point 4), at p = 1e4, 1e5, 1e6.
//
// The objective is separable with one row per coordinate,
// f(x) = sum_i log cosh(x_i - c_i), c_i uniform on [-5, 5]: its value and
// gradient are one Kokkos parallel_reduce on the default execution space
// (OpenMP or CUDA, by build), with x copied down and g copied back at each
// call, as an optimizer on the host needs them.
//
// Columns, in microseconds per call or per iteration:
//   value     objective.value(x), the copy down included
//   val+grad  objective.value_gradient(x, g), both copies included
//   host GD   the host vector work of one gradient descent or Barzilai-Borwein
//             iteration on one thread: d = -alpha g, the trial point
//             x + s d with its moved test, the four sums s'y, s's, y'y, g'g,
//             and the three copies x_previous, x, g (8 passes over p)
//   host Adam the host work of one Adam iteration on one thread: m, v, the
//             step u and g'Bg, g'g, the trial point, s'y, s's, y'y and the
//             three copies
//   dev GD    the same work as host GD written as Kokkos kernels on the
//             default execution space (one parallel_for, one parallel_reduce
//             of four sums, three deep_copy), with no copy between spaces
//   GD it, BB it, Adam it   one iteration of the real optimizer (30
//             iterations, no stopping rule fires), elapsed / iterations, with
//             its evaluations per iteration in brackets
//
// Usage: optimizers_first_order_vectors
// (OpenMP build: OMP_NUM_THREADS sets the kernel threads; the host vector
// work is always one thread.)

#include <laplace/optimizers/adam.hpp>
#include <laplace/optimizers/barzilai_borwein.hpp>
#include <laplace/optimizers/gradient_descent.hpp>
#include <laplace/optimizers/minimize.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace first_order_vectors_bench {

namespace lo = laplace::optimizers;
using device_vector = Kokkos::View<double*>;

// us per call: repeats until 0.3 s have passed (at least 5 calls)
template <class F>
double time_us(F&& f) {
  using clock = std::chrono::steady_clock;
  int calls = 0;
  const auto start = clock::now();
  double elapsed = 0.0;
  do {
    f();
    ++calls;
    elapsed = std::chrono::duration<double, std::micro>(clock::now() - start).count();
  } while (elapsed < 3e5 || calls < 5);
  return elapsed / calls;
}

// f = sum log cosh(x_i - c_i), value and gradient in one parallel_reduce
struct log_cosh_objective {
  device_vector c, xd, gd;
  explicit log_cosh_objective(int p) : c("c", p), xd("xd", p), gd("gd", p) {
    auto ch = Kokkos::create_mirror_view(c);
    std::uint64_t s = 42;
    for (int i = 0; i < p; ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      ch(i) = 10.0 * (static_cast<double>(s >> 11) * 0x1.0p-53) - 5.0;
    }
    Kokkos::deep_copy(c, ch);
  }
  double value(const lo::optimizer_point& x) {
    Kokkos::deep_copy(xd, x);
    double f = 0.0;
    const auto xv = xd, cv = c;
    Kokkos::parallel_reduce(
        "log_cosh_value", xv.extent(0),
        KOKKOS_LAMBDA(const int i, double& acc) {
          const double t = Kokkos::abs(xv(i) - cv(i));
          acc += t + Kokkos::log1p(Kokkos::exp(-2.0 * t)) - 0.69314718055994531;
        },
        f);
    return f;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    Kokkos::deep_copy(xd, x);
    double f = 0.0;
    const auto xv = xd, cv = c, gv = gd;
    Kokkos::parallel_reduce(
        "log_cosh_value_gradient", xv.extent(0),
        KOKKOS_LAMBDA(const int i, double& acc) {
          const double u = xv(i) - cv(i), t = Kokkos::abs(u);
          acc += t + Kokkos::log1p(Kokkos::exp(-2.0 * t)) - 0.69314718055994531;
          gv(i) = Kokkos::tanh(u);
        },
        f);
    Kokkos::deep_copy(g, gd);
    return f;
  }
};

// the host work of one gradient descent or Barzilai-Borwein iteration
double host_gd(int p) {
  lo::optimizer_vector x("x", p), g("g", p), d("d", p), xt("xt", p), gt("gt", p), xp("xp", p);
  for (int i = 0; i < p; ++i) x(i) = 0.1 * i / p, g(i) = 1.0 - 0.5 * i / p, gt(i) = 0.5 * g(i);
  double sink = 0.0;
  const double t = time_us([&] {
    const double alpha = 0.37, s = 1.0;
    for (int i = 0; i < p; ++i) d(i) = -alpha * g(i);
    bool moved = false;
    for (int i = 0; i < p; ++i) {
      xt(i) = x(i) + s * d(i);
      moved = moved || xt(i) != x(i);
    }
    double sy = 0.0, ss = 0.0, yy = 0.0, gg = 0.0;
    for (int i = 0; i < p; ++i) {
      const double si = xt(i) - x(i), yi = gt(i) - g(i);
      sy += si * yi, ss += si * si, yy += yi * yi, gg += gt(i) * gt(i);
    }
    Kokkos::deep_copy(xp, x);
    Kokkos::deep_copy(x, xt);
    Kokkos::deep_copy(g, gt);
    sink += sy + ss + yy + gg + moved;
  });
  if (sink == 12345.0) std::printf(" ");
  return t;
}

// the host work of one Adam iteration
double host_adam(int p) {
  lo::optimizer_vector x("x", p), g("g", p), xt("xt", p), gt("gt", p), xp("xp", p);
  std::vector<double> m(p, 0.0), v(p, 0.0), u(p);
  for (int i = 0; i < p; ++i) x(i) = 0.1 * i / p, g(i) = 1.0 - 0.5 * i / p, gt(i) = 0.5 * g(i);
  double sink = 0.0;
  const double t = time_us([&] {
    const double beta1 = 0.9, beta2 = 0.999, lr = 1e-3, c1 = 0.5, c2 = 0.01;
    double gbg = 0.0, gg = 0.0;
    for (int i = 0; i < p; ++i) {
      m[i] = beta1 * m[i] + (1.0 - beta1) * g(i);
      v[i] = beta2 * v[i] + (1.0 - beta2) * g(i) * g(i);
      const double b = lr / (std::sqrt(v[i] / c2) + 1e-8);
      u[i] = -b * m[i] / c1;
      gbg += b * g(i) * g(i), gg += g(i) * g(i);
    }
    bool moved = false;
    for (int i = 0; i < p; ++i) {
      xt(i) = x(i) + u[i];
      moved = moved || xt(i) != x(i);
    }
    double sy = 0.0, ss = 0.0, yy = 0.0;
    for (int i = 0; i < p; ++i) {
      const double si = xt(i) - x(i), yi = gt(i) - g(i);
      sy += si * yi, ss += si * si, yy += yi * yi;
    }
    Kokkos::deep_copy(xp, x);
    Kokkos::deep_copy(x, xt);
    Kokkos::deep_copy(g, gt);
    sink += sy + ss + yy + gbg + gg + moved;
  });
  if (sink == 12345.0) std::printf(" ");
  return t;
}

// the same work as host_gd as kernels on the default execution space
double device_gd(int p) {
  device_vector x("x", p), g("g", p), d("d", p), xt("xt", p), gt("gt", p), xp("xp", p);
  Kokkos::parallel_for(
      "device_gd_init", p, KOKKOS_LAMBDA(const int i) {
        x(i) = 0.1 * i / p, g(i) = 1.0 - 0.5 * i / p, gt(i) = 0.5 * g(i);
      });
  double sink = 0.0;
  const double t = time_us([&] {
    const double alpha = 0.37, s = 1.0;
    Kokkos::parallel_for(
        "device_gd_trial", p, KOKKOS_LAMBDA(const int i) {
          d(i) = -alpha * g(i);
          xt(i) = x(i) + s * d(i);
        });
    double sy = 0.0, ss = 0.0, yy = 0.0, gg = 0.0;
    Kokkos::parallel_reduce(
        "device_gd_sums", p,
        KOKKOS_LAMBDA(const int i, double& a, double& b, double& c, double& e) {
          const double si = xt(i) - x(i), yi = gt(i) - g(i);
          a += si * yi, b += si * si, c += yi * yi, e += gt(i) * gt(i);
        },
        sy, ss, yy, gg);
    Kokkos::deep_copy(xp, x);
    Kokkos::deep_copy(x, xt);
    Kokkos::deep_copy(g, gt);
    Kokkos::fence();
    sink += sy + ss + yy + gg;
  });
  if (sink == 12345.0) std::printf(" ");
  return t;
}

// a decrement rule that never fires: the run takes max_iterations
struct never_stop_rule {
  static constexpr unsigned needs = lo::stop_needs_decrement;
  lo::stop_verdict verdict(const lo::stop_rule_input&) const { return lo::stop_verdict::none; }
};

template <class Optimizer>
void full_run(Optimizer o, log_cosh_objective& f, int p) {
  o.max_iterations = 30;
  lo::optimizer_vector x0("x0", p);
  const lo::optimizer_report r = lo::minimize(o, f, x0);
  const double per = 1e6 * r.elapsed_seconds / std::max(1, r.iterations);
  std::printf(" %8.0f [%3.1f]", per, double(r.value_evaluations) / std::max(1, r.iterations));
}

void run() {
  std::printf("execution space %s, concurrency %d; us per call or per iteration\n",
              Kokkos::DefaultExecutionSpace::name(), Kokkos::DefaultExecutionSpace().concurrency());
  std::printf("%9s %9s %9s %9s %9s %9s %14s %14s %14s\n", "p", "value", "val+grad", "host GD", "host Adam", "dev GD",
              "GD it [ev]", "BB it [ev]", "Adam it [ev]");
  for (int p : {10000, 100000, 1000000}) {
    log_cosh_objective f(p);
    lo::optimizer_vector x("x", p), g("g", p);
    const double tv = time_us([&] { f.value(x); });
    const double tg = time_us([&] { f.value_gradient(x, g); });
    std::printf("%9d %9.1f %9.1f %9.1f %9.1f %9.1f", p, tv, tg, host_gd(p), host_adam(p), device_gd(p));
    full_run(lo::gradient_descent_optimizer<never_stop_rule>{}, f, p);
    full_run(lo::barzilai_borwein_optimizer<never_stop_rule>{}, f, p);
    lo::adam_optimizer<never_stop_rule> a;
    a.learning_rate = 0.1;
    full_run(a, f, p);
    std::printf("\n");
  }
}

}  // namespace first_order_vectors_bench

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  first_order_vectors_bench::run();
  Kokkos::finalize();
}
