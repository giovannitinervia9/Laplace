// The kernel of the shared check of the scalar bijectors with parameters
// (param_common.hpp) for the rows of one total order K = LAPLACE_PARAM_ORDER.
// tests/CMakeLists.txt compiles this file seven times, K = 0, ..., 6: the code
// of every map at every order is too large for one translation unit (ptxas
// took 700 s on it), and the seven parts compile in parallel.
#include "param_common.hpp"

#ifndef LAPLACE_PARAM_ORDER
#error "LAPLACE_PARAM_ORDER must be the total order 0..6 compiled by this unit"
#endif

namespace bijector_param_test {

namespace {

template <int K>
KOKKOS_INLINE_FUNCTION double partial_of(const lb::scalar_bijector& b, int w, double x, int idx) {
  lb::bijector_partials<K> r;
  if constexpr (K == 1) {
    r = w == 0 ? b.forward_partials_d1(x) : w == 1 ? b.inverse_partials_d1(x) : w == 2 ? b.forward_log_jacobian_partials_d1(x) : b.inverse_log_jacobian_partials_d1(x);
  } else if constexpr (K == 2) {
    r = w == 0 ? b.forward_partials_d2(x) : w == 1 ? b.inverse_partials_d2(x) : w == 2 ? b.forward_log_jacobian_partials_d2(x) : b.inverse_log_jacobian_partials_d2(x);
  } else if constexpr (K == 3) {
    r = w == 0 ? b.forward_partials_d3(x) : w == 1 ? b.inverse_partials_d3(x) : w == 2 ? b.forward_log_jacobian_partials_d3(x) : b.inverse_log_jacobian_partials_d3(x);
  } else if constexpr (K == 4) {
    r = w == 0 ? b.forward_partials_d4(x) : w == 1 ? b.inverse_partials_d4(x) : w == 2 ? b.forward_log_jacobian_partials_d4(x) : b.inverse_log_jacobian_partials_d4(x);
  } else if constexpr (K == 5) {
    r = w == 0 ? b.forward_partials_d5(x) : w == 1 ? b.inverse_partials_d5(x) : w == 2 ? b.forward_log_jacobian_partials_d5(x) : b.inverse_log_jacobian_partials_d5(x);
  } else {
    r = w == 0 ? b.forward_partials_d6(x) : w == 1 ? b.inverse_partials_d6(x) : w == 2 ? b.forward_log_jacobian_partials_d6(x) : b.inverse_log_jacobian_partials_d6(x);
  }
  return r.c[idx];
}

// the pure derivative of order K of quantity w (0 forward, 1 inverse, 2 and 3
// the log-Jacobians; 4 and 5, order 0 only, the distances to the bounds)
template <int K>
KOKKOS_INLINE_FUNCTION double pure_of(const lb::scalar_bijector& b, int w, double x) {
  if constexpr (K == 0) {
    switch (w) {
      case 0: return b.forward(x);
      case 1: return b.inverse(x);
      case 2: return b.forward_log_jacobian(x);
      case 3: return b.inverse_log_jacobian(x);
      case 4: return b.inverse_distance_upper(x);
      case 5: return b.inverse_distance_lower(x);
    }
  } else if constexpr (K == 1) {
    switch (w) {
      case 0: return b.forward_d1(x);
      case 1: return b.inverse_d1(x);
      case 2: return b.forward_log_jacobian_d1(x);
      case 3: return b.inverse_log_jacobian_d1(x);
    }
  } else if constexpr (K == 2) {
    switch (w) {
      case 0: return b.forward_d2(x);
      case 1: return b.inverse_d2(x);
      case 2: return b.forward_log_jacobian_d2(x);
      case 3: return b.inverse_log_jacobian_d2(x);
    }
  } else if constexpr (K == 3) {
    switch (w) {
      case 0: return b.forward_d3(x);
      case 1: return b.inverse_d3(x);
      case 2: return b.forward_log_jacobian_d3(x);
      case 3: return b.inverse_log_jacobian_d3(x);
    }
  } else if constexpr (K == 4) {
    switch (w) {
      case 0: return b.forward_d4(x);
      case 1: return b.inverse_d4(x);
      case 2: return b.forward_log_jacobian_d4(x);
      case 3: return b.inverse_log_jacobian_d4(x);
    }
  } else if constexpr (K == 5) {
    switch (w) {
      case 0: return b.forward_d5(x);
      case 1: return b.inverse_d5(x);
      case 2: return b.forward_log_jacobian_d5(x);
      case 3: return b.inverse_log_jacobian_d5(x);
    }
  } else {
    switch (w) {
      case 0: return b.forward_d6(x);
      case 1: return b.inverse_d6(x);
      case 2: return b.forward_log_jacobian_d6(x);
      case 3: return b.inverse_log_jacobian_d6(x);
    }
  }
  return laplace::math::detail::quiet_nan();
}

// y(i, 0): the value through the natural function; y(i, 1): for pure rows of
// order >= 1, the component c[0] of the partials (NaN-free 0 otherwise).
// Rows of another order are left to the kernel of their order.
template <int K>
struct eval_rows_of_order {
  map_t m;
  Rows a;
  Out y;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int w = static_cast<int>(a(i, 3)), n0 = static_cast<int>(a(i, 4)), n1 = static_cast<int>(a(i, 5)),
              n2 = static_cast<int>(a(i, 6));
    if (n0 + n1 + n2 != K) return;
    const lb::scalar_bijector b = make(m, a(i, 1), a(i, 2));
    const double x = a(i, 0);
    y(i, 1) = 0.0;
    if (w >= 4 || K == 0 || (n1 == 0 && n2 == 0)) {
      y(i, 0) = pure_of<K>(b, w, x);
      if constexpr (K >= 1)
        if (w < 4) y(i, 1) = partial_of<K>(b, w, x, 0);
    } else if constexpr (K >= 1) {
      const int idx = n2 * (K + 1) - n2 * (n2 - 1) / 2 + n1;  // bijector_partials<K>::index(n1, n2)
      y(i, 0) = partial_of<K>(b, w, x, idx);
    }
  }
};

}  // namespace

template <int K>
void eval_order(map_t m, Rows a, Out y) {
  Kokkos::parallel_for("bijector_param_check", Kokkos::RangePolicy<bench::Exec>(0, a.extent(0)),
                       eval_rows_of_order<K>{m, a, y});
}

template void eval_order<LAPLACE_PARAM_ORDER>(map_t, Rows, Out);

}  // namespace bijector_param_test
