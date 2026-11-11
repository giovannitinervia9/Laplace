<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/laplace-logo-dark.svg">
    <img alt="Laplace" src="docs/logo/laplace-logo-light.svg" width="420">
  </picture>
</p>

Laplace is a header-only C++20 library, in early development, for distributional
regression: statistical models in which every parameter of the response
distribution (location, scale, shape) has its own predictor, with smooth, random
and penalized terms. The models are meant to be fitted with exact derivatives,
either by penalized likelihood with the Laplace approximation for the latent
effects or by Hamiltonian Monte Carlo on the same gradients. Per-observation
computations are written as
[Kokkos](https://github.com/kokkos/kokkos) kernels, so the same source runs on
CPUs (Serial, OpenMP) and GPUs (CUDA).

## Status

Laplace is in early development and has no stable API yet. The model layer is
not available. What exists today is the numerical foundation on which it will
be built: special functions that can be called inside Kokkos kernels, on the
host and on the device, each one measured against 200-bit
[mpmath](https://mpmath.org) references and tested on CPU and GPU.

| Header (`include/laplace/math/`) | Functions (namespace `laplace::math`) |
|---|---|
| `elementary.hpp` | elementary functions (platform implementations, measured) |
| `special.hpp`, `polygamma.hpp` | `lgamma`, `digamma`, `trigamma`, `polygamma` (orders up to 6) |
| `incomplete_gamma.hpp` | `gamma_p`, `gamma_q`, `log_gamma_p`, `log_gamma_q` |
| `incomplete_beta.hpp` | `ibeta`, `ibetac`, `log_ibeta`, `log_ibetac` |
| `incomplete_inverse.hpp` | `gamma_p_inv`, `gamma_q_inv`, `ibeta_inv`, `ibetac_inv`, and versions taking log p |
| `normal.hpp` | `erfc`, `erfcx`, `Phi`, `log_Phi`, `log_Phi_d1` to `log_Phi_d5`, `Phi_inv`, `Phi_inv_logp` |
| `incomplete_gamma_deriv.hpp` | derivatives of log P(a, x) and log Q(a, x) in the shape a, orders 1 to 5 |
| `incomplete_beta_deriv.hpp` | derivatives of log I_x(a, b) and log(1 - I_x(a, b)) in a and b, orders 1 to 5 |
| `student.hpp` | log of the Student t cdf with real degrees of freedom, and its derivatives to order 5 |
| `owen.hpp` | Owen's T function, log of the skew-normal cdf and its derivatives to order 5 |
| `bessel_i.hpp` | `log_bessel_i0`, `log_bessel_i0_scaled`, the ratio I1/I0 and its inverse (von Mises), with derivatives to order 6, and the ratios I_j/I_0 (`bessel_i_ratios`) |
| `bessel_k.hpp` | `log_bessel_k` for real order, and its derivatives in the argument to order 7 |
| `faa_di_bruno.hpp` | `bell_partial`, and `compose_d1` to `compose_d7` (derivatives of f(g(x))) |
| `quadrature.hpp` | `quad::tanh_sinh`, `exp_sinh`, `sinh_sinh`, `gauss_kronrod`, `panels`, and fixed Gauss-Legendre, -Hermite and -Laguerre rules |

Every function is a separate function per derivative order, takes and returns
`double`, throws no exception and allocates no memory. The measured accuracy
and speed of each routine, and the reasons for each choice of algorithm, are in
[docs/numerics.md](docs/numerics.md).

Planned next, in this order: link functions and distributions with closed-form
derivatives; parameter maps, bases and penalties; optimizers; the fitting
engine, where a model is a graph of terms whose derivatives are composed by the
chain rule; Hamiltonian Monte Carlo (NUTS); a C interface with R and Python
bindings.

## Requirements

- A C++20 compiler. Tested with GCC 14.2 (CPU) and with MSVC 14.44 + CUDA 13.1 (GPU).
- CMake 3.25 or later.
- [Kokkos](https://github.com/kokkos/kokkos) 5, built with the Serial and/or OpenMP
  backend for the CPU, or with CUDA for the GPU.
- [Boost](https://www.boost.org) 1.90 or later, headers only (Boost.Math).
- Python 3 with mpmath, only to regenerate the reference data.

## Building and testing

```sh
cmake --preset cpu -DKokkos_ROOT=/path/to/kokkos/install -DBoost_ROOT=/path/to/boost
cmake --build --preset cpu
ctest --preset cpu
```

Instead of `Boost_ROOT`, `-DLAPLACE_BOOST_INCLUDE=/path/to/dir/holding/boost`
points to a directory that contains the Boost headers. The `cuda` preset builds
the same tests against a CUDA-enabled Kokkos (`CMAKE_CUDA_ARCHITECTURES` defaults
to `native`). With nvcc, `-DLAPLACE_CUDA_FMAD=OFF` stops the contraction of
multiply-adds into FMA, so that device results match host results bit for bit
more often.

Tests and benchmarks are built only when Laplace is the top-level project
(options `LAPLACE_BUILD_TESTS` and `LAPLACE_BUILD_BENCHMARKS`).

## Using Laplace

Laplace is an interface target. In a CMake project:

```cmake
add_subdirectory(Laplace)   # or FetchContent
target_link_libraries(my_target PRIVATE Laplace::laplace)
```

The functions are called inside Kokkos kernels:

```cpp
#include <Kokkos_Core.hpp>
#include <laplace/math/normal.hpp>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int n = 1000;
  Kokkos::View<double*> y("y", n);
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    const double x = -40.0 + 50.0 * i / (n - 1);
    y(i) = laplace::math::log_Phi(x);   // accurate far into the lower tail
  });
  Kokkos::fence();
}
```

## Repository layout

```
include/laplace/       the library (header-only)
  core/                configuration shared by all layers
  math/                special functions and their derivatives
tests/                 doctest suite (one executable, laplace_tests)
  data/reference/      mpmath reference values, as hex doubles
benchmarks/            accuracy and speed measurements (CPU and GPU)
tools/reference/       Python scripts that write tests/data/reference
third_party/doctest/   doctest 2.4.11 (MIT)
docs/                  numerics notes and the logo
```

## Reference data

`tests/data/reference/` holds the reference values used by the tests and the
benchmarks: for each function, its arguments and the value at 200-bit
precision, stored as a pair of doubles (`hi`, `lo`) in hexadecimal. The scripts
in `tools/reference/` regenerate them:

```sh
python tools/reference/normal.py
```

## License

Laplace is released under the [MIT License](LICENSE). doctest is MIT-licensed;
Kokkos (BSD-3-Clause) and Boost (Boost Software License) are external
dependencies.
