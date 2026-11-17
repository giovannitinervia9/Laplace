<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/laplace-logo-dark.svg">
    <img alt="Laplace" src="docs/logo/laplace-logo-light.svg" width="420">
  </picture>
</p>

Laplace is a C++20 library, in early development, for distributional
regression: statistical models in which every parameter of the response
distribution (location, scale, shape) has its own predictor, with smooth, random
and penalized terms. The models are meant to be fitted with exact derivatives,
either by penalized likelihood with the Laplace approximation for the latent
effects or by Hamiltonian Monte Carlo on the same gradients. Per-observation
computations are written as
[Kokkos](https://github.com/kokkos/kokkos) kernels, so the same source runs on
CPUs (Serial, OpenMP) and GPUs (CUDA).

The reference manual is at
[giovannitinervia9.github.io/Laplace](https://giovannitinervia9.github.io/Laplace/).

## Status

Laplace is in early development and has no stable API yet. It is being built
bottom-up: each layer below is complete enough to be used by the next one, and
the fitting engine comes once the layers it rests on are in place. The table
lists the layers in the order in which they entered the repository; each name
links to its page in the reference manual.

| Since | Layer | What it contains | State |
|---|---|---|---|
| 2026-11-09 | [Special functions](https://giovannitinervia9.github.io/Laplace/group__math.html) (`math/`) | gamma, digamma and polygamma; incomplete gamma and beta, their logarithms and inverses; the normal, Student t and skew-normal cdfs; Owen's T; Bessel I and K; derivatives in the shapes to order 5 or more; Faà di Bruno compositions; quadrature rules | available |
| 2026-11-12 | [Bijectors](https://giovannitinervia9.github.io/Laplace/group__bijector.html) (`bijector/`) | scalar maps with and without parameters (log, logit, softplus, bounded, Box-Cox, Yeo-Johnson, ...) and coupled maps (additive log-ratio, log-Cholesky, partial correlations, matrix logarithm), both directions, log-Jacobians and derivatives to order 6 | available |
| 2026-11-12 | [Structured matrices](https://giovannitinervia9.github.io/Laplace/group__matrices.html) (`pdmat/`) | diagonal, compound symmetry, AR(1), Toeplitz, unstructured, factor-analytic, reduced-rank, weighted sums: matrix, log-determinant, inverse, Cholesky factor and derivatives to order 6 | available |
| 2026-11-12 | [Distributions](https://giovannitinervia9.github.io/Laplace/group__distrib.html) (`distrib/`) | 38 univariate and multivariate distributions (Gaussian, Student t, gamma, beta, von Mises, negative binomial, Dirichlet, multivariate Gaussian and t, ...) with log-density derivatives in closed form, and wrappers for transformed, truncated, atom-mixture and independent-product laws | available |
| 2026-11-14 | [Bases](https://giovannitinervia9.github.io/Laplace/group__basis.html) (`basis/`) | B-splines (also cyclic), Fourier and Legendre bases, tensor products, differential operators and their penalty matrices | available |
| 2026-11-14 | [Dense linear algebra](https://giovannitinervia9.github.io/Laplace/group__dense.html) (`dense/`) | factorizations on host Kokkos views, through an Eigen backend compiled as a static library | available |
| 2026-11-15 | [Optimizers](https://giovannitinervia9.github.io/Laplace/group__optimizers.html) (`optimizers/`) | Newton, trust region, BFGS and L-BFGS with Armijo and Wolfe line searches, stop rules, multiple starts, a report of convergence for every run | available |
| 2026-11-17 | [Penalties](https://giovannitinervia9.github.io/Laplace/group__penalty.html) (`penalty/`) | the penalty concept, and penalties built from distributions | in progress |
| | Fitting engine | the model as a graph of terms whose derivatives are composed by the chain rule; Laplace approximation for latent effects | planned |
| | Hamiltonian Monte Carlo | NUTS on the same gradients | planned |
| | Interfaces | a C interface, with R and Python packages on top | planned |

The special functions, bijectors, structured matrices, distributions and bases
are callable inside Kokkos kernels, on the host and on the device, throw no
exception and allocate no memory; they are tested on CPU and GPU against
200-bit [mpmath](https://mpmath.org) references. Each derivative order is a
separate function. Dense linear algebra and the optimizers run on the host.
The measured accuracy and
speed of each routine, and the reasons for each choice of algorithm, are in
[docs/numerics.md](docs/numerics.md).

## Requirements

- A C++20 compiler. Tested with GCC 14.2 (CPU) and with MSVC 14.44 + CUDA 13.1 (GPU).
- CMake 3.25 or later.
- [Kokkos](https://github.com/kokkos/kokkos) 5, built with the Serial and/or OpenMP
  backend for the CPU, or with CUDA for the GPU.
- [Boost](https://www.boost.org) 1.90 or later, headers only (Boost.Math).
- [Eigen](https://eigen.tuxfamily.org) 3.4 or later, only for the dense linear
  algebra library (`Laplace::dense`); `-DLAPLACE_EIGEN_INCLUDE` points to the
  directory holding `Eigen/`, otherwise CMake looks for an installed Eigen3.
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

Laplace is an interface target (`Laplace::laplace`); the dense linear algebra is
a separate static library (`Laplace::dense`), linked only by code that uses it.
In a CMake project:

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
include/laplace/       the library headers
  core/                configuration shared by all layers
  math/                special functions and their derivatives
  bijector/            scalar and coupled bijectors
  pdmat/               structured positive-definite matrices
  distrib/             distributions and their wrappers
  basis/               bases and penalty matrices
  dense/               dense linear algebra (interface)
  optimizers/          Newton, trust region, BFGS, L-BFGS
  penalty/             penalties
src/dense/             the Eigen backend of dense/ (compiled, never by nvcc)
tests/                 doctest suite (one executable, laplace_tests)
  data/reference/      mpmath reference values, as hex doubles
benchmarks/            accuracy and speed measurements (CPU and GPU)
tools/reference/       Python scripts that write tests/data/reference
third_party/doctest/   doctest 2.4.11 (MIT)
docs/                  numerics notes, the reference manual and the logo
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
