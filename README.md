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
lists the layers in that order, each one using only those above it; each name
links to its page in the reference manual.

| Layer | What it contains | State |
|---|---|---|
| [Special functions](https://giovannitinervia9.github.io/Laplace/group__math.html) (`math/`) | gamma, digamma, polygamma and Stirling's series; incomplete gamma and beta, their logarithms, inverses and derivatives in the shapes; the normal, Student t and skew-normal cdfs; Owen's T; Bessel I and K; Faà di Bruno compositions; quadrature rules | available |
| [Bijectors](https://giovannitinervia9.github.io/Laplace/group__bijector.html) (`bijector/`) | scalar maps with and without parameters (log, logit, probit, softplus, bounded, Box-Cox, Yeo-Johnson, ...) and coupled maps (additive log-ratio, log-Cholesky, partial correlations, matrix logarithm), both directions, log-Jacobians and derivatives to order 6 | available |
| [Bases](https://giovannitinervia9.github.io/Laplace/group__basis.html) (`basis/`) | B-splines (also cyclic), Fourier and Legendre bases, tensor products, differential operators, penalty matrices and linear transforms of the coefficients | available |
| [Dense linear algebra](https://giovannitinervia9.github.io/Laplace/group__dense.html) (`dense/`) | Cholesky, triangular solves, condition estimate, pivoted QR, SVD, symmetric and generalized eigenproblems, null-space bases, on host Kokkos views through an Eigen backend | available |
| [Structured matrices](https://giovannitinervia9.github.io/Laplace/group__matrices.html) (`pdmat/`) | positive-definite structures (diagonal, compound symmetry, AR(1), Toeplitz, unstructured, factor-analytic, ...) and positive-semidefinite ones (reduced rank, weighted sums), with inverse, Cholesky factor, log-determinant and derivatives to order 6 | available |
| [Distributions](https://giovannitinervia9.github.io/Laplace/group__distrib.html) (`distrib/`) | 32 univariate distributions (on the real line, the positive half-line, a bounded interval, the circle, and discrete) and 6 multivariate ones (Gaussian and t in two parametrizations each, Dirichlet, multinomial), with derivatives of the log-density to order 6 and of the expected information to order 4; 5 wrappers (transformed, truncated, atom mixture, independent product, univariate as multivariate) | available |
| [Penalties](https://giovannitinervia9.github.io/Laplace/group__penalty.html) (`penalty/`) | the penalty concept with its normalization classes (proper, intrinsic, unnormalizable), penalties built from distributions and from independent products, proximal operators | in progress |
| [Optimizers](https://giovannitinervia9.github.io/Laplace/group__optimizers.html) (`optimizers/`) | Newton, trust region, BFGS and L-BFGS with Armijo and Wolfe line searches, stop rules, multiple starts, a report of convergence for every run | available |
| Fitting engine | the model as a graph of terms whose derivatives are composed by the chain rule; Laplace approximation for latent effects | planned |
| Hamiltonian Monte Carlo | NUTS on the same gradients | planned |
| Interfaces | a C interface, with R and Python packages on top | planned |

The per-observation routines of `math/`, `bijector/`, `basis/`, `pdmat/`,
`distrib/` and `penalty/` are marked `LAPLACE_HD`: they run on the host and
inside Kokkos kernels on the device, throw no exception and allocate no
memory. The routines that build and decompose (the constructors and transforms
of the bases, `weighted_sum_psdmat`, starting values), `dense/` and
`optimizers/` run on the host. Each derivative order is a separate function,
written out in C++ without automatic differentiation. The tests run on CPU and
GPU and check the routines against [mpmath](https://mpmath.org) references at
200 bits or more. The measured accuracy and speed of each routine, and the reasons for each
choice of algorithm, are in [docs/numerics.md](docs/numerics.md).

## Requirements

- A C++20 compiler. Tested with GCC 14.2 (CPU) and with MSVC 14.44 + CUDA 13.1 (GPU).
- CMake 3.25 or later.
- [Kokkos](https://github.com/kokkos/kokkos) 5, built with the Serial and/or OpenMP
  backend for the CPU, or with CUDA for the GPU.
- [Boost](https://www.boost.org) 1.90 or later, headers only (Boost.Math).
- [Eigen](https://eigen.tuxfamily.org) 3.4 or later, headers only. It is
  required to configure the project, since the dense linear algebra library
  (`Laplace::dense`) is always built.
- Python 3 with mpmath and SymPy, only to regenerate the reference data.
- Doxygen, only to build the reference manual.

## Building and testing

```sh
cmake --preset cpu -DKokkos_ROOT=/path/to/kokkos/install -DBoost_ROOT=/path/to/boost -DEigen3_DIR=/path/to/eigen/share/eigen3/cmake
cmake --build --preset cpu
ctest --preset cpu
```

Instead of `Boost_ROOT`, `-DLAPLACE_BOOST_INCLUDE=/path/to/dir/holding/boost`
points to a directory that contains the Boost headers; instead of `Eigen3_DIR`,
`-DLAPLACE_EIGEN_INCLUDE=/path/to/dir/holding/Eigen` points to a directory that
contains the Eigen headers. The `cpu-debug` preset is the same build without
optimization. The `cuda` preset builds
the same tests against a CUDA-enabled Kokkos (`CMAKE_CUDA_ARCHITECTURES` defaults
to `native`). With nvcc, `-DLAPLACE_CUDA_FMAD=OFF` stops the contraction of
multiply-adds into FMA, so that device results match host results bit for bit
more often.

Tests and benchmarks are built only when Laplace is the top-level project
(options `LAPLACE_BUILD_TESTS` and `LAPLACE_BUILD_BENCHMARKS`).

The reference manual is built with `doxygen docs/manual/Doxyfile` into
`build-docs/html`; `python -I tools/docs/check_manual.py` checks it for markup
that did not render.

## Using Laplace

Laplace is an interface target (`Laplace::laplace`); the dense linear algebra is
a separate static library (`Laplace::dense`). In a CMake project:

```cmake
add_subdirectory(Laplace)   # or FetchContent
target_link_libraries(my_target PRIVATE Laplace::laplace)
# code that uses dense/, the optimizers, the operators and transforms of the
# bases, or weighted_sum_psdmat also links the static library:
target_link_libraries(my_target PRIVATE Laplace::dense)
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
  basis/               bases and penalty matrices
  dense/               dense linear algebra (interface)
  pdmat/               structured positive-definite and positive-semidefinite matrices
  distrib/             distributions and their wrappers
  penalty/             penalties and proximal operators
  optimizers/          Newton, trust region, BFGS, L-BFGS
src/dense/             the Eigen backend of dense/ (compiled, never by nvcc)
tests/                 doctest suite (one executable, laplace_tests)
  data/reference/      mpmath reference values, as hex doubles
benchmarks/            accuracy and speed measurements (CPU and GPU)
tools/reference/       Python scripts that write tests/data/reference
tools/docs/            the check of the reference manual
third_party/doctest/   doctest 2.4.11 (MIT)
docs/                  numerics notes and the logo
  manual/              the reference manual (Doxygen)
```

## Reference data

`tests/data/reference/` holds the reference values used by the tests and the
benchmarks: for each function, its arguments and the value computed with
mpmath at 200 bits or more, stored as a pair of doubles (`hi`, `lo`) in
hexadecimal. The scripts in `tools/reference/` regenerate them; several derive
their formulas with SymPy first. They are run in isolated mode:

```sh
python -I tools/reference/normal.py
```

## License

Laplace is released under the [MIT License](LICENSE). doctest is MIT-licensed;
Kokkos (BSD-3-Clause), Boost (Boost Software License) and Eigen (MPL-2.0,
compiled with `EIGEN_MPL2_ONLY`) are external dependencies.
