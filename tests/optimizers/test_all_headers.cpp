// Every header of include/laplace/optimizers in one translation unit: a helper
// defined in two of them (as detail::max_abs was, in newton.hpp and bfgs.hpp)
// breaks this file's compilation. A new header is added to the list.

#include <doctest/doctest.h>

#include <laplace/optimizers/adam.hpp>
#include <laplace/optimizers/barzilai_borwein.hpp>
#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/bundle.hpp>
#include <laplace/optimizers/compass.hpp>
#include <laplace/optimizers/conjugate_gradient.hpp>
#include <laplace/optimizers/dense_curvature_solver.hpp>
#include <laplace/optimizers/gradient_descent.hpp>
#include <laplace/optimizers/lbfgs.hpp>
#include <laplace/optimizers/line_search.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/multistart.hpp>
#include <laplace/optimizers/nelder_mead.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/objective.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/primal_dual.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/optimizers/report.hpp>
#include <laplace/optimizers/stop_rules.hpp>
#include <laplace/optimizers/trust_region.hpp>

TEST_CASE("optimizers: every header compiles in one translation unit") {
  laplace::optimizers::optimizer_vector v("v", 2);
  v(0) = -3.0;
  v(1) = 2.0;
  CHECK(laplace::optimizers::detail::max_abs(v) == 3.0);
}
