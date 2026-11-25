#pragma once

// The linear nodes of a model as stored design rows (laplace::engine; plan,
// section 7.8): a linear predictor of unit i is
//
//   eta_r(i) = offset_r(i) + sum_e value(i, e) x[coefficient(i, e)]
//
// over the entries e that feed predictor r, x being the vector of coordinates.
// A block of entries has a fixed width w per unit (a covariate has width 1, a
// B-spline row of degree d has d + 1, a group effect has 1 per coefficient of
// the group), and an entry whose coefficient is -1 is empty. Every row of the
// stored design has the same width, the sum of the widths of the blocks, so
// device threads do not diverge.
//
// Coordinates come in two kinds (row_model.hpp): global ones, shared by every
// unit, and the coefficients of one grouping, n_groups blocks of group_width
// coordinates, of which a unit reads only those of its own group.

#include <vector>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief A block of design entries of fixed width that feeds one linear predictor.
 *
 * Stores \f$n \times w\f$ entries in row-major order, where \f$n\f$ is the number of units and
 * \f$w\f$ is `width`. For unit \f$i\f$ and column \f$j \in \{0, \dots, w - 1\}\f$, entry
 * \f$i w + j\f$ contributes \f$v_{ij} x_{c_{ij}}\f$ to linear predictor `predictor`; entries
 * with `coefficient == -1` are empty and skipped.
 *
 * \see design_column(), design_basis(), design_group_effect(), \ref engine
 */
struct design_entries {
  int predictor = 0;                ///< the linear predictor \f$r\f$ fed
  int width = 1;                    ///< entries per unit
  std::vector<double> value;        ///< \f$n \times w\f$, row-major
  std::vector<int> coefficient;     ///< \f$n \times w\f$, row-major: a coordinate, or \f$-1\f$ for none
};

/**
 * \ingroup engine
 * \brief Builds a width-1 design block for a single covariate column.
 *
 * Unit \f$i\f$ contributes `x[i]` to linear predictor `predictor` at coordinate `coefficient`.
 * Runs on the host.
 *
 * \param predictor Index of the linear predictor \f$r\f$ fed by this column.
 * \param x Covariate values of length \f$n\f$.
 * \param coefficient Global coordinate index (or `-1` for an empty column).
 * \return A width-1 design_entries block of length `x.size()`.
 *
 * \see design_entries, design_basis(), \ref engine
 */
inline design_entries design_column(int predictor, const std::vector<double>& x, int coefficient) {
  design_entries d;
  d.predictor = predictor;
  d.value = x;
  d.coefficient.assign(x.size(), coefficient);
  return d;
}

/**
 * \ingroup engine
 * \brief Evaluates a univariate basis at a covariate vector into a fixed-width design block.
 *
 * Calls `basis.row(x[i], ...)` for each unit \f$i\f$, giving width `basis.row_width()`.
 * Active basis column \f$c = (f + j) \bmod K\f$ (with \f$K = \text{basis.dimension}()\f$) maps to
 * coordinate `first_coefficient + c`, or to `-1` when \f$c\f$ appears in `dropped` for identifiability.
 * Runs on the host.
 *
 * \tparam Basis Basis type providing `row_width()`, `dimension()`, and `row(double, double*)`.
 * \param predictor Index of the linear predictor \f$r\f$ fed by the basis columns.
 * \param basis Host basis instance.
 * \param x Covariate values of length \f$n\f$.
 * \param first_coefficient Starting coordinate index for the \f$K\f$ basis coefficients.
 * \param dropped Basis column indices in \f$[0, K)\f$ to omit (mapped to `-1`).
 * \return A design_entries block of width `basis.row_width()` and \f$n\f$ rows.
 *
 * \see design_entries, design_column(), \ref basis "Bases", \ref engine
 */
template <class Basis>
design_entries design_basis(int predictor, const Basis& basis, const std::vector<double>& x, int first_coefficient,
                            const std::vector<int>& dropped = {}) {
  design_entries d;
  d.predictor = predictor;
  d.width = basis.row_width();
  const int K = basis.dimension(), n = static_cast<int>(x.size());
  d.value.resize(static_cast<std::size_t>(n) * d.width);
  d.coefficient.resize(d.value.size());
  std::vector<double> row(d.width);
  for (int i = 0; i < n; ++i) {
    const int f = basis.row(x[i], row.data());
    for (int j = 0; j < d.width; ++j) {
      const int c = (f + j) % K;
      bool drop = false;
      for (int k : dropped) drop = drop || k == c;
      d.value[i * d.width + j] = row[j];
      d.coefficient[i * d.width + j] = drop ? -1 : first_coefficient + c;
    }
  }
  return d;
}

}  // namespace laplace::engine
