#pragma once

// The nonlinear pointwise nodes of the general route (laplace::engine; plan,
// section 7.8): functions of a few values of the same unit, walked for each
// unit with a switch on their kind. One function per derivative order, each
// computing only its own order (plan, section 7.9, point 2).
//
//   link            o = g^-1(u), g a scalar_bijector (bijector/scalar.hpp):
//                   the inverse link from a predictor to a parameter
//   logistic_curve  o = A / (1 + exp((x_m - t) / s)), inputs (A, x_m, s) and a
//                   data column t: the three-parameter logistic growth curve
//
// For the logistic curve, with z = (x_m - t) / s, p = 1 / (1 + e^z),
// q = p (1 - p) and w = 1 - 2p, the derivatives in (A, x_m, s) are
//   f_A = p,  f_xm = -A q / s,  f_s = A q z / s,
//   f_AA = 0, f_A,xm = -q / s, f_A,s = q z / s,
//   f_xm,xm = A q w / s^2, f_xm,s = A q (1 - w z) / s^2, f_s,s = A q z (w z - 2) / s^2;
// orders three and four (logistic_curve_d3, logistic_curve_d4) follow from
// f = A h(x_m, s), h = g(z), with the derivatives of g = 1 / (1 + e^z)
//   g1 = -q, g2 = w q, g3 = 2 q^2 - w^2 q, g4 = w^3 q - 8 w q^2
// (all derived by SymPy in tools/reference/engine_nodes.py, which checks them
// against SymPy's own derivatives).

#include <laplace/bijector/scalar.hpp>
#include <laplace/core/config.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Kind of a nonlinear pointwise node.
 *
 * Selects whether a pointwise_node evaluates the inverse of a scalar bijector (`link`)
 * or the three-parameter logistic growth curve (`logistic_curve`).
 *
 * \see pointwise_node, \ref engine
 */
enum class pointwise_node_kind : int { link = 0, logistic_curve = 1 };

/**
 * \ingroup engine
 * \brief Largest number of inputs of a pointwise node.
 *
 * Set to 3 for `logistic_curve` inputs \f$(A, x_m, s)\f$; `link` nodes use 1 input.
 *
 * \see pointwise_node, \ref engine
 */
inline constexpr int pointwise_node_input_capacity = 3;

/**
 * \ingroup engine
 * \brief A nonlinear pointwise node: its kind, the values that feed its inputs, and its data column.
 *
 * Values are numbered as in row_model_spec: the linear predictors \f$0, \dots, R - 1\f$, then the
 * output of node \f$k\f$ as value \f$R + k\f$. Every input of node \f$k\f$ must reference an earlier
 * value index \f$< R + k\f$. A `link` node reads `input[0]` and applies `link.inverse()`; a
 * `logistic_curve` node reads `(input[0], input[1], input[2])` as \f$(A, x_m, s)\f$ together with
 * data column `data_column` as \f$t\f$.
 *
 * \see pointwise_node_kind, logistic_curve_value(), \ref bijector_scalar "Scalar bijectors", \ref engine
 */
struct pointwise_node {
  pointwise_node_kind kind = pointwise_node_kind::link;
  int input[pointwise_node_input_capacity] = {-1, -1, -1};
  int data_column = -1;               ///< logistic_curve: the column of t
  bijector::scalar_bijector link{};   ///< link: g, whose inverse is applied

  /**
   * \ingroup engine
   * \brief Number of value inputs read by this node (`1` for `link`, `3` for `logistic_curve`).
   * \return Input count for `kind`.
   */
  LAPLACE_HD int n_inputs() const { return kind == pointwise_node_kind::link ? 1 : 3; }
};

namespace detail_node {

struct logistic_state {
  double p, q, w, z, is;
};
LAPLACE_HD logistic_state logistic_curve_state(double xm, double s, double t) {
  logistic_state st;
  st.z = (xm - t) / s;
  st.p = 1.0 / (1.0 + Kokkos::exp(st.z));
  st.q = st.p * (1.0 - st.p);
  st.w = 1.0 - 2.0 * st.p;
  st.is = 1.0 / s;
  return st;
}

}  // namespace detail_node

/**
 * \ingroup engine
 * \brief Evaluates the three-parameter logistic growth curve \f$f(A, x_m, s; t) = A / (1 + e^{(x_m - t)/s})\f$.
 *
 * Requires \f$s \ne 0\f$; division by zero or non-finite inputs follow IEEE 754 arithmetic.
 *
 * \param A Asymptote parameter \f$A\f$.
 * \param xm Inflection point \f$x_m\f$.
 * \param s Scale parameter \f$s\f$.
 * \param t Data covariate \f$t\f$.
 * \return Curve value \f$A / (1 + e^{(x_m - t)/s})\f$.
 *
 * \see logistic_curve_d1(), logistic_curve_d2(), \ref engine
 */
LAPLACE_HD double logistic_curve_value(double A, double xm, double s, double t) {
  return A / (1.0 + Kokkos::exp((xm - t) / s));
}
/**
 * \ingroup engine
 * \brief Evaluates the first partial derivatives of the logistic growth curve in \f$(A, x_m, s)\f$.
 *
 * With \f$z = (x_m - t)/s\f$, \f$p = 1/(1 + e^z)\f$, and \f$q = p(1 - p)\f$, writes
 * \f$f_A = p\f$, \f$f_{x_m} = -A q / s\f$, and \f$f_s = A q z / s\f$ into `f1[0..2]`.
 *
 * \param A Asymptote parameter \f$A\f$.
 * \param xm Inflection point \f$x_m\f$.
 * \param s Scale parameter \f$s\f$ (\f$s \ne 0\f$).
 * \param t Data covariate \f$t\f$.
 * \param f1 Output array of length 3 receiving \f$(f_A, f_{x_m}, f_s)\f$.
 *
 * \see logistic_curve_value(), logistic_curve_d2(), \ref engine
 */
LAPLACE_HD void logistic_curve_d1(double A, double xm, double s, double t, double* f1) {
  const auto st = detail_node::logistic_curve_state(xm, s, t);
  f1[0] = st.p;
  f1[1] = -A * st.q * st.is;
  f1[2] = A * st.q * st.z * st.is;
}
/**
 * \ingroup engine
 * \brief Evaluates the second partial derivatives of the logistic growth curve in \f$(A, x_m, s)\f$.
 *
 * With \f$z = (x_m - t)/s\f$, \f$p = 1/(1 + e^z)\f$, \f$q = p(1 - p)\f$, and \f$w = 1 - 2p\f$,
 * writes the six derivatives packed on sorted pairs in lexicographic order into `f2[0..5]`:
 * \f$f_{AA} = 0\f$, \f$f_{A, x_m} = -q / s\f$, \f$f_{A, s} = q z / s\f$,
 * \f$f_{x_m, x_m} = A q w / s^2\f$, \f$f_{x_m, s} = A q (1 - w z) / s^2\f$, and
 * \f$f_{s, s} = A q z (w z - 2) / s^2\f$ (verified against SymPy in `tools/reference/engine_nodes.py`).
 *
 * \param A Asymptote parameter \f$A\f$.
 * \param xm Inflection point \f$x_m\f$.
 * \param s Scale parameter \f$s\f$ (\f$s \ne 0\f$).
 * \param t Data covariate \f$t\f$.
 * \param f2 Output array of length 6 receiving \f$(f_{AA}, f_{A,x_m}, f_{A,s}, f_{x_m,x_m}, f_{x_m,s}, f_{s,s})\f$.
 *
 * \see logistic_curve_value(), logistic_curve_d1(), \ref engine
 */
LAPLACE_HD void logistic_curve_d2(double A, double xm, double s, double t, double* f2) {
  const auto st = detail_node::logistic_curve_state(xm, s, t);
  const double is2 = st.is * st.is;
  f2[0] = 0.0;                                         // A A
  f2[1] = -st.q * st.is;                               // A xm
  f2[2] = st.q * st.z * st.is;                         // A s
  f2[3] = A * st.q * st.w * is2;                       // xm xm
  f2[4] = A * st.q * (1.0 - st.w * st.z) * is2;        // xm s
  f2[5] = A * st.q * st.z * (st.w * st.z - 2.0) * is2; // s s
}

namespace detail_node {

// The derivatives of g(z) = 1 / (1 + e^z) to order four in p, q, w
struct logistic_g {
  double g1, g2, g3, g4;
};
LAPLACE_HD logistic_g logistic_curve_g(const logistic_state& st) {
  const double q = st.q, w = st.w;
  return {-q, w * q, 2.0 * q * q - w * w * q, w * w * w * q - 8.0 * w * q * q};
}

}  // namespace detail_node

/**
 * \ingroup engine
 * \brief Evaluates the third partial derivatives of the logistic growth curve in \f$(A, x_m, s)\f$.
 *
 * Writes the ten derivatives packed on sorted triples of the inputs \f$(A, x_m, s)\f$ in lexicographic
 * order into `f3[0..9]`. With \f$f = A\,h(x_m, s)\f$, \f$h = g(z)\f$, \f$z = (x_m - t)/s\f$ and
 * \f$g(z) = 1/(1 + e^z)\f$, a triple with two or three \f$A\f$ is zero, one with one \f$A\f$ is a second
 * derivative of \f$h\f$, and the others are \f$A\f$ times a third derivative of \f$h\f$:
 * \f$h_{x_m x_m x_m} = g_3/s^3\f$, \f$h_{x_m x_m s} = -(2 g_2 + g_3 z)/s^3\f$,
 * \f$h_{x_m s s} = (2 g_1 + 4 g_2 z + g_3 z^2)/s^3\f$, \f$h_{sss} = -z (6 g_1 + 6 g_2 z + g_3 z^2)/s^3\f$, where
 * \f$g_1 = -q\f$, \f$g_2 = w q\f$, \f$g_3 = 2 q^2 - w^2 q\f$ are the derivatives of \f$g\f$ (derived by SymPy in
 * `tools/reference/engine_nodes.py`).
 *
 * \param A Asymptote parameter \f$A\f$.
 * \param xm Inflection point \f$x_m\f$.
 * \param s Scale parameter \f$s\f$ (\f$s \ne 0\f$).
 * \param t Data covariate \f$t\f$.
 * \param f3 Output array of length 10.
 *
 * \see logistic_curve_d2(), logistic_curve_d4(), \ref engine
 */
LAPLACE_HD void logistic_curve_d3(double A, double xm, double s, double t, double* f3) {
  const auto st = detail_node::logistic_curve_state(xm, s, t);
  const auto g = detail_node::logistic_curve_g(st);
  const double z = st.z, is2 = st.is * st.is, is3 = is2 * st.is;
  f3[0] = 0.0, f3[1] = 0.0, f3[2] = 0.0;                         // AAA AAxm AAs
  f3[3] = g.g2 * is2;                                            // A xm xm
  f3[4] = -(g.g1 + g.g2 * z) * is2;                              // A xm s
  f3[5] = z * (2.0 * g.g1 + g.g2 * z) * is2;                     // A s s
  f3[6] = A * g.g3 * is3;                                        // xm xm xm
  f3[7] = -A * (2.0 * g.g2 + g.g3 * z) * is3;                    // xm xm s
  f3[8] = A * (2.0 * g.g1 + z * (4.0 * g.g2 + g.g3 * z)) * is3;  // xm s s
  f3[9] = -A * z * (6.0 * g.g1 + z * (6.0 * g.g2 + g.g3 * z)) * is3;  // s s s
}

/**
 * \ingroup engine
 * \brief Evaluates the fourth partial derivatives of the logistic growth curve in \f$(A, x_m, s)\f$.
 *
 * Writes the fifteen derivatives packed on sorted quadruples of the inputs \f$(A, x_m, s)\f$ in
 * lexicographic order into `f4[0..14]`. In the notation of logistic_curve_d3(), a quadruple with two or
 * more \f$A\f$ is zero, one with one \f$A\f$ is a third derivative of \f$h\f$, and the others are \f$A\f$
 * times \f$h_{x_m^4} = g_4/s^4\f$, \f$h_{x_m^3 s} = -(3 g_3 + g_4 z)/s^4\f$,
 * \f$h_{x_m^2 s^2} = (6 g_2 + 6 g_3 z + g_4 z^2)/s^4\f$,
 * \f$h_{x_m s^3} = -(6 g_1 + 18 g_2 z + 9 g_3 z^2 + g_4 z^3)/s^4\f$ and
 * \f$h_{s^4} = z (24 g_1 + 36 g_2 z + 12 g_3 z^2 + g_4 z^3)/s^4\f$, with \f$g_4 = w^3 q - 8 w q^2\f$
 * (derived by SymPy in `tools/reference/engine_nodes.py`).
 *
 * \param A Asymptote parameter \f$A\f$.
 * \param xm Inflection point \f$x_m\f$.
 * \param s Scale parameter \f$s\f$ (\f$s \ne 0\f$).
 * \param t Data covariate \f$t\f$.
 * \param f4 Output array of length 15.
 *
 * \see logistic_curve_d3(), \ref engine
 */
LAPLACE_HD void logistic_curve_d4(double A, double xm, double s, double t, double* f4) {
  const auto st = detail_node::logistic_curve_state(xm, s, t);
  const auto g = detail_node::logistic_curve_g(st);
  const double z = st.z, is2 = st.is * st.is, is3 = is2 * st.is, is4 = is3 * st.is;
  for (int k = 0; k < 6; ++k) f4[k] = 0.0;                                           // two or more A
  f4[6] = g.g3 * is3;                                                                // A xm xm xm
  f4[7] = -(2.0 * g.g2 + g.g3 * z) * is3;                                            // A xm xm s
  f4[8] = (2.0 * g.g1 + z * (4.0 * g.g2 + g.g3 * z)) * is3;                          // A xm s s
  f4[9] = -z * (6.0 * g.g1 + z * (6.0 * g.g2 + g.g3 * z)) * is3;                     // A s s s
  f4[10] = A * g.g4 * is4;                                                           // xm^4
  f4[11] = -A * (3.0 * g.g3 + g.g4 * z) * is4;                                       // xm^3 s
  f4[12] = A * (6.0 * g.g2 + z * (6.0 * g.g3 + g.g4 * z)) * is4;                     // xm^2 s^2
  f4[13] = -A * (6.0 * g.g1 + z * (18.0 * g.g2 + z * (9.0 * g.g3 + g.g4 * z))) * is4;  // xm s^3
  f4[14] = A * z * (24.0 * g.g1 + z * (36.0 * g.g2 + z * (12.0 * g.g3 + g.g4 * z))) * is4;  // s^4
}

}  // namespace laplace::engine
