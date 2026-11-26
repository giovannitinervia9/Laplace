#pragma once

// The row kernels of the outer criterion (laplace::engine; plan, sections 7.9
// and 7.10): the passes over the units that the derivatives of the Laplace
// marginal criterion (marginal_criterion.hpp) read, one kernel for each
// derivative order, which computes that order and nothing above it.
//
// For the objective F(x) = -sum_i w_i l_i(x) + penalties, write T for the
// third derivatives of its first term in x and Q for the fourth (the
// penalties are quadratic, so they have none), Hi for a symmetric N x N
// matrix (the inverse of the Hessian, with the rows and columns of held
// coordinates zero) and v_1, ..., v_A for directions in x. A unit reads Hi
// only between coordinates of its own block, which are positions of the
// information layout, so Hi is given on that layout (one number per
// position): a dense inverse or a selected one gives it alike. The passes are
//
//   order 3   tau_c = sum_rs Hi_rs T_rsc                  (N numbers)
//   order 4   tau, and for each direction a and pair a <= b
//             T[v_a]_rs = sum_c T_rsc v_a,c               (on the information layout)
//             T[v_a, v_b]_c = sum_rs T_crs v_a,r v_b,s    (N numbers)
//             q_ab = sum_rstu Hi_rs Q_rstu v_a,t v_b,u    (one number)
//
// For one unit, with X the design row (X_sp the value of the entry of local
// slot s that feeds predictor p) and L3, L4 the third and fourth derivatives
// of -w_i l_i in the linear predictors, the contractions are taken once for
// the unit (plan, 7.9 point 4):
//
//   M_pq = sum_rs X_rp Hi_rs X_sq,   xi_a,p = sum_s X_sp v_a,s,
//   tau_c       += sum_t (sum_pq L3_pqt M_pq) X_ct
//   T[v_a]_rs   += sum_pq (sum_t L3_pqt xi_a,t) X_rp X_sq
//   T[v_a,v_b]_c += sum_t (sum_pq L3_pqt xi_a,p xi_b,q) X_ct
//   q_ab        += sum_tu (sum_pq L4_pqtu M_pq) xi_a,t xi_b,u
//
// Output, in one device vector: tau (N); then for order 4 the A matrices T[v_a]
// in the information layout of row_model.hpp, the A (A + 1) / 2 vectors
// T[v_a, v_b] (pairs a <= b in lexicographic order) and the A (A + 1) / 2
// numbers q_ab.
//
// Hi is read from device memory for each pair of design entries of a unit, at
// row_model_view::information_position() of the pair of local slots; the
// local sums of a block live on the thread's stack, so the local slots of a
// block are at most row_outer_local_capacity and the directions at most
// row_outer_direction_capacity.

#include <laplace/engine/row_kernels.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <stdexcept>
#include <utility>

namespace laplace::engine {

/**
 * \ingroup engine
 * \brief Largest number of local slots of a block (`n_global + group_width`) in the passes of the outer criterion.
 * \see assemble_outer_rows(), \ref engine
 */
// ponytail: the local sums of the order-4 pass live on the thread's stack (A x 136 numbers at 16
// slots); more slots need team scratch memory.
inline constexpr int row_outer_local_capacity = 16;
/**
 * \ingroup engine
 * \brief Largest number of outer directions in the order-4 pass of the outer criterion.
 * \see assemble_outer_rows(), \ref engine
 */
inline constexpr int row_outer_direction_capacity = 4;

/**
 * \ingroup engine
 * \brief Device matrix read by the outer passes: the directions (\f$N \times A\f$), column-major.
 * \see assemble_outer_rows(), \ref engine
 */
using outer_matrix_view = Kokkos::View<const double**, Kokkos::LayoutLeft, row_model_view::mem>;

/**
 * \ingroup engine
 * \brief Device vector read by the outer passes as \f$H^{-1}\f$ on the information layout: the entry at position \f$q\f$ is \f$H^{-1}_{rc}\f$ with \f$r\f$ = `row_model::information_row[q]` and \f$c\f$ = `information_column[q]`.
 * \see assemble_outer_rows(), \ref engine
 */
using outer_information_view = Kokkos::View<const double*, row_model_view::mem>;

/**
 * \ingroup engine
 * \brief Sizes of the output of an outer pass: \f$N\f$ coordinates, \f$A\f$ directions, the information layout.
 * \see assemble_outer_rows(), \ref engine
 */
struct outer_pass_layout {
  int n = 0, n_directions = 0, information = 0;
  /**
   * \ingroup engine
   * \brief Number of sorted pairs \f$0 \le a \le b < A\f$ of outer directions.
   * \return \f$A(A + 1)/2\f$, where \f$A = \f$ `n_directions`.
   */
  LAPLACE_HD int n_pairs() const { return n_directions * (n_directions + 1) / 2; }
  /**
   * \ingroup engine
   * \brief Offset of the vector \f$\tau\f$ of length \f$N\f$ in the pass output.
   * \return Zero.
   */
  LAPLACE_HD int tau() const { return 0; }
  /**
   * \ingroup engine
   * \brief Offset of the matrix \f$T[v_a]\f$, in the information layout, for direction `a` in an order-4 pass output.
   * \param a Direction index \f$0 \le a < \f$ `n_directions`.
   * \return Zero-based offset \f$N + a \cdot \f$ `information`.
   */
  LAPLACE_HD int third_direction(int a) const { return n + a * information; }
  /**
   * \ingroup engine
   * \brief Offset of the vector \f$T[v_a, v_b]\f$ of length \f$N\f$ for pair index `ab` in an order-4 pass output.
   * \param ab Lexicographic pair index \f$0 \le \mathrm{ab} < \f$ `n_pairs()`.
   * \return Zero-based offset \f$N + A \cdot \f$ `information` \f$+ \mathrm{ab} \cdot N\f$.
   */
  LAPLACE_HD int third_pair(int ab) const { return n + n_directions * information + ab * n; }
  /**
   * \ingroup engine
   * \brief Offset of the scalar \f$q_{ab}\f$ for pair index `ab` in an order-4 pass output.
   * \param ab Lexicographic pair index \f$0 \le \mathrm{ab} < \f$ `n_pairs()`.
   * \return Zero-based offset \f$N + A \cdot \f$ `information` \f$+ \f$ `n_pairs()` \f$\cdot N + \mathrm{ab}\f$.
   */
  LAPLACE_HD int fourth_pair(int ab) const { return n + n_directions * information + n_pairs() * n + ab; }
  /**
   * \ingroup engine
   * \brief Total number of scalars in the output of the outer pass of the given order.
   * \param order Pass order (`3` or `4`).
   * \return `n` when `order == 3`, otherwise `fourth_pair(n_pairs())`.
   */
  LAPLACE_HD int size(int order) const { return order == 3 ? n : fourth_pair(n_pairs()); }
};

namespace detail_outer {

template <class D, int R, int Order>
struct outer_sizes {
  static constexpr int P = D::n_params;
  static constexpr int LC = row_outer_local_capacity;
  static constexpr int AC = row_outer_direction_capacity;
  static constexpr int NH = Order == 4 ? LC * (LC + 1) / 2 : 1;
  static constexpr int NA = Order == 4 ? AC : 1;
  static constexpr int NAA = Order == 4 ? AC * (AC + 1) / 2 : 1;
};

}  // namespace detail_outer

/**
 * \ingroup engine
 * \brief Executes the outer pass of order 3 or 4 for distribution `D` and \f$R\f$ predictors, adding into `acc`.
 *
 * \tparam D Distribution type satisfying \ref distrib_core, with log-density derivatives to order 4.
 * \tparam R Number of linear predictors (\f$1 \le R \le 4\f$).
 * \tparam Order 3 (the vector \f$\tau\f$) or 4 (also the direction terms of the header).
 * \param M Built device model view (`n_local() <= row_outer_local_capacity`).
 * \param x Device coordinates, length \f$N\f$.
 * \param Hi Device \f$H^{-1}\f$ on the information layout (outer_information_view).
 * \param V Device \f$N \times A\f$ directions (read when `Order == 4`).
 * \param n_directions \f$A \le\f$ `row_outer_direction_capacity` (0 when `Order == 3`).
 * \param acc Device accumulator of length outer_pass_layout::size().
 *
 * \see outer_pass_layout, assemble_outer_row_model(), \ref engine
 */
template <class D, int R, int Order>
void assemble_outer_rows(const row_model_view& M, const Kokkos::View<const double*, row_model_view::mem>& x,
                         const outer_information_view& Hi,
                         const outer_matrix_view& V,
                         int n_directions, const row_model_view::dview& acc) {
  static_assert(Order == 3 || Order == 4, "an outer pass has order 3 or 4");
  using exec = row_model_view::exec;
  using SV = Kokkos::Experimental::ScatterView<double*, typename row_model_view::dview::array_layout, exec>;
  using S = detail_outer::outer_sizes<D, R, Order>;
  using Val = predictor_value<R, Order>;
  static_assert(S::P <= row_param_capacity, "the distrib has more parameters than row_param_capacity");
  const outer_pass_layout out{M.n_coordinates(), n_directions, M.information_size};
  const int nA = n_directions;
  SV sv(acc);
  Kokkos::parallel_for(
      "laplace::engine::assemble_outer_rows", Kokkos::RangePolicy<exec>(0, M.n_blocks), KOKKOS_LAMBDA(const int b) {
        auto a = sv.access();
        (void)out, (void)nA, (void)V;  // captured here: nvcc refuses a first capture inside if constexpr
        const int L = M.n_local();
        int gi[S::LC];
        double xl[S::LC], tau[S::LC];
        for (int s = 0; s < L; ++s) {
          gi[s] = M.global_index(s, b);
          xl[s] = x(gi[s]);
          tau[s] = 0.0;
        }
        double hd[S::NA][S::NH], rh[S::NAA][S::LC], qq[S::NAA], vl[S::NA][S::LC];
        if constexpr (Order == 4) {
          for (int d = 0; d < nA; ++d) {
            for (int q = 0; q < L * (L + 1) / 2; ++q) hd[d][q] = 0.0;
            for (int s = 0; s < L; ++s) vl[d][s] = V(gi[s], d);
          }
          for (int p = 0; p < nA * (nA + 1) / 2; ++p) {
            qq[p] = 0.0;
            for (int s = 0; s < L; ++s) rh[p][s] = 0.0;
          }
        }
        for (int i = M.block_begin(b); i < M.block_end(b); ++i) {
          double eta[R];
          for (int r = 0; r < R; ++r) eta[r] = M.has_offset ? M.offset(i, r) : 0.0;
          for (int e = 0; e < M.width; ++e) {
            const int s = M.design_slot(i, e);
            if (s >= 0) eta[M.entry_predictor(e)] += M.design_value(i, e) * xl[s];
          }
          const double y = M.y(i), wi = M.has_weight ? M.weight(i) : 1.0;
          Val val[row_value_capacity];
          for (int r = 0; r < R; ++r) val[r] = linear_predictor_value<R, Order>(r, eta[r]);
          detail_rows::walk_nodes<R, Order>(M, i, val);
          const Val* pin[S::P];
          double th[S::P];
          for (int j = 0; j < S::P; ++j) {
            pin[j] = &val[M.param_value[j]];
            th[j] = pin[j]->v;
          }
          const D d = detail_rows::make_distrib<D>(th, std::make_index_sequence<S::P>{});
          const auto l1 = detail_rows::log_density_d1(d, y);
          const auto l2 = detail_rows::log_density_d2(d, y);
          const auto l3 = detail_rows::log_density_d3(d, y);
          Val o;
          chain_rule_d1<R, S::P>(pin, l1.v, o);
          chain_rule_d3<R, S::P>(pin, l1.v, l2.v, l3.v, o);
          // L3 of -w l, full
          double L3[R][R][R];
          for (int p = 0; p < R; ++p)
            for (int q = p; q < R; ++q)
              for (int t = q; t < R; ++t) {
                const double v = -wi * o.d3[packed_triple<R>(p, q, t)];
                L3[p][q][t] = L3[p][t][q] = L3[q][p][t] = L3[q][t][p] = L3[t][p][q] = L3[t][q][p] = v;
              }
          // M_pq over the pairs of design entries
          double Mq[R][R];
          for (int p = 0; p < R; ++p)
            for (int q = 0; q < R; ++q) Mq[p][q] = 0.0;
          for (int e1 = 0; e1 < M.width; ++e1) {
            const int s1 = M.design_slot(i, e1);
            if (s1 < 0) continue;
            const double v1 = M.design_value(i, e1);
            const int p1 = M.entry_predictor(e1);
            for (int e2 = 0; e2 < M.width; ++e2) {
              const int s2 = M.design_slot(i, e2);
              if (s2 < 0) continue;
              const double h = Hi(s1 >= s2 ? M.information_position(b, s1, s2) : M.information_position(b, s2, s1));
              Mq[p1][M.entry_predictor(e2)] += v1 * M.design_value(i, e2) * h;
            }
          }
          double c3[R];
          for (int t = 0; t < R; ++t) {
            double v = 0.0;
            for (int p = 0; p < R; ++p)
              for (int q = 0; q < R; ++q) v += L3[p][q][t] * Mq[p][q];
            c3[t] = v;
          }
          for (int e = 0; e < M.width; ++e) {
            const int s = M.design_slot(i, e);
            if (s >= 0) tau[s] += M.design_value(i, e) * c3[M.entry_predictor(e)];
          }
          if constexpr (Order == 4) {
            const auto l4 = detail_rows::log_density_d4(d, y);
            chain_rule_d4<R, S::P>(pin, l1.v, l2.v, l3.v, l4.v, o);
            // xi_a,p
            double xi[S::NA][R];
            for (int dd = 0; dd < nA; ++dd)
              for (int p = 0; p < R; ++p) xi[dd][p] = 0.0;
            for (int e = 0; e < M.width; ++e) {
              const int s = M.design_slot(i, e);
              if (s < 0) continue;
              const double v = M.design_value(i, e);
              const int p = M.entry_predictor(e);
              for (int dd = 0; dd < nA; ++dd) xi[dd][p] += v * vl[dd][s];
            }
            // N4_tu = sum_pq L4_pqtu M_pq, with L4 unpacked to the full R^4 array
            double L4[R][R][R][R], N4[R][R];
            detail_tensor::unpack4<R>(o.d4, L4);
            for (int t = 0; t < R; ++t)
              for (int u = 0; u < R; ++u) {
                double v = 0.0;
                for (int p = 0; p < R; ++p)
                  for (int q = 0; q < R; ++q) v += L4[p][q][t][u] * Mq[p][q];
                N4[t][u] = -wi * v;
              }
            int ab = 0;
            for (int da = 0; da < nA; ++da)
              for (int db = da; db < nA; ++db, ++ab) {
                double z[R], qv = 0.0;
                for (int t = 0; t < R; ++t) {
                  double v = 0.0;
                  for (int p = 0; p < R; ++p)
                    for (int q = 0; q < R; ++q) v += L3[p][q][t] * xi[da][p] * xi[db][q];
                  z[t] = v;
                  for (int u = 0; u < R; ++u) qv += N4[t][u] * xi[da][t] * xi[db][u];
                }
                qq[ab] += qv;
                for (int e = 0; e < M.width; ++e) {
                  const int s = M.design_slot(i, e);
                  if (s >= 0) rh[ab][s] += M.design_value(i, e) * z[M.entry_predictor(e)];
                }
              }
            for (int dd = 0; dd < nA; ++dd) {
              double C[R][R];
              for (int p = 0; p < R; ++p)
                for (int q = 0; q < R; ++q) {
                  double v = 0.0;
                  for (int t = 0; t < R; ++t) v += L3[p][q][t] * xi[dd][t];
                  C[p][q] = v;
                }
              for (int e1 = 0; e1 < M.width; ++e1) {
                const int s1 = M.design_slot(i, e1);
                if (s1 < 0) continue;
                const double v1 = M.design_value(i, e1);
                const int p1 = M.entry_predictor(e1);
                for (int e2 = 0; e2 <= e1; ++e2) {
                  const int s2 = M.design_slot(i, e2);
                  if (s2 < 0) continue;
                  double c = v1 * M.design_value(i, e2) * C[p1][M.entry_predictor(e2)];
                  if (e1 != e2 && s1 == s2) c *= 2.0;  // both orders of the pair fall on the diagonal
                  hd[dd][s1 >= s2 ? detail_rows::lower_index(s1, s2) : detail_rows::lower_index(s2, s1)] += c;
                }
              }
            }
          }
        }
        for (int s = 0; s < L; ++s) a(out.tau() + gi[s]) += tau[s];
        if constexpr (Order == 4) {
          for (int dd = 0; dd < nA; ++dd)
            for (int r = 0; r < L; ++r)
              for (int c = 0; c <= r; ++c) {
                const int pos = M.information_position(b, r, c);
                a(out.third_direction(dd) + pos) += hd[dd][detail_rows::lower_index(r, c)];
              }
          for (int p = 0; p < nA * (nA + 1) / 2; ++p) {
            for (int s = 0; s < L; ++s) a(out.third_pair(p) + gi[s]) += rh[p][s];
            a(out.fourth_pair(p)) += qq[p];
          }
        }
      });
  Kokkos::Experimental::contribute(acc, sv);
}

namespace detail_outer {

template <class D, int R>
void run_order(const row_model& m, int order, const Kokkos::View<const double*, row_model_view::mem>& x,
               const outer_information_view& Hi,
               const outer_matrix_view& V, int nA,
               const row_model_view::dview& acc) {
  if (order == 3) assemble_outer_rows<D, R, 3>(m.view, x, Hi, V, nA, acc);
  else assemble_outer_rows<D, R, 4>(m.view, x, Hi, V, nA, acc);
}

template <class D>
void run_distrib(const row_model& m, int order, const Kokkos::View<const double*, row_model_view::mem>& x,
                 const outer_information_view& Hi,
                 const outer_matrix_view& V, int nA,
                 const row_model_view::dview& acc) {
  if (m.view.P != D::n_params) throw std::invalid_argument("assemble_outer_row_model: the model was built for another distrib");
  switch (m.view.R) {
    case 1: run_order<D, 1>(m, order, x, Hi, V, nA, acc); break;
    case 2: run_order<D, 2>(m, order, x, Hi, V, nA, acc); break;
    case 3: run_order<D, 3>(m, order, x, Hi, V, nA, acc); break;
    case 4: run_order<D, 4>(m, order, x, Hi, V, nA, acc); break;
    default: throw std::invalid_argument("assemble_outer_row_model: R must be 1 to 4");
  }
}

}  // namespace detail_outer

/**
 * \ingroup engine
 * \brief Runs the outer pass of order 3 or 4 and copies its output (outer_pass_layout) to the host.
 *
 * Switches on `m.distrib` within `List` and on \f$R\f$. Throws `std::invalid_argument` when the order is
 * not 3 or 4, when the block has more than `row_outer_local_capacity` local slots, or when there are
 * more than `row_outer_direction_capacity` directions.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \param m Built model.
 * \param order 3 or 4.
 * \param x Device coordinates.
 * \param Hi Device \f$H^{-1}\f$ on the information layout (outer_information_view).
 * \param V Device \f$N \times A\f$ directions (order 4).
 * \param n_directions \f$A\f$ (order 4; ignored at order 3).
 * \param out Host vector receiving outer_pass_layout::size() numbers.
 *
 * \see assemble_outer_rows(), outer_pass_layout, \ref engine
 */
template <class List>
void assemble_outer_row_model(const row_model& m, int order, const Kokkos::View<const double*, row_model_view::mem>& x,
                              const outer_information_view& Hi,
                              const outer_matrix_view& V,
                              int n_directions, std::vector<double>& out) {
  if (order != 3 && order != 4) throw std::invalid_argument("assemble_outer_row_model: the order must be 3 or 4");
  if (m.view.n_local() > row_outer_local_capacity)
    throw std::invalid_argument("assemble_outer_row_model: more local slots than row_outer_local_capacity");
  const int nA = order == 4 ? n_directions : 0;
  if (nA < 0 || nA > row_outer_direction_capacity)
    throw std::invalid_argument("assemble_outer_row_model: more directions than row_outer_direction_capacity");
  const outer_pass_layout lo{m.n_coordinates(), nA, m.view.information_size};
  row_model_view::dview acc("laplace::engine::outer_pass", lo.size(order));
  bool found = false;
  [&]<class... Ds>(distrib_list<Ds...>) {
    int k = 0;
    ((k++ == m.distrib ? (detail_outer::run_distrib<Ds>(m, order, x, Hi, V, nA, acc), found = true) : false), ...);
  }(List{});
  if (!found) throw std::invalid_argument("assemble_outer_row_model: the distrib index is outside the list");
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc);
  out.assign(h.data(), h.data() + lo.size(order));
}

}  // namespace laplace::engine
