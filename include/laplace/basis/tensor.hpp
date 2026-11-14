#pragma once

// Tensor product bases (laplace::basis); the design was decided with Giovanni
// on 2026-11-14 (diario.md, decisions P1-P10).
//
// The margins are D >= 2 univariate bases b_1, ..., b_D, of any types; margin j
// has K_j columns and rows of width w_j. The product has K = prod_j K_j columns
//   B_c(x) = b_{1,j_1}(x_1) b_{2,j_2}(x_2) ... b_{D,j_D}(x_D),
//   c = sum_j s_j j_j,  s_D = 1,  s_j = s_{j+1} K_{j+1},
// the last margin varying fastest, as kron(b_1, ..., b_D) (P2).
//
// A row at a point x = (x_1, ..., x_D) is first[D], the first column of each
// margin, and W = prod_j w_j values, the last margin fastest; value
// (i_1, ..., i_D) belongs to column sum_j s_j ((f_j + i_j) mod K_j) (the
// modulus only matters for a cyclic margin), which column() computes.
//
// Partial derivatives and integrals take a multi-index k[D] (P3, P5): k_j in
// 0..6 is the order of the derivative in x_j, k_j = -1 the integral from
// lower_j to x_j. An integrated margin contributes all its K_j integrals
// (first column 0, width K_j): the integrals of the columns before a B-spline
// window are not zero, so a window alone would lose values of the product.
// Hence row_width(k) = prod_j w_j(k_j) and scratch_size(k) = sum_j w_j(k_j),
// with w_j(-1) = K_j.
//
// Every kernel function takes a scratch of scratch_size(k) values of T (P4):
// the marginal rows are written there, then their product to out. Outside its
// domain each margin follows its own rule (P9).
//
// Host functions: cross_gram (P6), the penalties of differential operators on
// the product (P7) and the Kronecker transforms (P8).

#include <laplace/basis/core.hpp>
#include <laplace/basis/operator.hpp>
#include <laplace/basis/transform.hpp>
#include <laplace/core/config.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace laplace::basis {

namespace detail_tensor {

template <std::size_t I, class B>
struct margin_leaf {
  B b;
};

template <class Seq, class... B>
struct margin_pack;

template <std::size_t... I, class... B>
struct margin_pack<std::index_sequence<I...>, B...> : margin_leaf<I, B>... {
  margin_pack() = default;
  explicit margin_pack(const B&... b) : margin_leaf<I, B>{b}... {}
};

template <class B>
constexpr bool is_cyclic = requires(const B& b) { b.piece_totals; };

template <class B>
constexpr bool is_windowed = requires(const B& b) { b.knots; b.template integral_column_total<double>(0); };

// int_lower^x b_j for all K columns j of a univariate basis
template <class B, class T>
LAPLACE_HD void integral_vector(const B& b, T x, T* out) {
  const int K = b.dimension();
  if constexpr (is_cyclic<B>) {
    if (!Kokkos::isfinite(x)) {
      for (int j = 0; j < K; ++j) out[j] = x - x;
      return;
    }
    T n;
    const T y = b.reduce(x, n);
    T w[bspline_max_degree + 1];
    const int fe = b.ext.integral_row(y, w);
    for (int j = 0; j < K; ++j) out[j] = b.column_integral(j, n, fe, w);
  } else if constexpr (is_windowed<B>) {
    T w[bspline_max_degree + 1];
    const int f = b.integral_row(x, w), d = b.degree;
    if (!(w[0] == w[0])) {  // NaN x or the nan rule: the whole vector
      for (int j = 0; j < K; ++j) out[j] = w[0];
      return;
    }
    // columns before the window have finished: their integral over the domain
    for (int j = 0; j < f; ++j) out[j] = b.template integral_column_total<T>(j);
    for (int q = 0; q <= d; ++q) out[f + q] = w[q];
    for (int j = f + d + 1; j < K; ++j) out[j] = T(0);
  } else {
    b.integral_row(x, out);  // dense: all K columns, first column 0
  }
}

// the row of order k (-1 = the integral vector) at x into out; returns first
template <class B, class T>
LAPLACE_HD int margin_row(const B& b, int k, T x, T* out) {
  if (k < 0) {
    integral_vector(b, x, out);
    return 0;
  }
  return basis_row_of_order(b, k, x, out);
}

template <class B>
LAPLACE_HD int margin_width(const B& b, int k) {
  return k < 0 ? b.dimension() : b.row_width();
}

// f(index constant, margin) for every margin of a product, on the host only
// (for the host functions, whose bodies throw and allocate)
template <class TB, class F, std::size_t... I>
void host_each(const TB& b, F& f, std::index_sequence<I...>) {
  (f(std::integral_constant<std::size_t, I>{}, b.template margin<I>()), ...);
}
template <class TB, class F>
void host_each(const TB& b, F&& f) {
  host_each(b, f, std::make_index_sequence<TB::margin_count>{});
}

}  // namespace detail_tensor

// The concept of a basis of several variables (P1). x, first and k hold one
// entry per margin.
template <class B>
concept multivariate_basis = requires(const B& b, const double* x, const int* k, int* first, double* out,
                                      const double* beta, double* scratch) {
  { B::margin_count } -> std::convertible_to<int>;
  { b.dimension() } -> std::convertible_to<int>;
  { b.row_width(k) } -> std::convertible_to<int>;
  { b.scratch_size(k) } -> std::convertible_to<int>;
  { b.row(x, first, out, scratch) } -> std::convertible_to<int>;
  { b.row_partial(x, k, first, out, scratch) } -> std::convertible_to<int>;
  { b.expansion(x, beta, scratch) } -> std::convertible_to<double>;
  { b.expansion_partial(x, k, beta, scratch) } -> std::convertible_to<double>;
  { b.column(k, first, 0) } -> std::convertible_to<int>;
};

// The tensor product of D >= 2 univariate bases, a small struct copied by
// value into kernels; it holds copies of the margins' structs, whose memory
// stays with their owners (pass owner.device() for device kernels and
// owner.host() for the host functions).
template <univariate_basis... B>
struct tensor_product_basis {
  static constexpr int margin_count = sizeof...(B);
  static_assert(margin_count >= 2, "tensor_product_basis: at least two margins");

  detail_tensor::margin_pack<std::index_sequence_for<B...>, B...> margins;

  tensor_product_basis() = default;
  explicit tensor_product_basis(const B&... b) : margins(b...) {}

  template <std::size_t I>
  LAPLACE_HD const auto& margin() const {
    using M = std::tuple_element_t<I, std::tuple<B...>>;
    return static_cast<const detail_tensor::margin_leaf<I, M>&>(margins).b;
  }

  // f(index constant, margin) for every margin, in order
  template <class F>
  LAPLACE_HD void for_each_margin(F&& f) const {
    each(f, std::index_sequence_for<B...>{});
  }

  LAPLACE_HD int dimension() const {
    int K = 1;
    for_each_margin([&](auto, const auto& m) { K *= m.dimension(); });
    return K;
  }

  LAPLACE_HD int row_width(const int* k) const {
    int W = 1;
    for_each_margin([&](auto I, const auto& m) { W *= detail_tensor::margin_width(m, k[I]); });
    return W;
  }
  LAPLACE_HD int row_width() const { return row_width(zero_order().data()); }

  LAPLACE_HD int scratch_size(const int* k) const {
    int S = 0;
    for_each_margin([&](auto I, const auto& m) { S += detail_tensor::margin_width(m, k[I]); });
    return S;
  }
  LAPLACE_HD int scratch_size() const { return scratch_size(zero_order().data()); }

  // The column of value number flat (0 <= flat < row_width(k)) of a row.
  LAPLACE_HD int column(const int* k, const int* first, int flat) const {
    int c = 0, stride = 1;
    reverse_each([&](auto I, const auto& m) {
      const int w = detail_tensor::margin_width(m, k[I]), K = m.dimension();
      const int i = flat % w;
      flat /= w;
      int j = first[I] + i;
      if (j >= K) j -= K;
      c += stride * j;
      stride *= K;
    });
    return c;
  }
  LAPLACE_HD int column(const int* first, int flat) const { return column(zero_order().data(), first, flat); }

  // The row of the partial derivative (or integral) of multi-index k at x:
  // first[D] and row_width(k) values in out; returns row_width(k).
  template <class T>
  LAPLACE_HD int row_partial(const T* x, const int* k, int* first, T* out, T* scratch) const {
    int width[margin_count], offset[margin_count];
    marginal_rows(x, k, first, width, offset, scratch);
    return kron_rows(width, offset, scratch, out);
  }
  template <class T>
  LAPLACE_HD int row(const T* x, int* first, T* out, T* scratch) const {
    return row_partial(x, zero_order().data(), first, out, scratch);
  }

  // sum_c beta_c d^k B_c(x), beta flat in the column order above
  template <class T>
  LAPLACE_HD T expansion_partial(const T* x, const int* k, const T* beta, T* scratch) const {
    int first[margin_count], width[margin_count], offset[margin_count];
    marginal_rows(x, k, first, width, offset, scratch);
    return contract(first, width, offset, beta, scratch);
  }
  template <class T>
  LAPLACE_HD T expansion(const T* x, const T* beta, T* scratch) const {
    return expansion_partial(x, zero_order().data(), beta, scratch);
  }

  // ---- host functions (call them on a product of host() margins)

  // G_kl = kron_j G_j(k_j, l_j), the Gram matrix of d^k B and d^l B over the
  // box, dense K x K column-major; k_j, l_j in 0..6 (P6)
  void cross_gram(const int* k, const int* l, double* out) const;

 private:
  // out = r_1 kron r_2 kron ... kron r_D from the marginal rows in scratch,
  // built in place: at each step the product of length L becomes L w; entry a
  // is read before position a w, which is >= a, is written, so the loop runs
  // from a = L - 1 down. Returns the width of the row.
  template <class T>
  LAPLACE_HD static int kron_rows(const int* width, const int* offset, const T* scratch, T* out) {
    int L = width[0];
    for (int b = 0; b < L; ++b) out[b] = scratch[offset[0] + b];
    for (int j = 1; j < margin_count; ++j) {
      const T* r = scratch + offset[j];
      const int w = width[j];
      for (int a = L - 1; a >= 0; --a) {
        const T p = out[a];
        for (int b = w - 1; b >= 0; --b) out[a * w + b] = p * r[b];
      }
      L *= w;
    }
    return L;
  }

  // sum_c beta_c B_c from the marginal rows in scratch
  template <class T>
  LAPLACE_HD T contract(const int* first, const int* width, const int* offset, const T* beta, const T* scratch) const {
    int dim[margin_count];
    for_each_margin([&](auto I, const auto& m) { dim[I] = m.dimension(); });
    // odometer over the first D - 1 margins; the last margin innermost reads
    // consecutive columns of beta
    constexpr int D = margin_count;
    int i[D] = {};
    const int wl = width[D - 1], fl = first[D - 1], Kl = dim[D - 1];
    const T* rl = scratch + offset[D - 1];
    T sum = T(0);
    while (true) {
      T p = T(1);
      int base = 0;
      for (int j = 0; j < D - 1; ++j) {
        p *= scratch[offset[j] + i[j]];
        int c = first[j] + i[j];
        if (c >= dim[j]) c -= dim[j];
        base = base * dim[j] + c;
      }
      base *= Kl;
      T s = T(0);
      for (int b = 0; b < wl; ++b) {
        int c = fl + b;
        if (c >= Kl) c -= Kl;
        s += rl[b] * beta[base + c];
      }
      sum += p * s;
      int j = D - 2;
      while (j >= 0 && ++i[j] == width[j]) i[j--] = 0;
      if (j < 0) break;
    }
    return sum;
  }

  template <class F, std::size_t... I>
  LAPLACE_HD void each(F& f, std::index_sequence<I...>) const {
    (f(std::integral_constant<std::size_t, I>{}, margin<I>()), ...);
  }
  template <class F>
  LAPLACE_HD void reverse_each(F&& f) const {
    reverse(f, std::make_index_sequence<margin_count>{});
  }
  template <class F, std::size_t... I>
  LAPLACE_HD void reverse(F& f, std::index_sequence<I...>) const {
    constexpr std::size_t D = margin_count;
    (f(std::integral_constant<std::size_t, D - 1 - I>{}, margin<D - 1 - I>()), ...);
  }
  LAPLACE_HD static Kokkos::Array<int, margin_count> zero_order() { return Kokkos::Array<int, margin_count>{}; }

  // the marginal rows of order k_j at x_j, one after the other in scratch
  template <class T>
  LAPLACE_HD void marginal_rows(const T* x, const int* k, int* first, int* width, int* offset, T* scratch) const {
    int o = 0;
    for_each_margin([&](auto I, const auto& m) {
      offset[I] = o;
      width[I] = detail_tensor::margin_width(m, k[I]);
      first[I] = detail_tensor::margin_row(m, k[I], x[I], scratch + o);
      o += width[I];
    });
  }
};

template <univariate_basis... B>
tensor_product_basis(const B&...) -> tensor_product_basis<B...>;

// ---- host: Kronecker products, Gram matrices, penalties, transforms

// A dense matrix, column-major, rows x cols.
struct kronecker_factor {
  const double* data;
  int rows, cols;
};

// out = F_1 kron F_2 kron ... kron F_n, column-major, (prod rows) x (prod cols).
inline void kronecker_product(const std::vector<kronecker_factor>& factors, double* out) {
  std::vector<double> acc{1.0}, next;
  std::size_t r = 1, c = 1;
  for (const auto& f : factors) {
    const std::size_t R = r * f.rows, C = c * f.cols;
    next.assign(R * C, 0.0);
    for (std::size_t ja = 0; ja < c; ++ja)
      for (int jb = 0; jb < f.cols; ++jb)
        for (std::size_t ia = 0; ia < r; ++ia) {
          const double a = acc[ia + ja * r];
          if (a == 0.0) continue;
          const std::size_t col = ja * f.cols + jb;
          for (int ib = 0; ib < f.rows; ++ib)
            next[(ia * f.rows + ib) + col * R] = a * f.data[ib + static_cast<std::size_t>(jb) * f.rows];
        }
    acc.swap(next);
    r = R;
    c = C;
  }
  std::copy(acc.begin(), acc.end(), out);
}

template <univariate_basis... B>
void tensor_product_basis<B...>::cross_gram(const int* k, const int* l, double* out) const {
  std::vector<std::vector<double>> g(margin_count);
  std::vector<kronecker_factor> f;
  detail_tensor::host_each(*this, [&](auto I, const auto& m) {
    if (k[I] < 0 || k[I] > 6 || l[I] < 0 || l[I] > 6)
      throw std::invalid_argument("tensor_product_basis::cross_gram: orders must be in 0..6");
    const int K = m.dimension();
    g[I].resize(static_cast<std::size_t>(K) * K);
    m.cross_gram(k[I], l[I], g[I].data());
    f.push_back({g[I].data(), K, K});
  });
  kronecker_product(f, out);
}

// One term c d^k of a differential operator with constant coefficients on the
// product, k a multi-index with one order (0..6) per margin.
struct tensor_operator_term {
  double coefficient;
  std::vector<int> order;
};

// S_L = int over the box of (L f)^2 as a quadratic form in the coefficients,
// L = sum_a c_a d^(k^(a)): S_L = sum_{a,b} c_a c_b G_{k^(a) k^(b)}, dense
// K x K column-major (P7). A sum of squares of several operators (thin plate)
// is a sum of calls. For a polynomial margin (B-spline, Legendre) an order must
// not exceed its degree, or the penalty would not be the one of L.
template <univariate_basis... B>
void tensor_operator_penalty(const tensor_product_basis<B...>& basis, const std::vector<tensor_operator_term>& terms,
                             double* out) {
  constexpr int D = sizeof...(B);
  for (const auto& t : terms) {
    if (static_cast<int>(t.order.size()) != D)
      throw std::invalid_argument("tensor_operator_penalty: each order needs one entry per margin");
    detail_tensor::host_each(basis, [&](auto I, const auto& m) {
      const int k = t.order[I], deg = detail_operator::polynomial_degree(m);
      if (k < 0 || k > 6) throw std::invalid_argument("tensor_operator_penalty: orders must be in 0..6");
      if (deg >= 0 && k > deg)
        throw std::invalid_argument("tensor_operator_penalty: an order (" + std::to_string(k) +
                                    ") exceeds the degree of its margin (" + std::to_string(deg) + ")");
    });
  }
  const std::size_t KK = static_cast<std::size_t>(basis.dimension()) * basis.dimension();
  std::fill(out, out + KK, 0.0);
  std::vector<double> g(KK);
  for (const auto& a : terms)
    for (const auto& b : terms) {
      const double w = a.coefficient * b.coefficient;
      if (w == 0.0) continue;
      basis.cross_gram(a.order.data(), b.order.data(), g.data());
      for (std::size_t i = 0; i < KK; ++i) out[i] += w * g[i];
    }
}

// G_0 kron ... kron S_j kron ... kron G_0 = int over the box of (L_j f)^2, L_j
// the univariate operator op acting on x_j (j 0-based), S_j its
// operator_penalty on margin j and G_0 the Gram matrices of the others (P7).
template <univariate_basis... B>
void marginal_operator_penalty(const tensor_product_basis<B...>& basis, int j, const differential_operator& op,
                               double* out) {
  constexpr int D = sizeof...(B);
  if (j < 0 || j >= D) throw std::invalid_argument("marginal_operator_penalty: no such margin");
  std::vector<std::vector<double>> g(D);
  std::vector<kronecker_factor> f;
  detail_tensor::host_each(basis, [&](auto I, const auto& m) {
    const int K = m.dimension();
    g[I].resize(static_cast<std::size_t>(K) * K);
    if (static_cast<int>(I) == j) operator_penalty(m, op, g[I].data());
    else m.cross_gram(0, 0, g[I].data());
    f.push_back({g[I].data(), K, K});
  });
  kronecker_product(f, out);
}

// T = T_1 kron ... kron T_D for the coefficients of the product, from one
// transform per margin (P8); constraints, T' G T = I and Demmler-Reinsch carry
// over factor by factor, but basis_transform declares no property.
// ponytail: T is dense K x q (512 MB at D = 3, K_j = 20); apply it factored by
// mode products if a product that large is needed.
inline basis_transform kronecker_transform(const std::vector<basis_transform>& t) {
  if (t.size() < 2) throw std::invalid_argument("kronecker_transform: at least two transforms");
  std::size_t R = 1, C = 1;
  std::vector<kronecker_factor> f;
  for (const auto& m : t) {
    if (m.columns() < 1) throw std::invalid_argument("kronecker_transform: empty transform");
    f.push_back({m.matrix().data(), m.rows(), m.columns()});
    R *= m.rows();
    C *= m.columns();
  }
  detail_transform::host_matrix out("laplace::basis::kronecker_transform", R, C);
  kronecker_product(f, out.data());
  return basis_transform_from_matrix(out);
}

// T = T_a T_b: first a (K x q), then b (q x p) on a's coefficients (P8).
inline basis_transform compose_basis_transforms(const basis_transform& a, const basis_transform& b) {
  if (a.columns() != b.rows())
    throw std::invalid_argument("compose_basis_transforms: a's columns must equal b's rows");
  return basis_transform_from_matrix(detail_transform::product(a.matrix(), b.matrix(), false));
}

// The rows of multi-index k (k[j], one per margin) at every point of x, in a
// kernel on the execution space of the Views: x(i, 0..D-1), first(i, 0..D-1),
// values(i, 0..row_width(k) - 1), values a LayoutRight View of doubles.
// ponytail: the scratch is an n x scratch_size(k) View; use team scratch if
// that memory matters.
template <class TB, class XView, class FirstView, class ValuesView>
  requires multivariate_basis<TB>
void fill_tensor_basis_rows(const TB& b, const std::array<int, TB::margin_count>& k, const XView& x,
                            const FirstView& first, const ValuesView& values) {
  static_assert(std::is_same_v<typename ValuesView::array_layout, Kokkos::LayoutRight>,
                "fill_tensor_basis_rows writes each row in place: values must be LayoutRight");
  using exec = typename XView::execution_space;
  constexpr int D = TB::margin_count;
  Kokkos::Array<int, D> kk;
  for (int j = 0; j < D; ++j) kk[j] = k[j];
  const int S = b.scratch_size(k.data());
  Kokkos::View<double**, Kokkos::LayoutRight, typename XView::memory_space> scratch(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "laplace::basis::tensor_scratch"), x.extent(0), S);
  Kokkos::parallel_for(
      "laplace::basis::fill_tensor_basis_rows", Kokkos::RangePolicy<exec>(0, x.extent(0)), KOKKOS_LAMBDA(int i) {
        double xi[D];
        int fi[D];
        for (int j = 0; j < D; ++j) xi[j] = x(i, j);
        b.row_partial(xi, kk.data(), fi, &values(i, 0), &scratch(i, 0));
        for (int j = 0; j < D; ++j) first(i, j) = fi[j];
      });
}

}  // namespace laplace::basis
