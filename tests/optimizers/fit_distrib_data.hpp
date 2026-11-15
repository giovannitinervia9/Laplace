// Data for the fit_distrib bench: the frozen R references of
// tests/data/reference/optimizers/fit_distrib (README there), the simulation of
// data in C++ (counter-based draws by inversion, the same on every backend),
// and the library's starting values as a vector theta.
#pragma once

#include "fit_distrib_bench.hpp"

#include <laplace/distrib/bernoulli.hpp>
#include <laplace/distrib/cauchy.hpp>
#include <laplace/distrib/chisq.hpp>
#include <laplace/distrib/exponential.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/gaussian3.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/geometric.hpp>
#include <laplace/distrib/gumbel.hpp>
#include <laplace/distrib/invgauss.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/logistic.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <laplace/distrib/lognormal2.hpp>
#include <laplace/distrib/poisson.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace fit_distrib_bench {

inline std::vector<double> read_hex_csv(const std::string& path) {
  std::ifstream in(path);
  std::vector<double> v;
  std::string line;
  std::getline(in, line);  // header "y"
  while (std::getline(in, line))
    if (!line.empty()) v.push_back(std::strtod(line.c_str(), nullptr));
  return v;
}

inline std::vector<double> read_f64(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  std::vector<double> v(bytes / 8);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  return v;
}

// one row of fit_distrib.csv
struct reference_row {
  std::string distrib, param;
  long n = 0;
  double true_value = 0, estimate = 0, closed_form_mle = 0, loglik = 0;
  bool has_closed_form = false;
  int iterations = 0;
};

inline std::vector<reference_row> read_reference_rows(const std::string& path) {
  std::ifstream in(path);
  std::vector<reference_row> rows;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (header) { header = false; continue; }
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) {
      c.erase(std::remove(c.begin(), c.end(), '"'), c.end());
      f.push_back(c);
    }
    reference_row r;
    r.distrib = f[0];
    r.n = std::atol(f[2].c_str());
    r.param = f[4];
    r.true_value = std::strtod(f[5].c_str(), nullptr);
    r.estimate = std::strtod(f[6].c_str(), nullptr);
    r.has_closed_form = f[8] != "NA";
    r.closed_form_mle = r.has_closed_form ? std::strtod(f[8].c_str(), nullptr) : 0.0;
    r.loglik = std::strtod(f[9].c_str(), nullptr);
    r.iterations = std::atoi(f[11].c_str());
    rows.push_back(r);
  }
  return rows;
}

template <class ExecSpace = Kokkos::DefaultExecutionSpace>
Kokkos::View<const double*, typename ExecSpace::memory_space> to_device(const std::vector<double>& v) {
  Kokkos::View<double*, typename ExecSpace::memory_space> d("y", v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

// n draws from d by inversion, observation i from counter_uniform_source{seed, i}
template <class D, class MemSpace>
struct simulate_functor {
  Kokkos::View<double*, MemSpace> y;
  D d;
  std::uint64_t seed;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    y(i) = laplace::distrib::rng_by_inversion(d, seed, static_cast<std::uint64_t>(i));
  }
};

template <class D, class ExecSpace = Kokkos::DefaultExecutionSpace>
Kokkos::View<const double*, typename ExecSpace::memory_space> simulate(const D& d, long n, std::uint64_t seed) {
  Kokkos::View<double*, typename ExecSpace::memory_space> y("y", n);
  Kokkos::parallel_for("fit_distrib_simulate", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n)),
                       simulate_functor<D, typename ExecSpace::memory_space>{y, d, seed});
  Kokkos::fence();
  return y;
}

template <class View>
std::vector<double> to_host_vector(const View& y) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

// theta of a distrib, whose parameters are its leading double fields (true of every
// distrib here; binomial's size follows them)
template <class D>
std::vector<double> theta_of_distrib(const D& d) {
  static_assert(std::is_standard_layout_v<D> && std::is_trivially_copyable_v<D>);
  std::vector<double> t(D::n_params);
  std::memcpy(t.data(), &d, sizeof(double) * D::n_params);
  return t;
}

// the library start (generic_start.hpp) as theta
template <class D>
std::vector<double> library_start(const D& prototype, const std::vector<double>& y) {
  return theta_of_distrib(laplace::distrib::distrib_start_value(prototype, laplace::distrib::summarize_sample(y)));
}

}  // namespace fit_distrib_bench
