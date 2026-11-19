// Data for the fit_distrib bench: the frozen R references of
// tests/data/reference/optimizers/fit_distrib (README there), the simulation of
// data in C++ (counter-based draws by inversion, the same on every backend),
// and the library's starting values as a vector theta.
#pragma once

#include "fit_distrib_bench.hpp"

#include <laplace/distrib/bernoulli.hpp>
#include <laplace/distrib/beta.hpp>
#include <laplace/distrib/betabinom.hpp>
#include <laplace/distrib/cauchy.hpp>
#include <laplace/distrib/enet.hpp>
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
#include <laplace/distrib/mcp.hpp>
#include <laplace/distrib/negbin1.hpp>
#include <laplace/distrib/negbin2.hpp>
#include <laplace/distrib/pig.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <laplace/distrib/lognormal2.hpp>
#include <laplace/distrib/poisson.hpp>
#include <laplace/distrib/powerexp.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/distrib/scad.hpp>
#include <laplace/distrib/skewnormal.hpp>
#include <laplace/distrib/skewt.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <laplace/distrib/vonmises.hpp>
#include <laplace/distrib/weibull.hpp>

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

// the library start (generic_start.hpp) as theta, through the sample route: the distrib's
// start_value(y) where it has one (vonmises, powerexp), its start_value(summary) otherwise
template <class D>
std::vector<double> library_start(const D& prototype, const std::vector<double>& y) {
  return theta_of_distrib(laplace::distrib::distrib_start_value(prototype, y));
}

// D with its parameters from theta and its other fields (betabinom's size) from a
// prototype; the parameters are D's leading double fields (theta_of_distrib)
template <class D>
struct prototype_builder {
  D prototype;
  LAPLACE_HD D operator()(const double* theta) const {
    D d = prototype;
    double* p = reinterpret_cast<double*>(&d);
    for (int j = 0; j < D::n_params; ++j) p[j] = theta[j];
    return d;
  }
};

// one row of fit_distrib_laws.csv (tools/reference/optimizers_fit_distrib_laws.R)
struct laws_reference_row {
  std::string name, reference, param;
  long n = 0;
  double true_value = 0, mle = 0, loglik_mle = 0, statmodels7 = 0, loglik_statmodels7 = 0, score = 0;
  bool has_statmodels7 = false, statmodels7_converged = false;
};

inline std::vector<laws_reference_row> read_laws_reference_rows(const std::string& path) {
  std::ifstream in(path);
  std::vector<laws_reference_row> rows;
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
    laws_reference_row r;
    r.name = f[0];
    r.reference = f[1];
    r.n = std::atol(f[2].c_str());
    r.param = f[4];
    r.true_value = std::strtod(f[5].c_str(), nullptr);
    r.mle = std::strtod(f[6].c_str(), nullptr);
    r.loglik_mle = std::strtod(f[7].c_str(), nullptr);
    r.has_statmodels7 = f[8] != "NA";
    r.statmodels7 = r.has_statmodels7 ? std::strtod(f[8].c_str(), nullptr) : 0.0;
    r.loglik_statmodels7 = r.has_statmodels7 ? std::strtod(f[9].c_str(), nullptr) : 0.0;
    r.statmodels7_converged = f[10] == "TRUE";
    r.score = std::strtod(f[11].c_str(), nullptr);
    rows.push_back(r);
  }
  return rows;
}

}  // namespace fit_distrib_bench
