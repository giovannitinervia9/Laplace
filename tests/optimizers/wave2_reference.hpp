#pragma once

// Readers of the wave-2 references frozen from R (tests/data/reference/
// optimizers/wave2, README there; written by tools/reference/optimizers_wave2.R):
// the five datasets and the glmnet fits. Shared by the proximal tests and the
// LAD tests.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wave2_reference {

// one CSV line split at commas, quotes removed (no field holds a comma)
inline std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> f;
  std::string cur;
  for (char ch : line) {
    if (ch == ',') {
      f.push_back(cur);
      cur.clear();
    } else if (ch != '"' && ch != '\r') {
      cur += ch;
    }
  }
  f.push_back(cur);
  return f;
}

// the data rows of a CSV (comment lines and the header skipped)
inline std::vector<std::vector<std::string>> read_rows(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (header) {
      header = false;
      continue;
    }
    rows.push_back(split_csv(line));
  }
  return rows;
}

inline double to_double(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

// X (n x p, column-major, no intercept column) and the two responses
struct dataset {
  std::string id;
  int n = 0, p = 0;
  std::vector<double> x, y_gaussian, y_lad;
};

inline dataset load_dataset(const std::string& dir, const std::string& id) {
  dataset d;
  d.id = id;
  std::string file;
  for (const auto& r : read_rows(dir + "/datasets.csv"))
    if (r[0] == id) d.n = std::atoi(r[1].c_str()), d.p = std::atoi(r[2].c_str()), file = r[6];
  if (file.empty()) throw std::runtime_error("no dataset " + id);
  const std::size_t n = std::size_t(d.n), p = std::size_t(d.p);
  std::vector<double> all(n * (p + 2));
  if (file.size() > 4 && file.substr(file.size() - 4) == ".f64") {
    std::ifstream in(dir + "/" + file, std::ios::binary);
    in.read(reinterpret_cast<char*>(all.data()), std::streamsize(all.size() * sizeof(double)));
    if (!in) throw std::runtime_error("short file " + file);
  } else {
    const auto rows = read_rows(dir + "/" + file);
    if (rows.size() != n) throw std::runtime_error("wrong row count in " + file);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < p + 2; ++j) all[i + n * j] = to_double(rows[i][j]);
  }
  d.x.assign(all.begin(), all.begin() + std::ptrdiff_t(n * p));
  d.y_gaussian.assign(all.begin() + std::ptrdiff_t(n * p), all.begin() + std::ptrdiff_t(n * (p + 1)));
  d.y_lad.assign(all.begin() + std::ptrdiff_t(n * (p + 1)), all.end());
  return d;
}

// one glmnet fit: RSS/2 + a |beta|_1 + c |beta|^2 / 2 with an unpenalized intercept b0
struct glmnet_fit {
  std::string dataset;
  double alpha = 0, lambda_fraction = 0, lambda_laplace = 0, a = 0, c = 0;
  std::vector<int> support;  // 1-based indices of beta
  double objective_laplace = 0, penalty_constant_per_coef = 0;
  std::vector<double> coefficients;  // b0, beta1, ..., betap (the polished reference)
};

inline std::vector<glmnet_fit> read_glmnet_fits(const std::string& dir) {
  std::vector<glmnet_fit> fits;
  for (const auto& r : read_rows(dir + "/glmnet_fits.csv")) {
    glmnet_fit f;
    f.dataset = r[0];
    f.alpha = to_double(r[1]), f.lambda_fraction = to_double(r[2]), f.lambda_laplace = to_double(r[4]);
    f.a = to_double(r[5]), f.c = to_double(r[6]);
    std::istringstream s(r[8]);
    for (int j; s >> j;) f.support.push_back(j);
    f.objective_laplace = to_double(r[10]), f.penalty_constant_per_coef = to_double(r[13]);
    fits.push_back(f);
  }
  for (const auto& r : read_rows(dir + "/glmnet.csv"))
    for (auto& f : fits)
      if (f.dataset == r[0] && f.alpha == to_double(r[1]) && f.lambda_fraction == to_double(r[2]))
        f.coefficients.push_back(to_double(r[4]));  // in file order: b0, beta1, ...
  return fits;
}

}  // namespace wave2_reference
