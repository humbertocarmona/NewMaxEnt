#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace utils
{
arma::Col<double> center_xy(const arma::Col<double> &x, const arma::Col<double> &xy);

arma::Col<double> center_xyz(const arma::Col<double> &x, const arma::Col<double> &xy, const arma::Col<double> &xyz);

std::string today();

std::string now_30();

std::string now();

void make_path(const std::filesystem::path &path);

bool is_file(const std::filesystem::path &path);

bool is_dir(const std::filesystem::path &path);

int next_run_id(const std::string &directory, const std::regex &pattern);

template <typename T> std::string col_string(const arma::Col<T> &col);

// Generalized exponential
// and logarithm (Tsallis
// q-statistics)
// -------------------------------------------------------------
inline double exp_q(double x, double q)
{
    if (q == 1.0)
        return std::exp(x);
    double y = 1.0 + (1.0 - q) * x;
    if (y <= 0.0)
        return 0.0;
    return std::pow(y, 1.0 / (1.0 - q));
}

inline double log_q(double x, double q)
{
    if (q == 1.0)
        return std::log(x);
    return (std::pow(x, 1 - q) - 1.0) / (1.0 - q);
}

} // namespace utils
