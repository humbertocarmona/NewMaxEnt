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

std::string now();

void make_path(const std::filesystem::path &path);

bool is_file(const std::filesystem::path &path);

bool is_dir(const std::filesystem::path &path);


// Generalized exponential
// and logarithm (Tsallis q-statistics)
// -------------------------------------------------------------


} // namespace utils

inline double exp_q(double x, double q)
{
    if (q == 1.0)
        return std::exp(x);
    double y = 1.0 + (1.0 - q) * x;
    if (y <= 0.0)
        return 0.0;
    return std::pow(y, 1.0 / (1.0 - q));
}

inline std::string brief(const arma::Col<double>& v)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < std::min<std::size_t>(3, v.n_elem); ++i)
    {
        out << v(i);
        if (i < 2 && i + 1 < v.n_elem) out << ", ";
    }
    if (v.n_elem > 3) out << ", ...";
    out << "]";
    return out.str();
}




