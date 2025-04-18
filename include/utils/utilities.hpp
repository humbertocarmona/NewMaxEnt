#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace utils
{

std::string now();

inline void make_path(const std::filesystem::path &path)
{
    if (!std::filesystem::exists(path))
    {
        if (std::filesystem::create_directories(path))
        {
            std::cout << "Directory created: " << path << '\n';
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << '\n';
        }
    }
}

inline bool is_file(const std::filesystem::path &path)
{
    return std::filesystem::exists(path);
}

inline bool is_dir(const std::filesystem::path &path)
{
    return std::filesystem::exists(path);
}

inline std::string now()
{
    // Get the current time
    std::time_t t = std::time(nullptr);
    // Convert to local time
    std::tm tm = *std::localtime(&t);
    // Create a stringstream to format the date
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M");
    return oss.str();
}

inline double exp_q(double x, double q, double q_inv) { // q_inv = 1/(1-q) for q!=1
    if (q == 1.0)
        return std::exp(x);
    double y = 1.0 + (1.0 - q) * x;
    return (y > 0.0) ? std::pow(y, q_inv) : 0.0;
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

} // namespace utils
