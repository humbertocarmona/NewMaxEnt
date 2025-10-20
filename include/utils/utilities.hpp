#pragma once

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace utils
{

template <typename T> inline arma::Col<T> jsonToArmaCol(const nlohmann::json &j)
{
    arma::Col<double> vec(j.size());
    for (size_t i = 0; i < j.size(); ++i)
    {
        vec(i) = j.at(i).get<double>();
    }
    return vec;
}

inline bool isFileType(const std::string &fname, const std::string &suffix)
{
    int n   = suffix.length();
    int pos = fname.length() - n;
    if (pos > 0)
    {
        return fname.compare(pos, n, suffix) == 0;
    }
    return false;
}

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

inline std::filesystem::path get_available_filename(const std::filesystem::path &base)
{
    if (!std::filesystem::exists(base))
    {
        return base;
    }

    std::filesystem::path dir = base.parent_path();
    std::string stem          = base.stem().string();      // filename without extension
    std::string ext           = base.extension().string(); // includes the dot

    int counter = 1;
    while (true)
    {
        std::ostringstream new_name;
        new_name << stem << "_" << std::setw(2) << std::setfill('0') << counter << ext;
        std::filesystem::path candidate = dir / new_name.str();

        if (!std::filesystem::exists(candidate))
        {
            return candidate;
        }

        ++counter;
    }
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

inline double exp_q(double x, double q)
{ // q_inv = 1/(1-q) for q!=1
    if (q == 1.0)
        return std::exp(x);
    double qq = 1.0 - q;
    double y = 1.0 + qq * x;
    return (y > 0.0) ? std::pow(y, 1.0/qq) : 0.0;
}

inline double log_q(double x, double q)
{
    if (q == 1.0)
        return std::log(x);

    double qq = 1.0 - q;
    return (pow(x, qq) - 1.0) / qq;
}

inline std::string brief(const arma::Col<double> &v)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < std::min<std::size_t>(3, v.n_elem); ++i)
    {
        out << v(i);
        if (i < 2 && i + 1 < v.n_elem)
            out << ", ";
    }
    if (v.n_elem > 3)
        out << ", ...";
    out << "]";
    return out.str();
}

template <typename T> inline std::string colPrint(const arma::Col<T> &v)
{
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < v.n_elem; ++i)
    {
        out << v(i);
        if (i + 1 < v.n_elem)
            out << ", ";
    }
    out << "]";
    return out.str();
}

} // namespace utils
