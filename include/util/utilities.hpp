#pragma once

#include <armadillo>
#include <cmath>
#include <string>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <vector>

namespace utils {
    double rms(const arma::Col<double>& vec1, const arma::Col<double>& vec2);

    arma::Col<double> center_xy(const arma::Col<double> &x, const arma::Col<double> &xy);

    arma::Col<double> center_xyz(const arma::Col<double> &x, const arma::Col<double> &xy, const arma::Col<double> &xyz);
    
    std::string today();
    
    std::string now_30();
    
    std::string now();

    void make_path(const std::filesystem::path &path);
    
    bool is_file(const std::filesystem::path &path);
    
    bool is_dir(const std::filesystem::path &path);
    
    int next_run_id(const std::string& directory, const std::regex& pattern);

    template <typename T>
    std::string col_string(const arma::Col<T>& col);
}

