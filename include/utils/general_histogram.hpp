#pragma once

#include <armadillo>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>


template <typename T>
inline std::pair<std::vector<double>, std::vector<double>>
compute_histogram(const arma::Mat<T>& M, const double delta)
{
    // Build histogram
    std::unordered_map<int, double> raw_hist;
    for (arma::uword i = 0; i < M.n_rows; ++i) {
        for (arma::uword j = 0; j < M.n_cols; ++j) {
            int bin = static_cast<int>(M(i,j) / delta);
            raw_hist[bin] += 1.0;
        }
    }

    // Compute total area for normalization
    double total_area = 0.0;
    for (const auto& [bin, count] : raw_hist) {
        total_area += count * delta;
    }

    // Normalize the histogram
    for (auto& [bin, count] : raw_hist) {
        count /= total_area;
    }

    // Extract sorted vectors
    std::vector<int> bin_keys;
    for (const auto& [bin, _] : raw_hist) {
        bin_keys.push_back(bin);
    }
    std::sort(bin_keys.begin(), bin_keys.end());

    std::vector<double> bin_centers;
    std::vector<double> normalized_counts;
    for (int bin : bin_keys) {
        bin_centers.push_back(static_cast<double>(bin) * delta);
        normalized_counts.push_back(raw_hist[bin]);
    }

    return {bin_centers, normalized_counts};
}

