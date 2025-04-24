#pragma once

#include <armadillo>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>


// Returns: pair of (hist_values, bin_centers), both sorted by bin
inline std::pair<std::vector<double>, std::vector<double>>
compute_ordered_overlap_histogram(const arma::Mat<int>& M, bool log_peak = true)
{
    auto logger = getLogger();

    int n_rows = M.n_rows;
    int n_cols = M.n_cols;

    std::unordered_map<int, double> raw_hist;

    // Compute histogram of dot products
    for (int i = 0; i < n_rows; ++i) {
        for (int j = i + 1; j < n_rows; ++j) {
            int dot_product = arma::dot(M.row(i), M.row(j)); // in [-N, N], even only
            raw_hist[dot_product] += 1.0;
        }
    }

    // Normalize
    double norm = (n_rows * (n_rows - 1)) / 2.0;
    for (auto& kv : raw_hist) {
        kv.second /= norm;
    }

    // Sort keys
    std::vector<int> sorted_keys;
    for (const auto& kv : raw_hist) {
        sorted_keys.push_back(kv.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // Fill outputs
    std::vector<double> bin_centers;
    std::vector<double> hist_values;
    for (int dot_val : sorted_keys) {
        bin_centers.push_back(static_cast<double>(dot_val) / n_cols);
        hist_values.push_back(raw_hist[dot_val]);
    }

    return {hist_values, bin_centers};
}

