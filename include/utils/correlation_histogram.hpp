#pragma once

#include <algorithm>
#include <armadillo>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Computes the normalized histogram of pairwise correlations between rows or columns of a
 * matrix.
 *
 * This function computes dot products between all distinct pairs of rows (default) or columns
 * of the input matrix `M`, bins the resulting values into intervals of width `delta`, and
 * normalizes the histogram such that the total area under the histogram (sum of probabilities
 * times bin width) equals 1.
 *
 * The default bin width `delta = 2.0` is appropriate when `M` is an integer matrix
 * with entries ±1, where each row (or column) represents a configuration of a spin system
 * with `n_spins` spins. In such cases, the dot products lie within [-n_spins, n_spins]
 * and take only even or odd integer values depending on the parity of `n_spins`.
 *
 * @tparam T Numeric type of matrix elements (e.g., int, double).
 * @param M Matrix where each row or column represents a spin configuration or feature vector.
 * @param delta Bin width for histogram construction (default = 2.0).
 * @param by_row If true (default), computes correlations between rows; if false, between columns.
 * @return std::pair<std::vector<double>, std::vector<double>>
 *         - First vector: bin centers (real values corresponding to binned dot products).
 *         - Second vector: normalized histogram values (probability densities).
 */
template <typename T>
inline std::pair<std::vector<double>, std::vector<double>> correlation_histogram(
    const arma::Mat<T> &M,
    const double delta = 2.0,
    const bool by_row  = true)
{

    const arma::uword dim1 = by_row ? M.n_rows : M.n_cols;
    const arma::uword dim2 = by_row ? M.n_cols : M.n_rows;

    // Build histogram
    std::unordered_map<int, double> raw_hist;
    for (arma::uword i = 0; i < dim1; ++i)
    {
        for (arma::uword j = i + 1; j < dim1; ++j)
        {
            double dot;
            if (by_row)
            {
                dot = arma::dot(M.row(i), M.row(j));
            }
            else
            {
                dot = arma::dot(M.col(i), M.col(j));
            }

            int bin = static_cast<int>(dot / delta);
            raw_hist[bin] += 1.0;
        }
    }

    // Compute total area for normalization
    double total_area = 0.0;
    for (const auto &[bin, count] : raw_hist)
    {
        if (by_row)
        total_area += count * delta/M.n_cols;
        else
        total_area += count * delta/M.n_rows;
    }

    // Normalize the histogram
    for (auto &[bin, count] : raw_hist)
    {
        count /= total_area;
    }

    // Extract sorted vectors
    std::vector<int> bin_keys;
    for (const auto &[bin, _] : raw_hist)
    {
        bin_keys.push_back(bin);
    }
    std::sort(bin_keys.begin(), bin_keys.end());

    std::vector<double> bin_centers;
    std::vector<double> normalized_counts;
    for (int bin : bin_keys)
    {
        if (by_row)
        {
            bin_centers.push_back(static_cast<double>(bin) * delta / M.n_cols);
        }
        else
        {
            bin_centers.push_back(static_cast<double>(bin) * delta / M.n_rows);
        }
        // bin_centers.push_back(static_cast<double>(bin) * delta);
        normalized_counts.push_back(raw_hist[bin]);
    }

    return {bin_centers, normalized_counts};
}
