#include "core/centered_moments.hpp"
#include "util/logger.hpp"
#include "util/utilities.hpp"
#include <spdlog/spdlog.h>

inline int edge_index(int i, int j, int n)
{
    if (i > j) std::swap(i, j);
    return i * n - (i * (i + 1)) / 2 + (j - i - 1);
}

CenteredMoments compute_centered_moments(const arma::Col<double>& moment_1,
                                         const arma::Col<double>& moment_2,
                                         const arma::Col<double>& moment_3)
{
    auto logger = spdlog::get("bm");

    if (logger)
        logger->info("[compute_centered_moments] moment_3 (brief): {}", brief(moment_3));

    int n = moment_1.n_elem;
    int n_edges = n * (n - 1) / 2;
    int n_triplets = n * (n - 1) * (n - 2) / 6;

    // Check for expected sizes
    if (moment_2.n_elem != n_edges)
        throw std::runtime_error("moment_2 size does not match expected number of edges.");
    if (moment_3.n_elem != n_triplets)
        throw std::runtime_error("moment_3 size does not match expected number of triplets.");

    arma::Mat<double> corr2(n, n, arma::fill::zeros);
    arma::Col<double> centered_triplets(n_triplets, arma::fill::zeros);

    // Centered pairwise correlations
    for (int i = 0; i < n; ++i)
    {
        for (int j = i; j < n; ++j)
        {
            double centered = 0.0;
            if (i == j)
            {
                centered = 1.0 - std::pow(moment_1(i), 2);  // Var(x_i)
            }
            else
            {
                int idx = edge_index(i, j, n);
                centered = moment_2(idx) - moment_1(i) * moment_1(j);
            }
            corr2(i, j) = corr2(j, i) = centered;
        }
    }

    if (logger)
        logger->info("[compute_centered_moments] passed centered_moment_2");

    // Centered triplet correlations (connected 3rd order)
    int idx = 0;
    for (int i = 0; i < n - 2; ++i)
    {
        for (int j = i + 1; j < n - 1; ++j)
        {
            for (int k = j + 1; k < n; ++k)
            {
                double m3 = moment_3(idx);

                double c3 = m3
                    - moment_1(i) * moment_2(edge_index(j, k, n))
                    - moment_1(j) * moment_2(edge_index(i, k, n))
                    - moment_1(k) * moment_2(edge_index(i, j, n))
                    + 2.0 * moment_1(i) * moment_1(j) * moment_1(k);

                centered_triplets(idx++) = c3;
            }
        }
    }

    return {corr2, centered_triplets};
}
