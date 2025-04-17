#include "core/centered_moments.hpp"
#include "util/utilities.hpp"
#include <spdlog/spdlog.h>

inline int edge_index(int i, int j, int n)
{
    if (i > j)
        std::swap(i, j);
    return i * n - (i * (i + 1)) / 2 + (j - i - 1);
}

CenteredMoments compute_centered_moments(const arma::Col<double> &moment_1,
                                         const arma::Col<double> &moment_2,
                                         const arma::Col<double> &moment_3)
{
    auto logger = spdlog::get("bm");

    int n          = moment_1.n_elem;
    int n_edges    = n * (n - 1) / 2;
    int n_triplets = n * (n - 1) * (n - 2) / 6;

    // Check for expected sizes
    if (moment_2.n_elem != n_edges)
        throw std::runtime_error("moment_2 size does not match expected number of edges.");
    if (moment_3.n_elem != n_triplets)
        throw std::runtime_error("moment_3 size does not match expected number of triplets.");

    arma::Col<double> centered_moment_2(n_edges, arma::fill::zeros);
    arma::Col<double> centered_moment_3(n_triplets, arma::fill::zeros);

    // Centered pairwise correlations

    for (int i = 0; i < n-1; ++i)
    {
        for (int j = i+1; j < n; ++j)
        {
            int idx                = edge_index(i, j, n);

            double centered        = 0.0;
            centered_moment_2(idx) = moment_2(idx) - moment_1(i) * moment_1(j);
            ;
        }
    }


    // Centered triplet correlations (connected 3rd order)
    int idx = 0;
    for (int i = 0; i < n - 2; ++i)
    {
        for (int j = i + 1; j < n - 1; ++j)
        {
            int ij = edge_index(i, j, n);
            for (int k = j + 1; k < n; ++k)
            {
                int jk = edge_index(j, k, n);
                int ik = edge_index(i, k, n);

                double m3 = moment_3(idx);
                double c3 = m3 - moment_1(i) * moment_2(jk) - moment_1(j) * moment_2(ik) - moment_1(k) * moment_2(ij) +
                            2.0 * moment_1(i) * moment_1(j) * moment_1(k);

                centered_moment_3(idx++) = c3;
            }
        }
    }
    logger->debug("[compute_centered_moments] Centering moments completed");
    return {centered_moment_2, centered_moment_3};
}
