#include "utils/centered_moments.hpp"
#include "utils/utilities.hpp"
#include "utils/get_logger.hpp"

inline int edgeIndex(int i, int j, int n)
{
    if (i > j)
        std::swap(i, j);
    return i * n - (i * (i + 1)) / 2 + (j - i - 1);
}

CenteredMoments computeCenteredMoments(const arma::Col<double> &m1,
                                       const arma::Col<double> &m2,
                                       const arma::Col<double> &m3)
{
    // auto logger = getLogger();

    int n         = m1.n_elem;
    int nedges    = n * (n - 1) / 2;
    int ntriplets = n * (n - 1) * (n - 2) / 6;

    // Check for expected sizes
    if (m2.n_elem != nedges)
        throw std::runtime_error("moment_2 size does not match expected number of edges.");
    if (m3.n_elem != ntriplets){
        throw std::runtime_error("moment_3 size does not match expected number of triplets.");
    }
    arma::Col<double> m2_centered(nedges, arma::fill::zeros);
    arma::Col<double> m3_centered(ntriplets, arma::fill::zeros);

    // Centered pairwise correlations

    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            int idx = edgeIndex(i, j, n);

            double centered  = 0.0;
            m2_centered(idx) = m2(idx) - m1(i) * m1(j);
        }
    }

    // Centered triplet correlations (connected 3rd order)
    int idx = 0;
    for (int i = 0; i < n - 2; ++i)
    {
        for (int j = i + 1; j < n - 1; ++j)
        {
            int ij = edgeIndex(i, j, n);
            for (int k = j + 1; k < n; ++k)
            {
                int jk = edgeIndex(j, k, n);
                int ik = edgeIndex(i, k, n);

                double c3 = m3(idx) - m1(i) * m2(jk) - m1(j) * m2(ik) - m1(k) * m2(ij) + 2.0 * m1(i) * m1(j) * m1(k);

                m3_centered(idx++) = c3;
            }
        }
    }
    return {m2_centered, m3_centered};
}
