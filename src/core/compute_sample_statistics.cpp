#include "core/max_ent_core.hpp"
#include <armadillo>

void MaxEntCore::compute_sample_statistics()
{
    arma::Mat<double> samples_dbl = arma::conv_to<arma::Mat<double>>::from(raw_samples);
    int n_samples                 = samples_dbl.n_rows;
    int n_spins                   = samples_dbl.n_cols;

    // First moment: average magnetization of each spin
    sample_moment_1 = arma::mean(samples_dbl, 0).t(); // column vector

    // Second moment: correlations between pairs
    sample_moment_2.set_size(n_spins * (n_spins - 1) / 2);
    int idx = 0;
    for (int i = 0; i < n_spins - 1; ++i)
    {
        for (int j = i + 1; j < n_spins; ++j)
        {
            sample_moment_2(idx++) = arma::mean(samples_dbl.col(i) % samples_dbl.col(j));
        }
    }

    // Third moment: triplet correlations
    int n_triplets = (n_spins * (n_spins - 1) * (n_spins - 2)) / 6;
    sample_moment_3.set_size(n_triplets);
    idx = 0;
    for (int i = 0; i < n_spins - 2; ++i)
    {
        for (int j = i + 1; j < n_spins - 1; ++j)
        {
            for (int k = j + 1; k < n_spins; ++k)
            {
                sample_moment_3(idx++) = arma::mean(samples_dbl.col(i) % samples_dbl.col(j) % samples_dbl.col(k));
            }
        }
    }

    LOGGER->info("[compute_sample_statistics] Computed moments 1 (size {}), 2 (size {}), 3 (size {})",
                 sample_moment_1.n_elem, sample_moment_2.n_elem, sample_moment_3.n_elem);
    LOGGER->flush();
    
}
