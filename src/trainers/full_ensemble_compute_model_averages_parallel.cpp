#include "trainers/full_ensemble_trainer.hpp"
#include "utils/binary_permutations_sequence.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <algorithm> // Required for std::min
#include <armadillo>
#include <omp.h> // OpenMP

void FullEnsembleTrainer::computeModelAverages(double beta, bool triplets)
{
    auto logger = getLogger();

    int nspins = core.nspins;
    int nedges = core.nedges;
    m1_model.zeros(nspins);
    m2_model.zeros(core.nedges);
    m3_model.zeros(ntriplets);
    double q_inv = (params.q_val != 1) ? 1.0 / (1.0 - params.q_val) : 0.0;

    avg_energy         = 0.0;
    avg_energy_sq      = 0.0;
    avg_magnetization  = 0.0;
    double Z_partition = 0.0;
    size_t total       = 1ULL << nspins;
    ;

#pragma omp parallel
    {
        int num_threads   = omp_get_num_threads();
        int thread_id     = omp_get_thread_num();
        size_t chunk_size = (total + num_threads - 1) / num_threads; // ceil

        // Local accumulators per thread
        double local_avg_energy        = 0.0;
        double local_avg_energy_sq     = 0.0;
        double local_avg_magnetization = 0.0;
        double local_Z                 = 0.0;

        arma::Col<double> local_m1_model(nspins, arma::fill::zeros);
        arma::Col<double> local_m2_model(nedges, arma::fill::zeros);
        arma::Col<double> local_m3_model;
        if (triplets)
        {
            local_m3_model.set_size(ntriplets);
            local_m3_model.zeros();
        }

        double E = 0.0, P = 0.0;

        size_t start = thread_id * chunk_size;
        size_t end   = std::min(start + chunk_size, total);
        BinaryPermutationsSequence sequence(nspins, start, end);

        // logger->info("thread {} start {} end {}", thread_id, start, end);
        for (const auto &s : sequence)
        {
            E = energyAllPairs(s);
            P = utils::exp_q(-beta * E, params.q_val, q_inv);

            local_Z += P;
            local_avg_energy += P * E;
            local_avg_energy_sq += P * E * E;
            local_avg_magnetization += P * arma::mean(arma::conv_to<arma::vec>::from(s));

            // First-order moments
            for (size_t i = 0; i < nspins; ++i)
                local_m1_model(i) += P * s(i);

            // Second-order moments
            int idx = 0;
            for (size_t i = 0; i < nspins - 1; ++i)
            {
                for (size_t j = i + 1; j < nspins; ++j)
                {
                    local_m2_model(idx++) += P * s(i) * s(j);
                }
            }

            if (triplets)
            {
                // Third-order moments
                idx = 0;
                for (size_t i = 0; i < nspins - 2; ++i)
                {
                    for (size_t j = i + 1; j < nspins - 1; ++j)
                    {
                        for (size_t k = j + 1; k < nspins; ++k)
                        {
                            local_m3_model(idx++) += P * s(i) * s(j) * s(k);
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            avg_energy += local_avg_energy;
            avg_energy_sq += local_avg_energy_sq;
            avg_magnetization += local_avg_magnetization;
            m1_model += local_m1_model;
            m2_model += local_m2_model;
            Z_partition += local_Z;
            if (triplets)
                m3_model += local_m3_model;
        }
    } // End of parallel block

    // Normalize final averages
    avg_energy /= Z_partition;
    avg_energy_sq /= Z_partition;
    avg_magnetization /= Z_partition;

    m1_model /= Z_partition;
    m2_model /= Z_partition;
    if (triplets)
        m3_model /= Z_partition;
}
