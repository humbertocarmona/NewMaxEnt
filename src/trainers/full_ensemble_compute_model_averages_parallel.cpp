#include "trainers/full_ensemble_trainer.hpp"
#include "utils/binary_permutations_sequence.hpp"
#include "utils/get_logger.hpp"
// #include "utils/utilities.hpp"
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
    // double q_inv = (params.q_val != 1) ? 1.0 / (1.0 - params.q_val) : 0.0;

    // k-pairwise
    pK_model.zeros(nspins + 1);

    avg_energy         = 0.0;
    avg_energy_sq      = 0.0;
    avg_magnetization  = 0.0;
    double Z_partition = 0.0;
    size_t total       = 1ULL << nspins;
    // logger->info("Total number of configurations: {}", total);

    size_t N_supp = 0; // number of configurations with P>0
    // double f_supp            = 0.0; // fraction of supported configurations
    const double one_minus_q = 1.0 - params.q_val;

    max_weight        = -std::numeric_limits<double>::max();
    max_bracket       = -std::numeric_limits<double>::max();
    max_weight_energy = -std::numeric_limits<double>::max();
    ;
    // logger->info("Number of threads: {}", omp_get_num_threads());
    GE.clear();
    PE.clear();
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
        double local_bracket           = 1.0;
        size_t local_N_supp            = 0;
        double local_max_weight        = -std::numeric_limits<double>::max();
        double local_max_bracket       = -std::numeric_limits<double>::max();
        double local_max_weight_energy = -std::numeric_limits<double>::max();
        ;

        arma::Col<double> local_m1_model(nspins, arma::fill::zeros);
        arma::Col<double> local_m2_model(nedges, arma::fill::zeros);
        arma::Col<double> local_m3_model;
        std::unordered_map<int, double> local_GE, local_PE; // energy histogram

        if (triplets)
        {
            local_m3_model.set_size(ntriplets);
            local_m3_model.zeros();
            local_GE.clear();
            local_PE.clear();
        }

        // k-pairwise
        arma::Col<double> local_pK_model(nspins + 1, arma::fill::zeros);

        double E = 0.0, P = 0.0;

        size_t start = thread_id * chunk_size;
        size_t end   = std::min(start + chunk_size, total);
        BinaryPermutationsSequence sequence(nspins, start, end);

        // logger->info("thread {} start {} end {}", thread_id, start, end);
        for (const auto &s : sequence)
        {
            E             = energyAllPairs(s);
            P             = utils::exp_q(-beta * E, params.q_val);
            local_bracket = 1.0 - one_minus_q * beta * E;
            if (P > local_max_weight)
            {
                local_max_weight        = P;
                local_max_bracket       = local_bracket;
                local_max_weight_energy = E;
            }

            // Support counting: Tsallis q<1 typically yields exact zeros via cutoff.
            if (local_bracket > 0.0)
            {
                local_N_supp += 1;
            }

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

            // k_pairwise: always compute p(k)
            int k = static_cast<int>(arma::sum(s + 1) / 2);
            local_pK_model(k) += P;

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

                int E_bin = static_cast<int>(std::round(E / params.energy_bin));
                local_GE[E_bin] += 1.0;
                local_PE[E_bin] += P;
            }
        }

#pragma omp critical
        {
            if (local_max_weight > max_weight)
            {
                max_weight        = local_max_weight;
                max_bracket       = local_max_bracket;
                max_weight_energy = local_max_weight_energy;
            }
            avg_energy += local_avg_energy;
            avg_energy_sq += local_avg_energy_sq;
            avg_magnetization += local_avg_magnetization;
            m1_model += local_m1_model;
            m2_model += local_m2_model;
            Z_partition += local_Z;
            // k_pairwise: always compute p(k)
            pK_model += local_pK_model;

            N_supp += local_N_supp;

            if (triplets)
            {
                m3_model += local_m3_model;

                // merge local_H -> global H
                for (const auto &[bin, weight] : local_GE)
                {
                    GE[bin] += weight; // accumulate
                }
                for (const auto &[bin, weight] : local_PE)
                {
                    PE[bin] += weight; // accumulate
                }
            }
        }
    } // End of parallel block

    // Normalize final averages
    avg_energy /= Z_partition;
    avg_energy_sq /= Z_partition;
    avg_magnetization /= Z_partition;
    max_weight /= Z_partition;

    m1_model /= Z_partition;
    m2_model /= Z_partition;
    if (triplets)
    {
        m3_model /= Z_partition;
        for (auto &kv : PE)
        {
            kv.second /= Z_partition; // now H[bin] ~ P_q(E_bin)
        }
    }
    // k-pairwise
    pK_model /= Z_partition;
    f_supp = static_cast<double>(N_supp) / static_cast<double>(total);
    // logger->info(
    //     "Support fraction f_supp(beta={:.6g}, q={:.6g}) = {:.6g} (N_supp = {}), total = {:d}",
    //     beta, params.q_val, f_supp, N_supp, total);
}
