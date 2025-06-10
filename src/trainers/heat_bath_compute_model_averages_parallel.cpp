#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <cmath>
#include <omp.h> // OpenMP
#include <random>

// Perform Heat-Bath sampling and compute model averages
void HeatBathTrainer::computeModelAverages(double beta, bool triplets)
{
    auto logger   = getLogger();
    size_t nspins = core.nspins;
    size_t nedges = core.nedges;

    auto &h     = core.h;
    auto &J     = core.J;
    auto &K     = core.K;
    auto &edges = core.edges;

    // Initialize global averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    if (triplets)
        m3_model.zeros(ntriplets);

    // k-pairwise
    pK_model.zeros(nspins + 1);

    avg_energy        = 0.0;
    avg_energy_sq     = 0.0;
    avg_magnetization = 0.0;

    size_t global_sample_count = 0; // shared across threads
// Parallel block
#pragma omp parallel
    {
        int thread_id   = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        if (iter == 1 && thread_id == 0)
        {
            logger->info("[computeAverages] Running with {} threads.", num_threads);
        }

        size_t samples_per_thread =
            (total_number_samples + num_threads - 1) / num_threads; // ceil division
        size_t start_index = thread_id * samples_per_thread;

        // Local accumulators per thread
        arma::Col<double> local_m1_model(nspins, arma::fill::zeros);
        arma::Col<double> local_m2_model(nedges, arma::fill::zeros);
        arma::Col<double> local_m3_model;
        if (triplets)
            local_m3_model.set_size(ntriplets), local_m3_model.zeros();

        // k-pairwise
        arma::Col<double> local_pK_model(nspins + 1, arma::fill::zeros);

        double local_avg_energy        = 0.0;
        double local_avg_energy_sq     = 0.0;
        double local_avg_magnetization = 0.0;

        // Thread-local random number generator
        // std::random_device rd;
        // std::mt19937 rng(rd() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Local replicas collection (only if triplets are needed)
        arma::Mat<int> local_replicas;
        if (triplets)
            local_replicas.set_size(2 * samples_per_thread, nspins); // need to be safe

        size_t local_sample_count = 0;

#pragma omp for
        for (size_t n = 0; n < params.number_repetitions; ++n)
        {
            std::mt19937 rng(mc_seed + n);

            arma::Col<int> s(nspins, arma::fill::ones);
            s *= -1; // Initialize spins to -1
            double prob_plus;

            // Equilibration sweeps
            for (size_t sweep = 0; sweep < params.step_equilibration; ++sweep)
            {
                int ki = static_cast<int>(arma::sum(s + 1) / 2);
                for (size_t i = 0; i < nspins; ++i)
                {
                    double h_i = h(i);

                    for (size_t j = 0; j < nspins; ++j)
                    {
                        int ij = edges(i, j);
                        if (ij != -1)
                            h_i += J(ij) * s(j);
                    }
                    if (params.k_pairwise)
                    {

                        double exp_plus  = std::exp(beta * h_i);
                        double exp_minus = std::exp(-beta * h_i);
                        if (s(i) == -1)
                        {
                            exp_plus += K(ki + 1); // will increase ki by +1
                            exp_minus += K(ki);    // keep the same
                        }
                        else
                        {
                            exp_plus += K(ki);      // keep the same
                            exp_minus += K(ki - 1); // decrease ki by 1
                        }
                        prob_plus = exp_plus / (exp_plus + exp_minus);
                    }
                    else
                    {
                        prob_plus = 1.0 / (1.0 + std::exp(-2.0 * beta * h_i));
                    }
                    double r = dist(rng);
                    s(i)     = (r < prob_plus) ? 1 : -1;
                }
            }

            // Sampling phase
            size_t n_collected = 0;
            size_t sweep       = 0;
            while (n_collected < params.num_samples)
            {
                int ki = static_cast<int>(arma::sum(s + 1) / 2);

                for (size_t i = 0; i < nspins; ++i)
                {
                    double h_i = h(i);
                    for (size_t j = 0; j < nspins; ++j)
                    {
                        int ij = edges(i, j);
                        if (ij != -1)
                            h_i += J(ij) * s(j);
                    }
                    if (params.k_pairwise)
                    {
                        double exp_plus  = std::exp(beta * h_i);
                        double exp_minus = std::exp(-beta * h_i);
                        if (s(i) == -1)
                        {
                            exp_plus += K(ki + 1); // will increase ki by +1
                            exp_minus += K(ki);    // keep the same
                        }
                        else
                        {
                            exp_plus += K(ki);      // keep the same
                            exp_minus += K(ki - 1); // decrease ki by 1
                        }
                        prob_plus = exp_plus / (exp_plus + exp_minus);
                    }
                    else
                    {
                        prob_plus = 1.0 / (1.0 + std::exp(-2.0 * beta * h_i));
                    }
                    double r = dist(rng);
                    s(i)     = (r < prob_plus) ? 1 : -1;
                }

                if ((sweep % params.step_correlation) == 0)
                {
                    double E = energyAllPairs(s);
                    local_avg_energy += E;
                    local_avg_energy_sq += E * E;
                    local_avg_magnetization += arma::mean(arma::conv_to<arma::vec>::from(s));

                    for (size_t i = 0; i < nspins; ++i)
                        local_m1_model(i) += s(i);

                    size_t idx = 0;
                    for (size_t i = 0; i < nspins - 1; ++i)
                        for (size_t j = i + 1; j < nspins; ++j)
                            local_m2_model(idx++) += s(i) * s(j);

                    if (triplets)
                    {
                        idx = 0;
                        for (size_t i = 0; i < nspins - 2; ++i)
                            for (size_t j = i + 1; j < nspins - 1; ++j)
                                for (size_t k = j + 1; k < nspins; ++k)
                                    local_m3_model(idx++) += s(i) * s(j) * s(k);

                        local_replicas.row(local_sample_count) = s.t();
                    }
                    // k-pairwise
                    int k = static_cast<int>(arma::sum(s + 1) / 2);
                    local_pK_model(k) += 1.0;

                    ++local_sample_count; // because a thread may not collect all samples or collect
                                          // more

                    ++n_collected;
                }
                ++sweep;
            }
        }

        // Critical section: merge thread-local results
        size_t my_start_index = 0;

#pragma omp critical
        {
            logger->debug("thread: {} local_sample_count = {} /{}", thread_id, local_sample_count,
                          samples_per_thread);

            my_start_index = global_sample_count;
            global_sample_count += local_sample_count;

            avg_energy += local_avg_energy;
            avg_energy_sq += local_avg_energy_sq;
            avg_magnetization += local_avg_magnetization;
            m1_model += local_m1_model;
            m2_model += local_m2_model;
            if (triplets)
                m3_model += local_m3_model;

            // k-pairwise
            pK_model += local_pK_model;

            if (triplets)
            {
                for (size_t i = 0; i < local_sample_count; ++i)
                {
                    replicas.row(my_start_index + i) = local_replicas.row(i);
                }
            }
        }
    } // End of parallel block

    // Normalize final averages
    avg_energy /= static_cast<double>(global_sample_count);
    avg_energy_sq /= static_cast<double>(global_sample_count);
    avg_magnetization /= static_cast<double>(global_sample_count);

    m1_model /= static_cast<double>(global_sample_count);
    m2_model /= static_cast<double>(global_sample_count);
    if (triplets)
        m3_model /= static_cast<double>(global_sample_count);

    // k-pairwise
    pK_model /= static_cast<double>(global_sample_count);
}
