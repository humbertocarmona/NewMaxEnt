#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <cmath>
#include <random>
#include <omp.h> // OpenMP

// Perform Heat-Bath sampling and compute model averages
void HeatBathTrainer::computeModelAverages1(double beta, bool triplets)
{
    auto logger   = getLogger();
    size_t nspins = core.nspins;
    size_t nedges = core.nedges;
    

    auto &h     = core.h;
    auto &J     = core.J;
    auto &edges = core.edges;

    // Initialize global averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    if (triplets)
        m3_model.zeros(ntriplets);

    avg_energy        = 0.0;
    avg_energy_sq     = 0.0;
    avg_magnetization = 0.0;

    // Parallel block
    #pragma omp parallel
    {
        // Local accumulators per thread
        arma::Col<double> local_m1_model(nspins, arma::fill::zeros);
        arma::Col<double> local_m2_model(nedges, arma::fill::zeros);
        arma::Col<double> local_m3_model;
        if (triplets)
            local_m3_model.set_size(ntriplets), local_m3_model.zeros();

        double local_avg_energy = 0.0;
        double local_avg_energy_sq = 0.0;
        double local_avg_magnetization = 0.0;

        // Thread-local random number generator
        std::random_device rd;
        std::mt19937 rng(rd() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Local replicas collection (only if triplets are needed)
        arma::Mat<int> local_replicas;
        if (triplets)
            local_replicas.set_size(number_repetitions * num_samples, nspins);
        
        size_t local_sample_index = 0;

        #pragma omp for
        for (size_t n = 0; n < number_repetitions; ++n)
        {
            arma::Col<int> s(nspins, arma::fill::ones);
            s *= -1; // Initialize spins to -1

            // Equilibration sweeps
            for (size_t sweep = 0; sweep < step_equilibration; ++sweep)
            {
                for (size_t i = 0; i < nspins; ++i)
                {
                    double h_i = h(i);
                    for (size_t j = 0; j < nspins; ++j)
                    {
                        int ij = edges(i, j);
                        if (ij != -1)
                            h_i += J(ij) * s(j);
                    }
                    double exp_plus  = std::exp(beta * h_i);
                    double exp_minus = std::exp(-beta * h_i);
                    double prob_plus = exp_plus / (exp_plus + exp_minus);
                    double r         = dist(rng);
                    s(i)             = (r < prob_plus) ? 1 : -1;
                }
            }

            // Sampling phase
            size_t n_collected = 0;
            size_t sweep = 0;
            while (n_collected < num_samples)
            {
                for (size_t i = 0; i < nspins; ++i)
                {
                    double h_i = h(i);
                    for (size_t j = 0; j < nspins; ++j)
                    {
                        int ij = edges(i, j);
                        if (ij != -1)
                            h_i += J(ij) * s(j);
                    }
                    double exp_plus  = std::exp(beta * h_i);
                    double exp_minus = std::exp(-beta * h_i);
                    double prob_plus = exp_plus / (exp_plus + exp_minus);
                    double r         = dist(rng);
                    s(i)             = (r < prob_plus) ? 1 : -1;
                }

                if ((sweep % step_correlation) == 0)
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

                        local_replicas.row(local_sample_index++) = s.t();
                    }

                    ++n_collected;
                }
                ++sweep;
            }
        }

        // Critical section: merge thread-local results
        #pragma omp critical
        {
            avg_energy += local_avg_energy;
            avg_energy_sq += local_avg_energy_sq;
            avg_magnetization += local_avg_magnetization;
            m1_model += local_m1_model;
            m2_model += local_m2_model;
            if (triplets)
                m3_model += local_m3_model;

            if (triplets)
            {
                for (size_t i = 0; i < local_sample_index; ++i)
                {
                    replicas.row(i) = local_replicas.row(i);
                }
            }
        }
    }

    // Normalize final averages
    avg_energy /= static_cast<double>(total_number_samples);
    avg_energy_sq /= static_cast<double>(total_number_samples);
    avg_magnetization /= static_cast<double>(total_number_samples);

    m1_model /= static_cast<double>(total_number_samples);
    m2_model /= static_cast<double>(total_number_samples);
    if (triplets)
        m3_model /= static_cast<double>(total_number_samples);
}
