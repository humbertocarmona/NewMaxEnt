#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <cmath>
#include <random>

// Perform Heat-Bath sampling and compute model averages
void HeatBathTrainer::computeModelAverages1(double beta, bool triplets)
{

    auto logger   = getLogger();
    size_t nspins = core.nspins;
    size_t nedges = core.nedges;

    auto &h     = core.h;
    auto &J     = core.J;
    auto &edges = core.edges;
    // logger->info("iter={}", iter);
    // Initialize the model averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    m3_model.zeros(ntriplets);

    avg_energy        = 0.0;
    avg_energy_sq     = 0.0;
    avg_magnetization = 0.0;


    // Random number generator setup
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    arma::Col<int> s(nspins);

    replicas.fill(-1); // Initialize replicas to -1
    int n_samples_collected = 0;
    for (size_t n = 0; n < params.number_repetitions; ++n)
    {
        // std::mt19937 rng(std::random_device{}());
        std::mt19937 rng(mc_seed + n); // each repetition has a determined  seed
        s.fill(-1);

        for (size_t sweep = 0; sweep < params.step_equilibration; ++sweep)
        {
            for (size_t i = 0; i < nspins; ++i)
            {
                double h_i = h(i);
                for (size_t j = 0; j < nspins; ++j)
                {
                    int ij = edges(i, j);
                    if (ij != -1)
                    {
                        h_i += J(ij) * s(j);
                    }
                }
                // double exp_plus  = std::exp(beta * h_i);  // weight for s(i) = -1
                // double exp_minus = std::exp(-beta * h_i); // weight for s(i) = +1
                // double prob_plus = exp_plus / (exp_plus + exp_minus);
                double prob_plus = 1.0 / (1.0 + std::exp(-2.0 * beta * h_i));

                double r         = dist(rng);
                s(i)             = (r < prob_plus) ? 1 : -1;
            }
        }

        // Collect samples to compute averages
        size_t n_collected_rep = 0;
        for (size_t sweep = 0; n_collected_rep < params.num_samples; ++sweep)
        {
            // Perform a sweep for each spin i
            for (size_t i = 0; i < nspins; ++i)
            {
                double h_i = h(i);
                for (size_t j = 0; j < nspins; ++j) // sum over neighbors
                {

                    int ij = edges(i, j);
                    if (ij != -1)
                    {
                        h_i += J(ij) * s(j);
                    }
                }
                // double exp_plus  = std::exp(beta * h_i);  // weight for s(i) = -1
                // double exp_minus = std::exp(-beta * h_i); // weight for s(i) = +1
                // double prob_plus = exp_plus / (exp_plus + exp_minus);
                double prob_plus = 1.0 / (1.0 + std::exp(-2.0 * beta * h_i));

                double r         = dist(rng);
                s(i)             = (r < prob_plus) ? 1 : -1;
            }

            // Every sampleInterval sweeps, record the current configuration
            if ((sweep % params.step_correlation) == 0)
            {
                double E = energyAllPairs(s);
                avg_energy += E;
                avg_energy_sq += E * E;
                avg_magnetization += arma::mean(arma::conv_to<arma::vec>::from(s));

                for (size_t i = 0; i < nspins; ++i)
                {
                    m1_model(i) += s(i);
                }
                size_t idx = 0;
                for (size_t i = 0; i < nspins - 1; ++i)
                {
                    for (size_t j = i + 1; j < nspins; ++j)
                    {
                        m2_model(idx++) += s(i) * s(j);
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
                                m3_model(idx++) += s(i) * s(j) * s(k);
                            }
                        }
                    }
                    replicas.row(n_samples_collected) = s.t();
                }
                
                n_collected_rep++;
                n_samples_collected++;
            }
        }

    } // end for loop repetitions

    
    // Normalize the averages
    avg_energy /= static_cast<double>(total_number_samples);
    avg_energy_sq /= static_cast<double>(total_number_samples);
    avg_magnetization /= static_cast<double>(total_number_samples);

    m1_model /= static_cast<double>(total_number_samples);
    m2_model /= static_cast<double>(total_number_samples);
    if (triplets)
        m3_model /= static_cast<double>(total_number_samples);
}
