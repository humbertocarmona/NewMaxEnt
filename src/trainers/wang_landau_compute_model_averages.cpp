#include "io/make_file_names.hpp"
#include "trainers/wang_landau_trainer.hpp"

/**
 * @brief Implements BaseTrainer computeModelAverages.
 *
 * Here it is assumed that WangLandauTrainer::computeDensityOfStates is already run
 * needs:
 *      log_g_E:estimated log density of states g(E),
 *
 */

void WangLandauTrainer::computeModelAverages(double beta, bool triplets)
{
    auto logger = getLogger();

    auto &h    = core.h;
    auto &J    = core.J;
    int nspins = core.nspins;
    int nedges = core.nedges;

    // Initialize the model averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    m3_model.zeros(ntriplets);

    // RNG for spin updates
    std::mt19937 rng(wg_seed);

    // initial state: all spins up
    arma::Col<int> s = arma::ones<arma::Col<int>>(nspins);

    double E  = energyAllPairs(s);                            // compute energy of current config
    int E_bin = static_cast<int>(std::round(E / energy_bin)); // assign energy to bin

    size_t samplesCollected = 0;
    std::vector<double> log_weights(numSamples, 0.0);
    for (size_t sweep = 0; samplesCollected < numSamples; ++sweep)
    {
        arma::Col<int> s_trial = s;
        flip_random_spin(s_trial, rng); // propose a single-spin flip

        double E_trial  = energyAllPairs(s_trial);
        int E_trial_bin = static_cast<int>(std::round(E_trial / energy_bin));

        // Read log_g_E values for current and proposed energies
        // Note: if not previously visited, default-initialized to 0.0 so avoid directly
        // double ln_g_E       = log_g_E[E_bin];
        // double ln_g_E_trial = log_g_E[E_trial_bin];

        auto it_current = log_g_E.find(E_bin);
        auto it_trial   = log_g_E.find(E_trial_bin);
        if (it_current == log_g_E.end() || it_trial == log_g_E.end())
        {
            continue; // skip this move if either energy bin isn't in the density of states
        }

        double ln_g_E       = it_current->second;
        double ln_g_E_trial = it_trial->second;

        // Acceptance probability based on log density of states
        double p = std::exp(ln_g_E - ln_g_E_trial);
        auto r   = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        // Accept the move if...
        if (r < std::min(1.0, p))
        {
            s     = s_trial;
            E     = E_trial;
            E_bin = E_trial_bin;
        }

        if ((sweep % sampleInterval) == 0)
        {
            double log_P_E                = -beta * E - log_g_E[E_bin];
            double P_E                    = std::exp(log_P_E);
            log_weights[samplesCollected] = log_P_E; // ✅ store the log

            avg_energy += P_E * E;
            avg_energy_sq += P_E * E * E;
            avg_magnetization += P_E * arma::mean(arma::conv_to<arma::vec>::from(s));
            for (size_t i = 0; i < nspins; ++i)
            {
                m1_model(i) += P_E * s(i);
            }
            size_t idx = 0;
            for (size_t i = 0; i < nspins - 1; ++i)
            {
                for (size_t j = i + 1; j < nspins; ++j)
                {
                    m2_model(idx++) += P_E * s(i) * s(j);
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
                            m3_model(idx++) += P_E * s(i) * s(j) * s(k);
                        }
                    }
                }
                replicas.row(samplesCollected) = s.t();
            }
            samplesCollected++;
        }
    }

    double Z = std::exp(logsumexp(log_weights));
    // Normalize the averages
    avg_energy /= Z;
    avg_energy_sq /= Z;
    avg_magnetization /= Z;

    m1_model /= Z;
    m2_model /= Z;
    if (triplets)
        m3_model /= Z;
}