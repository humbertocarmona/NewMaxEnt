#include "io/make_file_names.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/utilities.hpp"

arma::Col<int> random_spin_config(int nspins, std::mt19937 &rng)
{
    arma::Col<int> s(nspins);
    std::uniform_int_distribution<int> dist(0, 1); // returns 0 or 1

    for (int i = 0; i < nspins; ++i)
    {
        s(i) = dist(rng) == 0 ? -1 : 1;
    }

    return s;
}

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

    // Prepare storage for observables
    avg_energy        = 0.0;
    avg_energy_sq     = 0.0;
    avg_magnetization = 0.0;

    // Initialize the model averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    m3_model.zeros(ntriplets);

    // RNG for spin updates
    std::mt19937 rng(wg_seed);

    // initial state: avoid all spins up or down
    arma::Col<int> s = random_spin_config(nspins, rng);

    // computer the energy now, and which bin it belongs to
    double E  = energyAllPairs(s);
    int E_bin = static_cast<int>(std::round(E / params.energy_bin));

    logger->debug("[computeModelAverages] Starting Wang Landau sampling");
    logger->debug("[computeModelAverages] E_0: {} E_bin: {}", E, E_bin);

    size_t sweep            = 0;
    size_t samplesCollected = 0;

    std::vector<double> log_weights(params.num_samples, 0.0);
    std::vector<double> energies(params.num_samples, 0.0);
    std::vector<double> magnetizations(params.num_samples, 0.0);

    // First moment
    arma::Col<double> m1(nspins);
    arma::Col<double> m2(nedges);
    arma::Col<double> m3(ntriplets);

    std::vector<arma::Col<double>> m1_list(params.num_samples, arma::Col<double>(nspins));
    std::vector<arma::Col<double>> m2_list(params.num_samples, arma::Col<double>(nedges));
    std::vector<arma::Col<double>> m3_list(params.num_samples, arma::Col<double>(ntriplets));
    ;

    size_t n_accepted = 0;
    size_t n_rejected = 0;

    while (samplesCollected < params.num_samples)
    {
        arma::Col<int> s_trial = s;
        flip_random_spin(s_trial, rng); // propose a single-spin flip

        double E_trial  = energyAllPairs(s_trial);
        int E_trial_bin = static_cast<int>(std::round(E_trial / params.energy_bin));

        // auto it_current = log_g_E.find(E_bin) reads log_g_E values for current and proposed
        // energies Note: if not previously visited, default-initialized to 0.0 so avoid directly
        // double ln_g_E       = log_g_E[E_bin];
        // double ln_g_E_trial = log_g_E[E_trial_bin];
        auto it_current = log_g_E.find(E_bin);
        auto it_trial   = log_g_E.find(E_trial_bin);
        if (it_current == log_g_E.end() || it_trial == log_g_E.end())
            continue;

        double ln_g_E       = it_current->second;
        double ln_g_E_trial = it_trial->second;

        double p = std::exp(ln_g_E - ln_g_E_trial);
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

        if (r < std::min(1.0, p))
        {
            n_accepted++;
            s     = s_trial;
            E     = E_trial;
            E_bin = E_trial_bin;
        }
        else
        {
            n_rejected++;
        }

        if (sweep % params.step_correlation == 0)
        {
            n_accepted                       = 0;
            n_rejected                       = 0;
            double log_P_E                   = -beta * E - log_g_E[E_bin];
            log_weights[samplesCollected]    = log_P_E;
            energies[samplesCollected]       = E;
            magnetizations[samplesCollected] = arma::mean(arma::conv_to<arma::vec>::from(s));

            for (size_t i = 0; i < nspins; ++i)
                m1(i) = s(i);
            m1_list[samplesCollected] = m1;

            // Second moment
            size_t idx = 0;
            for (size_t i = 0; i < nspins - 1; ++i)
                for (size_t j = i + 1; j < nspins; ++j)
                    m2(idx++) = s(i) * s(j);

            m2_list[samplesCollected] = m2;

            // Third moment (optional)
            if (triplets)
            {
                arma::Col<double> m3(ntriplets);
                idx = 0;
                for (size_t i = 0; i < nspins - 2; ++i)
                    for (size_t j = i + 1; j < nspins - 1; ++j)
                        for (size_t k = j + 1; k < nspins; ++k)
                            m3(idx++) = s(i) * s(j) * s(k);
                m3_list[samplesCollected]      = m3;
                replicas.row(samplesCollected) = s.t();
            }

            logger->debug("[wl train] ...................................");
            logger->debug("[wl train]  sweep {}  E: {} E_bin: {} p: {} r: {}", sweep, E, E_bin, p, r);
            logger->debug("[wl train] s: {}", utils::colPrint<int>(s));
            logger->debug("[wl train] sample {} n_accepted: {} n_rejected: {} ", samplesCollected, n_accepted, n_rejected);
            logger->debug("[wl train] log_P_E: {}", log_P_E);
            logger->debug("[wl train] ...................................");

            ++samplesCollected;
        }
        ++sweep;
    }

    // Normalize in log-space
    double logZ = logsumexp(log_weights);

    for (size_t i = 0; i < samplesCollected; ++i)
    {
        double weight = std::exp(log_weights[i] - logZ);

        avg_energy += weight * energies[i];
        avg_energy_sq += weight * energies[i] * energies[i];
        avg_magnetization += weight * magnetizations[i];
        m1_model += weight * m1_list[i];
        m2_model += weight * m2_list[i];
        if (triplets)
            m3_model += weight * m3_list[i];
    }

    logger->debug("[wl train] Averages computed from {} samples", samplesCollected);
    logger->debug("[wl train] avg_energy: {}", avg_energy);
    logger->debug("[wl train] avg_magnetization: {}", avg_magnetization);
    logger->debug("[wl train] m1_model: {}", utils::colPrint<double>(m1_model));
    logger->debug("[wl train] m1_data: {}", utils::colPrint<double>(m1_data));

}
