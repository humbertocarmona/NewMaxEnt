#include "trainers/wang_landau_trainer.hpp"

/**
 * @brief Randomly flips a single spin in the spin vector.
 *
 * Selects a random index in the input column vector `s` and flips the spin at that index
 * (i.e., multiplies it by -1). Assumes spins are represented as integers (+1 or -1).
 *
 * @param s   Reference to a column vector of spins (arma::Col<int>).
 * @param rng Reference to a random number generator (std::mt19937).
 */
void WangLandauTrainer::flip_random_spin(arma::Col<int> &s, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, s.n_elem - 1);
    int i = dist(rng);
    s(i) *= -1; // flip spin from +1 to -1 or vice versa
}

/**
 * @brief Checks if a histogram is flat based on a flatness threshold.
 *
 * Determines whether the histogram `H` is sufficiently flat by comparing the
 * minimum and maximum counts. The histogram is considered flat if the minimum
 * count is greater than `flatness_threshold` times the maximum count.
 *
 * @param H                  A histogram mapping energy values to their counts.
 * @param flatness_threshold Threshold ratio (default is 0.8) to determine flatness.
 * @return true if the histogram is flat, false otherwise.
 */
bool WangLandauTrainer::is_flat(const std::unordered_map<int, int> &H,
                                double flatness_threshold = 0.8)
{
    int min_H = std::numeric_limits<int>::max();
    int max_H = 0;
    for (auto &[E, count] : H)
    {
        min_H = std::min(min_H, count);
        max_H = std::max(max_H, count);
    }
    return (min_H > flatness_threshold * max_H);
}

void WangLandauTrainer::densityOfStates()
{
    auto logger = getLogger();

    auto &h    = core.h;
    auto &J    = core.J;
    int nspins = core.nspins;

    // RNG for spin updates

    std::mt19937 rng(1); // you have two RNGs declared: keep only this one or the next one
    // std::mt19937 rng(std::random_device{}()); // uncomment this for random seed per run


    arma::Col<int> s = arma::ones<arma::Col<int>>(nspins); // initial state: all spins up

    // log_g: estimated log density of states g(E), H: histogram of visits to energy bins


    double log_f  = 1.0;                             // initial modification factor f = e^1
    double E_real = energyAllPairs(s); // compute energy of current config
    int E_bin     = static_cast<int>(std::round(E_real / energy_bin)); // assign energy to bin

    size_t iter = 0;
    while (log_f > log_f_final && iter < 10000)
    {
        ++iter;
        H.clear(); // reset histogram for new round of sampling

        // Main Wang-Landau loop: perform a fixed number of Monte Carlo steps
        for (size_t sweep = 0; sweep < max_trials; ++sweep)
        {
            arma::Col<int> s_new = s;
            flip_random_spin(s_new, rng); // propose a single-spin flip

            double E_trial  = energyAllPairs(s_new);
            int E_trial_bin = static_cast<int>(std::round(E_trial / energy_bin));

            // Read log_g values for current and proposed energies
            // Note: if not previously visited, default-initialized to 0.0
            double ln_g_E       = log_g[E_bin];
            double ln_g_E_trial = log_g[E_trial_bin];

            // Acceptance probability based on log density of states
            double p = std::exp(ln_g_E - ln_g_E_trial);
            auto r   = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

            if (r < std::min(1.0, p))
            {
                // Accept the move
                s      = s_new;
                E_real = E_trial;
                E_bin  = E_trial_bin;
            }

            // Update log_g(E) and histogram H(E)
            log_g[E_bin] += log_f;
            H[E_bin]++;
        }

        // Check histogram flatness → reduce modification factor
        if (is_flat(H))
        {
            log_f /= 2.0; // reduce f multiplicatively (log(f) halves)
            logger->info("{} Histogram flat → reducing f to {:.2e}", iter, log_f);
        }
    }

    logger->info("saving result after {} iterations with log_f = {:.2e}", iter, log_f);

    std::map<int, double> g_normalized;
    double Z = 0.0;
    for (const auto &[E, logg] : log_g)
    {
        Z += std::exp(logg);
    }

    for (const auto &[E, logg] : log_g)
    {
        g_normalized[E] = std::exp(logg) / Z;
    }

    // Save log_g(E) to file (rescaled to original energy units)
    std::ofstream out("log_g.csv");
    out << "E" << "," << "log_g" << "\n";

    for (const auto &[E, val] : log_g)
    {
        out << E * energy_bin << "," << val << "\n"; // energy bin center, log density
    }
    out.close();
}

int howTouseIt()
{
    // main.cpp
    std::mt19937 rng(1);
    int nspins = 16; // number of spins

    // Example: random fields and couplings
    arma::vec h(nspins);
    int nedges = nspins * (nspins - 1) / 2;
    arma::vec J(nedges);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    int idx = 0;
    for (int i = 0; i < nspins; ++i)
    {
        h(i) = 0.01 * dist(rng) - 0.5;
        for (int j = i + 1; j < nspins; ++j)
        {
            J(idx++) = 0.01 * dist(rng);
        }
    }

    // wang_landau(h, J);

    return 0;
}