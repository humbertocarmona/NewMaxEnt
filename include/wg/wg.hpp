#pragma once

#include "utils/get_logger.hpp"

#include <armadillo>
#include <random>
#include <unordered_map>

using Field = arma::Col<double>;
using Conf  = arma::Col<int>;

// Compute energy of a configuration using full all-to-all pairwise interactions
inline double energyAllPairs(Conf s, Field h, Field J, int nspins)
{
    double En = 0.0;
    // Linear term: external field contribution
    for (int i = 0; i < nspins; ++i)
        En += h(i) * s(i);
    
    // Quadratic term: pairwise interactions (J is flattened upper triangle of J_ij)
    int idx = 0;
    for (int i = 0; i < nspins - 1; ++i)
        for (int j = i + 1; j < nspins; ++j)
            En += J(idx++) * s(i) * s(j);

    return -En; // convention: minus sign for Boltzmann weight exp(-E)
}

// Randomly flip one spin in the configuration
inline void flip_random_spin(Conf &s, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, s.n_elem - 1);
    int i = dist(rng);
    s(i) *= -1; // flip spin from +1 to -1 or vice versa
}

// Check whether histogram is "flat" enough by comparing min and max counts
inline bool is_flat(const std::unordered_map<int, int> &H, double flatness_threshold = 0.8)
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

inline void wang_landau(const arma::vec &h,
                        const arma::mat &J,
                        double log_f_final = 1e-5,
                        double energy_bin  = 0.2)
{
    auto logger=getLogger();
    // RNG for spin updates

    std::mt19937 rng(1); // you have two RNGs declared: keep only this one or the next one
    // std::mt19937 rng(std::random_device{}()); // uncomment this for random seed per run

    int nspins        = h.n_elem;
    size_t max_trials = 100000;

    Conf s = arma::ones<Conf>(nspins);     // initial state: all spins up

    // log_g: estimated log density of states g(E), H: histogram of visits to energy bins
    std::unordered_map<int, double> log_g;
    std::unordered_map<int, int> H;

    double log_f      =  1.0;                   // initial modification factor f = e^1
    double E_real = energyAllPairs(s, h, J, nspins); // compute energy of current config
    int E_bin     = static_cast<int>(std::round(E_real / energy_bin)); // assign energy to bin

    
    size_t safe_while = 0;
    while (log_f > log_f_final && safe_while <10000)
    {
        ++safe_while;
        H.clear(); // reset histogram for new round of sampling

        // Main Wang-Landau loop: perform a fixed number of Monte Carlo steps
        for (size_t sweep = 0; sweep < max_trials; ++sweep)
        {
            Conf s_new = s;
            flip_random_spin(s_new, rng); // propose a single-spin flip

            double E_trial      = energyAllPairs(s_new, h, J, nspins);
            int E_trial_bin     = static_cast<int>(std::round(E_trial / energy_bin));

            // Read log_g values for current and proposed energies
            // Note: if not previously visited, default-initialized to 0.0
            double ln_g_E       = log_g[E_bin];
            double ln_g_E_trial = log_g[E_trial_bin];

            // Acceptance probability based on log density of states
            double p = std::exp(ln_g_E - ln_g_E_trial);
            auto r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

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
            logger->info("{} Histogram flat → reducing f to {:.2e}",safe_while, log_f);
        }
    }

    logger->info("saving result after {} iterations with log_f = {:.2e}",safe_while, log_f);


    std::map<int, double> g_normalized;
    double Z = 0.0;
    for (const auto& [E, logg] : log_g) {
        Z += std::exp(logg);
    }
    
    for (const auto& [E, logg] : log_g) {
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
