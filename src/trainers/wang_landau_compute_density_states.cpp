#include "io/make_file_names.hpp"
#include "trainers/wang_landau_trainer.hpp"

void WangLandauTrainer::computeDensityOfStates()
{
    auto logger = getLogger();

    auto &h    = core.h;
    auto &J    = core.J;
    int nspins = core.nspins;

    // RNG for spin updates

    std::mt19937 rng(1); // you have two RNGs declared: keep only this one or the next one
    // std::mt19937 rng(std::random_device{}()); // uncomment this for random seed per run

    arma::Col<int> s = arma::ones<arma::Col<int>>(nspins); // initial state: all spins up

    // log_g_E: estimated log density of states g(E), H: histogram of visits to energy bins

    double log_f  = 1.0;               // initial modification factor f = e^1
    double E_real = energyAllPairs(s); // compute energy of current config
    int E_bin     = static_cast<int>(std::round(E_real / energy_bin)); // assign energy to bin

    log_g_E.clear();
    H.clear();

    logger->info("[computeDensityOfStates] Wang-Landau started computing the DOS");
    size_t iter = 0;
    while (log_f > log_f_final && iter < 10000)
    {
        ++iter;
        H.clear(); // reset histogram for new round of sampling

        // Main Wang-Landau loop: perform a random walk
        for (size_t sweep = 0; sweep < equilibrationSweeps; ++sweep)
        {
            arma::Col<int> s_new = s;
            flip_random_spin(s_new, rng); // propose a single-spin flip

            double E_trial  = energyAllPairs(s_new);
            int E_trial_bin = static_cast<int>(std::round(E_trial / energy_bin));

            // Read log_g_E values for current and proposed energies
            // Note: if not previously visited, default-initialized to 0.0
            double ln_g_E       = log_g_E[E_bin];
            double ln_g_E_trial = log_g_E[E_trial_bin];

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

            // Update log_g_E(E) and histogram H(E)
            log_g_E[E_bin] += log_f;
            H[E_bin]++;
        }

        // Check histogram flatness → reduce modification factor
        if (is_flat(H))
        {
            log_f /= 2.0; // reduce f multiplicatively (log(f) halves)
            // logger->info("[computeDensityOfStates] {} Histogram flat → reducing f to {:.2e}",
            // iter, log_f);
        }
    }

    if (iter >= 10000)
    {
        logger->warn("[computeDensityOfStates] Consider increasing iter > {}", iter);
    }
    // logger->info("saving result after {} iterations with log_f = {:.2e}", iter, log_f);
    // // Save log_g_E(E) to file (rescaled to original energy units)
    // auto output = io::make_DensOfStates_filename(params)
    // std::ofstream out("log_g_E.csv");
    // out << "E" << "," << "log_g_E" << "\n";

    // for (const auto &[E, val] : log_g_E)
    // {
    //     out << E * energy_bin << "," << val << "\n"; // energy bin center, log density
    // }
    // out.close();
    logger->info("[computeDensityOfStates] Wang-Landau finished computing the DOS {:.22}", log_f);
}