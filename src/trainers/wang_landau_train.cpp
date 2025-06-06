#include "trainers/compute_cost.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <chrono> // Add at the top of your file

void WangLandauTrainer::train()
{
    auto logger        = getLogger();
    using clock        = std::chrono::high_resolution_clock;
    auto last_log_time = clock::now(); // 🔥 Start the timer

    for (iter = iter; iter < params.maxIterations; ++iter)
    {
        computeDensityOfStates();

        computeModelAverages(1.0, false);

        if (params.updateType == 'g')
        {
            gradUpdateModel(iter);
        }
        else if (params.updateType == 's')
        {
            gradUpdateModelSeq(iter);
        }
        else
        {
            plawUpdateModel(iter);
        }
        
        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(params.tolerance_h, params.tolerance_J))
        {
            logger->info("[wl train] Wang Landau converged at iteration {}", iter);
            break;
        }
        if (iter % 10 == 0)
        {
            auto now       = clock::now();
            double elapsed = std::chrono::duration<double>(now - last_log_time).count();
            last_log_time  = now;
            double h_mean  = arma::mean(core.h);
            double J_mean  = arma::mean(core.J);
            double h_max   = arma::max(arma::abs(core.h));

            logger->info("[hb train] Iter {:5d} | M1: {:9.6f} | M2: {:9.6f} | elapsed: {:5.2f} | eta_h_t: {:4.2e} | eta_J_t: {:4.2e}",
                         iter, cost.cost_m1, cost.cost_m2, elapsed, eta_h_t, eta_J_t);
        }
    }

    logger->debug("[wl train] Finished Wang Landau training.");
}
