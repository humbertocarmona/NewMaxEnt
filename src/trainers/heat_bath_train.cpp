#include "trainers/compute_cost.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <chrono> // Add at the top of your file

/**
 * TODO:
 * 1 - save checkpoint files every n iterations
 * 2 - add an iter_0 option to start from a previous iteration, do it as an argument
 * 3 - modify log removing total cost
 */

void HeatBathTrainer::train()
{
    auto logger = getLogger();
    logger->info("[hb train] Starting Heat Bath training q_val = {}", params.q_val);

    using clock        = std::chrono::high_resolution_clock;
    auto last_log_time = clock::now(); // 🔥 Start the timer

    for (iter = iter; iter < params.maxIterations; ++iter)
    {
        computeModelAverages(1.0, false);
        // updateModelParameters(iter);
        oldUpdateModel(iter);

        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(params.tolerance_h, params.tolerance_J))
        {
            logger->info("[hb train] Monte Carlo converged at iteration {}", iter);
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
            logger->info("[hb train] Iter {:5d} |h_mean: {:5.2f}| |h_max: {:5.2f}| M1: {:9.6f} | "
                         "M2: {:9.6f} | elapsed: {:5.2f} {:4.2e} {:4.2e}",
                         iter, h_mean, h_max, cost.cost_m1, cost.cost_m2, elapsed, eta_h_t,
                         eta_J_t);
        }
        if (iter % 100 == 0)
        {
            saveModel("checkpoint");
        }
    }

    logger->debug("[hb train] Finished Monte Carlo training.");
}
