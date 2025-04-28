#include "trainers/compute_cost.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <chrono> // Add at the top of your file


void HeatBathTrainer::train()
{
    auto logger = getLogger();
    logger->info("[hb train] Starting Heat Bath training q_val = {}", q_val);
    
    using clock = std::chrono::high_resolution_clock;
    auto last_log_time = clock::now(); // 🔥 Start the timer

    for (iter = iter; iter < maxIterations; ++iter)
    {
        computeModelAverages(
            1.0, false); // beta = 1 for training, triplets=false don't need m3_model here
        updateModelParameters(iter);

        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(tolerance_h, tolerance_J))
        {
            logger->info("[hb train] Monte Carlo converged at iteration {}", iter);
            break;
        }
        if (iter % 10 == 0)
        {
            auto now = clock::now();
            double elapsed = std::chrono::duration<double>(now - last_log_time).count();
            last_log_time = now;

            logger->info("[hb train] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f} | elapsed: {:.3f} s", iter,
                         cost.cost_total, cost.cost_m1, cost.cost_m2, elapsed);
        }
    }
    computeModelAverages(1.0, true);

    logger->debug("[hb train] Finished Monte Carlo training.");
}
