#include "trainers/compute_cost.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void WangLandauTrainer::train()
{
    auto logger = getLogger();
    logger->info("[wl train] Starting Wang Landau training q_val = {}", params.q_val);

    for (iter = iter; iter < params.maxIterations; ++iter)
    {
        computeDensityOfStates();
        computeModelAverages(1.0, false);
        updateModelParameters(iter);

        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(params.tolerance_h, params.tolerance_J))
        {
            logger->info("[wl train] Wang Landau converged at iteration {}", iter);
            break;
        }
        if (iter % 10 == 0)
        {
            logger->info("[wl train] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f}", iter,
                         cost.cost_total, cost.cost_m1, cost.cost_m2);
        }
    }

    logger->debug("[wl train] Finished Wang Landau training.");
}
