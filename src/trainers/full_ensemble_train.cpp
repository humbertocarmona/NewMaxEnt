#include "trainers/full_ensemble_trainer.hpp"
#include "trainers/compute_cost.hpp"
#include "utils/get_logger.hpp"

void FullEnsembleTrainer::train()
{
    auto logger = getLogger();
    logger->info("[train] Starting full enumeration training");
    auto &h = core.h;
    auto &J = core.J;

    h.fill(0);
    J.fill(0);

    for (size_t iter=1; iter<maxIterations; ++iter){
        computeFullEnumerationAverages(1.0, false); // beta = 1 for training, triplets=false don't need m3_model here
        updateModelParameters(iter);
        auto cost = compute_cost(m1_data, m2_data, m1_model, m2_model);
        if (cost.check_convergence(tolerance_h, tolerance_J))
        {
            logger->info("[train] Full ensemble converged at iteration {}", iter);
             break;
        }
        if (iter % 10 == 0)
        {
            logger->debug("[train] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f}",
                         iter,
                         cost.cost_total,
                         cost.cost_m1,
                         cost.cost_m2);
        }
    }
    computeFullEnumerationAverages(1.0, true);
    logger->debug("[train] Finished full enumeration training.");
}
