#include "trainers/full_ensemble_trainer.hpp"
#include "trainers/compute_cost.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void FullEnsembleTrainer::train()
{
    auto logger = getLogger();
    logger->info("[f train] Starting full enumeration training q_val = {}", params.q_val);
    
    for (iter=iter; iter<params.maxIterations; ++iter){
        computeModelAverages(false); // beta = 1 for training, triplets=false don't need m3_model here
        updateModelParameters(iter);

        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(params.tolerance_h, params.tolerance_J))
        {
            logger->info("[f train] Full ensemble converged at iteration {}", iter);
             break;
        }
        if (iter % 10 == 0)
        {
            logger->info("[f train] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f}",
                         iter,
                         cost.cost_total,
                         cost.cost_m1,
                         cost.cost_m2);
        }

    }

    logger->debug("[f train] Finished full enumeration training.");
}
