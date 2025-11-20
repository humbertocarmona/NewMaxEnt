#include "trainers/compute_cost.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void FullEnsembleTrainer::train()
{
    auto logger = getLogger();
    logger->info("[f train] Starting full enumeration training q_val = {}", params.q_val);
    using clock        = std::chrono::high_resolution_clock;
    auto last_log_time = clock::now(); // 🔥 Start the timer

    for (iter = iter; iter < params.maxIterations; ++iter)
    {
        computeModelAverages(1.0, false);
        if (params.updateType == 2)
        {
            gradUpdateModel(iter);
        }
        else if (params.updateType == 3)
        {
            gradUpdateModelSeq(iter);
        }
        else
        {
            plawUpdateModel(iter);
        }
        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model, pK_data, pK_model);

        if (cost.check_convergence(params.tolerance_h, params.tolerance_J))
        {
            logger->info("[f train] Full ensemble converged at iteration {}", iter);
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
            logger->info("[full train] Iter {:5d} | M1: {:11.8f} | M2: {:11.8f} | pk: {:9.6f} | "
                         "eta_h_t: {:4.2e} | eta_J_t: {:4.2e}",
                         iter, cost.cost_m1, cost.cost_m2, cost.cost_pk, eta_h_t, eta_J_t);
        }
        if (iter % params.save_checkpoint == 0)
        {
            saveModel(params.file_checkpoint, false); 
        }
    }

    logger->debug("[f train] Finished full enumeration training.");
}
