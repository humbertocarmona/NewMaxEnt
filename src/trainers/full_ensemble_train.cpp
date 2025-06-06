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
            logger->info("[full train] Iter {:5d} | elapsed: {:5.2f} | eta_h: {:5.3f} | eta_J: "
                         "{:5.3f} | M1: {:9.6f} | "
                         "M2: {:9.6f} grad_h: {:9.6f} grad_J: {:9.6f}",
                         iter, elapsed, eta_h_t, eta_J_t, cost.cost_m1, cost.cost_m2,
                         last_grad_norm_h, last_grad_norm_J);
        }
        if (iter % params.save_checkpoint == 0)
        {
            saveModel("checkpoint");
        }
    }

    logger->debug("[f train] Finished full enumeration training.");
}
