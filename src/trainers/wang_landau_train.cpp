#include "trainers/compute_cost.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

/** TODO:
 *  - precisa usar HeatBathTrainer para iniciar o problema com h e J != 0.
 *  - inserr no BaseTrainer um método que inclua:
 * arma::Col<int> s_up = arma::ones<arma::Col<int>>(nspins);
 * arma::Col<int> s_down = -arma::ones<arma::Col<int>>(nspins);
 * double E_up = energyAllPairs(s_up);      // usually very negative
 * double E_down = energyAllPairs(s_down);  // usually very positive
 * double E_range = std::abs(E_max - E_min)
 * int expected_num_bins = static_cast<int>(E_range / energy_bin) + 1;
 * ou use isso apara determinar energy_bin
 * */

void WangLandauTrainer::train()
{
    auto logger = getLogger();
    logger->info("[train] Starting Wang Landau training q_val = {}", q_val);

    for (iter = iter; iter < maxIterations; ++iter)
    {
        computeDensityOfStates();
        computeModelAverages(1.0, false);
        updateModelParameters(iter);

        auto cost = compute_cost(m1_data, m1_model, m2_data, m2_model);

        if (cost.check_convergence(tolerance_h, tolerance_J))
        {
            logger->info("[train] Wang Landau converged at iteration {}", iter);
            break;
        }
        if (iter % 10 == 0)
        {
            logger->info("[train] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f}", iter,
                         cost.cost_total, cost.cost_m1, cost.cost_m2);
        }
    }
    computeModelAverages(1.0, true);

    logger->debug("[train] Finished Wang Landau training.");
}
