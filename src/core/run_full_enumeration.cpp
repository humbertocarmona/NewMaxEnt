#include "core/centered_moments.hpp" // ← needed for centered correlations
#include "core/compute_model_statistics_full.hpp"
#include "core/cost_function.hpp"
#include "core/max_ent_core.hpp"

void MaxEntCore::run_full_enumeration()
{
    auto logger = LOGGER;

    logger->info("Starting full enumeration training...");

    initialize_fields();

    momentum_m_1 = arma::zeros<arma::Col<double>>(n_spins);
    momentum_m_2 = arma::zeros<arma::Col<double>>(n_edges);

    compute_sample_statistics(); // fills sample_moment_1 and sample_moment_2

    for (iter = 1; iter <= run_parameters.max_iter; ++iter)
    {
        compute_model_statistics_full(n_spins,
                                      h,
                                      J,
                                      model_moment_1,
                                      model_moment_2,
                                      model_moment_3,
                                      run_parameters.q_val,
                                      run_parameters.beta,
                                      false // skip triplets and energy during training
        );

        update_model_parameters();

        auto cost = compute_cost(sample_moment_1, model_moment_1, sample_moment_2, model_moment_2);

        if (cost.check_convergence(run_parameters.tol_1, run_parameters.tol_2))
        {
            logger->info("[run_full_enumeration] Converged at iteration {}", iter);
            break;
        }

        if (iter % 10 == 0)
        {
            logger->info("[run_full_enumeration] Iter {} | Cost: {:.6f} | M1: {:.6f} | M2: {:.6f}",
                         iter,
                         cost.total,
                         cost.moment_1,
                         cost.moment_2);
        }
    }

    logger->info("[run_full_enumeration] Post-convergence analysis...");

    // Final model statistics including triplets and energy info
    double energy_sq_mean = 0.0;

    compute_model_statistics_full(n_spins,
                                  h,
                                  J,
                                  model_moment_1,
                                  model_moment_2,
                                  model_moment_3,
                                  run_parameters.q_val,
                                  run_parameters.beta,
                                  true, // compute triplets
                                  &energy_mean,
                                  &energy_sq_mean);

    energy_fluctuation = energy_sq_mean - std::pow(energy_mean, 2.0);

    logger->info("[run_full_enumeration] Full enumeration run completed.");
}
