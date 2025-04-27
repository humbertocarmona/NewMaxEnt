#pragma once

#include "core/max_ent_core.hpp"
#include "core/run_parameters.hpp"
#include "trainers/heat_bath_trainer.hpp"

inline void pretrain_with_heatbath(MaxEntCore &core,
                                   const RunParameters &params,
                                   const std::string &data_filename)
{
    spdlog::info("[WL Pretrain] Starting. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
    HeatBathTrainer tmp_model(core, params.q_val,
                              params.pre_maxIterations, // << shorter!
                              params.tolerance_h, params.tolerance_J, params.eta_h, params.eta_J,
                              params.alpha_h, params.alpha_J, params.gamma_h, params.gamma_J,
                              data_filename);

    tmp_model.configureMonteCarlo(params.pre_step_equilibration, params.pre_num_samples,
                                  params.pre_step_correlation, params.pre_number_repetitions);

    spdlog::info("[WL Pretrain] Starting HeatBath pretraining for {} iterations, {} samples...",
                 params.pre_maxIterations, params.pre_num_samples);

    tmp_model.train();

    spdlog::info("[WL Pretrain] Finished. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
}
