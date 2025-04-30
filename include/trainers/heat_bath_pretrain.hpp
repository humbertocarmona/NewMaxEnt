#pragma once

#include "core/max_ent_core.hpp"
#include "core/run_parameters.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/get_logger.hpp"

inline void pretrain_with_heatbath(MaxEntCore &core,
                                   RunParameters &params,
                                   const std::string &data_filename)
{
    auto logger = getLogger();
    logger->info("[WL Pretrain] Starting. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
    
    auto params_pre = params;
    params_pre.num_samples = params.pre_num_samples;
    params_pre.maxIterations = params.pre_maxIterations;
    params_pre.step_equilibration = params.pre_step_equilibration;
    params_pre.step_correlation = params.pre_step_correlation;
    params_pre.step_equilibration = params.pre_step_equilibration;
    params_pre.step_correlation = params.pre_step_correlation;

    params_pre.loginfo();              

    HeatBathTrainer tmp_model(core, params_pre, data_filename);

    spdlog::info("[WL Pretrain] Starting HeatBath pre-training for {} iterations, {} samples...",
                 params.pre_maxIterations, params.pre_num_samples);

    tmp_model.train();

    logger->info("[WL Pretrain] Finished. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
}
