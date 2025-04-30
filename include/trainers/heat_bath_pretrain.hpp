#pragma once

#include "core/max_ent_core.hpp"
#include "core/run_parameters.hpp"
#include "trainers/heat_bath_trainer.hpp"

inline void pretrain_with_heatbath(MaxEntCore &core,
                                   RunParameters &params,
                                   const std::string &data_filename)
{
    spdlog::info("[WL Pretrain] Starting. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
    HeatBathTrainer tmp_model(core, params, data_filename);

    spdlog::info("[WL Pretrain] Starting HeatBath pre-training for {} iterations, {} samples...",
                 params.pre_maxIterations, params.pre_num_samples);

    tmp_model.train();

    spdlog::info("[WL Pretrain] Finished. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
}
