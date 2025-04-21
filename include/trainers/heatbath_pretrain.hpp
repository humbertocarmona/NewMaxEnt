#pragma once

#include "trainers/heat_bath_trainer.hpp"
#include "core/max_ent_core.hpp"
#include "core/run_parameters.hpp"

// inline void pretrain_with_heatbath(MaxEntCore &core, const RunParameters &params, const std::string &data_filename) 
// {
//     HeatBathTrainer tmp_model(core, params.q_val,
//                               params.pretrain_maxIterations,      // << shorter!
//                               params.tolerance_h,
//                               params.tolerance_J,
//                               params.eta_h,
//                               params.eta_J,
//                               params.alpha_h,
//                               params.alpha_J,
//                               params.gamma_h,
//                               params.gamma_J,
//                               data_filename);

//     tmp_model.configureMonteCarlo(params.pretrain_equilibration_sweeps,
//                                   params.pretrain_numSamples,
//                                   params.pretrain_sampleInterval);

//     spdlog::info("[WL Pretrain] Starting HeatBath pretraining for {} iterations, {} samples...",
//                  params.pretrain_maxIterations, params.pretrain_numSamples);

//     tmp_model.train();

//     spdlog::info("[WL Pretrain] Finished. Norm h = {:.4e}, Norm J = {:.4e}", 
//                  arma::norm(core.h), arma::norm(core.J));
// }


inline void pretrain_with_heatbath(MaxEntCore &core, const RunParameters &params, const std::string &data_filename) 
{
    // maxIterations
    // numSamples

    HeatBathTrainer tmp_model(core, params.q_val, params.maxIterations, params.tolerance_h,
                              params.tolerance_J, params.eta_h, params.eta_J, params.alpha_h,
                              params.alpha_J, params.gamma_h, params.gamma_J, data_filename);

    tmp_model.configureMonteCarlo(params.equilibration_sweeps,
                                  params.numSamples,
                                  params.sampleInterval);

    tmp_model.train();
}