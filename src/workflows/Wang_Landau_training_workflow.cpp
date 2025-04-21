#include "io/write_trained_json.hpp"
#include "trainers/heat_bath_pretrain.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "trainers/wang_landau_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/training_workflow.hpp"

void WangLandauTrainingWorkflow(RunParameters params)
{
    auto logger        = getLogger();
    auto data_filename = params.raw_data_file;
    if (data_filename == "none")
        data_filename = params.trained_model_file;

    MaxEntCore core(params.nspins, params.runid);
    pretrain_with_heatbath(core, params, data_filename);

    WangLandauTrainer model(core, params.q_val,
                            params.pre_maxIterations, // << shorter!
                            params.tolerance_h, params.tolerance_J, params.eta_h, params.eta_J,
                            params.alpha_h, params.alpha_J, params.gamma_h, params.gamma_J,
                            data_filename);
    model.configureWangLandau(params.log_f_final, params.energy_bin, params.flatness_threshold,
                              params.equilibrationSweeps, params.numSamples, params.sampleInterval);

    model.train();
    CenteredMoments c_model =
        computeCenteredMoments(model.get_m1_model(), model.get_m2_model(), model.get_m3_model());

    CenteredMoments c_data =
        computeCenteredMoments(model.get_m1_data(), model.get_m2_data(), model.get_m3_data());
    writeTrainedModel<WangLandauTrainer>(params, model, c_data, c_model);

    spdlog::info("[WL] Finished. Norm h = {:.4e}, Norm J = {:.4e}", arma::norm(core.h),
                 arma::norm(core.J));
}