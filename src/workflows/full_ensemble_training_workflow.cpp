#include "io/write_trained_json.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/training_workflow.hpp"

void fullEnsembleTrainingWorkflow(RunParameters params)
{
    auto logger = getLogger();

    MaxEntCore core(params.nspins, params.runid);
    FullEnsembleTrainer model(core, params.q_val, params.maxIterations, params.tolerance_h,
                              params.tolerance_J, params.eta_h, params.eta_J, params.alpha_h,
                              params.alpha_J, params.gamma_h, params.gamma_J, params.raw_data_file);

    model.train();

    CenteredMoments c_model =
        computeCenteredMoments(model.get_m1_model(), model.get_m2_model(), model.get_m3_model());

    CenteredMoments c_data =
        computeCenteredMoments(model.get_m1_data(), model.get_m2_data(), model.get_m3_data());
    writeTrainedModel<FullEnsembleTrainer>(params, model, c_data, c_model);
}