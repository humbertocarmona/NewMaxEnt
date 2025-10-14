#include "io/write_json.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/generateSyntheticWorkFlow.hpp"

void generateSyntheticWorkflow(RunParameters params)
{
    auto logger = getLogger();

    auto data_filename = params.trained_model_file;

    MaxEntCore core(params.nspins, params.runid);

    FullEnsembleTrainer model(core, params, data_filename);

    //model.train();
    model.saveModel("gen_final");
}