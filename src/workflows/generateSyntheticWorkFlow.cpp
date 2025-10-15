#include "io/write_json.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/generateSyntheticWorkFlow.hpp"

void generateSyntheticWorkflow(RunParameters params)
{
    auto logger = getLogger();

    // here need to make sure that the trained_model_file has:
    // h, J, q, nspins, runid
    auto data_filename = params.trained_model_file;

    // initialize core model
    MaxEntCore core(params.nspins, params.runid);

    // construct model (using FullEnsembleTrainer to generate means)
    FullEnsembleTrainer model(core, params, data_filename);
    // compute model averages in parallel
    model.computeModelAverages(1.0, true);

    model.copySyntheticMeans();

    model.saveModel("gen_final");
}