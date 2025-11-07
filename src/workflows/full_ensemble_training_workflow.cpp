#include "io/write_json.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/training_workflow.hpp"

void fullEnsembleTrainingWorkflow(RunParameters params)
{
    auto logger = getLogger();

    auto data_filename = params.raw_data_file;
    if (data_filename == "none")
        data_filename = params.trained_model_file;

    // create an empty core with nspins and runid
    MaxEntCore core(params.nspins, params.runid);

    // create a BaseTrainer
    FullEnsembleTrainer model(core, params, data_filename);
    std::cout << "updateType=" << params.updateType << std::endl;
    model.train();

    model.saveModel("final_");
}
