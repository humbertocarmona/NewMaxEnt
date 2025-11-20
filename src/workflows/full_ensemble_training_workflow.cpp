#include "trainers/full_ensemble_trainer.hpp"
#include "workflows/training_workflow.hpp"
#include "io/make_file_names.hpp"
#include <iostream>
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
    
    model.saveModel(params.file_final);
}
