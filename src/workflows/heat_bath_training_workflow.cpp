#include "io/write_json.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/centered_moments.hpp"
#include "utils/get_logger.hpp"
#include "workflows/training_workflow.hpp"

void heatBathTrainingWorkflow(RunParameters params)
{
    auto logger        = getLogger();
    auto data_filename = params.raw_data_file;
    if (data_filename == "none")
        data_filename = params.trained_model_file;

    MaxEntCore core(params.nspins, params.runid);
    HeatBathTrainer model(core, params, data_filename);

    
    model.train();

    model.saveModel("final");
}
