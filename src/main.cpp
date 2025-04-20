#include "core/run_parameters.hpp"
#include "workflows/training_workflow.hpp"
#include "workflows/run_temperature_dependence.hpp"
#include "utils/get_logger.hpp"

int main(int argc, char **argv)
{
    auto logger=getLogger();
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " path/to/params.json\n";
        return 1;
    }
    std::string param_file = argv[1];

    RunParameters params = parseParameters(param_file);
    params.loginfo();
    if (params.run_type == "Full_Ensemble")
    {
        fullEnsembleTrainingWorkflow(params);
    }else if (params.run_type == "Monte_Carlo"){
        heatBathTrainingWorkflow(params);
    }else if (params.run_type == "Temperature_Dep"){
        runTemperatureDependence(params);
    }else{
        logger->warn("{} not recognized", params.run_type);
    }

    return 0;
}
