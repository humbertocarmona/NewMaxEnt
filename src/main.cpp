#include "core/run_parameters.hpp"
#include "utils/get_logger.hpp"
#include "workflows/full_ensemble_no_update.hpp"
#include "workflows/run_temperature_dependence.hpp"
#include "workflows/training_workflow.hpp"
#include <iostream>
int main(int argc, char **argv)
{
    auto logger = getLogger();

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " path/to/params.json\n";
        return 1;
    }
    std::string param_file = argv[1];

    RunParameters params = parseParameters(param_file);

    if (params.run_type == "Full_Ensemble" || params.run_type == "Full")
    {
        fullEnsembleTrainingWorkflow(params);
    }
    else if (params.run_type == "Heat_Bath" || params.run_type == "MC")
    {
        heatBathTrainingWorkflow(params);
    }
    else if (params.run_type == "Temperature_Dep" || params.run_type == "TDep")
    {
        runTemperatureDependence(params);
    }
    else if (params.run_type == "Gen_Full" || params.run_type == "Gen_MC" || params.run_type == "Copy")
    {
        full_ensemble_no_update(params);
    }
    else
    {
        logger->warn("{} not recognized", params.run_type);
    }

    return 0;
}
