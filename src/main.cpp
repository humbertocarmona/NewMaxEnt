#include "core/run_parameters.hpp"
#include "utils/get_logger.hpp"
#include "workflows/run_temperature_dependence.hpp"
#include "workflows/training_workflow.hpp"

int main(int argc, char **argv)
{
    auto logger = getLogger();

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " path/to/params.json\n";
        return 1;
    }
    std::string param_file = argv[1];

    RunParameters params = parseParameters(param_file);

    if (params.run_type == "Full_Ensemble")
    {
        fullEnsembleTrainingWorkflow(params);
    }
    else if (params.run_type == "Heat_Bath")
    {
        heatBathTrainingWorkflow(params);
    }
    else if (params.run_type == "Wang_Landau")
    {
        WangLandauTrainingWorkflow(params);
    }
    else if (params.run_type == "Temperature_Dep")
    {
        runTemperatureDependence(params);
    }
    else
    {
        logger->warn("{} not recognized", params.run_type);
    }

    return 0;
}
