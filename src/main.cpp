#include "core/run_parameters.hpp"
#include "utils/get_logger.hpp"
#include "workflows/run_temperature_dependence.hpp"
#include "workflows/training_workflow.hpp"
#include "workflows/generateSyntheticWorkFlow.hpp"

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
    else if (params.run_type == "Gen_Full" || params.run_type == "Gen_MC")
    {
        // create new workflow to compute means using full or mc
        //
        // this needs:
        // - read h_i and J_ij
        // - produce m1_data, m2_data, m3_data, pk_data for T=1
        // - save the result as an untrained model (not raw spin file)
        // genUntrainedWorkflow(params)
        //
        //  n<=20
        // need to run full_ensemble_compute_model_averages once...
        //
        //  n>20
        // need to run heat_bath_compute_model_averages one...
        //
        // save the result averages...
        generateSyntheticWorkflow(params);

    }
    else
    {
        logger->warn("{} not recognized", params.run_type);
    }

    return 0;
}
