#include "core/run_parameters.hpp"
#include "workflows/training_workflow.hpp"
#include "workflows/run_temperature_dependence.hpp"
#include "utils/get_logger.hpp"
#include "wg/wg.hpp"

int main(int argc, char **argv)
{
    auto logger=getLogger();

    // ------------------------------------------------------------------
    std::mt19937 rng(1);
    int nspins = 16; // number of spins

    // Example: random fields and couplings
    Field h(nspins);
    int nedges = nspins*(nspins -1)/2;
    Field J(nedges);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    int idx = 0;
    for (int i = 0; i < nspins; ++i) {
        h(i) = 0.01*dist(rng)-0.5;
        for (int j = i + 1; j < nspins; ++j) {
            J(idx++) = 0.01*dist(rng);
        }
    }

    wang_landau(h, J);


    return 0;
    // -----------------------------------------------------------------



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
