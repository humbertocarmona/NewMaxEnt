#include "core/max_ent_core.hpp"
#include "core/run_parameters.hpp"
#include "workflows/full_ensemble_training_workflow.hpp"


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " path/to/params.json\n";
        return 1;
    }    
    std::string param_file = argv[1];

    RunParameters params = parseParameters(param_file);
    params.loginfo();

    fullEnsembleTrainingWorkflow(params);

    return 0;
}