
#include "io/parse_parameters.hpp"
#include "workflows/run_thermo_sweep.hpp"
#include "workflows/run_full_enumeration_workflow.hpp"
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " path/to/params.json\n";
        return 1;
    }

    // Load parameters
    std::string param_file = argv[1];
    Params run_parameters  = parse_parameters(param_file);

    run_parameters.log_info(true);

    if (run_parameters.run_type == "full enumeration")
    {
        run_full_enumeration_workflow(run_parameters);
    }
    else if (run_parameters.run_type == "thermo sweep")
    {
        run_thermo_sweep_workflow(run_parameters);
    }
    return 0;
}
