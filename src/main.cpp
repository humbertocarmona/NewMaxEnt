
#include "core/centered_moments.hpp"
#include "core/max_ent_core.hpp"
#include "io/parse_parameters.hpp"
#include "io/write_output_json.hpp"
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

    // Build and run model
    MaxEntCore model(run_parameters, true);

    model.run_full_enumeration();

    // Prepare centered correlations
    CenteredMoments centered =
        compute_centered_moments(model.get_model_moment_1(), model.get_model_moment_2(), model.get_model_moment_3());

    // Compute centered sample correlations
    CenteredMoments centered_sample =
        compute_centered_moments(model.get_sample_moment_1(), model.get_sample_moment_2(), model.get_sample_moment_3());

    // Final energy stats
    double e_mean  = model.get_energy_mean();
    double e_fluct = model.get_energy_fluctuation();

    // Compose output filename
    std::ostringstream filename;
    // clang-format off
    filename << run_parameters.result_dir << "/"
             << "ens_" << run_parameters.runid 
             << "_nspins_" << run_parameters.gen_nspins 
             << "_q_"  << std::fixed << std::setprecision(2) << run_parameters.q_val 
             << "_id_" << run_parameters.id 
             << "_final.json";
    // clang-format on

    // Save results
    // clang-format off
    write_output_json(filename.str(), 
                      run_parameters, 
                      model.get_h(), 
                      model.get_J(), 
                      model.get_sample_moment_1(),
                      model.get_model_moment_1(), 
                      model.get_sample_moment_2(), 
                      model.get_model_moment_2(),
                      centered_sample.correlation_matrix_2.as_col(), 
                      centered.correlation_matrix_2.as_col(),
                      model.get_model_moment_3(), 
                      centered.centered_moment_3.value_or(arma::Col<double>()), e_mean,
                      e_fluct, model.get_iteration());
    // clang-format on

    std::cout << "Results written to: " << filename.str() << "\n";
    return 0;
}
