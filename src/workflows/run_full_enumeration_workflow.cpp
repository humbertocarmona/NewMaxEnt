#include "workflows/run_full_enumeration_workflow.hpp"
#include "core/max_ent_core.hpp"
#include "core/centered_moments.hpp"
#include "io/write_output_json.hpp"

void run_full_enumeration_workflow(const Params& run_parameters){
        // Build and run model
        MaxEntCore model(run_parameters, true);

        model.run_full_enumeration();

        // Prepare centered correlations
        CenteredMoments centered = compute_centered_moments(model.get_model_moment_1(), model.get_model_moment_2(),
                                                            model.get_model_moment_3());

        // Compute centered sample correlations
        CenteredMoments centered_sample = compute_centered_moments(
            model.get_sample_moment_1(), model.get_sample_moment_2(), model.get_sample_moment_3());

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

}
