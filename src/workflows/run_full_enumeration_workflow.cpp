#include "workflows/run_full_enumeration_workflow.hpp"
#include "core/centered_moments.hpp"
#include "core/max_ent_core.hpp"
#include "io/write_output_json.hpp"
#include "io/make_file_names.hpp"


void run_full_enumeration_workflow(const Params &run_parameters)
{
    auto logger = spdlog::get("bm");

    // Build and run model
    MaxEntCore model(run_parameters, true);

    model.run_full_enumeration();

    // Prepare centered correlations
    CenteredMoments centered_model =
        compute_centered_moments(model.get_model_moment_1(), model.get_model_moment_2(), model.get_model_moment_3());

    // Compute centered sample correlations
    CenteredMoments centered_sample =
        compute_centered_moments(model.get_sample_moment_1(), model.get_sample_moment_2(), model.get_sample_moment_3());

    // Final energy stats
    double e_mean  = model.get_energy_mean();
    double e_fluct = model.get_energy_fluctuation();

    // Compose output filename
    auto filename = io::make_full_enumeration_filename(run_parameters);

    // Save results
    write_output_json(filename,
                      run_parameters,
                      model.get_h(),
                      model.get_J(),
                      model.get_sample_moment_1(),
                      model.get_model_moment_1(),
                      model.get_sample_moment_2(),
                      model.get_model_moment_2(),
                      model.get_sample_moment_3(),
                      model.get_model_moment_3(),
                      centered_sample.centered_moment_2,
                      centered_model.centered_moment_2,
                      centered_sample.centered_moment_3,
                      centered_model.centered_moment_3,
                      e_mean,
                      e_fluct,
                      model.get_iteration());

    
    logger->info("[run_full_enumeration_workflow] Results written to: {}", filename);
}
