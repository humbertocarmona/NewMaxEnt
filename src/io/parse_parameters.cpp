#include "core/parameters.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

// This function reads a JSON file containing run configuration
// and converts it into a Params struct used to control the model
Params parse_params(const std::string &filename)
{
    Params p;

    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        throw std::runtime_error("Could not open parameter file: " + filename);
    }

    nlohmann::json json_data;
    infile >> json_data;

    // Basic identification and run information
    p.id       = json_data.value("id", "");
    p.runid    = json_data.value("runid", 0);
    p.comment  = json_data.value("comment", "");
    p.run_type = json_data.value("run_type", "ens");

    // Input and checkpoint files
    p.raw_samples_file = json_data.value("raw_samples_file", "none");
    p.stats_file       = json_data.value("stats_file", "none");
    p.checkpoint_file  = json_data.value("checkpoint_file", "none");

    // Result storage options
    p.result_dir    = json_data.value("result_dir", "./Testing");
    p.save_state    = json_data.value("save_state", 1000);
    p.save_result   = json_data.value("save_result", true);
    p.save_energies = json_data.value("save_energies", false);

    // Optimization parameters
    p.n_relax_steps = json_data.value("n_relax_steps", 1000);
    p.init_step     = json_data.value("init_step", 0);
    p.eta_h         = json_data.value("eta_h", 0.1);
    p.eta_J         = json_data.value("eta_J", 0.1);
    p.gamma_h       = json_data.value("gamma_h", 0.2);
    p.gamma_J       = json_data.value("gamma_J", 0.2);
    p.alpha         = json_data.value("alpha", 0.1);
    p.tol_mag       = json_data.value("tol_mag", 0.001);
    p.tol_cor       = json_data.value("tol_cor", 0.001);

    // Entropic parameters (typically only one q value, but can be extended)
    if (json_data.contains("q_val") && json_data["q_val"].is_array())
    {
        p.q_val = arma::Col<double>(json_data["q_val"].size());
        for (size_t i = 0; i < p.q_val.n_elem; ++i)
        {
            p.q_val[i] = json_data["q_val"][i];
        }
    }

    // Generation parameters (only used in run_type = "gen")
    p.gen_nspins  = json_data.value("gen_nspins", 16);
    p.gen_seed    = json_data.value("gen_seed", -1);
    p.gen_h_mean  = json_data.value("gen_h_mean", -1.0);
    p.gen_h_width = json_data.value("gen_h_width", 2.0);
    p.gen_J_mean  = json_data.value("gen_J_mean", 0.0);
    p.gen_J_width = json_data.value("gen_J_width", 0.5);

    // Monte Carlo generation parameters (used for synthetic data)
    p.mc_n_samples     = json_data.value("mc_n_samples", 50000);
    p.mc_n_coherence   = json_data.value("mc_n_coherence", 40);
    p.mc_n_equilibrium = json_data.value("mc_n_equilibrium", 3000);
    p.mc_n_rept        = json_data.value("mc_n_rept", 40);
    p.mc_seed          = json_data.value("mc_seed", 1234);

    infile.close();
    return p;
}

