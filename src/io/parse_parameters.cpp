#include "core/run_parameters.hpp"
#include "io/read_trained_json.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
RunParameters parseParameters(const std::string &filename)
{
    auto logger = getLogger();
    RunParameters p;
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        throw std::runtime_error("Could not open parameter file: " + filename);
    }

    std::set<std::string> valid_run_types = {"Full_Ensemble",   "Full", "Heat_Bath", "MC",
                                             "Temperature_Dep", "TDep", "Gen_Full",  "Gen_MC"};

    nlohmann::json json_data;
    infile >> json_data;

    if (!json_data.contains("run_type"))
    {
        throw std::runtime_error("run_type is required ");
    }

    p.run_type = json_data.value("run_type", "");
    if (valid_run_types.count(p.run_type) == 0)
    {
        throw std::runtime_error("Invalid run_type in " + filename + ": " + p.run_type);
    }

    // main parameters
    p.runid              = json_data.value("runid", "testing");
    p.raw_data_file      = json_data.value("raw_data_file", "none");
    p.trained_model_file = json_data.value("trained_model_file", "none");
    p.result_dir         = json_data.value("result_dir", "./results");
    p.nspins             = json_data.value("nspins", 16);
    p.q_val              = json_data.value("q_val", 1.0);
    p.beta               = json_data.value("beta", 1.0);
    p.iter               = json_data.value("iter", 1);
    p.continue_run       = json_data.value("continue_run", 0);

    if (p.continue_run == 1)
    {
        std::cout << "reading model" + p.trained_model_file << std::endl;
        // read the model file
        auto obj = readJSONData(p.trained_model_file);
        if (!obj.contains("run_parameters"))
        {
            throw std::runtime_error("run_parameters required in " + p.trained_model_file);
        }
        auto run_parameters = obj["run_parameters"];
        p.runid             = run_parameters["runid"];
        p.q_val             = run_parameters["q_val"];
        p.nspins            = run_parameters["nspins"];
        p.iter              = obj["iter"];
    }

    bool isTraining = p.run_type == "Full_Ensemble" || p.run_type == "Heat_Bath";
    bool isMc       = p.run_type == "Heat_Bath" || p.run_type == "Wang_Landau";
    bool isTdep     = p.run_type == "Temperature_Dep";

    if (isTraining && !json_data.contains("training"))
    {
        throw std::runtime_error(p.run_type + " requires 'training' in " + filename);
    }

    if (isTdep && !json_data.contains("temperature_range"))
    {
        throw std::runtime_error(p.run_type + " requires 'temperature_range' in " + filename);
    }

    if (p.trained_model_file == "none" && p.raw_data_file == "none")
    {
        throw std::runtime_error("'raw_data_file' or 'trained_model_file' required in " + filename);
    }

    if (json_data.contains("training"))
    {
        auto tr           = json_data["training"];
        p.maxIterations   = tr.value("maxIterations", 1000);
        p.save_checkpoint = tr.value("save_checkpoint", 10000);
        p.tolerance_h     = tr.value("tolerance_h", 1.0e-4);
        p.tolerance_J     = tr.value("tolerance_J", 1.0e-4);
        p.eta_h           = tr.value("eta_h", 0.1);
        p.eta_J           = tr.value("eta_J", 0.1);
        p.alpha_h         = tr.value("alpha_h", 0.1);
        p.alpha_J         = tr.value("alpha_J", 0.1);
        p.gamma_h         = tr.value("gamma_h", 0.2);
        p.gamma_J         = tr.value("gamma_J", 0.2);
        p.updateType =
            tr.value("update_type", 'p'); // 'p' power law, 'g' gradient 's' gradient sequential
    }

    if (json_data.contains("k-pairwise"))
    {
        auto pw = json_data["k-pairwise"];

        p.k_pairwise  = pw.value("k_pairwise", true);
        p.tolerance_k = pw.value("tolerance_k", 1.0e-4);
        p.eta_k       = pw.value("eta_k", 0.1);
        p.alpha_k     = pw.value("alpha_k", 0.1);
        p.gamma_k     = pw.value("gamma_k", 0.2);
        logger->info("[parseParameters] p.k_pairwise = {}", p.k_pairwise);
    }

    if (isTdep && p.trained_model_file == "none")
    {
        throw std::runtime_error(p.run_type + " requires 'trained_model_file' in " + filename);
    }

    if (json_data.contains("temperature_range") && json_data["temperature_range"].is_array())
    {
        const auto &arr = json_data["temperature_range"];

        // Check if all elements are numbers
        bool all_numeric =
            std::all_of(arr.begin(), arr.end(), [](const auto &el) { return el.is_number(); });

        if (!all_numeric)
            throw std::runtime_error("All elements of 'temperature_range' must be numeric");

        if (arr.size() == 3)
        {
            double t_begin = arr[0];
            double t_end   = arr[1];
            double t_step  = arr[2];

            if (t_step <= 0)
                throw std::runtime_error("temperature_range step must be > 0");

            for (double T = t_begin; T <= t_end + 1e-10; T += t_step)
                p.temperature_range.push_back(T);
        }
        else
        {
            // just copy the arr
            p.temperature_range = arr.get<std::vector<double>>();
        }
    }

    if (json_data.contains("Monte_Carlo"))
    {
        auto mc              = json_data["Monte_Carlo"];
        p.step_equilibration = mc.value("step_equilibration", 1000);
        p.num_samples        = mc.value("num_samples", 1000);
        p.step_correlation   = mc.value("step_correlation", 100);
        p.number_repetitions = mc.value("num_repetitions", 20);
        p.rng_seed           = mc.value("rng_seed", 1);
    }

    if (json_data.contains("Wang_Landau"))
    {
        auto wl                  = json_data["Wang_Landau"];
        p.pre_maxIterations      = wl.value("pre_maxIterations", 200);
        p.pre_step_equilibration = wl.value("pre_step_equilibration", 1000);
        p.pre_step_correlation   = wl.value("pre_step_correlation", 100);
        p.pre_num_samples        = wl.value("pre_num_samples", 1000);
        p.pre_number_repetitions = wl.value("pre_num_repetitions", 10);
        p.log_f_final            = wl.value("log_f_final", 1e-6);
        p.energy_bin             = wl.value("energy_bin", 0.2);
        p.flatness_threshold     = wl.value("flatness_threshold", 0.8);
        if (p.rng_seed == 1)
        {
            p.rng_seed = wl.value("rng_seed", 1);
        }
    }

    

    p.loginfo("parsedParameters");
    return p;
}
