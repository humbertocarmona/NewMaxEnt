#include "core/run_parameters.hpp"
#include "utils/utilities.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

RunParameters parseParameters(const std::string &filename)
{
    RunParameters p;
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        throw std::runtime_error("Could not open parameter file: " + filename);
    }
    nlohmann::json json_data;
    infile >> json_data;

    p.run_type      = json_data.value("run_type", "fun ensemble train");
    p.runid         = json_data.value("runid", "testing");
    p.raw_data_file = json_data.value("raw_data_file", "raw_data.csv");
    p.result_dir    = json_data.value("result_dir", "./results");
    utils::make_path(p.result_dir);

    // needed by MaxEntCore
    p.nspins = json_data.value("nspins", 16);
    p.q_val  = json_data.value("q_val", 1.0);
    p.beta   = json_data.value("beta", 1.0);

    std::stringstream ss;
    ss << p.result_dir << "/" << p.nspins;
    p.result_dir = ss.str();
    utils::make_path(p.result_dir);

    // needed by FullEnsembleTrainer
    p.maxIterations = json_data.value("maxIterations", 1000);
    p.tolerance_h   = json_data.value("tolerance_h", 1.0e-4);
    p.tolerance_J   = json_data.value("tolerance_J", 1.0e-4);
    p.eta_h         = json_data.value("eta_h", 0.1);
    p.eta_J         = json_data.value("eta_J", 0.1);
    p.alpha_h       = json_data.value("alpha_h", 0.1);
    p.alpha_J       = json_data.value("alpha_J", 0.1);
    p.gamma_h       = json_data.value("gamma_h", 0.2);
    p.gamma_J       = json_data.value("gamma_J", 0.2);

    return p;
}