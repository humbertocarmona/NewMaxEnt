#include "core/run_parameters.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
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
    nlohmann::json json_data;
    infile >> json_data;

    p.run_type           = json_data.value("run_type", "fun ensemble");
    p.runid              = json_data.value("runid", "testing");
    p.raw_data_file      = json_data.value("raw_data_file", "none");
    p.trained_model_file = json_data.value("trained_model_file", "none");
    p.result_dir         = json_data.value("result_dir", "./results");
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

    if (json_data.contains("temperature_range") && json_data["temperature_range"].is_array())
    {
        const auto &arr = json_data["temperature_range"];
        if (arr.size() == 3)
        {
            double t_begin = arr[0];
            double t_end   = arr[1];
            double t_step  = arr[2];

            if (t_step <= 0)
                throw std::runtime_error("temperature_range step must be > 0");

            for (double T = t_begin; T <= t_end + 1e-10; T += t_step)
                p.temperature_range.push_back(T);

            logger->info("[parseParameters] T range={}", utils::colPrint(p.temperature_range));
        }
        else
        {
            p.temperature_range = arr.get<std::vector<double>>();
        }
    }
    else if (p.run_type == "Temperature_Dep")
    {
        throw std::runtime_error("JSON 'Temperature_Dep' require 'temperature_range': " + filename);
    }

    if (json_data.contains("Monte_Carlo"))
    {
        auto mc                = json_data["Monte_Carlo"];
        p.equilibration_sweeps = mc.value("equilibration_sweeps", 1000);
        p.numSamples           = mc.value("numSamples", 1000);
        p.sampleInterval       = mc.value("sampleInterval", 100);
    }
    else if (p.run_type == "Monte_Carlo" || p.run_type == "Temperature_Dep")
    {
        throw std::runtime_error("JSON Monte_Carlo  and Temperature_Dep need 'Monte_Carlo': " +
                                 filename);
    }

    if (p.trained_model_file == "none" && p.raw_data_file == "none")
    {
        throw std::runtime_error("JSON need 'raw_data_file' or 'trained_model_file'");
    }
    if (p.trained_model_file == "none" && p.run_type == "Temperature_Dep")
    {
        throw std::runtime_error("JSON  Temperature_DepJSON need 'trained_model_file'");
    }

    return p;
}