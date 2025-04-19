#pragma once
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <nlohmann/json.hpp>

#include <string>

struct RunParameters
{
    std::string run_type      = "fun ensemble train";
    std::string runid         = "testing";
    std::string raw_data_file = "raw_data.csv"; // filename with raw data samples to compute means
    std::string result_dir    = "./results"; // result_dir/runid/ is where all results will be saved

    // needed by MaxEntCore
    int nspins   = 16;
    double q_val = 1.0;
    double beta  = 1.0;

    // needed by FullEnsembleTrainer
    int maxIterations          = 1000;
    double tolerance_h         = 1.0e-4;
    double tolerance_J         = 1.0e-4;
    double eta_h               = 0.1;
    double eta_J               = 0.1;
    double alpha_h             = 0.1;
    double alpha_J             = 0.1;
    double gamma_h             = 0.2;
    double gamma_J             = 0.2;
    int numEquilibrationSweeps = 1000;
    int numSamples             = 1000;
    int sampleInterval         = 100;

    RunParameters() = default;

    void loginfo(std::string caption = "") const
    {

        if (caption == "")
            caption = "RunParameters";
        auto logger = getLogger();

        logger->info("[{}] run_type               {}", caption, run_type);
        logger->info("[{}] runid                  {}", caption, runid);
        logger->info("[{}] raw_data_file          {}", caption, raw_data_file);
        logger->info("[{}] result_dir             {}", caption, result_dir);
        logger->info("[{}] nspins                 {}", caption, nspins);
        logger->info("[{}] q_val                  {}", caption, q_val);
        logger->info("[{}] beta                   {}", caption, beta);

        logger->info("[{}] maxIterations          {}", caption, maxIterations);
        logger->info("[{}] tolerance_h            {}", caption, tolerance_h);
        logger->info("[{}] tolerance_J            {}", caption, tolerance_J);

        logger->info("[{}] eta_h                  {}", caption, eta_h);
        logger->info("[{}] eta_J                  {}", caption, eta_J);
        logger->info("[{}] alpha_h                {}", caption, alpha_h);
        logger->info("[{}] alpha_J                {}", caption, alpha_J);
        logger->info("[{}] gamma_h                {}", caption, gamma_h);
        logger->info("[{}] gamma_J                {}", caption, gamma_J);
        if (run_type == "Monte Carlo")
        {
            logger->info("[{}] numEquilibrationSweeps {}", caption, numEquilibrationSweeps);
            logger->info("[{}] numSamples             {}", caption, numSamples);
            logger->info("[{}] sampleInterval         {}", caption, sampleInterval);
        }
    };

    nlohmann::json to_json() const
    {
        nlohmann::json obj;

        obj["run_type"]      = run_type;
        obj["runid"]         = runid;
        obj["raw_data_file"] = raw_data_file;
        obj["result_dir"]    = result_dir;
        obj["nspins"]        = nspins;
        obj["q_val"]         = q_val;
        obj["beta"]          = beta;
        obj["maxIterations"] = maxIterations;
        obj["tolerance_h"]   = tolerance_h;
        obj["tolerance_J"]   = tolerance_J;
        obj["eta_h"]         = eta_h;
        obj["eta_J"]         = eta_J;
        obj["alpha_h"]       = alpha_h;
        obj["alpha_J"]       = alpha_J;
        obj["gamma_h"]       = gamma_h;
        obj["gamma_J"]       = gamma_J;
        if (run_type == "Monte Carlo")
        {
            obj["numEquilibrationSweeps"] = numEquilibrationSweeps;
            obj["numSamples"]             = numSamples;
            obj["sampleInterval"]         = sampleInterval;
        }
        return obj;
    };
};

RunParameters parseParameters(const std::string &filename);