#pragma once
#include "utils/utilities.hpp"
#include <armadillo>
#include <nlohmann/json.hpp>
#include <spdlog/sinks/stdout_color_sinks.h> // Add this for stdout_color_mt
#include <spdlog/spdlog.h>

#include <string>

struct RunParameters
{
    std::string run_type      = "fun ensemble train";
    std::string runid         = "testing";
    std::string raw_data_file = "raw_data.csv"; // filename with raw data samples to compute means
    std::string result_dir    = "./results";    // result_dir/runid/ is where all results will be saved

    // needed by MaxEntCore
    int nspins   = 16;
    double q_val = 1.0;
    double beta  = 1.0;

    // needed by FullEnsembleTrainer
    int maxIterations  = 1000;
    double tolerance_h = 1.0e-4;
    double tolerance_J = 1.0e-4;
    double eta_h       = 0.1;
    double eta_J       = 0.1;
    double alpha_h     = 0.1;
    double alpha_J     = 0.1;
    double gamma_h     = 0.2;
    double gamma_J     = 0.2;

    RunParameters() = default;

    void loginfo() const
    {
        auto logger = spdlog::stdout_color_mt("core_logger");
        logger->info("[RunParameters] run_type          {}", run_type);
        logger->info("[RunParameters] runid             {}", runid);
        logger->info("[RunParameters] raw_data_file     {}", raw_data_file);
        logger->info("[RunParameters] result_dir        {}", result_dir);
        logger->info("[RunParameters] nspins            {}", nspins);
        logger->info("[RunParameters] q_val             {}", q_val);
        logger->info("[RunParameters] beta              {}", beta);

        logger->info("[RunParameters] maxIterations     {}", maxIterations);
        logger->info("[RunParameters] tolerance_h       {}", tolerance_h);
        logger->info("[RunParameters] tolerance_J       {}", tolerance_J);

        logger->info("[RunParameters] eta_h             {}", eta_h);
        logger->info("[RunParameters] eta_J             {}", eta_J);
        logger->info("[RunParameters] alpha_h           {}", alpha_h);
        logger->info("[RunParameters] alpha_J           {}", alpha_J);
        logger->info("[RunParameters] gamma_h           {}", gamma_h);
        logger->info("[RunParameters] gamma_J           {}", gamma_J);
    }
};

RunParameters parseParameters(const std::string &filename);