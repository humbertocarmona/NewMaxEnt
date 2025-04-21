#pragma once
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <nlohmann/json.hpp>

#include <string>

struct RunParameters
{
    // main parameters
    std::string run_type           = "fun ensemble train";
    std::string runid              = "testing";
    std::string raw_data_file      = "none"; // filename with raw data samples to compute means
    std::string trained_model_file = "none"; // filename with trained model to compute means
    std::string result_dir = "./results";    // result_dir/runid/ is where all results will be saved
    int nspins             = 16;
    double q_val           = 1.0;
    double beta            = 1.0;

    // model training parameters
    size_t maxIterations = 1000;
    double tolerance_h   = 1.0e-4;
    double tolerance_J   = 1.0e-4;
    double eta_h         = 0.1;
    double eta_J         = 0.1;
    double alpha_h       = 0.1;
    double alpha_J       = 0.1;
    double gamma_h       = 0.2;
    double gamma_J       = 0.2;
    // Monte Carlo parameters
    int rng_seed               = 1;
    size_t equilibrationSweeps = 100000;
    size_t numSamples          = 1000;
    size_t sampleInterval      = 100;
    size_t _trials             = 100000;
    // Wang-Landau
    size_t pre_maxIterations        = 200;
    size_t pre_equilibration_sweeps = 1000;
    size_t pre_numSamples           = 1000;
    size_t pre_sampleInterval       = 100;
    double log_f_final              = 1.0e-6;
    double energy_bin               = 0.2;
    double flatness_threshold       = 0.8;

    // post-processing temperature dependence
    std::vector<double> temperature_range = std::vector<double>();

    RunParameters() = default;

    void loginfo(std::string caption = "") const
    {

        if (caption == "")
            caption = "RunParameters";
        auto logger = getLogger();

        logger->info("[{}] run_type               {}", caption, run_type);
        logger->info("[{}] runid                  {}", caption, runid);
        if (raw_data_file != "none")
            logger->info("[{}] raw_data_file          {}", caption, raw_data_file);
        if (trained_model_file != "none")
            logger->info("[{}] trained_model_file     {}", caption, trained_model_file);
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
        if (run_type == "Heat_Bath" || run_type == "Wang_Landau")
        {
            logger->info("[{}] rng_seed               {}", caption, rng_seed);
            logger->info("[{}] equilibrationSweeps    {}", caption, equilibrationSweeps);
            logger->info("[{}] numSamples             {}", caption, numSamples);
            logger->info("[{}] sampleInterval         {}", caption, sampleInterval);
        }
        if (run_type == "Temperature_Dep")
        {
            logger->info("[{}] temperature_range =    {}", caption,
                         utils::colPrint(arma::Col<double>(temperature_range)));
        }
        if (run_type == "Wang_Landau")
        {
            logger->info("pre_maxIterations           {}", caption, pre_maxIterations);
            logger->info("pre_equilibration_sweeps    {}", caption, pre_equilibration_sweeps);
            logger->info("pre_numSamples              {}", caption, pre_numSamples);
            logger->info("pre_sampleInterval          {}", caption, pre_sampleInterval);
            logger->info("log_f_final                 {}", caption, log_f_final);
            logger->info("energy_bin                  {}", caption, energy_bin);
            logger->info("flatness_threshold          {}", caption, flatness_threshold);
        }
    };

    nlohmann::json to_json() const
    {
        nlohmann::json obj;

        obj["run_type"] = run_type;
        obj["runid"]    = runid;
        if (raw_data_file != "none")
            obj["raw_data_file"] = raw_data_file;
        if (trained_model_file != "none")
            obj["trained_model_file"] = trained_model_file;
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
        if (run_type == "Heat_Bath" || run_type == "Wang_Landau")
        {
            obj["rng_seed"]            = rng_seed;
            obj["equilibrationSweeps"] = equilibrationSweeps;
            obj["numSamples"]          = numSamples;
            obj["sampleInterval"]      = sampleInterval;
        }
        if (run_type == "Temperature_Dep")
            obj["temperature_range"] = temperature_range;

        if (run_type == "Wang_Landau")
        {
            obj["pre_maxIterations"]        = pre_maxIterations;
            obj["pre_equilibration_sweeps"] = pre_equilibration_sweeps;
            obj["pre_numSamples"]           = pre_numSamples;
            obj["pre_sampleInterval"]       = pre_sampleInterval;
            obj["log_f_final"]              = log_f_final;
            obj["energy_bin"]               = energy_bin;
            obj["flatness_threshold"]       = flatness_threshold;
        }

        return obj;
    };
};

RunParameters parseParameters(const std::string &filename);