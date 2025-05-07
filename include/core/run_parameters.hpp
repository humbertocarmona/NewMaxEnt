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
    size_t save_checkpoint =  10000;
    double tolerance_h   = 1.0e-4;
    double tolerance_J   = 1.0e-4;
    double eta_h         = 0.1;
    double eta_J         = 0.1;
    double alpha_h       = 0.1;
    double alpha_J       = 0.1;
    double gamma_h       = 0.2;
    double gamma_J       = 0.2;

    // for adaptive learning rate
    double eta_h_min           = 1.0e-4;
    double eta_J_min           = 1.0e-4;
    double grad_drop_threshold = 1.0e-3;

    // k-pairwise (force P(K))
    bool k_pairwise = true;
    double tolerance_k   = 1.0e-4;
    double eta_k         = 0.1;
    double alpha_k       = 0.1;
    double gamma_k       = 0.2;
    double eta_K_min           = 1.0e-4;


    // Monte Carlo parameters
    int rng_seed              = 1;
    size_t step_equilibration = 100000;
    size_t num_samples        = 1000;
    size_t step_correlation   = 100;
    size_t number_repetitions = 20;
    // Wang-Landau
    size_t pre_maxIterations      = 200;
    size_t pre_step_equilibration = 1000;
    size_t pre_num_samples        = 1000;
    size_t pre_step_correlation   = 100;
    size_t pre_number_repetitions = 10;
    double log_f_final            = 1.0e-6;
    double energy_bin             = 0.2;
    double flatness_threshold     = 0.8;

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
        logger->info("[{}] save_checkpoint          {}", caption, save_checkpoint);
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
            logger->info("[{}] step_equilibration    {}", caption, step_equilibration);
            logger->info("[{}] num_samples             {}", caption, num_samples);
            logger->info("[{}] step_correlation         {}", caption, step_correlation);
            logger->info("[{}] number_repetitions         {}", caption, number_repetitions);
        }
        if (run_type == "Temperature_Dep")
        {
            logger->info("[{}] temperature_range =    {}", caption,
                         utils::colPrint(arma::Col<double>(temperature_range)));
        }
        if (run_type == "Wang_Landau")
        {
            logger->info("[{}] pre_maxIterations           {}", caption, pre_maxIterations);
            logger->info("[{}] pre_step_equilibration    {}", caption, pre_step_equilibration);
            logger->info("[{}] pre_num_samples              {}", caption, pre_num_samples);
            logger->info("[{}] pre_step_correlation          {}", caption, pre_step_correlation);
            logger->info("[{}] pre_number_repetitions          {}", caption,
                         pre_number_repetitions);
            logger->info("[{}] log_f_final                 {}", caption, log_f_final);
            logger->info("[{}] energy_bin                  {}", caption, energy_bin);
            logger->info("[{}] flatness_threshold          {}", caption, flatness_threshold);
        }

        if (k_pairwise){
            logger->info("[{}] k_pairwise                  {}",caption,k_pairwise);
            logger->info("[{}] tolerance_k                 {}",caption,tolerance_k);
            logger->info("[{}] eta_k                       {}",caption,eta_k);
            logger->info("[{}] alpha_k                     {}",caption,alpha_k);
            logger->info("[{}] gamma_k                     {}",caption,gamma_k);
            logger->info("[{}] eta_K_min                   {}",caption,eta_K_min);   
        }
    };

    nlohmann::json to_json() const
    {
        nlohmann::json obj, tr, mc, wl, pw;

        obj["run_type"] = run_type;
        obj["runid"]    = runid;
        if (raw_data_file != "none")
            obj["raw_data_file"] = raw_data_file;
        if (trained_model_file != "none")
            obj["trained_model_file"] = trained_model_file;
        obj["result_dir"] = result_dir;
        obj["nspins"]     = nspins;
        obj["q_val"]      = q_val;
        obj["beta"]       = beta;

        tr["maxIterations"] = maxIterations;
        tr["save_checkpoint"] = save_checkpoint;
        tr["tolerance_h"]   = tolerance_h;
        tr["tolerance_J"]   = tolerance_J;
        tr["eta_h"]         = eta_h;
        tr["eta_J"]         = eta_J;
        tr["alpha_h"]       = alpha_h;
        tr["alpha_J"]       = alpha_J;
        tr["gamma_h"]       = gamma_h;
        tr["gamma_J"]       = gamma_J;
        obj["training"]     = tr;

        if (run_type == "Heat_Bath" || run_type == "Wang_Landau")
        {
            mc["rng_seed"]           = rng_seed;
            mc["step_equilibration"] = step_equilibration;
            mc["num_samples"]        = num_samples;
            mc["step_correlation"]   = step_correlation;
            mc["number_repetitions"] = number_repetitions;
            obj["Monte_Carlo"]       = mc;
        }
        if (run_type == "Temperature_Dep")
            obj["temperature_range"] = temperature_range;

        if (run_type == "Wang_Landau")
        {
            wl["pre_maxIterations"]      = pre_maxIterations;
            wl["pre_step_equilibration"] = pre_step_equilibration;
            wl["pre_num_samples"]        = pre_num_samples;
            wl["pre_step_correlation"]   = pre_step_correlation;
            wl["pre_number_repetitions"] = pre_number_repetitions;
            wl["log_f_final"]            = log_f_final;
            wl["energy_bin"]             = energy_bin;
            wl["flatness_threshold"]     = flatness_threshold;
            obj["Wang_Landau"]           = wl;
        }

        pw["k_pairwise"]     = k_pairwise;
        pw["tolerance_k"]    = tolerance_k;
        pw["eta_k"]          = eta_k;
        pw["alpha_k"]        = alpha_k;
        pw["gamma_k"]        = gamma_k;
        pw["eta_K_min"]      = eta_K_min;
        obj["k-pairwise"]    = pw;

        return obj;
    };
};

RunParameters parseParameters(const std::string &filename);