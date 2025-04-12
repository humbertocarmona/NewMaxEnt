#pragma once

#include "util/logger.hpp"
#include "util/utilities.hpp"
#include <armadillo>
#include <string>

struct Params
{
    /* data */
    std::string id;
    int runid;                    // runid could be the gen seed
    std::string raw_samples_file; // filename with raw samples to compute means
    std::string stats_file;       // JSON file with means, and correlations
    std::string checkpoint_file;  // JSON checkpoint file with means, and correlations
    std::string run_type;         //
    std::string comment;
    std::string result_dir;
    int save_state;
    bool save_result;
    bool save_energies;
    int max_iter;
    int init_step;
    float q_val;
    float beta;
    float eta_h;
    float eta_J;
    float gamma_h;
    float gamma_J;
    float alpha;
    float tol_1;
    float tol_2;
    // Generate means from h0, J0
    int gen_nspins;
    int gen_seed; // could define the runid for this run, and also for any run
    // that starts from it
    double gen_h_mean;
    double gen_h_width;
    double gen_J_mean;
    double gen_J_width;
    arma::Col<double> gen_h0; // do not initialize it, it will have n_elem=0
    arma::Col<double> gen_J0;
    // Monte Carlo
    int mc_n_samples;
    int mc_n_coherence;
    int mc_n_equilibrium;
    int mc_n_rept;
    int mc_seed;
    // clang-format off
    Params(std::string id_ = "",
           int runid_ = 0,
           std::string raw_sample_file_ = "none",
           std::string stats_file_ = "none",
           std::string checkpoint_file_ = "none",
           std::string run_type_ = "ens",
           std::string comment_ = "auto",
           std::string result_dir_ = "./Testing",
           int save_sate_ = 1000,
           bool save_result_ = true,
           bool save_energies_ = false,
           int max_iter_ = 1000,
           int init_step_ = 0,
           float q_val_ = 1.0,
           float beta_ = 1.0,
           double eta_h_ = 0.1,
           double eta_J_ = 0.1,
           double gamma_h_ = 0.2,
           double gamma_J_ = 0.2,
           double alpha_ = 0.1,
           double tol_1_ = 0.001,
           double tol_2_ = 0.001,
           // generate means from h0, J0
           int gen_nspins_ = 16,
           int gen_seed_ = -1,
           double gen_h_mean_ = -1.0,
           double gen_h_width_ = 2.0,
           double gen_J_mean_ = 0.0,
           double gen_J_width_ = 0.5,
           // Monte Carlo
           int mc_n_samples_ = 50000,
           int mc_n_equilibrium_ = 3000,
           int mc_n_coherence_ = 40,
           int mc_n_rept_ = 40,
           int mc_seed_ = 1234): runid(runid_),
        id(id_),
        raw_samples_file(raw_sample_file_),
        stats_file(stats_file_),
        checkpoint_file(checkpoint_file_),
        run_type(run_type_),
        comment(comment_),
        result_dir(result_dir_),
        save_result(save_result_),
        save_energies(save_energies_),
        save_state(save_sate_),
        max_iter(max_iter_),
        init_step(init_step_),
        q_val(q_val_),
        beta(beta_),
        eta_h(eta_h_),
        eta_J(eta_J_),
        gamma_h(gamma_h_),
        gamma_J(gamma_J_),
        alpha(alpha_),
        tol_1(tol_1_),
        tol_2(tol_2_),
        gen_seed(gen_seed_),
        gen_nspins(gen_nspins_),
        gen_h_mean(gen_h_mean_),
        gen_h_width(gen_h_width_),
        gen_J_mean(gen_J_mean_),
        gen_J_width(gen_h_width_),
        mc_n_samples(mc_n_samples_),
        mc_n_coherence(mc_n_coherence_),
        mc_n_equilibrium(mc_n_equilibrium_),
        mc_n_rept(mc_n_rept_),
        mc_seed(mc_seed_) {};
    //clang-format on


    void log_info(bool verbose = false) const
    {
        set_console_verbosity(verbose); // Adjust verbosity based on the flag
        auto logger = create_logger();

        logger->info("[Params] id                {}", id);
        logger->info("[Params] runid             {}", runid);
        logger->info("[Params] raw_samples_file  {}", raw_samples_file);
        logger->info("[Params] stats_file        {}", stats_file);
        logger->info("[Params] checkpoint_file   {}", checkpoint_file);
        logger->info("[Params] run_type          {}", run_type);
        logger->info("[Params] comment           {}", comment);
        logger->info("[Params] result_dir        {}", result_dir);
        logger->info("[Params] save_state        {}", save_state);
        logger->info("[Params] save_energies     {}", save_energies);
        logger->info("[Params] max_iter          {}", max_iter);
        logger->info("[Params] init_step         {}", init_step);
        logger->info("[Params] q_val             {}", q_val);
        logger->info("[Params] beta              {}", beta);
        logger->info("[Params] eta_h             {}", eta_h);
        logger->info("[Params] eta_J             {}", eta_J);
        logger->info("[Params] gamma_h           {}", gamma_h);
        logger->info("[Params] gamma_J           {}", gamma_J);
        logger->info("[Params] alpha             {}", alpha);
        logger->info("[Params] tol_1             {}", tol_1);
        logger->info("[Params] tol_2             {}", tol_2);
        // Gen section
        if (run_type=="gen"){
            logger->info("[Params] gen_nspins        {}", gen_nspins);
            logger->info("[Params] gen_h_mean        {}", gen_h_mean);
            logger->info("[Params] gen_h_width       {}", gen_h_width);
            logger->info("[Params] gen_J_mean        {}", gen_J_mean);
            logger->info("[Params] gen_J_width       {}", gen_J_width);
        }
        // Monte Carlo
        if (run_type == "mc"){
            logger->info("[Params] mc_n_samples      {}", mc_n_samples);
            logger->info("[Params] mc_n_coherence    {}", mc_n_coherence);
            logger->info("[Params] mc_n_equilibrium  {}", mc_n_equilibrium);
            logger->info("[Params] mc_n_rept         {}", mc_n_rept);
            logger->info("[Params] mc_seed           {}", mc_seed);
        }
        logger->flush();
    }
};

