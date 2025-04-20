#include "trainers/base_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <memory>
#include <nlohmann/json.hpp>

// Constructor
BaseTrainer::BaseTrainer(MaxEntCore &core,
                         double q_val,
                         size_t maxIterations,
                         double tolerance_h,
                         double tolerance_J,
                         double eta_h,
                         double eta_J,
                         double alpha_h,
                         double alpha_J,
                         double gamma_h,
                         double gamma_J,
                         const std::string &data_filename) :
    core(core),
    q_val(q_val),
    maxIterations(maxIterations),
    tolerance_h(tolerance_h),
    tolerance_J(tolerance_J),
    eta_h(eta_h),
    eta_J(eta_J),
    alpha_h(alpha_h),
    alpha_J(alpha_J),
    gamma_h(gamma_h),
    gamma_J(gamma_J)
{
    auto logger = getLogger();
    int n       = core.nspins;
    ntriplets   = n * (n - 1) * (n - 2) / 6;

    logger->info("data_filename = {}", data_filename);
    if (utils::isFileType(data_filename, "csv"))
    {
        // reads raw data file
        DataStatisticsBreakdown res = compute_data_statistics(data_filename);
        m1_data                     = res.m1_data;
        m2_data                     = res.m2_data;
        m3_data                     = res.m3_data;

        core.h.fill(0);
        core.J.fill(0);
        iter = 1;
    }
    else if (utils::isFileType(data_filename, "json"))
    {
        // re-start from last run (more iterations)
        auto obj = readTrainedModel(data_filename);
        if (!obj.contains("type") || obj["type"] != className)
        {
            logger->error("Incorrect or missing object type in JSON");
            throw std::runtime_error("Invalid JSON object type");
        }

        int n = obj["run_parameters"]["nspins"];
        if (n != core.nspins)
        {
            logger->error("wrong number of spins {}, expected {} ", n, core.nspins);
            throw std::runtime_error("Wrong number of spins");
        }
        iter        = obj["iter"].get<int>();
        m1_data     = utils::jsonToArmaCol<double>(obj["m1_data"]);
        m2_data     = utils::jsonToArmaCol<double>(obj["m2_data"]);
        m3_data     = utils::jsonToArmaCol<double>(obj["m3_data"]);
        core.h      = utils::jsonToArmaCol<double>(obj["h"]);
        core.J      = utils::jsonToArmaCol<double>(obj["J"]);
        q_val       = obj["run_parameters"]["q_val"];
        tolerance_h = obj["run_parameters"]["tolerance_h"];
        tolerance_J = obj["run_parameters"]["tolerance_J"];
        eta_h       = obj["run_parameters"]["eta_h"];
        eta_J       = obj["run_parameters"]["eta_J"];
        alpha_h     = obj["run_parameters"]["alpha_h"];
        alpha_J     = obj["run_parameters"]["alpha_J"];
        gamma_h     = obj["run_parameters"]["gamma_h"];
        gamma_J     = obj["run_parameters"]["gamma_J"];
    }
    else
    {
        m1_data = arma::zeros<arma::Col<double>>(core.nspins);
        m2_data = arma::zeros<arma::Col<double>>(core.nedges);
        m3_data = arma::zeros<arma::Col<double>>(ntriplets);
        logger->warn("here now");
    }

    m1_model = arma::zeros<arma::Col<double>>(core.nspins);
    m2_model = arma::zeros<arma::Col<double>>(core.nedges);
    m3_model = arma::zeros<arma::Col<double>>(ntriplets);

    delta_h = arma::zeros<arma::Col<double>>(core.nspins);
    delta_J = arma::zeros<arma::Col<double>>(core.nedges);

};


// void BaseTrainer::train()
// {
//     // Provide a default or empty implementation if not overridden.
// }