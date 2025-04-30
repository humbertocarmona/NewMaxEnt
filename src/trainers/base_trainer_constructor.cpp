#include "trainers/base_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <nlohmann/json.hpp>
#include <string>

// Constructor
BaseTrainer::BaseTrainer(MaxEntCore &core_,
                         RunParameters &params_,
                         const std::string &data_filename) :
    core(core_),
    params(params_)
{
    auto logger = getLogger();
    int n       = core.nspins;
    ntriplets   = n * (n - 1) * (n - 2) / 6;

    eta_h_t = params.eta_h;
    eta_J_t = params.eta_J;

    if (utils::isFileType(data_filename, "csv"))
    {
        // reads raw data file
        DataStatisticsBreakdown res = compute_data_statistics(data_filename);
        m1_data                     = res.m1_data;
        m2_data                     = res.m2_data;
        m3_data                     = res.m3_data;

        core.J.fill(0);
        // initialize h[i] to match magnetization
        for (int i = 0; i < n; i++)
        {
            core.h[i] = m1_data[i];
        }
        iter = 1;
    }
    else if (utils::isFileType(data_filename, "json"))
    {
        // re-start from last run (more iterations)
        auto obj = readTrainedModel(data_filename);
        if (!obj.contains("type") || obj["type"] != className)
        {
            logger->error("Incorrect or missing object type in JSON");
            throw std::runtime_error("Not a BaseTrainer file");
        }

        int n = obj["run_parameters"]["nspins"];
        if (n != core.nspins)
        {
            logger->error("wrong number of spins {}, expected {} ", n, core.nspins);
            throw std::runtime_error("Wrong number of spins");
        }
        iter    = obj["iter"].get<int>();
        m1_data = utils::jsonToArmaCol<double>(obj["m1_data"]);
        m2_data = utils::jsonToArmaCol<double>(obj["m2_data"]);
        m3_data = utils::jsonToArmaCol<double>(obj["m3_data"]);
        core.h  = utils::jsonToArmaCol<double>(obj["h"]);
        core.J  = utils::jsonToArmaCol<double>(obj["J"]);

        // if commented, will run with the new run_parameters
        // q_val       = obj["run_parameters"]["q_val"];
        // tolerance_h = obj["run_parameters"]["tolerance_h"];
        // tolerance_J = obj["run_parameters"]["tolerance_J"];
        // eta_h       = obj["run_parameters"]["eta_h"];
        // eta_J       = obj["run_parameters"]["eta_J"];
        // alpha_h     = obj["run_parameters"]["alpha_h"];
        // alpha_J     = obj["run_parameters"]["alpha_J"];
        // gamma_h     = obj["run_parameters"]["gamma_h"];
        // gamma_J     = obj["run_parameters"]["gamma_J"];
    }
    else
    {
        throw std::runtime_error("Invalid data file type");
    }

    m1_model = arma::zeros<arma::Col<double>>(core.nspins);
    m2_model = arma::zeros<arma::Col<double>>(core.nedges);
    m3_model = arma::zeros<arma::Col<double>>(ntriplets);

    delta_h = arma::zeros<arma::Col<double>>(core.nspins);
    delta_J = arma::zeros<arma::Col<double>>(core.nedges);
};
