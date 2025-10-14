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

    delta_h = arma::zeros<arma::vec>(core.nspins);
    delta_J = arma::zeros<arma::vec>(core.nedges);

    grad_h = arma::zeros<arma::vec>(core.nspins);
    grad_J = arma::zeros<arma::vec>(core.nedges);

    h_p = arma::zeros<arma::vec>(core.nspins);
    J_p = arma::zeros<arma::vec>(core.nedges);

    last_grad_norm_h = 0.0;
    last_grad_norm_J = 0.0;

    eta_h_t = params.eta_h;
    eta_J_t = params.eta_J;

    iter = 1;
    if (utils::isFileType(data_filename, "csv"))
    {
        // reads raw data file
        DataStatisticsBreakdown res = compute_data_statistics(data_filename);
        m1_data                     = res.m1_data;
        m2_data                     = res.m2_data;
        m3_data                     = res.m3_data;

        core.J.fill(0);
        core.h.fill(0);
        // initialize h[i] to match magnetization
        // for (int i = 0; i < n; i++)
        //     core.h[i] = m1_data[i];

        // std::mt19937 rng(params.rng_seed);

        // std::normal_distribution<double> J_dist(0.0, 1.0 / std::sqrt(core.nspins));
        // for (size_t i = 0; i < core.J.n_elem; ++i)
        //     core.J(i) = J_dist(rng);

        // k-pairwise
        pK_data = res.pK_data;
        core.K.fill(0);
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

        // if (params.iter < 1)
        //     iter = obj["iter"].get<int>();
        // else
        // {
        //     iter = params.iter;
        // }
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

        // k-pairwise
        if (obj.contains("pK_data"))
        {
            pK_data = utils::jsonToArmaCol<double>(obj["pK_data"]);
        }
        if (obj.contains("K"))
        {
            core.K = utils::jsonToArmaCol<double>(obj["K"]);
        }
        core.K = arma::zeros<arma::Col<double>>(core.nspins + 1);
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

    // k-pairwise
    pK_model = arma::zeros<arma::Col<double>>(core.nspins + 1);
    delta_K  = arma::zeros<arma::Col<double>>(core.nspins + 1);
};
