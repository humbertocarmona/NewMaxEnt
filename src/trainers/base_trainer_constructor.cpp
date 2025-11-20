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
    auto logger   = getLogger();
    int n         = core.nspins;
    auto run_type = params.run_type;
    ntriplets     = n * (n - 1) * (n - 2) / 6;

    // from parameters file
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

    // let's zero this just in case...
    m1_model = arma::zeros<arma::Col<double>>(core.nspins);
    m2_model = arma::zeros<arma::Col<double>>(core.nedges);
    m3_model = arma::zeros<arma::Col<double>>(ntriplets);

    delta_h = arma::zeros<arma::Col<double>>(core.nspins);
    delta_J = arma::zeros<arma::Col<double>>(core.nedges);

    // k-pairwise
    pK_model = arma::zeros<arma::Col<double>>(core.nspins + 1);
    delta_K  = arma::zeros<arma::Col<double>>(core.nspins + 1);

    iter = params.iter;
    // run_type == Full_Ensemble (Full) or Heat_Bath (MC)
    bool train          = (run_type == "Full" || run_type == "Full_Ensemble");
    train               = train || (run_type == "MC" || run_type == "Heat_Bath");
    bool read_raw_data  = train && utils::isFileType(data_filename, "csv");
    bool read_model     = train && utils::isFileType(data_filename, "json");

    bool read_model_gen = (run_type == "Gen_Full" || run_type == "Gen_MC");
    read_model_gen      = read_model_gen && utils::isFileType(data_filename, "json");

    read_model = read_model || (run_type == "Copy");


    if (read_raw_data)
    {   // reads raw data file

        // need to compute the moments
        DataStatisticsBreakdown res = compute_data_statistics(data_filename);
        m1_data                     = res.m1_data;
        m2_data                     = res.m2_data;
        m3_data                     = res.m3_data;
        pK_data = res.pK_data;
                
        core.J.fill(0);
        core.h.fill(0);
        core.K.fill(0);
        
        /* -------------------------------------------------------------------------
        // initialize h[i] to match magnetization
        for (int i = 0; i < n; i++)
                 core.h[i] = m1_data[i];

                 
        // initialize J[i] with centered normal distribution
        std::mt19937 rng(params.rng_seed);
        std::normal_distribution<double> J_dist(0.0, 1.0 / std::sqrt(core.nspins));
        for (size_t i = 0; i < core.J.n_elem; ++i)
             core.J(i) = J_dist(rng);
        ------------------------------------------------------------------------- */
    }
    else if (read_model)
    {
        // re-start from last run (more iterations)
        auto obj    = readJSONData(data_filename);
        bool obj_ok = obj.contains("m1_data") || obj.contains("x_obs");
        if (!obj_ok)
        {
            logger->error("Incorrect or missing object type in JSON");
            throw std::runtime_error("Can't read this file");
        }
        if (params.ver == "1.1")
        { // latest version of the output file
            int n = obj["run_parameters"]["nspins"];
            if (n != core.nspins)
            {
                logger->error("wrong number of spins {}, expected {} ", n, core.nspins);
                throw std::runtime_error("Wrong number of spins");
            }

            // keep old h and J for continuation
            core.h = utils::jsonToArmaCol<double>(obj["h"]);
            core.J = utils::jsonToArmaCol<double>(obj["J"]);
            // observations and last parameters are read from file
            m1_data = utils::jsonToArmaCol<double>(obj["m1_data"]);
            m2_data = utils::jsonToArmaCol<double>(obj["m2_data"]);
            m3_data = utils::jsonToArmaCol<double>(obj["m3_data"]);
            if (obj.contains("pK_data"))
            {
                pK_data = utils::jsonToArmaCol<double>(obj["pK_data"]);
            }
            else
            {
                pK_data = arma::zeros<arma::Col<double>>(core.nspins + 1);
            }

            m1_model = utils::jsonToArmaCol<double>(obj["m1_model"]);
            m2_model = utils::jsonToArmaCol<double>(obj["m2_model"]);
            m3_model = utils::jsonToArmaCol<double>(obj["m3_model"]);
            pK_model = utils::jsonToArmaCol<double>(obj["pK_model"]);
        }
        else if (params.ver == "1.0")
        { // for backwards compatibility
            int n = core.nspins;
            if(obj.contains("nspins")){  // ver1.0 synth files don't have nspins
                n = obj["nspins"];
            }
            
            if (n != core.nspins)
            {
                logger->error("wrong number of spins {}, expected {} ", n, core.nspins);
                throw std::runtime_error("Wrong number of spins");
            }

            // keep old h and J for continuation
            core.h = utils::jsonToArmaCol<double>(obj["h"]);
            core.J = utils::jsonToArmaCol<double>(obj["J"]);

            m1_data = utils::jsonToArmaCol<double>(obj["x_obs"]);
            m2_data = utils::jsonToArmaCol<double>(obj["xy_obs"]);
            m3_data = utils::jsonToArmaCol<double>(obj["xyz_obs"]);
            pK_data = utils::jsonToArmaCol<double>(obj["P_K_obs"]);

            m1_model = utils::jsonToArmaCol<double>(obj["x_mod"]);
            m2_model = utils::jsonToArmaCol<double>(obj["xy_mod"]);
            m3_model = utils::jsonToArmaCol<double>(obj["xyz_mod"]);
            pK_model = utils::jsonToArmaCol<double>(obj["P_K_mod"]);
        }

        // only if the model was run with k_pairwise originally
        // K is are the Lagrange multipliers associated with p(k)
        // only the latest version
        if (obj.contains("K"))
        {
            core.K = utils::jsonToArmaCol<double>(obj["K"]);
        }
        else
        {
            core.K.fill(0.0);
        }

        // need to reset the h and J fields in case of reading a synthetic sample
        if (params.reset_fields)
        { // ! this is the case when reading a synth_ file with data observations
            logger->info("reseting model parameters h,J and K");
            core.h.fill(0.0);
            core.J.fill(0.0);
            core.K.fill(0.0);
        }
        // keep
    }
    else if (read_model_gen)
    { // this file contains only nspins, h and J
        auto obj = readJSONData(data_filename);

        if (!obj.contains("nspins"))
        {
            logger->error("[base_trainer_constructor] JSON data need field 'nspins'");
            throw std::runtime_error("'nspins' required");
        }

        if (!obj.contains("h"))
        {
            logger->error("[base_trainer_constructor] JSON data need field 'h'");
            throw std::runtime_error("'h' required");
        }
        if (!obj.contains("J"))
        {
            logger->error("[base_trainer_constructor] JSON data need field 'J'");
            throw std::runtime_error("'J' required");
        }
        std::cout << "ok read obj" << std::endl;

        int n = obj["nspins"];
        if (n != core.nspins)
        {
            logger->error("[base_trainer_constructor] wrong number of spins {}, expected {} ", n, core.nspins);
            throw std::runtime_error("Wrong number of spins");
        }
        core.h = utils::jsonToArmaCol<double>(obj["h"]);
        core.J = utils::jsonToArmaCol<double>(obj["J"]);

        m1_data = arma::zeros<arma::Col<double>>(core.nspins);
        m2_data = arma::zeros<arma::Col<double>>(core.nedges);
        m3_data = arma::zeros<arma::Col<double>>(ntriplets);
        pK_data = arma::zeros<arma::Col<double>>(core.nspins + 1);
    }
    else
    {
        throw std::runtime_error("[base_trainer_constructor] Invalid model data type");
    }
};