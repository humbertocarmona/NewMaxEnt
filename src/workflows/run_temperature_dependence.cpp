#include "workflows/run_temperature_dependence.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "io/make_file_names.hpp"
#include "utils/get_logger.hpp"

#include <fstream>
#include <iomanip>

#include <fstream>

void save_replicas_to_csv(const arma::Mat<int> &replicas, const std::string &filename)
{
    std::ofstream out(filename);

    if (!out)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    for (size_t i = 0; i < replicas.n_rows; ++i)
    {
        for (size_t j = 0; j < replicas.n_cols; ++j)
        {
            out << replicas(i, j);
            if (j + 1 < replicas.n_cols)
                out << ",";
        }
        out << "\n";
    }

    out.close();
}

void runTemperatureDependence(const RunParameters &params)
{
    auto logger = getLogger();

    int nspins = params.nspins;
    MaxEntCore core(nspins, params.runid);
    FullEnsembleTrainer model_full(core, params.q_val, params.maxIterations, params.tolerance_h,
                                   params.tolerance_J, params.eta_h, params.eta_J, params.alpha_h,
                                   params.alpha_J, params.gamma_h, params.gamma_J,
                                   params.trained_model_file);

    HeatBathTrainer model_mc(core, params.q_val, params.maxIterations, params.tolerance_h,
                             params.tolerance_J, params.eta_h, params.eta_J, params.alpha_h,
                             params.alpha_J, params.gamma_h, params.gamma_J,
                             params.trained_model_file);
    model_mc.configureMonteCarlo(params.equilibration_sweeps, params.numSamples,
                                 params.sampleInterval);

    double energy;
    double specific_heat;
    double magnetization;
    auto file_tdep = io::make_tdep_filename(params);
    logger->info("opening {}", file_tdep);
    std::ofstream out(file_tdep);
    out << "T,beta,energy,specific_heat,magnetization\n";
    out << std::fixed << std::setprecision(6);

    if (nspins < 21)
    { // loop to compute T, beta, <E>, <CV> <mag>, full ensemble is more accurate
        for (double T : params.temperature_range)
        {
            double beta = 1.0 / T;

            model_full.computeModelAverages(beta, true);
            energy        = model_full.get_avg_energy();
            specific_heat = (model_full.get_avg_energy_sq() - std::pow(energy, 2.0)) / (T * T);
            magnetization = model_full.get_avg_magnetization();

            out << T << "," << beta << "," << energy << ","  << specific_heat << "," << magnetization
            << "\n";
            logger->info("[runTemperatureDependence] T={:.2f} E={:.2f} CV={:.2f} M={:.2f}", 
                T, energy, specific_heat, magnetization);

            model_mc.computeModelAverages(beta, true);
            auto replicas = model_mc.get_replicas();
            
            auto file_replicas = io::make_replicas_filename(params, T);
            save_replicas_to_csv(replicas, file_replicas);
        }
    }
    else
    {   // have to do with the heat_bath already used to train the model
        for (double T : params.temperature_range)
        {
            double beta = 1.0 / T;

            model_mc.computeModelAverages(beta, true);
            energy        = model_mc.get_avg_energy();
            specific_heat = (model_mc.get_avg_energy_sq() - std::pow(energy, 2.0)) / (T * T);
            magnetization = model_mc.get_avg_magnetization();

            out << T << "," << beta << "," << energy << ","  << specific_heat << "," << magnetization
            << "\n";
            logger->info("[runTemperatureDependence] T={:.2f} E={:.2f} CV={:.2f} M={:.2f}", 
                T, energy, specific_heat, magnetization);

            auto replicas = model_mc.get_replicas();

            auto file_replica = io::make_replicas_filename(params, T);
            save_replicas_to_csv(replicas, file_replica);
        }
    }
}