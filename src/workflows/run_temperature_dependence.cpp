#include "workflows/run_temperature_dependence.hpp"
#include "io/make_file_names.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/correlation_histogram.hpp"
#include "utils/get_logger.hpp"

#include <fstream>
#include <iomanip>

#include <fstream>
void save_histogram_to_csv(const std::vector<double> &bin_centers,
                           const std::vector<double> &hist_values,
                           const std::string &filename)
{
    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Could not open histogram file for writing: " + filename);
    }

    out << "q,P_q\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < bin_centers.size(); ++i)
    {
        out << bin_centers[i] << "," << hist_values[i] << "\n";
    }
}

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

void runTemperatureDependence(RunParameters &params)
{
    auto logger = getLogger();

    int nspins = params.nspins;
    MaxEntCore core(nspins, params.runid);
    FullEnsembleTrainer model_full(core, params,params.trained_model_file);

    HeatBathTrainer model_mc(core, params, params.trained_model_file);
    
    double energy;
    double specific_heat;
    double magnetization;
    auto file_tdep = io::make_filename(params, "tdep-");
    std::ofstream out(file_tdep);
    logger->info("[runTemperatureDependence] Saving to {}", file_tdep);
    out << "T,beta,energy,specific_heat,magnetization,q_max\n";
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

            model_mc.computeModelAverages(beta, true);
            auto replicas = model_mc.get_replicas();

            auto [bin_centers, hist_values] = correlation_histogram<int>(replicas, 2.0, true);

            auto max_it    = std::max_element(hist_values.begin(), hist_values.end());
            size_t max_idx = std::distance(hist_values.begin(), max_it);
            double q_max   = bin_centers[max_idx];
            double p_q_max = hist_values[max_idx]; //*max_it;

            logger->info("[runTemperatureDependence] T={:.2f} E={:.2f} CV={:.2f} M={:.2f}, "
                         "q_max={:.2f}, p_q_max={:.2f}",
                         T, energy, specific_heat, magnetization, q_max, p_q_max);
            out << T << "," << beta << "," << energy << "," << specific_heat << "," << magnetization
                << "," << q_max << "\n";

            if (std::abs(T - 1.0) < 1e-6 * std::max(1.0, std::abs(T)))
            {
                auto file_replicas = io::make_replicas_filename(params, T);
                auto file_corr     = io::make_replica_correlation_filename(params, T);
                save_replicas_to_csv(replicas, file_replicas);
                save_histogram_to_csv(bin_centers, hist_values, file_corr);
            }
        }
    }
    else
    { // heat_bath already used to train the model
        for (double T : params.temperature_range)
        {
            double beta = 1.0 / T;

            model_mc.computeModelAverages(beta, true);
            energy        = model_mc.get_avg_energy();
            specific_heat = (model_mc.get_avg_energy_sq() - std::pow(energy, 2.0)) / (T * T);
            magnetization = model_mc.get_avg_magnetization();

            auto replicas = model_mc.get_replicas();

            auto [bin_centers, hist_values] = correlation_histogram<int>(replicas, 2.0, true);

            auto max_it    = std::max_element(hist_values.begin(), hist_values.end());
            size_t max_idx = std::distance(hist_values.begin(), max_it);
            double q_max   = bin_centers[max_idx];
            double p_q_max = hist_values[max_idx]; //*max_it;

            logger->info("[runTemperatureDependence] T={:.2f} E={:.2f} CV={:.2f} M={:.2f}, "
                         "q_max={:.2f}, p_q_max={:.2f}",
                         T, energy, specific_heat, magnetization, q_max, p_q_max);
            out << T << "," << beta << "," << energy << "," << specific_heat << "," << magnetization
                << "," << q_max << "\n";

            if (std::abs(T - 1.0) < 1e-6 * std::max(1.0, std::abs(T)))
            {
                auto file_replicas = io::make_replicas_filename(params, T);
                auto file_corr     = io::make_replica_correlation_filename(params, T);
                save_replicas_to_csv(replicas, file_replicas);
                save_histogram_to_csv(bin_centers, hist_values, file_corr);
            }
        }
    }
}
