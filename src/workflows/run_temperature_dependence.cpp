#include "workflows/run_temperature_dependence.hpp"
#include "io/make_file_names.hpp"
#include "trainers/full_ensemble_trainer.hpp"
#include "trainers/heat_bath_trainer.hpp"
#include "utils/correlation_histogram.hpp"
#include "utils/get_logger.hpp"

#include <armadillo>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include <fstream>
void save_histogram_to_csv(const std::vector<double> &bin_centers,
                           const std::vector<double> &hist_values,
                           const std::string &filename)
{
    std::ofstream hist_out(filename);
    if (!hist_out)
    {
        throw std::runtime_error("Could not open histogram file for writing: " + filename);
    }

    hist_out << "q,P_q\n";
    hist_out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < bin_centers.size(); ++i)
    {
        hist_out << bin_centers[i] << "," << hist_values[i] << "\n";
    }
}

void save_replicas_to_csv(const arma::Mat<int> &replicas, const std::string &filename)
{
    std::ofstream replicas_out(filename);

    if (!replicas_out)
    {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    for (size_t i = 0; i < replicas.n_rows; ++i)
    {
        for (size_t j = 0; j < replicas.n_cols; ++j)
        {
            replicas_out << replicas(i, j);
            if (j + 1 < replicas.n_cols)
                replicas_out << ",";
        }
        replicas_out << "\n";
    }

    replicas_out.close();
}

void runTemperatureDependence(RunParameters &params)
{
    auto logger      = getLogger();
    bool use_T_range = false;
    if (params.beta_range.empty() && !params.T_range.empty())
    {
        use_T_range = true;
        params.beta_range.resize(params.T_range.size());
        for (size_t i = 0; i < params.T_range.size(); ++i)
        {
            params.beta_range[i] = 1.0 / params.T_range[i];
        }
    }
    int nspins = params.nspins;
    MaxEntCore core(nspins, params.runid);

    FullEnsembleTrainer model_full(core, params, params.trained_model_file);
    HeatBathTrainer model_mc(core, params, params.trained_model_file);

    std::size_t nt = params.beta_range.size();
    arma::vec beta_range(params.beta_range.data(), nt, /* copy_aux_mem = */ true);
    arma::vec E(nt, arma::fill::zeros);
    arma::vec FSUPP(nt, arma::fill::zeros);
    arma::vec CV(nt, arma::fill::zeros);
    arma::vec M(nt, arma::fill::zeros);
    arma::vec Qmax(nt, arma::fill::zeros);
    arma::vec PQmax(nt, arma::fill::zeros);
    arma::vec MaxWeight(nt, arma::fill::zeros);
    arma::vec MaxBracket(nt, arma::fill::zeros);
    arma::vec MaxWeightEnergy(nt, arma::fill::zeros);

    if (nspins < 21)
    { // loop to compute T, beta, <E>, <CV> <mag>, full ensemble is more accurate
        std::size_t i = 0;
        for (double beta : params.beta_range)
        {
            double T = 1.0 / beta;

            model_full.computeModelAverages(beta, true);
            double energy = model_full.get_avg_energy();
            double specific_heat =
                (model_full.get_avg_energy_sq() - std::pow(energy, 2.0)) / (T * T);
            double magnetization = model_full.get_avg_magnetization();
            double f_supp        = model_full.get_f_supp();

            logger->info("[runTemperatureDependence]  T={:.2f} beta={:.2f} E={:.2f} CV={:.2f} "
                         "M={:.2f} fsupp={:.2e}",
                         T, beta, energy, specific_heat, magnetization, f_supp);

            E(i)               = energy;
            CV(i)              = specific_heat;
            M(i)               = magnetization;
            FSUPP(i)           = f_supp;
            MaxWeight(i)       = model_full.get_max_weight();
            MaxBracket(i)      = model_full.get_max_bracket();
            MaxWeightEnergy(i) = model_full.get_max_weight_energy();

            if (params.compute_replica_cor)
            {
                // need model_mc to compute replica correlations
                model_mc.computeModelAverages(beta, true);
                auto replicas                   = model_mc.get_replicas();
                auto [bin_centers, hist_values] = correlation_histogram<int>(replicas, 2.0, true);
                auto max_it    = std::max_element(hist_values.begin(), hist_values.end());
                size_t max_idx = std::distance(hist_values.begin(), max_it);
                double q_max   = bin_centers[max_idx];
                double p_q_max = hist_values[max_idx]; //*max_it;
                Qmax(i)        = q_max;
                PQmax(i)       = p_q_max;

                logger->info("[runTemperatureDependence] q_max={:.2f}, p_q_max={:.2f}", q_max,
                             p_q_max);
                if (std::abs(T - 1.0) < 1e-6 * std::max(1.0, std::abs(T)))
                {
                    auto file_replicas = io::make_replicas_filename(params, T);
                    auto file_corr     = io::make_replica_correlation_filename(params, T);
                    save_replicas_to_csv(replicas, file_replicas);
                    save_histogram_to_csv(bin_centers, hist_values, file_corr);
                }
            }

            i++;
        }
    }
    else
    { // heat_bath already used to train the model
        std::size_t i = 0;
        for (double beta : params.beta_range)
        {
            double T = 1.0 / beta;

            model_mc.computeModelAverages(beta, true);
            double energy = model_mc.get_avg_energy();
            double specific_heat =
                beta * beta * (model_mc.get_avg_energy_sq() - std::pow(energy, 2.0));
            double magnetization = model_mc.get_avg_magnetization();

            auto replicas                   = model_mc.get_replicas();
            auto [bin_centers, hist_values] = correlation_histogram<int>(replicas, 2.0, true);
            auto max_it    = std::max_element(hist_values.begin(), hist_values.end());
            size_t max_idx = std::distance(hist_values.begin(), max_it);
            double q_max   = bin_centers[max_idx];
            double p_q_max = hist_values[max_idx]; //*max_it;

            E(i)     = energy;
            CV(i)    = specific_heat;
            M(i)     = magnetization;
            Qmax(i)  = q_max;
            PQmax(i) = p_q_max;

            logger->info(
                "[runTemperatureDependence] T={:.2f} beta={:.2f} E={:.2f} CV={:.2f} M={:.2f}, "
                "q_max={:.2f}, p_q_max={:.2f}",
                T, beta, energy, specific_heat, magnetization, q_max, p_q_max);

            if (std::abs(T - 1.0) < 1e-6 * std::max(1.0, std::abs(T)))
            {
                auto file_replicas = io::make_replicas_filename(params, T);
                auto file_corr     = io::make_replica_correlation_filename(params, T);
                save_replicas_to_csv(replicas, file_replicas);
                save_histogram_to_csv(bin_centers, hist_values, file_corr);
            }
            i++;
        }
    }

    // creates a tde- file to save temperature dependence
    auto file_tdep = io::make_filename(params, "tdep");
    // ./results/pairwise/tdep_$(runid)_n_$(nspins).csv <----------
    std::ofstream tdep_fout(file_tdep);
    if (!tdep_fout)
    {
        logger->error("Could not open {}", file_tdep);
        return;
    }

    // Header
    tdep_fout << "T,beta,E,CV,M,fsupp,Qmax,PQmax,MaxWeight,MaxBracket,MaxWeightEnergy\n";

    // Data
    size_t n = beta_range.size();
    for (size_t i = 0; i < n; ++i)
    {
        tdep_fout << std::setprecision(12) << 1.0 / beta_range(i) << "," << beta_range(i) << ","
                  << E(i) << "," << CV(i) << "," << M(i) << "," << FSUPP(i) << "," << Qmax(i) << ","
                  << PQmax(i) << "," << MaxWeight(i) << "," << MaxBracket(i) << ","
                  << MaxWeightEnergy(i) << "\n";
    }

    tdep_fout.close();
    logger->info("[runTemperatureDependence] Saved CSV to {}", file_tdep);

    // nlohmann::json obj;
    // obj["beta"]  = beta_range;
    // obj["E"]     = E;
    // obj["CV"]    = CV;
    // obj["M"]     = M;
    // obj["Qmax"]  = Qmax;
    // obj["PQmax"] = PQmax;
    //
    // std::ofstream fout(file_tdep);
    // fout << obj.dump(2);
    // fout.close();
    // logger->info("[runTemperatureDependence] Saved to {}", file_tdep);
}
