#include "workflows/run_thermo_sweep.hpp"
#include "io/read_model_minimal_from_json.hpp"
#include "core/compute_model_statistics.hpp"
#include "util/logger.hpp"

#include <fstream>
#include <iomanip>

void run_thermo_sweep_workflow(const Params& run_parameters)
{
    auto logger = spdlog::get("bm");
    std::ostringstream filename;

    const auto model = read_model_minimal_from_json(run_parameters.checkpoint_file);
    const int n = model.nspins;
    const int n_edges = n * (n - 1) / 2;
    const int n_triplets = n * (n - 1) * (n - 2) / 6;
    filename << run_parameters.result_dir << "/"
    << "thermo_sweep_" << run_parameters.runid 
    << "_nspins_" << run_parameters.gen_nspins 
    << "_q_"  << std::fixed << std::setprecision(2) << run_parameters.q_val 
    << ".csv";

    std::ofstream out(run_parameters.result_dir + "/thermo_sweep.csv");
    out << "T,beta,energy,energy_sq,specific_heat,magnetization\n";
    out << std::fixed << std::setprecision(6);


    for (double T : run_parameters.temperature_range)
    {
        double beta = 1.0 / T;

        arma::Col<double> m1(n, arma::fill::zeros);
        arma::Col<double> m2(n_edges, arma::fill::zeros);
        arma::Col<double> m3(n_triplets, arma::fill::zeros);

        double energy = 0.0;
        double energy_sq = 0.0;

        compute_model_statistics(
            n,
            model.h,
            model.J,
            m1,
            m2,
            m3,
            model.q_val,
            beta,
            true, // compute means of energy and energy_sq
            &energy,
            &energy_sq
        );

        double specific_heat = (energy_sq - std::pow(energy, 2.0)) / (T * T);
        double magnetization = arma::mean(m1);

        out << T << "," << beta << "," << energy << "," << energy_sq << ","
            << specific_heat << "," << magnetization << "\n";
            logger->info("[run_thermo_sweep_workflow] T={:.2f} | E={:.4f} | Cv={:.4f} | M={:.4f}",
                T, energy, specific_heat, magnetization);
    }

    out.close();
}
