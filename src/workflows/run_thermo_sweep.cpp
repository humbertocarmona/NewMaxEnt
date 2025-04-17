#include "workflows/run_thermo_sweep.hpp"
#include "core/compute_model_statistics.hpp"
#include "io/make_file_names.hpp"
#include "io/read_model_minimal_from_json.hpp"
// #include "util/logger.hpp"
#include "util/replica_state.hpp"

#include <fstream>
#include <iomanip>

void write_top_k_replicas(const std::string &filename, const std::vector<ReplicaState> &sorted_replicas, int n_spins)
{
    std::ofstream out_k(filename);
    out_k << "prob";
    for (int i = 0; i < n_spins; ++i)
        out_k << ",s" << std::setfill('0') << std::setw(2) << i + 1;
    out_k << "\n";

    for (const auto &rep : sorted_replicas)
    {
        out_k << rep.probability;
        for (int val : rep.spins)
            out_k << "," << val;
        out_k << "\n";
    }
}

void run_thermo_sweep_workflow(const Params &run_parameters)
{
    auto logger = spdlog::get("bm");

    const auto model     = read_model_minimal_from_json(run_parameters.checkpoint_file);
    const int n          = model.nspins;
    const int n_edges    = n * (n - 1) / 2;
    const int n_triplets = n * (n - 1) * (n - 2) / 6;
    auto filename        = io::make_tdep_filename(run_parameters);
    std::ofstream out(filename);
    out << "T,beta,energy,energy_sq,specific_heat,magnetization\n";
    out << std::fixed << std::setprecision(6);

    for (double T : run_parameters.temperature_range)
    {
        double beta = 1.0 / T;

        arma::Col<double> m1(n, arma::fill::zeros);
        arma::Col<double> m2(n_edges, arma::fill::zeros);
        arma::Col<double> m3(n_triplets, arma::fill::zeros);

        double energy    = 0.0;
        double energy_sq = 0.0;

        //clang-format off
        auto top_replicas = compute_model_statistics(n, model.h, model.J, m1, m2, m3, model.q_val, beta,
                                                     true, // compute means of energy and energy_sq
                                                     &energy, &energy_sq, run_parameters.top_k_states);
        //clang-format on

        double specific_heat = (energy_sq - std::pow(energy, 2.0)) / (T * T);
        double magnetization = arma::mean(m1);

        out << T << "," << beta << "," << energy << "," << energy_sq << "," << specific_heat << "," << magnetization
            << "\n";
        logger->info("[run_thermo_sweep_workflow] T={:.2f} | E={:.4f} | Cv={:.4f} | M={:.4f}", T, energy, specific_heat,
                     magnetization);

        if (run_parameters.top_k_states > 0)
        {
            auto filename_top_k = io::make_top_k_filename(run_parameters, T);
            write_top_k_replicas(filename_top_k, top_replicas, n);
        }
    }

    out.close();
}
