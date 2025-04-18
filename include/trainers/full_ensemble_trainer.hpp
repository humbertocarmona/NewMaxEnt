#pragma once

#include "core/max_ent_core.hpp" // Your minimal core class header
#include "utils/compute_data_statistics.hpp"
#include <armadillo>
#include <memory>
#include <spdlog/spdlog.h>

class FullEnsembleTrainer
{
  public:
    // Constructor
    FullEnsembleTrainer(MaxEntCore &core,
                        size_t maxIterations,
                        double tolerance_h,
                        double tolerance_J,
                        double eta_h,
                        double eta_J,
                        double alpha_h,
                        double alpha_J,
                        double delta_h,
                        double delta_J,
                        const std::string &data_filename)
        : core(core), maxIterations(maxIterations), tolerance_h(tolerance_h), tolerance_J(tolerance_J), eta_h(eta_h),
          eta_J(eta_J), alpha_h(alpha_h), alpha_J(alpha_J), delta_h(delta_h), delta_J(delta_J)
    {
        int n                       = core.nspins;
        ntriplets                   = n * (n - 1) * (n - 2) / 6;
        DataStatisticsBreakdown res = compute_data_statistics(data_filename);
        m1_data                     = res.m1_data;
        m2_data                     = res.m2_data;
        m3_data                     = res.m3_data;

        m1_model = arma::zeros<arma::Col<double>>(core.nspins);
        m2_model = arma::zeros<arma::Col<double>>(core.nedges);
        m3_model = arma::zeros<arma::Col<double>>(ntriplets);
    };

    // Main training function
    void train();

    // may be used by TemperatureDependence of the trained model
    void computeFullEnumerationAverages(double beta, bool triplets);

    const arma::Col<double> &get_m1_data() const
    {
        return m1_data;
    }
    const arma::Col<double> &get_m2_data() const
    {
        return m2_data;
    }
    const arma::Col<double> &get_m3_data() const
    {
        return m3_data;
    }

    const arma::Col<double> &get_m1_model() const
    {
        return m1_model;
    }
    const arma::Col<double> &get_m2_model() const
    {
        return m2_model;
    }
    const arma::Col<double> &get_m3_model() const
    {
        return m2_model;
    }

  private:
    // Reference to the core model
    MaxEntCore &core;
    int ntriplets;

    // Training parameters
    size_t maxIterations;
    double tolerance_h;
    double tolerance_J;
    double q_val;   // for exp_q(x,q) Tsallis q-exponential
    double eta_h;   // training rate for h
    double eta_J;   // training rate for J
    double alpha_h; // training momentum alpha_h * delta_h
    double alpha_J; // training momentum alpha_J * delta_J
    double delta_h; // training momentum alpha_h * delta_h
    double delta_J; // training momentum alpha_J * delta_J

    // sample and model averages
    double avg_energy;
    double avg_energy_sq;

    // averages used for training
    arma::Col<double> m1_data;  // sample fist momentum: <s_i>
    arma::Col<double> m2_data;  // sample second momentum: <s_i*s_j>
    arma::Col<double> m1_model; // model's fist momentum: <s_i>
    arma::Col<double> m2_model; // model's second momentum: <s_i*s_j>

    // averages used to compare predictions
    arma::Col<double> m3_data;  // sample third momentum: <s_i*s_j*s_j>
    arma::Col<double> m3_model; // model's third momentum: <s_i*s_j*s_j>

    // Private helper functions

    void updateModelParameters();
    double energyAllPairs(arma::Col<int> s);
};
