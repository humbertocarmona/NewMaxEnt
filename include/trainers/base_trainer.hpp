#pragma once
#include "core/max_ent_core.hpp" // minimal core class header
#include "io/read_trained_json.hpp"
#include "utils/compute_data_statistics.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"
#include <armadillo>
#include <memory>
#include <nlohmann/json.hpp>

class BaseTrainer
{
  public:
    BaseTrainer(MaxEntCore &core,
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
                const std::string &data_filename);

    virtual ~BaseTrainer() = default;

    // This method will be overridden by each specific trainer
    virtual void computeModelAverages(double beta, bool triplets = false) = 0;

    // Other common tasks (e.g., compute temperature dependence)
    // virtual void computeTemperatureDependence(double betaStart, double betaEnd, double step) = 0;

    virtual void train() = 0;

    const arma::Col<double> &get_h() const
    {
        return core.h;
    }
    const arma::Col<double> &get_J() const
    {
        return core.J;
    }

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
        return m3_model;
    }
    const double get_avg_energy() const
    {
        return avg_energy;
    }
    const double get_avg_energy_sq() const
    {
        return avg_energy_sq;
    }
    const double get_avg_magnetization() const
    {
        return avg_magnetization;
    }
    const size_t get_iter() const
    {
        return iter;
    }

    const std::string get_type() const
    {
        return className;
    }

  protected:
    MaxEntCore &core;
    int ntriplets;

    // Training parameters
    size_t maxIterations;
    size_t iter;

    double tolerance_h;
    double tolerance_J;
    double q_val;   // for exp_q(x,q) Tsallis q-exponential
    double eta_h;   // training rate for h
    double eta_J;   // training rate for J
    double alpha_h; // training momentum alpha_h * delta_h
    double alpha_J; // training momentum alpha_J * delta_J
    double gamma_h; // training momentum alpha_h * delta_h
    double gamma_J; // training momentum alpha_J * delta_J

    arma::Col<double> delta_h; // training momentum alpha_h * delta_h(i)
    arma::Col<double> delta_J; // training momentum alpha_J * delta_J(idx)

    // sample and model averages
    double avg_energy;
    double avg_energy_sq;
    double avg_magnetization;

    // averages used for training
    arma::Col<double> m1_data;  // sample fist momentum: <s_i>
    arma::Col<double> m2_data;  // sample second momentum: <s_i*s_j>
    arma::Col<double> m1_model; // model's fist momentum: <s_i>
    arma::Col<double> m2_model; // model's second momentum: <s_i*s_j>

    // averages used to compare predictions
    arma::Col<double> m3_data;  // sample third momentum: <s_i*s_j*s_j>
    arma::Col<double> m3_model; // model's third momentum: <s_i*s_j*s_j>

    // Private helper functions

    void updateModelParameters(size_t t);
    double energyAllPairs(arma::Col<int> s);

  private:
    std::string className = "BasicTrainer";
};
