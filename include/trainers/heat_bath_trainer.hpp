#pragma once

#include "base_trainer.hpp"

class HeatBathTrainer : public BaseTrainer
{
  public:
    HeatBathTrainer(MaxEntCore &core,
                    double q_val,
                    size_t max_iterations,
                    double tolerance_h,
                    double tolerance_J,
                    double eta_h,
                    double eta_J,
                    double alpha_h,
                    double alpha_J,
                    double gamma_h,
                    double gamma_J,
                    const std::string &data_filename) :
        BaseTrainer(core,
                    q_val,
                    max_iterations,
                    tolerance_h,
                    tolerance_J,
                    eta_h,
                    eta_J,
                    alpha_h,
                    alpha_J,
                    gamma_h,
                    gamma_J,
                    data_filename) {};

    void configureMonteCarlo(size_t step_equilibration_,
                             size_t num_samples_,
                             size_t step_correlation_,
                             int number_repetitions_)
    {
        step_equilibration   = step_equilibration_;
        step_correlation     = step_correlation_;
        num_samples          = num_samples_;
        number_repetitions   = number_repetitions_;
        total_number_samples = num_samples * number_repetitions;

        int nspins = core.nspins;
        replicas.set_size(total_number_samples, nspins);
        replicas.fill(-1);
    }

    void computeModelAverages(double beta, bool triplets) override;
    void computeModelAverages1(double beta, bool triplets);

    void train() override;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }

  private:
    std::string className = "FullEnsembleTrainer";
    int mc_seed           = 1;

    size_t step_equilibration;   // Number of equilibration sweeps
    size_t step_correlation;     // Number of sweeps between samples
    size_t num_samples;          // Number of samples to collect
    size_t number_repetitions;   // Number of repetitions for averaging
    size_t total_number_samples; // Total number of samples
    arma::Mat<int> replicas;
};