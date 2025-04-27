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

    void configureMonteCarlo(size_t equilibration_weeps, size_t num_samples, size_t sample_interval)
    {
        int nspins          = core.nspins;
        equilibrationSweeps = equilibration_weeps;
        numSamples          = num_samples;
        sampleInterval      = sample_interval;

        replicas.set_size(numSamples, nspins);
        replicas.fill(-1);
    }

    void computeModelAverages(double beta, bool triplets) override;
    void train() override;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }

  private:
    std::string className = "FullEnsembleTrainer";
    int mc_seed           = 1;

    size_t equilibrationSweeps; // Number of equilibration sweeps
    size_t numSamples;          // Number of samples to collect
    size_t sampleInterval;      // Number of sweeps between samples
    size_t numRepetitions;      // Number of repetitions for averaging
    arma::Mat<int> replicas;
};