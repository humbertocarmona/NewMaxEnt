#pragma once

#include "base_trainer.hpp"

class HeatBathTrainer : public BaseTrainer
{
  public:
    HeatBathTrainer(MaxEntCore &core,
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
                    const std::string &data_filename) :
        BaseTrainer(core,
                    q_val,
                    maxIterations,
                    tolerance_h,
                    tolerance_J,
                    eta_h,
                    eta_J,
                    alpha_h,
                    alpha_J,
                    gamma_h,
                    gamma_J,
                    data_filename) {};

    void configureMonteCarlo(size_t equilibrationSweeps, size_t samples, size_t interval)
    {
        numEquilibrationSweeps = equilibrationSweeps;
        numSamples             = samples;
        sampleInterval         = interval;
    }

    void computeModelAverages(double beta, bool triplets) override;
    void train() override;

    // void computeTemperatureDependence(double betaStart, double betaEnd, double step) override {
    // Implement temperature dependence logic here
    // }
  private:
    std::string className = "FullEnsembleTrainer";

    size_t numEquilibrationSweeps; // Number of equilibration sweeps
    size_t numSamples;             // Number of samples to collect
    size_t sampleInterval;         // Number of sweeps between samples
};