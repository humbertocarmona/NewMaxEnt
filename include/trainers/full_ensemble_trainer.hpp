#pragma once

#include "base_trainer.hpp"

class FullEnsembleTrainer : public BaseTrainer {
public:
    FullEnsembleTrainer(MaxEntCore &core,
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
        const std::string &data_filename)
        : BaseTrainer(core, q_val, maxIterations, tolerance_h, tolerance_J,
                      eta_h, eta_J, alpha_h, alpha_J, gamma_h, gamma_J,data_filename) {};


    void computeModelAverages(double beta, bool triplets ) override;
    void train() override;

    // void computeTemperatureDependence(double betaStart, double betaEnd, double step) override {
        // Implement temperature dependence logic here
    // }
    private:
    std::string className = "FullEnsembleTrainer";
};
