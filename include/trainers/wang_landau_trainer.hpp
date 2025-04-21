#pragma once

#include "base_trainer.hpp"
#include <armadillo>
#include <random>

class WangLandauTrainer : public BaseTrainer
{
  public:
    WangLandauTrainer(MaxEntCore &core,
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

    void computeModelAverages(double beta, bool triplets) override;
    void train() override;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }



  private:
    std::string className = "WangLandauTrainer";
    arma::Mat<int> replicas;
    size_t max_trials = 100000;
    double log_f_final = 1e-5;
    double energy_bin = 0.2;

    std::unordered_map<int, double> log_g;
    std::unordered_map<int, int> H;

    void flip_random_spin(arma::Col<int> &s, std::mt19937 &rng);

    bool is_flat(const std::unordered_map<int, int> &H, double flatness_threshold);

    void densityOfStates();

};