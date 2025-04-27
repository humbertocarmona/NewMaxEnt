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

    void configureWangLandau(double log_f_final_,
                             double energy_bin_,
                             double flatness_threshold_,
                             size_t equilibration_weeps,
                             size_t num_samples,
                             size_t sample_interval,
                             size_t n_repetitions)
    {
        if (num_samples == 0 || sample_interval == 0)
            throw std::invalid_argument(
                "num_samples and step_correlation must be greater than zero.");

        step_equilibration   = equilibration_weeps;
        step_correlation     = sample_interval;
        num_samples          = num_samples;
        number_repetitions   = n_repetitions;
        total_number_samples = num_samples * number_repetitions;
        
        int nspins = core.nspins;
        replicas.set_size(total_number_samples, nspins);
        replicas.fill(-1);

        log_f_final        = log_f_final_;
        energy_bin         = energy_bin_;
        flatness_threshold = flatness_threshold_;

    }

    void computeModelAverages(double beta, bool triplets) override;
    void train() override;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }

  private:
    std::string className = "WangLandauTrainer";
    int wg_seed           = 1;

    size_t step_equilibration;
    size_t step_correlation;     // Number of sweeps between samples
    size_t num_samples;          // Number of samples to collect
    size_t number_repetitions;   // Number of repetitions for averaging
    size_t total_number_samples; // Total number of samples
    arma::Mat<int> replicas;

    double log_f_final;
    double energy_bin;
    double flatness_threshold;

    std::unordered_map<int, double> log_g_E; // ln(G(E) density of states
    std::unordered_map<int, int> H;          // energy histogram

    void flip_random_spin(arma::Col<int> &s, std::mt19937 &rng);

    bool is_flat(const std::unordered_map<int, int> &H);

    void computeDensityOfStates();

    inline double logsumexp(const std::vector<double> &vector_of_logs)
    {
        double max_log = *std::max_element(vector_of_logs.begin(), vector_of_logs.end());
        double sum     = 0.0;
        for (double x : vector_of_logs)
            sum += std::exp(x - max_log);
        return max_log + std::log(sum);
    }
};