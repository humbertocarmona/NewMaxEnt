#pragma once

#include "base_trainer.hpp"
#include <armadillo>
#include <random>

class WangLandauTrainer : public BaseTrainer
{
  public:
    WangLandauTrainer(MaxEntCore &core,
                        RunParameters &params,
                      const std::string &data_filename) :
        BaseTrainer(core,params,
                    data_filename) {
                        configureWangLandau();
                    };

    void configureWangLandau()
    {
        if (params.num_samples == 0 || params.step_correlation == 0)
            throw std::invalid_argument(
                "num_samples and step_correlation must be greater than zero.");

        total_number_samples = params.num_samples * params.number_repetitions;
        
        int nspins = core.nspins;
        replicas.set_size(total_number_samples, nspins);
        replicas.fill(-1);


    }

    void computeModelAverages(double beta=1.0, bool triplets=false) override;
    void train() override;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }

  private:
    std::string className = "WangLandauTrainer";
    int wg_seed           = 1;

    size_t total_number_samples; // Total number of samples
    arma::Mat<int> replicas;

    std::unordered_map<int, double> log_g_E; // ln(G(E) density of states
    std::unordered_map<int, int> H;          // energy histogram

    void flip_random_spin(arma::Col<int> &s, std::mt19937 &rng);

    bool is_flat(const std::unordered_map<int, int> &H, 
                 double flatness_threshold = 0.8);

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