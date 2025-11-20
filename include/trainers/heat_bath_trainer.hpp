#pragma once

#include "base_trainer.hpp"
#include "core/run_parameters.hpp"
#include "io/make_file_names.hpp"
#include "io/write_json.hpp"
#include "utils/centered_moments.hpp"

class HeatBathTrainer : public BaseTrainer
{
  public:
    HeatBathTrainer(MaxEntCore &core, RunParameters &params, const std::string &data_filename) :
        BaseTrainer(core, params, data_filename)
    {
        configureMonteCarlo();
    };

    void configureMonteCarlo()
    {
        total_number_samples = params.num_samples * params.number_repetitions;

        int nspins = core.nspins;
        replicas.set_size(total_number_samples, nspins);
        replicas.fill(-1);
    }

    void computeModelAverages(double beta = 1.0, bool triplets = false) override;
    void computeModelAverages1(double beta = 1.0, bool triplets = false);

    void train() override;

    void saveModel(std::string filename) const;

    const arma::Mat<int> &get_replicas() const
    {
        return replicas;
    }
    const std::unordered_map<int, double> &get_GE() const
    {
        return GE;
    }

        const std::unordered_map<int, double> &get_PE() const
    {
        return PE;
    }
  private:
    std::string className = "FullEnsembleTrainer";
    int mc_seed           = 1;

    size_t total_number_samples; // Total number of samples
    arma::Mat<int> replicas;

    std::unordered_map<int, double> PE; // energy histogram
    std::unordered_map<int, double> GE; // energy histogram
};