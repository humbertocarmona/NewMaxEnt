#pragma once

#include "base_trainer.hpp"
#include "core/run_parameters.hpp"
#include "io/make_file_names.hpp"
#include "io/write_json.hpp"
#include "utils/centered_moments.hpp"

class FullEnsembleTrainer : public BaseTrainer
{
  public:
    FullEnsembleTrainer(MaxEntCore &core, RunParameters &params, const std::string &data_filename) :
        BaseTrainer(core, params, data_filename) {};

    void computeModelAverages(double beta = 1.0, bool triplets = false) override;
    void computeModelAverages1(double beta = 1.0, bool triplets = false);
    void train() override;
    void saveModel(std::string prefix) const;
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
    std::unordered_map<int, double> PE; // energy histogram
    std::unordered_map<int, double> GE; // energy histogram
};
