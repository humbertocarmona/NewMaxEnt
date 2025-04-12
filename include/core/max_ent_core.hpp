#pragma once

#include "core/parameters.hpp"
#include <armadillo>
#include <memory>
#include <spdlog/spdlog.h>

class MaxEntCore
{
  public:
    MaxEntCore(const Params &params, bool verbose = false);

    void initialize_fields();
    void run_full_enumeration();

    void set_samples(const arma::Mat<int> &samples);
    void set_h(double mean, double width);
    void set_J(double mean, double width);
    
    const arma::Col<double> &get_h() const;
    const arma::Col<double> &get_J() const;
    const arma::Mat<int> &get_raw_samples() const;

    std::shared_ptr<spdlog::logger> get_logger() const
    {
        return LOGGER;
    }

    const Params &get_params() const
    {
        return run_parameters;
    }

    void compute_sample_statistics();

    const arma::Col<double> &get_sample_moment_1() const
    {
        return sample_moment_1;
    }
    const arma::Col<double> &get_sample_moment_2() const
    {
        return sample_moment_2;
    }
    const arma::Col<double> &get_sample_moment_3() const
    {
        return sample_moment_3;
    }

    const arma::Col<double> &get_model_moment_1() const
    {
        return model_moment_1;
    }
    const arma::Col<double> &get_model_moment_2() const
    {
        return model_moment_2;
    }
    const arma::Col<double> &get_model_moment_3() const
    {
        return model_moment_3;
    }

  private:
    std::shared_ptr<spdlog::logger> LOGGER;
    bool verbose;

    Params run_parameters;
    int n_spins;
    int n_edges;
    int iter;

    arma::Col<double> h;
    arma::Col<double> J;
    arma::Mat<int> raw_samples;

    arma::Mat<int> edge_index;

    arma::Col<double> sample_moment_1;
    arma::Col<double> sample_moment_2;
    arma::Col<double> sample_moment_3;

    arma::Col<double> model_moment_1;
    arma::Col<double> model_moment_2;
    arma::Col<double> model_moment_3;

    arma::Col<double> momentum_m_1;
    arma::Col<double> momentum_m_2;

    void initialize_network();
    void initialize_random_fields();
    void initialize_couplings();
    void update_model_parameters();
};
