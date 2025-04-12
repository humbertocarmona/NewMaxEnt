#pragma once

#include "core/parameters.hpp"
#include <armadillo>
#include <memory>
#include <vector>

class MaxEntCore
{
  public:
    MaxEntCore(const Params &params, bool verbose = false);

    void initialize_from_params();
    void relax(double q = 1.0);

    void set_samples(const arma::Mat<int> &samples);
    void set_h(double mean, double width);
    void set_J(double mean, double width);

    const arma::Col<double> &get_h() const;
    const arma::Col<double> &get_J() const;
    const arma::Mat<int> &get_samples() const;

    // Add other needed accessors...
    std::shared_ptr<spdlog::logger> get_logger() const
    {
        return LOGGER;
    }

    const Params &get_params() const
    {
        return run_parameters;
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

    void initialize_network();
    void initialize_random_fields();
    void initialize_couplings();
};
