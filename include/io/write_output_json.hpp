#pragma once

#include "core/centered_moments.hpp"
#include "core/parameters.hpp"
#include <armadillo>
#include <nlohmann/json.hpp>

void write_output_json(const std::string &filename,
                       const Params &run_parameters,
                       const arma::Col<double> &h,
                       const arma::Col<double> &J,
                       const arma::Col<double> &sample_moment_1,
                       const arma::Col<double> &model_moment_1,
                       const arma::Col<double> &sample_moment_2,
                       const arma::Col<double> &model_moment_2,
                       const arma::Col<double> &sample_moment_3,
                       const arma::Col<double> &model_moment_3,
                       const arma::Col<double> &centered_sample_moment_2,
                       const arma::Col<double> &centered_model_moment_2,
                       const arma::Col<double> &centered_sample_moment_3,
                       const arma::Col<double> &centered_model_moment_3,
                       double energy_mean,
                       double energy_fluctuation,
                       int final_iter);
