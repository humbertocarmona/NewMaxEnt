#pragma once
#include <armadillo>
void compute_model_statistics(const int &n_spins, const arma::Col<double> &h, const arma::Col<double> &J,
                              arma::Col<double> &model_moment_1, arma::Col<double> &mode_moment_2,
                              arma::Col<double> &model_moment_3, double q, double beta);
