#pragma once
#include <armadillo>
void compute_model_statistics(const arma::Col<double> &h, const arma::Col<double> &J, arma::Col<double> &moment_1,
                              arma::Col<double> &moment_2, arma::Col<double> &moment_3, double q, double beta = 1.0);
