#pragma once

#include <armadillo>

struct CostBreakdown
{
    double total;
    double moment_1;
    double moment_2;
};

CostBreakdown computeCost(const arma::Col<double> &moment_1_data, const arma::Col<double> &moment_1_model,
                          const arma::Col<double> &moment_2_data, const arma::Col<double> &moment_2_model);
