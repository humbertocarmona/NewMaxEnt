#pragma once

#include <armadillo>

struct CostBreakdown
{
    double total;
    double moment_1;
    double moment_2;

    // Convergence check using separate tolerances for moment_1 and moment_2
    bool check_convergence(double tol_1, double tol_2) const
    {
        return moment_1 < tol_1 && moment_2 < tol_2;
    }
};

CostBreakdown compute_cost(const arma::Col<double> &moment_1_data, const arma::Col<double> &moment_1_model,
                          const arma::Col<double> &moment_2_data, const arma::Col<double> &moment_2_model);
