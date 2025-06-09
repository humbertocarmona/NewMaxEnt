#pragma once
#include "utils/get_logger.hpp"
#include <armadillo>

struct CostBreakdown
{
    double cost_total;
    double cost_m1;
    double cost_m2;

    // Convergence check using separate tolerances for moment_1 and moment_2
    bool check_convergence(double tol_1, double tol_2) const
    {
        return cost_m1 < tol_1 && cost_m2 < tol_2;
    }
};

inline CostBreakdown compute_cost(const arma::Col<double> &m1_data,
                                  const arma::Col<double> &m1_model,
                                  const arma::Col<double> &m2_data,
                                  const arma::Col<double> &m2_model)
{

    double cost1 = arma::accu(arma::square(m1_model - m1_data));
    double cost2 = arma::accu(arma::square(m2_model - m2_data));
    cost1 /= arma::accu(arma::square(m1_data));
    cost2 /= arma::accu(arma::square(m2_data));

    return {cost1 + cost2, cost1, cost2};
}