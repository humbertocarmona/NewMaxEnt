#include "core/cost_function.hpp"

CostBreakdown compute_cost(const arma::Col<double> &moment_1_data, const arma::Col<double> &moment_1_model,
                          const arma::Col<double> &moment_2_data, const arma::Col<double> &moment_2_model)
{
    double cost1 = arma::accu(arma::square(moment_1_model - moment_1_data));
    double cost2 = arma::accu(arma::square(moment_2_model - moment_2_data));
    return {cost1 + cost2, cost1, cost2};
}
