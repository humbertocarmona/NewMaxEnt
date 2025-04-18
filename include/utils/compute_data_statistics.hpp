#pragma once
#include <armadillo>

#include <string>

struct DataStatisticsBreakdown
{

    arma::Col<double> m1_data; // sample fist momentum: <s_i>
    arma::Col<double> m2_data; // sample second momentum: <s_i*s_j>;
    arma::Col<double> m3_data; // sample third momentum: <s_i*s_j*s_j>

    DataStatisticsBreakdown(size_t n)
        : m1_data(n, arma::fill::zeros), m2_data(n * (n - 1) / 2, arma::fill::zeros),
          m3_data(n * (n - 1) * (n - 2) / 6, arma::fill::zeros)
    {
    }
};

arma::Mat<int> read_raw_data(const std::string &filename);

DataStatisticsBreakdown compute_data_statistics(const std::string &filename);
