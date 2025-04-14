#pragma once
#include <string>
#include <armadillo>

struct ModelMinimal
{
    arma::Col<double> h;
    arma::Col<double> J;
    int nspins;
    double q_val;
};

ModelMinimal read_model_minimal_from_json(const std::string& filename);
