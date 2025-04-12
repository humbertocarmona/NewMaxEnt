#pragma once
#include <armadillo>

inline double energy(const arma::Col<int> &s, const arma::Col<double> &h, const arma::Col<double> &J, const int &n)
{
    double e = 0.0;
    for (int i = 0; i < n; ++i)
        e += h(i) * s(i);
    int idx = 0;
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            e += J(idx++) * s(i) * s(j);
    return -e;
}
