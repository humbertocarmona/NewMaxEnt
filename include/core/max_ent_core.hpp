#pragma once
#include "utils/get_logger.hpp"
#include <armadillo>
#include <string>
class MaxEntCore
{
  public:
    std::string runid;
    int nspins;
    int nedges;

    arma::Col<double> h;
    arma::Col<double> J;
    arma::Mat<int> edges;

    MaxEntCore(size_t n, const std::string &runid_) : nspins(n), runid(runid_)
    {
        nedges = nspins * (nspins - 1) / 2;
        h.set_size(nspins);
        h.fill(0);
        J.set_size(nedges);
        J.fill(0);
        edges.set_size(nspins, nspins);
        edges.fill(-1);
        int idx = 0;
        for (int i = 0; i < nspins - 1; ++i)
        {
            for (int j = i + 1; j < nspins; ++j)
            {
                edges(i, j) = edges(j, i) = idx++;
            }
        }
    };
};