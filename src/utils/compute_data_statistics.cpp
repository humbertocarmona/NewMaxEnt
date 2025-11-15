#include "utils/compute_data_statistics.hpp"
#include "io/read_raw_data.hpp"
#include "utils/get_logger.hpp"
#include <armadillo>
#include <cctype>
#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h> // Add this for stdout_color_mt
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

DataStatisticsBreakdown compute_data_statistics(const std::string &filename)
{

    auto logger = getLogger();

    arma::Mat<int> raw_data = readRawData(filename);

    arma::Mat<double> data_dbl = arma::conv_to<arma::Mat<double>>::from(raw_data);
    int nspins                 = data_dbl.n_cols;
    int nsamples               = data_dbl.n_rows;

    // initialize the DataStatisticsBreakdown structure column vectors
    DataStatisticsBreakdown res(nspins);

    // First moment: average magnetization of each spin
    res.m1_data = arma::mean(data_dbl, 0).t(); // column vector

    // Second moment: correlations between pairs
    ;
    int idx = 0;
    for (int i = 0; i < nspins - 1; ++i)
    {
        for (int j = i + 1; j < nspins; ++j)
        {
            res.m2_data(idx++) = arma::mean(data_dbl.col(i) % data_dbl.col(j));
        }
    }

    // Third moment: triplet correlations

    idx = 0;
    for (int i = 0; i < nspins - 2; ++i)
    {
        for (int j = i + 1; j < nspins - 1; ++j)
        {
            for (int k = j + 1; k < nspins; ++k)
            {
                res.m3_data(idx++) =
                    arma::mean(data_dbl.col(i) % data_dbl.col(j) % data_dbl.col(k));
            }
        }
    }

    res.pK_data.fill(0);
    for (int n = 0; n < nsamples; ++n)
    {
        auto s = raw_data.row(n);
        int k  = static_cast<int>(arma::sum(s + 1) / 2);
        res.pK_data(k) += 1.0;
    }
    res.pK_data /= nsamples;

    logger->debug(
        "[compute_data_statistics] Computed moments 1 (size {}), 2 (size {}), 3 (size {})",
        res.m1_data.n_elem, res.m2_data.n_elem, res.m3_data.n_elem);

    return res;
}
