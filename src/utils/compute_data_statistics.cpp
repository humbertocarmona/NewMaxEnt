#include "utils/compute_data_statistics.hpp"
#include <armadillo>
#include <cctype>
#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h> // Add this for stdout_color_mt
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

arma::Mat<int> read_raw_data(const std::string &filename)
{
    auto logger = spdlog::stdout_color_mt("core_logger");

    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open " + filename);
    }
    logger->debug("[read_raw_data] Reading raw data from {}", filename);

    std::string line;
    std::vector<std::vector<int>> tempMatrix;
    size_t n_cols       = 0;
    int n_rows          = 0;
    bool header_checked = false;

    while (std::getline(file, line))
    {
        // Skip comment lines
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        // Skip header line if it starts with an alphabet character
        if (!header_checked)
        {
            std::stringstream ss(line);
            std::string first_val;
            std::getline(ss, first_val, ',');
            if (!first_val.empty() && std::isalpha(static_cast<unsigned char>(first_val[0])))
            {
                header_checked = true;
                continue;
            }
            header_checked = true; // Header checked even if not skipped
        }

        n_rows++;
        std::stringstream ss(line);
        std::string value;
        std::vector<int> row;

        while (std::getline(ss, value, ','))
        {
            row.push_back(std::stoi(value));
        }

        if (tempMatrix.empty())
        {
            n_cols = row.size();
        }

        if (row.size() != n_cols)
        {
            throw std::runtime_error("Inconsistent number of columns in the CSV file.");
        }

        tempMatrix.push_back(row);
    }

    arma::Mat<int> samples(n_rows, n_cols);

    for (size_t i = 0; i < tempMatrix.size(); ++i)
    {
        for (size_t j = 0; j < tempMatrix[i].size(); ++j)
        {
            samples(i, j) = tempMatrix[i][j];
        }
    }

    return samples;
}

DataStatisticsBreakdown compute_data_statistics(const std::string &filename)
{
    auto logger = spdlog::stdout_color_mt("core_logger");

    arma::Mat<int> raw_data = read_raw_data(filename);

    arma::Mat<double> data_dbl = arma::conv_to<arma::Mat<double>>::from(raw_data);
    int nspins                 = data_dbl.n_cols;
    int nsamples               = data_dbl.n_rows;

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
                res.m3_data(idx++) = arma::mean(data_dbl.col(i) % data_dbl.col(j) % data_dbl.col(k));
            }
        }
    }

    logger->debug("[compute_data_statistics] Computed moments 1 (size {}), 2 (size {}), 3 (size {})",
                 res.m1_data.n_elem,
                 res.m2_data.n_elem,
                 res.m3_data.n_elem);

    return res;
}
