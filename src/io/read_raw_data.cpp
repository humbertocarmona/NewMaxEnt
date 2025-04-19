#include "utils/get_logger.hpp"
#include <armadillo>


arma::Mat<int> readRawData(const std::string &filename)
{
    auto logger = getLogger();

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

    // just convert it to arma::Mat
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