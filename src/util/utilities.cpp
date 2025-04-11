#include "util/utilities.hpp"
#include "util/logger.hpp"
#include <cassert>
#include <regex>
#include <sstream>

namespace utils
{
namespace fs = std::filesystem;

double rms(const arma::Col<double> &vec1, const arma::Col<double> &vec2)
{
    // Ensure the vectors are the same size
    if (vec1.n_elem != vec2.n_elem)
    {
        throw std::invalid_argument("Vectors must be the same length");
    }

    // Calculate the squared differences
    arma::Col<double> diff = vec1 - vec2;
    double rms             = std::sqrt(arma::accu(arma::square(diff)) / diff.n_elem);

    return rms;
}

/**
 * @brief Centers the given vector `xy` based on the values from vector `x`.
 *
 * The function computes a new column vector `xy_centered` by adjusting
 * the values in `xy` based on the pairwise products of elements in `x`.
 * The size of `xy` must be n = s(s-1)/2, where s is the number of spins
 * and is computed from n using the discriminant formula. This function
 * asserts that the discriminant is an integer and odd.
 *
 * @param x A row vector of size `nspins` containing the spin values.
 * @param xy A column vector of size `nspins(nspins-1)/2` containing
 *           pairwise interaction values between the spins.
 * @return arma::Col<double> A column vector `xy_centered` of the same size
 *         as `xy`, where each value is centered using the corresponding
 *         pairwise product of elements in `x`.
 * @note Uses `assert` to check that the discriminant is an integer and odd.
 */
arma::Col<double> center_xy(const arma::Col<double> &x, const arma::Col<double> &xy)
{
    auto logger = get_logger();

    // Compute the discriminant for the quadratic equation to get nspins
    double discriminant = std::sqrt(1 + 8 * xy.n_elem);

    // Assert that the discriminant is an integer
    assert(std::floor(discriminant) == discriminant && "Discriminant is not an integer");

    // Assert that the discriminant is odd
    assert(static_cast<int>(discriminant) % 2 == 1 && "Discriminant is not an odd integer");

    // Calculate number of spins
    int nspins = static_cast<int>((1 + discriminant) / 2);

    // Initialize centered vector with zeros
    arma::Col<double> xy_centered(xy.n_elem, arma::fill::zeros);

    // Main loop for centering
    int idx = 0;
    for (int i = 0; i < nspins - 1; ++i)
    {
        for (int j = i + 1; j < nspins; ++j)
        {
            xy_centered(idx) = xy(idx) - x(i) * x(j);
            ++idx; // Increment the index
        }
    }

    return xy_centered; // Return by value (move semantics used automatically)
}

/**
 * @brief Centers the given vector `xyz` based on the values from vectors `x` and `xy`.
 *
 * This function computes a new column vector `xyz_centered` by adjusting
 * the values in `xyz` based on pairwise products of elements in `x` and `xy`.
 * The size of `xy` must be n = s(s-1)/2, and the size of `xyz` must be
 * n_triplets = s(s-1)(s-2)/6, where s is the number of spins.
 *
 * @param x A row vector of size `nspins` containing the spin values.
 * @param xy A column vector of size `nspins(nspins-1)/2` containing pairwise interaction values.
 * @param xyz A column vector of size `nspins(nspins-1)(nspins-2)/6` containing triplet interaction values.
 * @return arma::Col<double> A column vector `xyz_centered` of the same size as `xyz`, centered using `x` and `xy`.
 */
arma::Col<double> center_xyz(const arma::Col<double> &x, const arma::Col<double> &xy, const arma::Col<double> &xyz)
{
    std::ostringstream oss;
    // Compute the discriminant for the quadratic equation to get nspins
    double discriminant = std::sqrt(1 + 8 * xy.n_elem);

    // Assert that the discriminant is an integer
    assert(std::floor(discriminant) == discriminant && "Discriminant is not an integer");

    // Assert that the discriminant is odd
    assert(static_cast<int>(discriminant) % 2 == 1 && "Discriminant is not an odd integer");

    // Calculate number of spins
    int nspins = static_cast<int>((1 + discriminant) / 2);

    // Corrected calculation for the number of triplets
    int ntriplets = static_cast<int>(nspins * (nspins - 1) * (nspins - 2) / 6);

    assert(xyz.n_elem == ntriplets && "Number of triplets");

    // Initialize centered vector with zeros
    arma::Col<double> xyz_centered(xyz.n_elem, arma::fill::zeros);

    // Initialize xy matrix (symmetric)
    arma::Mat<double> xy_mat(nspins, nspins, arma::fill::zeros);
    int idx = 0;
    for (int i = 0; i < nspins - 1; ++i)
    {
        for (int j = i + 1; j < nspins; ++j)
        {
            xy_mat(i, j) = xy(idx);
            xy_mat(j, i) = xy(idx);
            ++idx; // Increment the index
        }
    }

    // Main loop for centering the xyz vector
    idx = 0;
    for (int i = 0; i < nspins - 2; ++i)
    {
        for (int j = i + 1; j < nspins - 1; ++j)
        {
            for (int k = j + 1; k < nspins; ++k)
            {
                xyz_centered(idx) =
                    xyz(idx) - x(i) * xy_mat(j, k) - x(j) * xy_mat(i, k) - x(k) * xy_mat(i, j) + 2 * x(i) * x(j) * x(k);
                ++idx; // Increment the index
            }
        }
    }

    return xyz_centered; // Return by value (move semantics used automatically)
}

std::string today()
{
    // Get the current time
    std::time_t t = std::time(nullptr);
    // Convert to local time
    std::tm tm = *std::localtime(&t);
    // Create a stringstream to format the date
    std::ostringstream oss;
    // oss << std::put_time(&tm, "%Y%m%d-%H%M");
    oss << std::put_time(&tm, "%Y%m%d");
    return oss.str();
}

std::string now()
{
    // Get the current time
    std::time_t t = std::time(nullptr);
    // Convert to local time
    std::tm tm = *std::localtime(&t);
    // Create a stringstream to format the date
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M");
    return oss.str();
}

std::string now_30()
{
    // Get the current time
    std::time_t t = std::time(nullptr);
    // Convert to local time
    std::tm tm = *std::localtime(&t);
    if (tm.tm_min < 20)
    {
        tm.tm_min = 0;
    }
    else if (tm.tm_min < 40)
    {
        tm.tm_min = 30;
    }
    else
    {
        tm.tm_min = 0;
        tm.tm_hour += 1;
        if (tm.tm_hour == 24)
        {
            tm.tm_hour = 0;
            tm.tm_mday += 1;
        }
    }
    // Create a stringstream to format the date
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M");
    return oss.str();
}

bool is_file(const std::filesystem::path &path)
{
    return fs::exists(path);
}

bool is_dir(const std::filesystem::path &path)
{
    return fs::exists(path);
}

void make_path(const std::filesystem::path &path)
{
    if (!fs::exists(path))
    {
        if (fs::create_directories(path))
        {
            std::cout << "Directory created: " << path << '\n';
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << '\n';
        }
    }
}

int next_run_id(const std::string &directory, const std::regex &pattern)
{
    std::vector<int> run_ids;

    // Iterate over files in the directory
    for (const auto &entry : fs::directory_iterator(directory))
    {
        const std::string filename = entry.path().filename().string();
        std::smatch match;

        // Check if the filename matches the pattern
        if (std::regex_match(filename, match, pattern))
        {
            // Extract the run ID and convert it to an integer
            int run_id = std::stoi(match[1]);
            run_ids.push_back(run_id);
        }
    }

    // Determine the next available run ID
    return run_ids.empty() ? 1 : *std::max_element(run_ids.begin(), run_ids.end()) + 1;
};

template <typename T> std::string col_string(const arma::Col<T> &col)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < col.n_elem; ++i)
    {
        oss << col(i);
        if (i < col.n_elem - 1)
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

// Explicit template instantiation (if needed)
template std::string col_string(const arma::Col<double> &);
template std::string col_string(const arma::Col<int> &);
} // namespace utils
