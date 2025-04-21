#pragma once
#include "core/run_parameters.hpp"
#include "utils/utilities.hpp" // utils::get_available_filename

#include <iomanip>
#include <sstream>
#include <string>

namespace io
{

/**
 * @brief Construct the filename for the main thermodynamic sweep CSV output.
 */
inline std::string make_tdep_filename(const RunParameters &params)
{
    std::ostringstream fname;
    fname << params.result_dir << "/"
          << "sweep-" << params.run_type << "-" << params.runid << ".csv";
    std::filesystem::path output = utils::get_available_filename(fname.str());
    return output.string();
}

/**
 * @brief Construct the filename for top-K replicas at a given temperature.
 */
inline std::string make_replicas_filename(const RunParameters &params, double T)
{
    std::ostringstream fname;
    fname << params.result_dir << "/"
          << "replicas-" << params.run_type << "-" << params.runid << "-T-" << std::fixed
          << std::setprecision(2) << T << ".csv";

    std::filesystem::path output = utils::get_available_filename(fname.str());
    return output.string();
}

/**
 * @brief Construct the filename for full enumeration run.
 */
inline std::string make_trained_filename(const RunParameters &params)
{
    std::ostringstream fname;
    fname << params.result_dir << "/final-" << params.run_type << "-" << params.runid << ".json";
    std::filesystem::path output = utils::get_available_filename(fname.str());

    return output.string();
}

inline std::string make_DensOfStates_filename(const RunParameters &params)
{
    std::ostringstream fname;
    fname << params.result_dir << "/log_g_E-" << params.run_type << "-" << params.runid << ".csv";
    std::filesystem::path output = utils::get_available_filename(fname.str());

    return output.string();
}

} // namespace io
