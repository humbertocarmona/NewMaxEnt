#pragma once
#include "core/parameters.hpp" // Ensure this provides the definition of `Params`
#include <iomanip>
#include <sstream>
#include <string>

namespace io
{

/**
 * @brief Construct the filename for the main thermodynamic sweep CSV output.
 */
inline std::string make_tdep_filename(const Params &params)
{
    std::ostringstream filename;
    // clang-format off
    filename << params.result_dir << "/"
             << "thermo_sweep_" << params.runid 
             << "_q_" << std::fixed << std::setprecision(2) << params.q_val << ".csv";
    // clang-format on
    return filename.str();
}

/**
 * @brief Construct the filename for top-K replicas at a given temperature.
 */
inline std::string make_top_k_filename(const Params &params, double T)
{
    std::ostringstream fname;
    // clang-format off
    fname << params.result_dir << "/top_replicas_" << params.runid 
          << "T_" << std::fixed << std::setprecision(2) << T
          << "_q_" << std::fixed << std::setprecision(2) << params.q_val 
          << ".csv";
    return fname.str();
    // clang-format on
}

/**
 * @brief Construct the filename for full enumeration run.
 */
inline std::string make_full_enumeration_filename(const Params &params)
{
    std::ostringstream fname;
    // clang-format off
    fname << params.result_dir << "/"
          << "ens_" << params.runid 
          << "_nspins_" << params.gen_nspins 
          << "_q_"  << std::fixed << std::setprecision(2) << params.q_val 
          << "_id_" << params.id 
          << "_final.json";
    return fname.str();
    // clang-format on
}

} // namespace io
