#pragma once
#include "core/run_parameters.hpp"
#include "utils/utilities.hpp" // utils::get_available_filename

#include <iomanip>
#include <sstream>
#include <string>

namespace io
{
/**
 * @brief Construct the filename for saving trained model.
 */
inline std::string make_filename(const RunParameters &params, std::string prefix)
{
    std::ostringstream outdir;

    outdir << params.result_dir << "/n" << params.nspins;
    // example: result_dir = "./Results/n20"
    //
    if (params.run_type == "Gen_Full")
    {
        outdir << "/synth";
        // example: result_dir = "./Results/n20/synth"
        // outdir << "/qobs_" << std::fixed << std::setprecision(2) << params.q_val;
    }
    else if (params.k_pairwise)
    {
        outdir << "/k_pairwise";
        // example: result_dir = "./Results/n20/k_pairwise"
    }
    else
    {
        outdir << "/pairwise";
        // example: result_dir = "./Results/n20/pairwise"
    }

    // outdir << "/" << params.run_type;

    utils::make_path(outdir.str());

    std::ostringstream fname;

    fname << outdir.str() << "/" << prefix << "_" << params.runid << ".json";
    // example: result_dir = "./Results/n20/pairwise/final_$(runid).json"

    std::filesystem::path output = utils::get_available_filename(fname.str());

    return output.string();
}

/**
 * @brief Construct the filename for top-K replicas at a given temperature.
 */
inline std::string make_replicas_filename(const RunParameters &params, double T)
{
    std::ostringstream outdir;
    outdir << params.result_dir << "/" << params.run_type;
    utils::make_path(outdir.str());

    std::ostringstream fname;
    fname << outdir.str() << "/replicas-" << params.runid << "-T-" << std::fixed
          << std::setprecision(2) << T << ".csv";

    std::filesystem::path output = utils::get_available_filename(fname.str());
    return output.string();
}

/**
 * @brief Construct the filename for top-K replicas at a given temperature.
 */
inline std::string make_replica_correlation_filename(const RunParameters &params, double T)
{
    std::ostringstream outdir;
    outdir << params.result_dir << "/" << params.run_type;
    utils::make_path(outdir.str());

    std::ostringstream fname;
    fname << outdir.str() << "/corr-" << params.runid << "-T-" << std::fixed << std::setprecision(2)
          << T << ".csv";

    std::filesystem::path output = utils::get_available_filename(fname.str());
    return output.string();
}

inline std::string make_DensOfStates_filename(const RunParameters &params)
{
    std::ostringstream outdir;
    utils::make_path(outdir.str());

    std::ostringstream fname;
    std::filesystem::path output = utils::get_available_filename(fname.str());

    return output.string();
}

} // namespace io
