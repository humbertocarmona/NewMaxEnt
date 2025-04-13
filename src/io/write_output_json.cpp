#include "io/write_output_json.hpp"
#include <fstream>

using json = nlohmann::json;

void write_output_json(
    const std::string& filename,
    const Params& run_parameters,
    const arma::Col<double>& h,
    const arma::Col<double>& J,
    const arma::Col<double>& sample_moment_1,
    const arma::Col<double>& model_moment_1,
    const arma::Col<double>& sample_moment_2,
    const arma::Col<double>& model_moment_2,
    const arma::Col<double>& centered_sample_moment_2,
    const arma::Col<double>& centered_model_moment_2,
    const arma::Col<double>& model_moment_3,
    const arma::Col<double>& centered_model_moment_3,
    double energy_mean,
    double energy_fluctuation,
    int final_iter
)
{
    json j;

    j["iter"] = final_iter;
    j["nspins"] = run_parameters.gen_nspins;
    j["nedges"] = run_parameters.gen_nspins * (run_parameters.gen_nspins - 1) / 2;
    j["beta"] = run_parameters.beta;
    j["q_val"] = run_parameters.q_val;
    j["id"] = run_parameters.id;

    j["h"] = std::vector<double>(h.begin(), h.end());
    j["J"] = std::vector<double>(J.begin(), J.end());

    j["x_obs"] = std::vector<double>(sample_moment_1.begin(), sample_moment_1.end());
    j["x_mod"] = std::vector<double>(model_moment_1.begin(), model_moment_1.end());

    j["xy_obs"] = std::vector<double>(sample_moment_2.begin(), sample_moment_2.end());
    j["xy_mod"] = std::vector<double>(model_moment_2.begin(), model_moment_2.end());

    j["xy_obs_centered"] = std::vector<double>(centered_sample_moment_2.begin(), centered_sample_moment_2.end());
    j["xy_mod_centered"] = std::vector<double>(centered_model_moment_2.begin(), centered_model_moment_2.end());

    j["model_moment_3"] = std::vector<double>(model_moment_3.begin(), model_moment_3.end());
    j["model_moment_3_centered"] = std::vector<double>(centered_model_moment_3.begin(), centered_model_moment_3.end());

    j["energy_mean"] = energy_mean;
    j["energy_fluctuation"] = energy_fluctuation;

    // Include full Params struct as "params" object
    j["params"] = run_parameters.to_json();

    std::ofstream out(filename);
    out << j.dump(2);
}
