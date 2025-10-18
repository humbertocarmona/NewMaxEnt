#pragma once

#include "core/run_parameters.hpp"
#include "io/make_file_names.hpp"
#include "utils/centered_moments.hpp"

#include <nlohmann/json.hpp>
#include <ostream>
#include <sstream>
#include <string>

template <typename T>
void writeTrainedModel(const T &model,
                       const CenteredMoments m_data,
                       const CenteredMoments m_model,
                       std::string prefix = "final-")
{
    auto logger = getLogger();

    nlohmann::json obj;
    obj["type"] = model.get_type();

    obj["iter"] = model.get_iter();

    obj["h"] = model.get_h();
    obj["J"] = model.get_J();

    obj["avg_energy"]        = model.get_avg_energy();
    obj["avg_energy_sq"]     = model.get_avg_energy_sq();
    obj["m1_data"]           = model.get_m1_data();
    obj["m2_data"]           = model.get_m2_data();
    obj["m3_data"]           = model.get_m3_data();
    obj["m2_data_centered"]  = m_data.centered_moment_2;
    obj["m3_data_centered"]  = m_data.centered_moment_3;
    obj["m1_model"]          = model.get_m1_model();
    obj["m2_model"]          = model.get_m2_model();
    obj["m3_model"]          = model.get_m3_model();
    obj["m2_model_centered"] = m_model.centered_moment_2;
    obj["m3_model_centered"] = m_model.centered_moment_3;

    // k-pairwise
    obj["K"]        = model.get_K();
    obj["pK_data"]  = model.get_pK_data();
    obj["pK_model"] = model.get_pK_model();

    obj["run_parameters"] = model.get_params().to_json();

    if (model.get_params().run_type == "Gen_Full")
    {
        obj["sample"] = "true";
    }

    auto output = io::make_filename(model.get_params(), prefix);
    std::ofstream out(output);
    out << obj.dump(2);
    logger->info("[writeTrainedModel] saved {}", output);
};
