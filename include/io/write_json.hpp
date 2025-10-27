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
    
    auto map_ge = model.get_GE();
    auto map_pe = model.get_PE();
    
    arma::Col<double> en, we;
    auto wbin = model.get_params().energy_bin;
    
    std::vector<std::pair<int, double>> sorted_GE(map_ge.begin(), map_ge.end());
    std::sort(sorted_GE.begin(), sorted_GE.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<std::pair<int, double>> sorted_PE(map_pe.begin(), map_pe.end());
    std::sort(sorted_PE.begin(), sorted_PE.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    
    for (const auto &[bin, weight] : sorted_GE)
    {
        en.insert_rows(en.n_rows, 1);
        en(en.n_rows - 1) = bin * wbin;

        we.insert_rows(we.n_rows, 1);
        we(en.n_rows - 1) = weight;
    }
    obj["E"]  = en;
    obj["GE"] = we;


    en.clear();
    we.clear();
    for (const auto &[bin, weight] : sorted_PE)
    {
        en.insert_rows(en.n_rows, 1);
        en(en.n_rows - 1) = bin * wbin;

        we.insert_rows(we.n_rows, 1);
        we(en.n_rows - 1) = weight;
    }

    obj["PE"] = we;
    auto output = io::make_filename(model.get_params(), prefix);
    std::ofstream out(output);
    out << obj.dump(2);
    logger->info("[writeTrainedModel] saved {}", output);
};
