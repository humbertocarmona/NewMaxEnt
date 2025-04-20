#include "core/run_parameters.hpp"
#include "utils/centered_moments.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

template <typename T>
void writeTrainedModel(const RunParameters params,
                       const T &model,
                       const CenteredMoments m_data,
                       const CenteredMoments m_model)
{
    auto logger = getLogger();

    std::ostringstream fname;
    fname << params.result_dir << "/final-"
          << params.run_type <<"-" << params.runid << ".json";

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

    obj["run_parameters"] = params.to_json();

    std::filesystem::path output = utils::get_available_filename(fname.str());
    std::ofstream out(output);
    out << obj.dump(2);
    logger->info("[writeTrainedModel] saved {}", output.string());
};
