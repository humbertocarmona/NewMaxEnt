#include "io/write_trained_json.hpp"
#include "utils/get_logger.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

void writeTrainedModel(RunParameters params, FullEnsembleTrainer model, CenteredMoments m_data, CenteredMoments m_model)
{
    auto logger = getLogger();

    std::ostringstream fname;
    fname << params.result_dir << "/"
          << "run_" << params.nspins << ".json";
    
    nlohmann::json obj;

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

    std::ofstream out(fname.str());
    out << obj.dump(2);
    logger->info("[writeTrainedModel] saved {}", fname.str());

}