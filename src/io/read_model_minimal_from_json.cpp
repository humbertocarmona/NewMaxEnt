#include "io/read_model_minimal_from_json.hpp"
#include "util/logger.hpp"
#include "util/utilities.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

ModelMinimal read_model_minimal_from_json(const std::string& filename)
{
    auto logger = spdlog::get("bm");

    std::ifstream in(filename);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open model JSON: " + filename);
    }

    nlohmann::json j;
    in >> j;

    ModelMinimal model;
    model.nspins = j.at("nspins").get<int>();
    model.q_val  = j.value("q_val", 1.0);

    model.h = arma::Col<double>(j.at("h").get<std::vector<double>>());
    model.J = arma::Col<double>(j.at("J").get<std::vector<double>>());


    logger->info("[read_model_minimal_from_json] h =  {}", brief(model.h));
    logger->info("[read_model_minimal_from_json] J =  {}", brief(model.J));

    return model;
}
