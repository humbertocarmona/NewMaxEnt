#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

inline nlohmann::json readTrainedModel(const std::string &filepath)
{
    std::ifstream in(filepath);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    nlohmann::json obj;
    in >> obj;
    return obj;
}
