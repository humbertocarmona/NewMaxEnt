#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "util/utilities.hpp"
#include <memory>
// #include <sstream>
#include <string>
#include <vector>

inline std::shared_ptr<spdlog::logger> create_logger(const std::string &logger_name = "bm")
{

    // Reuse if already created
    if (auto logger = spdlog::get(logger_name))
    {
        return logger;
    }

    // Create console sink (disabled by default)
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::off);

    // Create file sink
    utils::make_path("./logs");
    const std::string timestamp = utils::now();
    const std::string log_file  = "./logs/bm_" + timestamp + ".log";

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_level(spdlog::level::debug);

    // Combine sinks
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);

    // Register globally
    spdlog::register_logger(logger);
    return logger;
}

inline void set_console_verbosity(bool verbose)
{
    auto logger = create_logger();
    if (auto console_sink = std::dynamic_pointer_cast<spdlog::sinks::stdout_color_sink_mt>(logger->sinks()[0]))
    {
        console_sink->set_level(verbose ? spdlog::level::debug : spdlog::level::off);
    }
}
