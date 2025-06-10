#pragma once

#include <memory>
#include <mutex>
#include <spdlog/sinks/stdout_color_sinks.h> // Add this for stdout_color_mt
#include <spdlog/spdlog.h>

#pragma once

#include <chrono>
#include <ctime>
#include <memory>
#include <mutex>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

// Function to get the global logger instance
inline std::shared_ptr<spdlog::logger> getLogger()
{
    static std::shared_ptr<spdlog::logger> logger = nullptr;
    static std::mutex loggerMutex;

    std::lock_guard<std::mutex> lock(loggerMutex);
    if (!logger)
    {
        // Create a console sink
        auto console_sink    = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto now             = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm *tm_ptr      = std::localtime(&now_time);

        std::ostringstream oss;
        oss << std::put_time(tm_ptr, "%Y%m%d_%H%M%S");
        std::string filename = "logs/output_" + oss.str() + ".log";
        // Create a file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);

        // Combine the sinks into a single logger
        // std::vector<spdlog::sink_ptr> sinks = {console_sink, file_sink};
        std::vector<spdlog::sink_ptr> sinks = {console_sink};
        logger = std::make_shared<spdlog::logger>("MaxEnt", begin(sinks), end(sinks));

        // Set the logging level and pattern if needed
        logger->set_level(spdlog::level::info); // Example: set default level to info
        // logger->set_pattern("[%d/%m] %v");
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
        // logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

        spdlog::register_logger(logger);
    }
    return logger;
}
