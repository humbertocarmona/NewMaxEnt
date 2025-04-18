#include "util/utilities.hpp"
#include "util/logger.hpp"
#include <cassert>
#include <regex>
#include <sstream>

namespace utils
{
namespace fs = std::filesystem;

std::string now()
{
    // Get the current time
    std::time_t t = std::time(nullptr);
    // Convert to local time
    std::tm tm = *std::localtime(&t);
    // Create a stringstream to format the date
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M");
    return oss.str();
}


bool is_file(const std::filesystem::path &path)
{
    return fs::exists(path);
}

bool is_dir(const std::filesystem::path &path)
{
    return fs::exists(path);
}

void make_path(const std::filesystem::path &path)
{
    if (!fs::exists(path))
    {
        if (fs::create_directories(path))
        {
            std::cout << "Directory created: " << path << '\n';
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << '\n';
        }
    }
}


} // namespace utils
