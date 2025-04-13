cmake_minimum_required(VERSION 3.14)
project(MaxEntModel VERSION 0.1.0)
cmake_policy(SET CMP0167 NEW)

# Use system-installed fmt
add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(CUSTOM_OUTPUT_DIR "$ENV{HOME}/.config/bin")
file(MAKE_DIRECTORY ${CUSTOM_OUTPUT_DIR})


# Dependencies
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Build the main library
add_library(maxent_lib ${SOURCES})

target_link_libraries(maxent_lib
    PRIVATE
    armadillo
    Boost::program_options
    spdlog::spdlog
    fmt::fmt
)

# Build the test executable
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(test_maxent ${TEST_SOURCES})
target_link_libraries(test_maxent
    PRIVATE
    maxent_lib
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Optional: set output directory for test binary
set_target_properties(test_maxent PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable testing and test discovery
# enable_testing()
# include(GoogleTest)
# gtest_discover_tests(test_maxent)

enable_testing()

include(GoogleTest)

gtest_discover_tests(test_maxent
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    DISCOVERY_TIMEOUT 30
    DISCOVERY_MODE PRE_TEST
)


# Build the main executable
add_executable(maxent_main src/main.cpp)

target_link_libraries(maxent_main
    PRIVATE
    maxent_lib
    armadillo
    Boost::program_options
    spdlog::spdlog
    fmt::fmt
)

set_target_properties(maxent_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_OUTPUT_DIR}
)
