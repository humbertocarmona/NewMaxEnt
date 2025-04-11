cmake_minimum_required(VERSION 3.14)
project(MaxEntModel VERSION 0.1.0)
cmake_policy(SET CMP0167 NEW)

add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Build the library
add_library(maxent_lib ${SOURCES})
target_link_libraries(maxent_lib
    PRIVATE
    armadillo
    Boost::program_options
    spdlog::spdlog
    fmt::fmt
)

# Build the test executable
add_executable(test_maxent tests/test_main.cpp)
target_link_libraries(test_maxent
    PRIVATE
    maxent_lib
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Set test output dir
set_target_properties(test_maxent PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest
enable_testing()
add_test(NAME unit_tests COMMAND test_maxent)
