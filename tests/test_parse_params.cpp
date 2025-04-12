#include "core/parameters.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

// Helper: Create a temporary JSON file with given content
std::string create_temp_json_file(const nlohmann::json &data, const std::string &filename = "test_config.json")
{
    std::ofstream out(filename);
    out << data.dump(4);
    out.close();
    return filename;
}

Params parse_params(const std::string &filename); // Assume this exists elsewhere

TEST(ParseParamsTest, ReadsBasicFields)
{
    nlohmann::json test_json = {{"id", "test_run"},   {"runid", 123},        {"raw_samples_file", "samples.csv"},
                                {"run_type", "ens"},  {"save_result", true}, {"eta_h", 0.25},
                                {"q_val", {1.0, 1.5}}};

    auto file = create_temp_json_file(test_json);
    Params p  = parse_params(file);

    EXPECT_EQ(p.id, "test_run");
    EXPECT_EQ(p.runid, 123);
    EXPECT_EQ(p.raw_samples_file, "samples.csv");
    EXPECT_EQ(p.run_type, "ens");
    EXPECT_TRUE(p.save_result);
    EXPECT_NEAR(p.eta_h, 0.25, 1e-6);
    ASSERT_EQ(p.q_val.n_elem, 2);
    EXPECT_NEAR(p.q_val[0], 1.0, 1e-6);
    EXPECT_NEAR(p.q_val[1], 1.5, 1e-6);
}
