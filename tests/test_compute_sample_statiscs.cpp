#include "core/MaxEntCore.hpp"
#include <armadillo>
#include <gtest/gtest.h>

TEST(ComputeSampleStatisticsTest, ComputesMomentsCorrectly)
{
    Params params;
    params.gen_nspins = 3;

    arma::Mat<int> raw_samples(4, 3);
    raw_samples = {{1, -1, 1}, {-1, -1, 1}, {1, 1, -1}, {-1, 1, -1}};

    MaxEntCore model(params, false);
    model.set_samples(raw_samples);
    model.compute_sample_statistics();

    arma::Col<double> expected_m1 = {0.0, 0.0, 0.0};  // means over columns
    arma::Col<double> expected_m2 = {0.0, 0.0, -1.0}; // (±1)^2 = 1
    arma::Col<double> expected_m3 = {0.0};            // (±1)^3 = ±1

    EXPECT_TRUE(arma::approx_equal(model.get_sample_moment_1(), expected_m1, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(model.get_sample_moment_2(), expected_m2, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(model.get_sample_moment_3(), expected_m3, "absdiff", 1e-10));
}
