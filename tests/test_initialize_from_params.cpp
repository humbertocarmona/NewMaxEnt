#include "core/MaxEntCore.hpp"
#include <armadillo>
#include <gtest/gtest.h>

TEST(MaxEntCoreTest, InitializeFromParams_UsesPredefinedFields)
{
    Params params;
    params.gen_nspins = 4;
    params.gen_h0     = arma::vec({0.1, 0.2, 0.3, 0.4});
    params.gen_J0     = arma::vec({1.0, 1.1, 1.2, 1.3, 1.4, 1.5});

    MaxEntCore model(params, true);

    EXPECT_TRUE(arma::approx_equal(model.get_h(), params.gen_h0, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(model.get_J(), params.gen_J0, "absdiff", 1e-10));
}

TEST(MaxEntCoreTest, InitializeFromParams_GeneratesFieldsIfEmpty)
{
    Params params;
    params.gen_nspins  = 4;
    params.gen_h_mean  = 0.5;
    params.gen_h_width = 0.1;
    params.gen_J_mean  = 1.0;
    params.gen_J_width = 0.2;
    params.gen_seed    = 42;

    MaxEntCore model(params, true);

    EXPECT_EQ(model.get_h().n_elem, 4);
    EXPECT_EQ(model.get_J().n_elem, 6); // 4*(4-1)/2

    EXPECT_TRUE(arma::approx_equal(model.get_h(), model.get_params().gen_h0, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(model.get_J(), model.get_params().gen_J0, "absdiff", 1e-10));
}
