#include "core/max_ent_core.hpp"
#include <armadillo>
#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(MaxEntCoreSmokeTest, LoggerInitialization)
{
    Params params;
    params.gen_nspins = 4;
    params.gen_seed   = 42;

    MaxEntCore model(params, true);
    auto logger = model.get_logger();

    ASSERT_TRUE(logger != nullptr);
    logger->info("[Test] Logger works inside MaxEntCore.");
    logger->flush();
}

TEST(MaxEntCoreSmokeTest, ParamsLogInfo)
{
    Params params;
    params.gen_nspins = 4;
    params.gen_seed   = 42;

    MaxEntCore model(params, true);
    model.get_params().log_info(true); // should just print info
}
