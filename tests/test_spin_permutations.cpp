#include "util/spin_permutations_iterator.hpp"
#include <armadillo>
#include <gtest/gtest.h>
#include <set>

TEST(SpinPermutationsTest, GeneratesAllPermutationsFor2Spins)
{
    SpinPermutationsSequence sequence(2);
    std::set<std::vector<int>> expected = {{+1, +1}, {+1, -1}, {-1, +1}, {-1, -1}};

    for (const auto &spin : sequence)
    {
        std::vector<int> s(spin.begin(), spin.end());
        EXPECT_TRUE(expected.count(s) > 0) << "Unexpected spin config: [" << spin(0) << ", " << spin(1) << "]";
        expected.erase(s);
    }

    EXPECT_TRUE(expected.empty()) << "Not all expected configurations were generated.";
}

TEST(SpinPermutationsTest, GeneratesCorrectCountFor3Spins)
{
    SpinPermutationsSequence sequence(3);
    int count = 0;
    for (const auto &spin : sequence)
    {
        ASSERT_EQ(spin.n_elem, 3);
        for (int i = 0; i < 3; ++i)
            ASSERT_TRUE(spin(i) == -1 || spin(i) == 1);
        ++count;
    }
    EXPECT_EQ(count, 8); // 2^3 = 8 permutations
}
