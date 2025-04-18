#include "utils/gray_code_sequence.hpp"
#include <gtest/gtest.h>
#include <vector>
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(GrayCodeSequenceTest, CorrectStatesFor3Spins)
{
    // Arrange
    int nspins = 3;

    // Expected sequence for 3 spins (Gray code order):
    std::vector<std::vector<int>> expected_states = {{+1, +1, +1},
                                                     {+1, +1, -1},
                                                     {+1, -1, -1},
                                                     {+1, -1, +1},
                                                     {-1, -1, +1},
                                                     {-1, -1, -1},
                                                     {-1, +1, -1},
                                                     {-1, +1, +1}};

    // 000, 001, 011, 010, 110, 111, 101, 100

    // Act & Assert
    GrayCodeSequence sequence(nspins);
    size_t index = 0;
    // for (const auto &pair : sequence)
    // {
    //     const auto& state = pair.first;
    //     const auto& flipped_index = pair.second;

    //     std::cout << "state " << index << ": ";
    //     for (int spin : state)
    //     {
    //         std::cout << spin << " ";
    //     }
    //     std::cout  << "flipped: "<< flipped_index << " \n";
    //     std::cout << "Expected state at index " << index << ": ";

    //     for (int spin : expected_states[index])
    //     {
    //         std::cout << spin << " ";
    //     }
    //     std::cout << "\n";
    //     std::cout << "\n";

    //     ++index;
    // }

    index = 0;
    for (const auto &pair : sequence)
    {
        const auto& state = pair.first;
        const auto& flipped_index = pair.second;
        ASSERT_LT(index, expected_states.size()) << "Generated too many states.";
        EXPECT_EQ(state, expected_states[index]) << "State at index " << index << " does not match expected.";
        index++;
    }

    ASSERT_EQ(index, expected_states.size()) << "Number of generated states does not match expected.";
}
