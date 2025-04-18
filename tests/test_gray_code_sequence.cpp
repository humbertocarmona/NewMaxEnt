#include "utils/gray_code_sequence.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(GrayCodeSequenceTest, CorrectStatesFor3Spins) {
    // Arrange
    int nspins = 3;

    // Expected sequence for 3 spins (Gray code order):
    std::vector<std::vector<int>> expected_states = {
        {+1, +1, +1},
        {+1, +1, -1},
        {+1, -1, -1},
        {+1, -1, +1},
        {-1, -1, +1},
        {-1, +1, +1},
        {-1, +1, -1},
        {-1, -1, -1}
    };

    // Act & Assert
    GrayCodeSequence sequence(nspins);
    size_t index = 0;

    for (const auto& state : sequence) {
        ASSERT_LT(index, expected_states.size()) << "Generated too many states.";
        EXPECT_EQ(state, expected_states[index]) << "State at index " << index << " does not match expected.";
        index++;
    }

    ASSERT_EQ(index, expected_states.size()) << "Number of generated states does not match expected.";
}
