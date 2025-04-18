#include "utils/binary_permutations_sequence.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(BinaryPermutationsSequenceTest, FullSequence) {
    // Arrange
    int nspins = 3;

    // Expected sequence for 3 spins:
    // {+1, +1, +1}, {+1, +1, -1}, {+1, -1, +1}, {+1, -1, -1},
    // {-1, +1, +1}, {-1, +1, -1}, {-1, -1, +1}, {-1, -1, -1}
    std::vector<std::vector<int>> expected_states = {
        {+1, +1, +1},
        {+1, +1, -1},
        {+1, -1, +1},
        {+1, -1, -1},
        {-1, +1, +1},
        {-1, +1, -1},
        {-1, -1, +1},
        {-1, -1, -1}
    };

    // Act & Assert
    BinaryPermutationsSequence sequence(nspins);
    size_t index = 0;

    for (const auto& state : sequence) {
        ASSERT_LT(index, expected_states.size()) << "Generated too many states.";
        EXPECT_EQ(state.n_elem, nspins) << "State does not have the correct number of spins.";
        std::vector<int> state_vector(state.begin(), state.end());
        EXPECT_EQ(state_vector, expected_states[index]) << "State at index " << index << " does not match expected.";
        ++index;
    }

    ASSERT_EQ(index, expected_states.size()) << "Number of generated states does not match expected.";
}

TEST(BinaryPermutationsSequenceTest, PartialSequence) {
    // Arrange
    int nspins = 3;
    int start = 4;  // Start from binary 100
    int end = 8;    // Stop at binary 111

    // Expected sequence for this range:
    // {-1, +1, +1}, {-1, +1, -1}, {-1, -1, +1}, {-1, -1, -1}
    std::vector<std::vector<int>> expected_states = {
        {+1, +1, +1},
        {+1, +1, -1},
        {+1, -1, +1},
        {+1, -1, -1},
        {-1, +1, +1},
        {-1, +1, -1},
        {-1, -1, +1},
        {-1, -1, -1}
    };

    // Act & Assert
    BinaryPermutationsSequence sequence(nspins,start,end);
    
    size_t index = start;
    for (const auto &state : sequence)
    {
        std::cout << "state " << index << ": ";
        for (int spin : state)
        {
            std::cout << spin << " ";
        }
        std::cout << " \n";
        std::cout << "Expected state at index " << index << ": ";

        for (int spin : expected_states[index])
        {
            std::cout << spin << " ";
        }
        std::cout << "\n";
        std::cout << "\n";

        ++index;
    }

    index=start;
    for (const auto& state : sequence) {
        ASSERT_LT(index, expected_states.size()) << "Generated too many states.";
        EXPECT_EQ(state.n_elem, nspins) << "State does not have the correct number of spins.";
        std::vector<int> state_vector(state.begin(), state.end());
        EXPECT_EQ(state_vector, expected_states[index]) << "State at index " << index << " does not match expected.";
        ++index;
    }

    ASSERT_EQ(index, expected_states.size()) << "Number of generated states does not match expected.";
}
