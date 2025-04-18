#include "utils/spin_permutations_iterator.hpp"

// Constructor for SpinPermutationsIterator
SpinPermutationsIterator::SpinPermutationsIterator(int n, bool end) : n(n), finished(end)
{
    if (!finished)
    {
        current_permutation =
            arma::Col<int>(n, arma::fill::ones) * spin_values(0); // Initialize all spins to the first spin value
    }
}

// Dereference operator to get the current permutation
SpinPermutationsIterator::value_type SpinPermutationsIterator::operator*() const
{
    return current_permutation;
}

// Pre-increment operator (moves to the next permutation)
SpinPermutationsIterator &SpinPermutationsIterator::operator++()
{
    for (int i = n - 1; i >= 0; --i)
    {
        // If the current spin matches the first spin value, flip to the second spin value
        if (current_permutation(i) == spin_values(0))
        {
            current_permutation(i) = spin_values(1); // Flip to the second spin value
            return *this;
        }
        else
        {
            current_permutation(i) = spin_values(0); // Reset back to the first spin value
        }
    }
    finished = true; // Mark as finished when all permutations are done
    return *this;
}

// Inequality operator
bool SpinPermutationsIterator::operator!=(const SpinPermutationsIterator &other) const
{
    return finished != other.finished;
}

// SpinPermutationsSequence constructor
SpinPermutationsSequence::SpinPermutationsSequence(int n) : n(n)
{
}

// Return the beginning iterator
SpinPermutationsIterator SpinPermutationsSequence::begin() const
{
    return SpinPermutationsIterator(n);
}

// Return the end iterator
SpinPermutationsIterator SpinPermutationsSequence::end() const
{
    return SpinPermutationsIterator(n, true); // Mark end as true
}
