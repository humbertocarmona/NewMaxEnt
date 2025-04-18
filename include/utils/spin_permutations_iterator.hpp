#pragma once

#include <armadillo>
#include <iterator>

// Class to generate spin permutations using iterators
class SpinPermutationsIterator
{
  public:
    using value_type        = arma::Col<int>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const value_type *;
    using reference         = const value_type &;
    using iterator_category = std::input_iterator_tag;

    SpinPermutationsIterator(int n, bool end = false);

    value_type operator*() const;
    SpinPermutationsIterator &operator++();
    bool operator!=(const SpinPermutationsIterator &other) const;

  private:
    int n;
    arma::Col<int> spin_values = {+1, -1};
    arma::Col<int> current_permutation;
    bool finished;
};

// Helper class to manage the spin permutations sequence
class SpinPermutationsSequence
{
  public:
    SpinPermutationsSequence(int n);
    SpinPermutationsIterator begin() const;
    SpinPermutationsIterator end() const;

  private:
    int n;
};
