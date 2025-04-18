#pragma once

#include <armadillo>
#include <iterator>

class BinaryPermutationsIterator {
public:
    using value_type = arma::Col<int>;
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    BinaryPermutationsIterator(int n, int start_index, int end_index, bool end = false)
        : n(n), current_index(end ? end_index : start_index), end_index(end_index), finished(end) {
        if (!finished) {
            current_permutation = arma::Col<int>(n, arma::fill::zeros);
            updateCurrentPermutation();
        }
    }

    value_type operator*() const {
        return current_permutation;
    }

    BinaryPermutationsIterator& operator++() {
        ++current_index;
        if (current_index >= end_index) {
            finished = true;
        } else {
            updateCurrentPermutation();
        }
        return *this;
    }

    bool operator!=(const BinaryPermutationsIterator& other) const {
        return finished != other.finished;
    }

private:
    void updateCurrentPermutation() {
        for (int i = 0; i < n; ++i) {
            current_permutation[i] = (current_index & (1 << (n - i - 1))) ? -1 : +1;
        }
    }

    int n;
    int current_index;
    int end_index;
    bool finished;
    arma::Col<int> current_permutation;
};

class BinaryPermutationsSequence {
public:
    BinaryPermutationsSequence(int n, int start = 0, int end = -1)
        : n(n), start_index(start), end_index(end == -1 ? (1 << n) : end) {}

    BinaryPermutationsIterator begin() const {
        return BinaryPermutationsIterator(n, start_index, end_index);
    }

    BinaryPermutationsIterator end() const {
        return BinaryPermutationsIterator(n, start_index, end_index, true);
    }

private:
    int n;
    int start_index;
    int end_index;
};
