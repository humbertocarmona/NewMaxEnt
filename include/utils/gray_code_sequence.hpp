#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

class GrayCodeIterator 
{
public:
    using value_type = std::pair<std::vector<int>, int>; // State and flipped index

    GrayCodeIterator(int n, bool end = false)
        : n(n), k(end ? (1 << n) : 0), finished(end), last_flipped_index(-1) {}

    value_type operator*() const {
        std::vector<int> spin_config(n);
        int gray_code = k ^ (k >> 1);
        for (int i = 0; i < n; ++i) {
            int bit = (gray_code >> (n - 1 - i)) & 1;
            spin_config[i] = spin_values[bit];
        }
        return {spin_config, last_flipped_index};
    }

    GrayCodeIterator& operator++() {
        int new_k = k + 1;
        int flipped_bit = __builtin_ctz(k ^ new_k); // Find the position of the flipped bit
        last_flipped_index = n - 1 - flipped_bit;   // Map to the corresponding index in the spin vector
        k = new_k;
        finished = (k >= (1 << n));
        return *this;
    }

    bool operator!=(const GrayCodeIterator& other) const {
        return finished != other.finished;
    }

private:
    int n;                 // Number of spins
    int k;                 // Current index in the Gray code sequence
    bool finished;         // Indicates whether the sequence has finished
    int last_flipped_index; // The index of the bit flipped in the current step
    const std::vector<int> spin_values = {+1, -1}; // Spin values to use
};

class GrayCodeSequence {
public:
    GrayCodeSequence(int n) : n(n) {}

    GrayCodeIterator begin() const {
        return GrayCodeIterator(n);
    }

    GrayCodeIterator end() const {
        return GrayCodeIterator(n, true);
    }

private:
    int n;
};
