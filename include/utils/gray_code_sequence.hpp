#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

class GrayCodeIterator
{
  public:
    using value_type = std::vector<int>;

    GrayCodeIterator(int n, bool end = false) : n(n), k(end ? (1 << n) : 0), finished(end)
    {
    }

    const value_type operator*() const
    {
        value_type spin_config(n);
        int gray_code = k ^ (k >> 1); // Generate the Gray code for current k
        for (int i = 0; i < n; ++i)
        {
            int bit        = (gray_code >> (n - 1 - i)) & 1; // Get the i-th bit of the Gray code
            spin_config[i] = spin_values[bit];
        }
        return spin_config;
    }

    GrayCodeIterator &operator++()
    {
        ++k;
        finished = (k >= (1 << n));
        return *this;
    }

    bool operator!=(const GrayCodeIterator &other) const
    {
        return finished != other.finished;
    }

  private:
    int n;                                         // Number of spins
    int k;                                         // Current index in the sequence
    bool finished;                                 // Whether we’ve reached the end
    const std::vector<int> spin_values = {+1, -1}; // Spin values to use
};

class GrayCodeSequence
{
  public:
    GrayCodeSequence(int n) : n(n)
    {
    }

    GrayCodeIterator begin() const
    {
        return GrayCodeIterator(n);
    }

    GrayCodeIterator end() const
    {
        return GrayCodeIterator(n, true);
    }

  private:
    int n;
};


