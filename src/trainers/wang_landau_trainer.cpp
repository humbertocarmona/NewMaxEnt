#include "trainers/wang_landau_trainer.hpp"

/**
 * @brief Randomly flips a single spin in the spin vector.
 *
 * Selects a random index in the input column vector `s` and flips the spin at that index
 * (i.e., multiplies it by -1). Assumes spins are represented as integers (+1 or -1).
 *
 * @param s   Reference to a column vector of spins (arma::Col<int>).
 * @param rng Reference to a random number generator (std::mt19937).
 */
void WangLandauTrainer::flip_random_spin(arma::Col<int> &s, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, s.n_elem - 1);
    int i = dist(rng);
    s(i) *= -1; // flip spin from +1 to -1 or vice versa
}

/**
 * @brief Checks if a histogram is flat based on a flatness threshold.
 *
 * Determines whether the histogram `H` is sufficiently flat by comparing the
 * minimum and maximum counts. The histogram is considered flat if the minimum
 * count is greater than `flatness_threshold` times the maximum count.
 *
 * @param H                  A histogram mapping energy values to their counts.
 * @param flatness_threshold Threshold ratio (default is 0.8) to determine flatness.
 * @return true if the histogram is flat, false otherwise.
 */
bool WangLandauTrainer::is_flat(const std::unordered_map<int, int> &H)
{
    int min_H = std::numeric_limits<int>::max();
    int max_H = 0;
    for (auto &[E, count] : H)
    {
        min_H = std::min(min_H, count);
        max_H = std::max(max_H, count);
    }
    return (min_H > flatness_threshold * max_H);
}



int howTouseIt()
{
    // main.cpp
    std::mt19937 rng(1);
    int nspins = 16; // number of spins

    // Example: random fields and couplings
    arma::vec h(nspins);
    int nedges = nspins * (nspins - 1) / 2;
    arma::vec J(nedges);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    int idx = 0;
    for (int i = 0; i < nspins; ++i)
    {
        h(i) = 0.01 * dist(rng) - 0.5;
        for (int j = i + 1; j < nspins; ++j)
        {
            J(idx++) = 0.01 * dist(rng);
        }
    }

    // wang_landau(h, J);

    return 0;
}