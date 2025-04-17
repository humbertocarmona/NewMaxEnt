#pragma once
#include <armadillo>

/**
 * @brief Lightweight struct to store a spin configuration and its associated probability.
 *
 * This struct is primarily used to track the top-K most probable spin configurations
 * during full enumeration workflows, such as thermodynamic sweeps.
 *
 * The comparison operator is intentionally reversed (using `>`) so that instances
 * can be stored in a std::priority_queue as a min-heap — i.e., the least probable
 * state is at the top of the heap and can be efficiently removed when a more
 * probable one is encountered.
 */
struct ReplicaState {
    double probability;       ///< The probability associated with the spin configuration
    arma::Col<int> spins;     ///< Spin configuration (e.g., -1/+1 values)

    /**
     * @brief Reversed comparison operator for use in a min-heap (std::priority_queue).
     *
     * Returns true if this ReplicaState has a higher probability than another,
     * making the priority queue prioritize lower-probability states.
     */
    bool operator<(const ReplicaState& other) const {
        return probability > other.probability;
    }
};
