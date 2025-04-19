#include <armadillo>
#include <random>
#include <cmath>

// Perform Heat-Bath sampling and compute model averages
void computeModelAverages(
    arma::Col<int>& spins,              // Current spin configuration
    const arma::Col<double>& J,         // Linearized interaction matrix
    const arma::Col<double>& h,         // External fields
    const arma::Mat<int>& edges,        // Matrix providing linear indices
    double beta,                        // Inverse temperature
    size_t numEquilibrationSweeps,      // Number of equilibration sweeps
    size_t numSamples,                  // Number of samples to collect
    size_t sampleInterval,              // Number of sweeps between samples
    arma::Col<double>& m1_model,        // First moments (output)
    arma::Col<double>& m2_model         // Second moments (output)
) {
    size_t nspins = spins.n_elem;

    // Initialize the model averages to zero
    m1_model.zeros(nspins);
    m2_model.zeros(J.n_elem);

    // Random number generator setup
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Perform equilibration sweeps (no data collection)
    for (size_t sweep = 0; sweep < numEquilibrationSweeps; ++sweep) {
        for (size_t i = 0; i < nspins; ++i) {
            double local_field = h(i);
            for (size_t j = 0; j < nspins; ++j) {
                if (j != i) {
                    int edgeIndex = edges(i, j);
                    if (edgeIndex != -1) {
                        local_field += J(edgeIndex) * spins(j);
                    }
                }
            }
            double exp_plus = std::exp(beta * local_field);
            double exp_minus = std::exp(-beta * local_field);
            double prob_plus = exp_plus / (exp_plus + exp_minus);
            double r = dist(rng);
            spins(i) = (r < prob_plus) ? 1 : -1;
        }
    }

    // Collect samples to compute averages
    size_t samplesCollected = 0;
    for (size_t sweep = 0; samplesCollected < numSamples; ++sweep) {
        // Perform a sweep
        for (size_t i = 0; i < nspins; ++i) {
            double local_field = h(i);
            for (size_t j = 0; j < nspins; ++j) {
                if (j != i) {
                    int edgeIndex = edges(i, j);
                    if (edgeIndex != -1) {
                        local_field += J(edgeIndex) * spins(j);
                    }
                }
            }
            double exp_plus = std::exp(beta * local_field);
            double exp_minus = std::exp(-beta * local_field);
            double prob_plus = exp_plus / (exp_plus + exp_minus);
            double r = dist(rng);
            spins(i) = (r < prob_plus) ? 1 : -1;
        }

        // Every sampleInterval sweeps, record the current configuration
        if ((sweep % sampleInterval) == 0) {
            for (size_t i = 0; i < nspins; ++i) {
                m1_model(i) += spins(i);
            }
            size_t idx = 0;
            for (size_t i = 0; i < nspins - 1; ++i) {
                for (size_t j = i + 1; j < nspins; ++j) {
                    int edgeIndex = edges(i, j);
                    if (edgeIndex != -1) {
                        m2_model(edgeIndex) += spins(i) * spins(j);
                    }
                }
            }
            samplesCollected++;
        }
    }

    // Normalize the averages
    m1_model /= static_cast<double>(numSamples);
    m2_model /= static_cast<double>(numSamples);
}
