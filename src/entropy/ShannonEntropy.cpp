#include "entropy/ShannonEntropy.hpp"
#include "core/MaxEntCore.hpp"
#include <armadillo>
#include <cmath>


void ShannonEntropy::compute_expectations(MaxEntCore& model, double q) {
    // TODO: Replace with actual ensemble enumeration from ensemble_means.cpp
    // Example stub:
    model.get_logger()->info("[ShannonEntropy] computing expectations with q = 1.0 (Shannon)");
}