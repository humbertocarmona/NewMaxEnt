// --- EntropyModel.hpp ---
#pragma once

class MaxEntCore; // Forward declaration

class EntropyModel {
public:
    virtual void compute_expectations(MaxEntCore& model, double q = 1.0) = 0;
    virtual ~EntropyModel() = default;
};