#pragma once

#include "EntropyModel.hpp"

class ShannonEntropy : public EntropyModel {
public:
    void compute_expectations(MaxEntCore& model, double q = 1.0) override;
};