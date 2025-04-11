#pragma once

#include "entropy/EntropyModel.hpp"  // ✅ Must include base class!

class MaxEntCore;

class TsallisEntropy : public EntropyModel
{
  public:
    void compute_expectations(MaxEntCore &model, double q = 1.0) override;
};
