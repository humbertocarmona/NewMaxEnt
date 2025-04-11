#include "entropy/TsallisEntropy.hpp"
#include "core/MaxEntCore.hpp"

void TsallisEntropy::compute_expectations(MaxEntCore& model, double q) {
    model.get_logger()->info("[TsallisEntropy] computing expectations with q = {} (Tsallis)", q);
}
