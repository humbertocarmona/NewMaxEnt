#include "core/MaxEntCore.hpp"
#include "core/parameters.hpp"
#include "entropy/ShannonEntropy.hpp"
#include <iostream>

int main()
{
    Params params;
    params.gen_nspins = 4;
    params.gen_seed   = 42;

    MaxEntCore model(params, true);

    ShannonEntropy entropy;
    entropy.compute_expectations(model);

    std::cout << "[Test] Basic MaxEntCore + ShannonEntropy ran successfully.\n";
    return 0;
}
