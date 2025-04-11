#include "core/MaxEntCore.hpp"
#include "core/parameters.hpp"
#include "entropy/ShannonEntropy.hpp"
#include <iostream>

int main()
{
    Params params;
    params.gen_nspins = 4;

    MaxEntCore model(params, true);
    ShannonEntropy entropy;
    entropy.compute_expectations(model);

    std::cout << "Basic test ran successfully.\n";
    return 0;
}