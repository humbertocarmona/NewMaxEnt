#include "core/max_ent_core.hpp"
#include "core/parameters.hpp"
#include <iostream>

int main()
{
    Params params;
    params.gen_nspins = 4;

    MaxEntCore model(params, true);

    std::cout << "Basic test ran successfully.\n";
    return 0;
}
