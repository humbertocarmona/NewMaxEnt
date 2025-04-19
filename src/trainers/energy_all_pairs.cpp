#include "trainers/full_ensemble_trainer.hpp"
#include <armadillo>

double FullEnsembleTrainer::energyAllPairs(arma::Col<int> s)
{

    double En = 0.0;
    for (int i = 0; i < core.nspins; ++i)
        En += core.h(i) * s(i);
    int idx = 0;
    for (int i = 0; i < core.nspins - 1; ++i)
        for (int j = i + 1; j < core.nspins; ++j)
            En += core.J(idx++) * s(i) * s(j);

    return -En;
}
