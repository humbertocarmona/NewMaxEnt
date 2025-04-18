#include "trainers/full_ensemble_trainer.hpp"
#include "utils/spin_permutations_iterator.hpp"
#include "utils/utilities.hpp"

void FullEnsembleTrainer::computeFullEnumerationAverages(double beta, bool triplets = false)
{
    auto logger = core.logger;
    logger->info("[computeFullEnumerationAverages] Starting full enumeration averages");

    auto &h = core.h;
    auto &J = core.J;

    int nspins = core.nspins;
    int nedges = core.nedges;
    m1_model.zeros(nspins);
    m2_model.zeros(nedges);
    m3_model.zeros(ntriplets);
    double q_inv = (q_val != 1) ? 1.0 / (1.0 - q_val) : 0.0; 
    
    avg_energy    = 0.0;
    avg_energy_sq = 0.0;

    arma::Col<int> s(nspins);
    double Z = 0.0;
    double E = 0.0;
    double P = 0.0;
    for (auto p = SpinPermutationsSequence(nspins).begin(); p != SpinPermutationsSequence(nspins).end(); ++p)
    {
        s = *p;
        E = energyAllPairs(s);
        P = utils::exp_q(-beta * E, q_val, q_inv);
        Z += P;
        avg_energy += P * E;
        avg_energy_sq += P * E * E;

        // First-order moments
        for (int i = 0; i < nspins; ++i)
            m1_model(i) += P * s(i);

        // Second-order moments
        int idx = 0;
        for (int i = 0; i < nspins - 1; ++i)
        {
            for (int j = i + 1; j < nspins; ++j)
            {
                m2_model(idx++) += P * s(i) * s(j);
            }
        }

        if (triplets)
        {
            // Third-order moments
            idx = 0;
            for (int i = 0; i < nspins - 2; ++i)
            {
                for (int j = i + 1; j < nspins - 1; ++j)
                {
                    for (int k = j + 1; k < nspins; ++k)
                    {
                        m3_model(idx++) += P * s(i) * s(j) * s(k);
                    }
                }
            }
        }
    }
    avg_energy /= Z;
    avg_energy_sq /= Z;
    m1_model /= Z;
    m2_model /= Z;
    if (triplets)
        m3_model /= Z;
}
