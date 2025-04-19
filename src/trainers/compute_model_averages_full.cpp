#include "trainers/full_ensemble_trainer.hpp"
#include "utils/binary_permutations_sequence.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void FullEnsembleTrainer::computeModelAverages(double beta, bool triplets)
{
    // auto logger = getLogger();

    int nspins = core.nspins;
    m1_model.zeros(nspins);
    m2_model.zeros(core.nedges);
    m3_model.zeros(ntriplets);
    double q_inv = (q_val != 1) ? 1.0 / (1.0 - q_val) : 0.0;


    avg_energy    = 0.0;
    avg_energy_sq = 0.0;

    // arma::Col<int> s(nspins);
    double Z = 0.0;
    double E = 0.0;
    double P = 0.0;
    BinaryPermutationsSequence sequence(nspins);

    for (const auto &s : sequence)
    {
        E = energyAllPairs(s);
        P = utils::exp_q(-beta * E, q_val, q_inv);

        Z += P;
        avg_energy += P * E;
        avg_energy_sq += P * E * E;

        // First-order moments
        for (size_t i = 0; i < nspins; ++i)
            m1_model(i) += P * s(i);

        // Second-order moments
        int idx = 0;
        for (size_t i = 0; i < nspins - 1; ++i)
        {
            for (size_t j = i + 1; j < nspins; ++j)
            {
                m2_model(idx++) += P * s(i) * s(j);
            }
        }

        if (triplets)
        {
            // Third-order moments
            idx = 0;
            for (size_t i = 0; i < nspins - 2; ++i)
            {
                for (size_t j = i + 1; j < nspins - 1; ++j)
                {
                    for (size_t k = j + 1; k < nspins; ++k)
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

    // logger->warn("[computeFullEnumerationAverages] ntriplets = {} {}",ntriplets, m3_model.n_elem);

}
