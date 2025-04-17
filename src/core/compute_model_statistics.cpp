#include "core/compute_model_statistics.hpp"
#include "core/energy.hpp"
#include "util/spin_permutations_iterator.hpp"
#include "util/utilities.hpp"

std::vector<ReplicaState> compute_model_statistics(
    const int &n_spins, const arma::Col<double> &h, const arma::Col<double> &J, arma::Col<double> &model_moment_1,
    arma::Col<double> &model_moment_2, arma::Col<double> &model_moment_3, double q_val, double beta,
    bool compute_triplets, double *avg_energy, double *avg_energy_sq, int top_k_replicas)
{
    std::priority_queue<ReplicaState> top_replicas;

    int n_edges    = n_spins * (n_spins - 1) / 2;
    int n_triplets = n_spins * (n_spins - 1) * (n_spins - 2) / 6;

    model_moment_1.zeros(n_spins);
    model_moment_2.zeros(n_edges);
    model_moment_3.zeros(n_triplets);

    double energy_accum    = 0.0;
    double energy_sq_accum = 0.0;

    arma::Col<int> s(n_spins);
    double Z = 0.0;

    for (auto p = SpinPermutationsSequence(n_spins).begin(); p != SpinPermutationsSequence(n_spins).end(); ++p)
    {
        s        = *p;
        double E = energy(s, h, J, n_spins);
        double P = utils::exp_q(-beta * E, q_val);
        Z += P;

        // First-order moments
        for (int i = 0; i < n_spins; ++i)
            model_moment_1(i) += P * s(i);

        // Second-order moments
        int idx = 0;
        for (int i = 0; i < n_spins - 1; ++i)
        {
            for (int j = i + 1; j < n_spins; ++j)
            {
                model_moment_2(idx++) += P * s(i) * s(j);
            }
        }

        if (compute_triplets)
        {
            // Third-order moments
            idx = 0;
            for (int i = 0; i < n_spins - 2; ++i)
            {
                for (int j = i + 1; j < n_spins - 1; ++j)
                {
                    for (int k = j + 1; k < n_spins; ++k)
                    {
                        model_moment_3(idx++) += P * s(i) * s(j) * s(k);
                    }
                }
            }

            // Energy stats
            energy_accum += P * E;
            energy_sq_accum += P * E * E;
        }
        if (top_k_replicas > 0)
        {
            ReplicaState candidate = {P, s};

            if (top_replicas.size() < top_k_replicas)
            {
                top_replicas.push(candidate);
            }
            else if (P > top_replicas.top().probability)
            {
                top_replicas.pop();
                top_replicas.push(candidate);
            }
        }
    }

    // Normalize moments
    model_moment_1 /= Z;
    model_moment_2 /= Z;
    model_moment_3 /= Z;

    // Normalize and output energy stats
    if (compute_triplets && avg_energy && avg_energy_sq)
    {
        *avg_energy    = energy_accum / Z;
        *avg_energy_sq = energy_sq_accum / Z;
    }

    if (top_k_replicas > 0)
    {
        // sort top-k replicas
        std::vector<ReplicaState> sorted;
        while (!top_replicas.empty())
        {
            sorted.push_back(top_replicas.top());
            top_replicas.pop();
        }
        std::reverse(sorted.begin(), sorted.end()); // highest P first
        return sorted;
    }
    return {};
}
