#include "core/max_ent_core.hpp"
#include <random>

void MaxEntCore::initialize_fields()
{
    if (run_parameters.gen_h0.n_elem > 0)
        h = run_parameters.gen_h0;
    else
    {
        set_h(run_parameters.gen_h_mean, run_parameters.gen_h_width);
        run_parameters.gen_h0 = h;
    }

    if (run_parameters.gen_J0.n_elem > 0)
        J = run_parameters.gen_J0;
    else
    {
        set_J(run_parameters.gen_J_mean, run_parameters.gen_J_width);
        run_parameters.gen_J0 = J;
    }
}

void MaxEntCore::set_h(double mean, double width)
{
    std::mt19937_64 gen(run_parameters.gen_seed > 0 ? run_parameters.gen_seed : std::random_device{}());
    h.set_size(n_spins);
    if (width < 1e-5)
        h.fill(mean);
    else
    {
        std::uniform_real_distribution<double> dist(mean - width / 2, mean + width / 2);
        for (int i = 0; i < n_spins; ++i)
            h[i] = dist(gen);
    }
}

void MaxEntCore::set_J(double mean, double width)
{
    std::mt19937_64 gen(run_parameters.gen_seed > 0 ? run_parameters.gen_seed : std::random_device{}());
    J.set_size(n_edges);
    if (width < 1e-5)
        J.fill(mean);
    else
    {
        std::normal_distribution<double> dist(mean, width);
        for (int i = 0; i < n_edges; ++i)
            J[i] = dist(gen);
    }
}