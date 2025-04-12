#include "core/max_ent_core.hpp"

void MaxEntCore::update_model_parameters()
{
    double eta_h = run_parameters.eta_h * std::pow(iter, -run_parameters.gamma_h);
    double eta_J = run_parameters.eta_J * std::pow(iter, -run_parameters.gamma_J);

    double delta_h = 0.0;
    double delta_J = 0.0;

    for (size_t i = 0; i < n_spins; i++)
    {
        delta_h = eta_h * (sample_moment_1(i) - model_moment_1(i));
        h(i) = h(i) + delta_h + run_parameters.alpha * momentum_m_1(i);
        momentum_m_1(i) = delta_h;
    }

    for (size_t i = 0; i < n_edges; i++)
    {
        delta_J = eta_J * (sample_moment_2(i) - model_moment_2(i));
        J(i) = J(i) + delta_J + run_parameters.alpha * momentum_m_2(i);
        momentum_m_2(i) = delta_J;
    }
}