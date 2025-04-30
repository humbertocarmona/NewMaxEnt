#include "trainers/base_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void BaseTrainer::updateModelParameters(size_t t)
{

    auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    // compute the gradients
    arma::Col<double> grad_h = m1_data - m1_model;
    arma::Col<double> grad_J = m2_data - m2_model;

    double norm_grad_h = arma::norm(grad_h);
    double norm_grad_J = arma::norm(grad_J);

    if (t == 0) // just to be sure
    {
        eta_h_t = params.eta_h;
        eta_J_t = params.eta_J;
    }

    if (params.adaptive_eta_h)
    {
        if (norm_grad_h > last_grad_norm_h - params.grad_drop_threshold)
        {
            eta_h_t = eta_h_t * std::exp(-params.gamma_h * norm_grad_h);
            if (eta_h_t < params.eta_h_min)
                eta_h_t = params.eta_h_min;
        }
    }
    if (params.adaptive_eta_J)
    {
        if (norm_grad_J > last_grad_norm_J - params.grad_drop_threshold)
        {
            eta_J_t = eta_J_t * std::exp(-params.gamma_J * norm_grad_J);
            if (eta_J_t < params.eta_J_min)
                eta_J_t = params.eta_J_min;
        }
    }

    last_grad_norm_h = norm_grad_h;
    last_grad_norm_J = norm_grad_J;

    for (size_t i = 0; i < core.nspins; i++)
    {
        double delta_h_t = eta_h_t * grad_h(i);
        h(i)             = h(i) + delta_h_t + params.alpha_h * delta_h(i);
        delta_h(i)       = delta_h_t;
    }

    for (size_t i = 0; i < core.nedges; i++)
    {
        double delta_J_t = eta_J_t * grad_J(i);
        J(i)             = J(i) + delta_J_t + params.alpha_J * delta_J(i);
        delta_J(i)       = delta_J_t;
    }
}