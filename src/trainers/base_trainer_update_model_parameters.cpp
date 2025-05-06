#include "trainers/base_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void BaseTrainer::parallelUpdateModel(size_t t)
{

    auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    auto &alpha_h = params.alpha_h;
    auto &alpha_J = params.alpha_J;

    // compute the gradients
    grad_h = m1_data - m1_model;
    grad_J = m2_data - m2_model;

    double grad_norm_h = arma::norm(grad_h);
    double grad_norm_J = arma::norm(grad_J);
    // > 0.01 means large drop (keep), < 0 means increase norm (decrease)
    double delta_grad_h = std::abs(last_grad_norm_h - grad_norm_h) / last_grad_norm_h;
    double delta_grad_J = std::abs(last_grad_norm_J - grad_norm_J) / last_grad_norm_J;

    if (t % 10 == 0)
        logger->info("delta_grad_h={}", delta_grad_h);

    if (delta_grad_h < params.grad_drop_threshold) // only decrease if small drop, or negative
    {
        double fac = params.gamma_h * grad_norm_h / (1.0 + params.gamma_h * grad_norm_h);
        eta_h_t    = params.eta_h * fac;
        eta_h_t    = std::max(eta_h_t, params.eta_h_min);
    }
    if (delta_grad_J < params.grad_drop_threshold) // only decrease if small drop, or negative
    {
        double fac = params.gamma_J * grad_norm_J / (1.0 + params.gamma_J * grad_norm_J);
        eta_J_t    = params.eta_J * fac;
        eta_J_t    = std::max(eta_J_t, params.eta_J_min);
    }

    for (size_t i = 0; i < core.nspins; i++)
    {
        double delta_h_t = eta_h_t * grad_h(i);
        h(i)             = h(i) + delta_h_t + alpha_h * delta_h(i);
        delta_h(i)       = delta_h_t;
    }

    for (size_t i = 0; i < core.nedges; i++)
    {
        double delta_J_t = eta_J_t * grad_J(i);
        J(i)             = J(i) + delta_J_t + alpha_J * delta_J(i);
        delta_J(i)       = delta_J_t;
    }

    last_grad_norm_h = grad_norm_h;
    last_grad_norm_J = grad_norm_J;
}

void BaseTrainer::sequentialUpdateModel(size_t t)
{
    auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    // Compute gradients
    grad_h = m1_data - m1_model;
    grad_J = m2_data - m2_model;

    double grad_norm_h = arma::norm(grad_h);
    double grad_norm_J = arma::norm(grad_J);
    // > 0.01 means large drop (keep), < 0 means increase norm (decrease)
    double delta_grad_h = std::abs(last_grad_norm_h - grad_norm_h) / last_grad_norm_h;
    double delta_grad_J = std::abs(last_grad_norm_J - grad_norm_J) / last_grad_norm_J;

    if (t % 10 == 0)
        logger->info("delta_grad_h={}", delta_grad_h);

    if (delta_grad_h < params.grad_drop_threshold) // only decrease if small drop, or negative
    {
        double fac = params.gamma_h * grad_norm_h / (1.0 + params.gamma_h * grad_norm_h);
        eta_h_t    = params.eta_h * fac;
        eta_h_t    = std::max(eta_h_t, params.eta_h_min);
    }
    if (delta_grad_J < params.grad_drop_threshold) // only decrease if small drop, or negative
    {
        double fac = params.gamma_J * grad_norm_J / (1.0 + params.gamma_J * grad_norm_J);
        eta_J_t    = params.eta_J * fac;
        eta_J_t    = std::max(eta_J_t, params.eta_J_min);
    }

    // Sequential update for h
    arma::uword max_h_idx = arma::abs(grad_h).index_max();
    double delta_h_t      = eta_h_t * grad_h(max_h_idx);
    delta_h(max_h_idx)    = delta_h_t + params.alpha_h * delta_h(max_h_idx);
    h(max_h_idx) += delta_h(max_h_idx);

    // Sequential update for J
    arma::uword max_J_idx = arma::abs(grad_J).index_max();
    double delta_J_t      = eta_J_t * grad_J(max_J_idx);
    delta_J(max_J_idx)    = delta_J_t + params.alpha_J * delta_J(max_J_idx);
    J(max_J_idx) += delta_J(max_J_idx);

    // in the case of updating top K
    // arma::uvec top_J_indices = arma::sort_index(arma::abs(grad_J), "descend");
    // size_t K = J.size()/4;
    // for (size_t k = 0; k < K; ++k)
    // {
    //     arma::uword idx = top_J_indices(k);
    //     double delta    = eta_J_t * grad_J(idx);
    //     delta_J(idx)    = delta + params.alpha_J * delta_J(idx);
    //     J(idx) += delta_J(idx);

    // in the case of updating all
    // for (size_t i = 0; i < core.nedges; i++)
    // {
    //     double delta_J_t = eta_J_t * grad_J(i);
    //     J(i)             = J(i) + delta_J_t + params.alpha_J * delta_J(i);
    //     delta_J(i)       = delta_J_t;
    // }
    last_grad_norm_h = grad_norm_h;
    last_grad_norm_J = grad_norm_J;
}

void BaseTrainer::oldUpdateModel(size_t t)
{

    // auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    double eta_h_t = params.eta_h * std::pow(t, -params.gamma_h);
    double eta_J_t = params.eta_J * std::pow(t, -params.gamma_J);

    double delta_h_t = 0.0;
    for (size_t i = 0; i < core.nspins; i++)
    {
        delta_h_t  = eta_h_t * (m1_data(i) - m1_model(i));
        h(i)       = h(i) + delta_h_t + params.alpha_h * delta_h(i);
        delta_h(i) = delta_h_t;
    }

    double delta_J_t = 0.0;
    for (size_t i = 0; i < core.nedges; i++)
    {
        delta_J_t  = params.eta_J * (m2_data(i) - m2_model(i));
        J(i)       = J(i) + delta_J_t + params.alpha_J * delta_J(i);
        delta_J(i) = delta_J_t;
    }
}

void BaseTrainer::secantUpdateModel(size_t t)
{

    auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    auto alpha_h = params.alpha_h;
    auto alpha_J = params.alpha_J;

    arma::vec grad_h_p = grad_h; // grad_h(t-1)
    arma::vec grad_J_p = grad_J; // grad_J(t-1)

    // compute the gradients
    grad_h = m1_data - m1_model; // grad_h(t)
    grad_J = m2_data - m2_model; // grad_J(t)

    if (t == 1) // just to be sure
    {

        h_p = h; // this will be h(t=0)
        J_p = J; // this will be h(t=0)

        for (size_t i = 0; i < core.nspins; i++)
        {
            double delta_h_t = eta_h_t * grad_h(i);
            h(i)             = h(i) + delta_h_t + alpha_h * delta_h(i);
            delta_h(i)       = delta_h_t;
        }

        for (size_t i = 0; i < core.nedges; i++)
        {
            double delta_J_t = eta_J_t * grad_J(i);
            J(i)             = J(i) + delta_J_t + alpha_J * delta_J(i);
            delta_J(i)       = delta_J_t;
        }
        // now h =  h(t=1)
        // now h =  h(t=1)
    }
    else
    { // say t
        double grad_norm_h = arma::norm(grad_h);
        double grad_norm_J = arma::norm(grad_J);

        // > 0.01 means large drop (keep), < 0 means increase norm (decrease)
        double delta_grad_h = std::abs(last_grad_norm_h - grad_norm_h) / last_grad_norm_h;
        double delta_grad_J = std::abs(last_grad_norm_J - grad_norm_J) / last_grad_norm_J;

        if (t % 10 == 0)
            logger->info("delta_grad_h={}", delta_grad_h);

        if (delta_grad_h < params.grad_drop_threshold) // only decrease if small drop, or negative
        {
            double fac = params.gamma_h * grad_norm_h / (1.0 + params.gamma_h * grad_norm_h);
            eta_h_t    = params.eta_h * fac;
            eta_h_t    = std::max(eta_h_t, params.eta_h_min);
        }
        if (delta_grad_J < params.grad_drop_threshold) // only decrease if small drop, or negative
        {
            double fac = params.gamma_J * grad_norm_J / (1.0 + params.gamma_J * grad_norm_J);
            eta_J_t    = params.eta_J * fac;
            eta_J_t    = std::max(eta_J_t, params.eta_J_min);
        }

        const double epsilon = 1e-8; // or a configurable parameter

        for (size_t i = 0; i < core.nspins; i++)
        {

            double dh           = h(i) - h_p(i);           // h(t-1)-h(t-2)
            double delta_grad_h = grad_h(i) - grad_h_p(i); // grad_h(t) - grad_h(t-1)

            double denom     = (std::abs(dh) > epsilon) ? (delta_grad_h / dh) : 0.0;
            double delta_h_t = (denom != 0.0) ? eta_h_t * grad_h(i) / denom
                                              : 0.0; // grad_h(t)*delta_h/delta_grad_h

            h_p(i) = h(i); // store h(t) before updating now h_p <- h(t-1)

            h(i) = h(i) + delta_h_t +
                   alpha_h * delta_h(i); // h(t+1) = h(t) + eta_h*grad_h(t)*(h(t)-h(t-1))/(grad_h(t)
                                         // - grad_h(t-1))
            delta_h(i) = delta_h_t;
        }

        for (size_t i = 0; i < core.nedges; i++)
        {
            double dJ           = J(i) - J_p(i);           // J(t-1)-J(t-2)
            double delta_grad_J = grad_J(i) - grad_J_p(i); // grad_J(t) - grad_J(t-1)

            double denom     = (std::abs(dJ) > epsilon) ? (delta_grad_J / dJ) : 0.0;
            double delta_J_t = (denom != 0.0) ? eta_J_t * grad_J(i) / denom : 0.0;

            J_p(i) = J(i); // store h(t) before updating, now J_p <- J(t-1)

            J(i)       = J(i) + delta_J_t + alpha_J * delta_J(i);
            delta_J(i) = delta_J_t;
        }

        last_grad_norm_h = grad_norm_h;
        last_grad_norm_J = grad_norm_J;
    }
}
