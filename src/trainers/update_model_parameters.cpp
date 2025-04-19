#include "trainers/full_ensemble_trainer.hpp"
#include "utils/get_logger.hpp"
#include "utils/utilities.hpp"

void FullEnsembleTrainer::updateModelParameters(size_t t)
{

    // auto logger = getLogger();

    auto &h = core.h;
    auto &J = core.J;

    double eta_h_t = eta_h * std::pow(t, -gamma_h);
    double eta_J_t = eta_J * std::pow(t, -gamma_J);

    
    double delta_h_t = 0.0;
    for (size_t i = 0; i < core.nspins; i++)
    {
        delta_h_t  = eta_h_t * (m1_data(i) - m1_model(i));
        h(i)       = h(i) + delta_h_t + alpha_h * delta_h(i);
        delta_h(i) = delta_h_t;
    }


    double delta_J_t = 0.0;
    for (size_t i = 0; i < core.nedges; i++)
    {
        delta_J_t  = eta_J * (m2_data(i) - m2_model(i));
        J(i)       = J(i) + delta_J_t + alpha_J * delta_J(i);
        delta_J(i) = delta_J_t;
    }


}