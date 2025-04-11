#include "core/MaxEntCore.hpp"

#include "util/logger.hpp"
#include <cassert>
#include <random>

MaxEntCore::MaxEntCore(const Params &params, bool verbose_) : par(params), verbose(verbose_)
{
    LOGGER = get_logger();
    set_console_verbosity(verbose);

    nspins = par.gen_nspins;
    nedges = nspins * (nspins - 1) / 2;
    iter   = 1;

    initialize_network();
    initialize_from_params();
}

void MaxEntCore::initialize_from_params()
{
    if (par.gen_h0.n_elem > 0)
        h = par.gen_h0;
    else
    {
        set_h(par.gen_h_mean, par.gen_h_width);
        par.gen_h0 = h;
    }

    if (par.gen_J0.n_elem > 0)
        J = par.gen_J0;
    else
    {
        set_J(par.gen_J_mean, par.gen_J_width);
        par.gen_J0 = J;
    }
}

void MaxEntCore::initialize_network()
{
    edge.resize(nedges);
    edge_index.set_size(nspins, nspins);
    edge_index.fill(-1);
    int idx = 0;
    for (int i = 0; i < nspins - 1; ++i)
    {
        for (int j = i + 1; j < nspins; ++j)
        {
            edge[idx]        = Edge(i, j);
            edge_index(i, j) = idx;
            edge_index(j, i) = idx;
            ++idx;
        }
    }
}

void MaxEntCore::set_samples(const arma::Mat<int> &input)
{
    samples = input;
    nspins  = input.n_cols;
    nedges  = nspins * (nspins - 1) / 2;
}

void MaxEntCore::set_h(double mean, double width)
{
    std::mt19937_64 gen(par.gen_seed > 0 ? par.gen_seed : std::random_device{}());
    h.set_size(nspins);
    if (width < 1e-5)
        h.fill(mean);
    else
    {
        std::uniform_real_distribution<double> dist(mean - width / 2, mean + width / 2);
        for (int i = 0; i < nspins; ++i)
            h[i] = dist(gen);
    }
}

void MaxEntCore::set_J(double mean, double width)
{
    std::mt19937_64 gen(par.gen_seed > 0 ? par.gen_seed : std::random_device{}());
    J.set_size(nedges);
    if (width < 1e-5)
        J.fill(mean);
    else
    {
        std::normal_distribution<double> dist(mean, width);
        for (int i = 0; i < nedges; ++i)
            J[i] = dist(gen);
    }
}

const arma::Col<double> &MaxEntCore::get_h() const
{
    return h;
}
const arma::Col<double> &MaxEntCore::get_J() const
{
    return J;
}
const arma::Mat<int> &MaxEntCore::get_samples() const
{
    return samples;
}

void MaxEntCore::relax(double q)
{
    LOGGER->info("[MaxEntCore] relax() with q = {} [stub]", q);
    // TODO: Call entropy model to compute ensemble means
    // TODO: Update h, J to match sample moments
    // Placeholder implementation
}