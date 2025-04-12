#include "core/max_ent_core.hpp"

#include "util/logger.hpp"
#include <cassert>
#include <random>

MaxEntCore::MaxEntCore(const Params &params, bool verbose_) : run_parameters(params), verbose(verbose_)
{
    LOGGER = create_logger();
    set_console_verbosity(verbose);

    n_spins = run_parameters.gen_nspins;
    n_edges = n_spins * (n_spins - 1) / 2;

    iter = 1;
}

void MaxEntCore::set_samples(const arma::Mat<int> &input)
{
    raw_samples = input;
    n_spins = input.n_cols;
    n_edges = n_spins * (n_spins - 1) / 2;
}

const arma::Col<double> &MaxEntCore::get_h() const
{
    return h;
}
const arma::Col<double> &MaxEntCore::get_J() const
{
    return J;
}
const arma::Mat<int> &MaxEntCore::get_raw_samples() const
{
    return raw_samples;
}


