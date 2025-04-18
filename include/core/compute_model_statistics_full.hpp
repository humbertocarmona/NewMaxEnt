#pragma once
#include "util/replica_state.hpp"
#include <armadillo>
#include <queue> // for std::priority_queue

std::vector<ReplicaState> compute_model_statistics_full(const int &n_spins,
                                                        const arma::Col<double> &h,
                                                        const arma::Col<double> &J,
                                                        arma::Col<double> &model_moment_1,
                                                        arma::Col<double> &mode_moment_2,
                                                        arma::Col<double> &model_moment_3,
                                                        double q,
                                                        double beta,
                                                        bool compute_triplets,
                                                        double *avg_energy    = nullptr,
                                                        double *avg_energy_sq = nullptr,
                                                        int top_k             = 0);
