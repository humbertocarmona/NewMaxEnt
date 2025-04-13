#pragma once

#include <armadillo>
#include <optional>

struct CenteredMoments
{
    arma::Mat<double> correlation_matrix_2;                 // ⟨(x_i- ⟨x_i⟩) (x_j- ⟨x_i⟩)⟩ 
    std::optional<arma::Col<double>> centered_moment_3;     // ⟨(x_i- ⟨x_i⟩) (x_j- ⟨x_j) (x_k)- ⟨x_k⟩⟩

    double get_variance(int i) const
    {
        return correlation_matrix_2(i, i);
    }

    double get_covariance(int i, int j) const
    {
        return correlation_matrix_2(i, j);
    }

    double get_centered_triplet(int idx) const
    {
            return centered_moment_3.value()(idx);
    }
};

/**
 * Computes centered second- and (optionally) third-order moments.
 * @param moment_1    Vector of first-order model means ⟨x_i⟩
 * @param moment_2    Vectorized ⟨x_i x_j⟩ in row-major [i * n + j]
 * @param moment_3    Optional ⟨x_i x_j x_k⟩ with i < j < k (natural triplet index order)
 */
CenteredMoments compute_centered_moments(
    const arma::Col<double>& moment_1,
    const arma::Col<double>& moment_2,
    const arma::Col<double>& moment_3
);
