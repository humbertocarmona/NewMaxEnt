#pragma once

#include <armadillo>

/**
 * Structure with vectorized centered moments m2 and m3
 */
struct CenteredMoments
{
    arma::Col<double> centered_moment_2; // ⟨(x_i- ⟨x_i⟩) (x_j- ⟨x_i⟩)⟩
    arma::Col<double> centered_moment_3; // ⟨(x_i- ⟨x_i⟩) (x_j- ⟨x_j) (x_k)- ⟨x_k⟩⟩
};

/**
 * Computes centered second- and (optionally) third-order moments.
 * @param moment_1    Vector of first-order model means ⟨x_i⟩
 * @param moment_2    Vectorized ⟨x_i x_j⟩ in row-major [i * n + j]
 * @param moment_3    Optional ⟨x_i x_j x_k⟩ with i < j < k (natural triplet index order)
 */
CenteredMoments computeCenteredMoments(const arma::Col<double> &moment_1,
                                       const arma::Col<double> &moment_2,
                                       const arma::Col<double> &moment_3);
