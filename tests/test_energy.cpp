#include "core/energy.hpp"
#include <armadillo>
#include <gtest/gtest.h>

TEST(EnergyTest, ComputesCorrectEnergy)
{
    // 3 spins, full connectivity: 3 edges (0,1), (0,2), (1,2)
    arma::Col<double> h = {0.5, -0.3, 0.1};
    arma::Col<double> J = {1.0, 0.5, -0.8}; // edge indices: (0,1)=0, (0,2)=1, (1,2)=2
    arma::Col<int> spin = {1, -1, 1};

    // Expected:
    // h·s = 0.5*1 + (-0.3)*(-1) + 0.1*1 = 0.5 + 0.3 + 0.1 = 0.9
    // J·(s_i * s_j) = 1.0*(1*-1) + 0.5*(1*1) + (-0.8)*(-1*1) = -1 + 0.5 + 0.8 = 0.3
    // Energy = - (0.9 + 0.3) = -1.2

    double E = energy(spin, h, J, 3);
    EXPECT_NEAR(E, -1.2, 1e-10);
}
