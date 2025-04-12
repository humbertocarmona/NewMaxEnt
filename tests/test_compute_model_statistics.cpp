#include "core/compute_model_statistics.hpp"
#include "core/energy.hpp"
#include "util/spin_permutations_iterator.hpp"
#include <armadillo>
#include <gtest/gtest.h>

// Helper to manually compute expected moments for 4 spins, with h = {0.1, 0.2, 0.3, 0.4}, J = {0.1, 0.2, 0.3, 0.4, 0.5,
// 0.6}
TEST(ComputeModelStatisticsTest, ComputesCorrectMomentsForFourSpins)
{
    arma::Col<double> h = {0.1, 0.2, 0.3, 0.4};
    arma::Col<double> J = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}; // Upper triangle (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    arma::Col<double> moment_1, moment_2, moment_3;

    // Brute-force computation for verification
    int n = 4;
    int n_edges = n * (n - 1) / 2;
    int n_triplets = n * (n - 1) * (n - 2) / 6;
    arma::Col<double> expected_1(n, arma::fill::zeros);
    arma::Col<double> expected_2(n_edges, arma::fill::zeros);
    arma::Col<double> expected_3(n_triplets, arma::fill::zeros);

    compute_model_statistics(n, h, J, moment_1, moment_2, moment_3, /*q=*/1.0, /*beta=*/1.0);

    double Z = 0.0;

    for (auto p = SpinPermutationsSequence(n).begin(); p != SpinPermutationsSequence(n).end(); ++p)
    {
        arma::Col<int> s = *p;
        double E = energy(s, h, J, 4);
        double P = std::exp(-E);
        Z += P;

        expected_1 += P * arma::conv_to<arma::Col<double>>::from(s);

        int idx2 = 0;
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                expected_2(idx2++) += P * s(i) * s(j);

        int idx3 = 0;
        for (int i = 0; i < n - 2; ++i)
            for (int j = i + 1; j < n - 1; ++j)
                for (int k = j + 1; k < n; ++k)
                    expected_3(idx3++) += P * s(i) * s(j) * s(k);
    }

    expected_1 /= Z;
    expected_2 /= Z;
    expected_3 /= Z;

    EXPECT_TRUE(arma::approx_equal(moment_1, expected_1, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(moment_2, expected_2, "absdiff", 1e-10));
    EXPECT_TRUE(arma::approx_equal(moment_3, expected_3, "absdiff", 1e-10));
}
