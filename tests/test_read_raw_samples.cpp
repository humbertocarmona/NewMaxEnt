#include "io/read_raw_samples.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <string>

TEST(ReadRawSamplesTest, HandlesHeaderAndComments) {
    // Create a temporary CSV file with comments and a header
    std::string filename = "test_samples.csv";
    std::ofstream file(filename);
    ASSERT_TRUE(file.is_open());

    file << "spin1,spin2,spin3\n";
    file << "1,-1,1\n";
    file << "# This is a comment line\n";
    file << "-1,1,-1\n";
    file << "1,1,-1\n";
    file.close();

    arma::Mat<int> expected(3, 3);
    expected = {
        {1, -1, 1},
        {-1, 1, -1},
        {1,1,-1} };


    arma::Mat<int> result = read_raw_samples(filename);

    result.print("Parsed results:");

    ASSERT_EQ(result.n_rows, 3);
    ASSERT_EQ(result.n_cols, 3);
    ASSERT_TRUE(arma::approx_equal(result, expected, "absdiff", 0));

    std::remove(filename.c_str());
}

