#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

void write_g_E(const std::unordered_map<int, double> &H,
               double bin_width,
               const std::string &filename)
{
    // Copy to vector and sort by key (energy bin)
    std::vector<std::pair<int, double>> sorted_H(H.begin(), H.end());
    std::sort(sorted_H.begin(), sorted_H.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        return;
    }

    // Write header
    file << "Energy,Weight\n";

    // Write sorted data
    for (const auto &[bin, weight] : sorted_H)
        file << bin * bin_width << "," << weight << "\n";

    file.close();
    std::cout << "Saved energy density to " << filename << std::endl;
};
