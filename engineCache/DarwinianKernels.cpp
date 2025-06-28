#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include "kernel.h"

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
    os << "Kernel Window:\n";
    bool corrupted = false;
    int indexCounter = 0;
    std::ostringstream windowStr;
    std::vector<float> weights = kernel.copy().second; // weights
    std::vector<char> structure = kernel.copy().first; // shape
    for (auto& letter : structure){
        if (letter == 'n') {
            windowStr << '\n'; // new line for next layer
        }
        else if (letter == 'v') {
            windowStr << letter << ": " << std::to_string(kernel.copy().second[indexCounter++]) << " "; // add value or absence of value
        }
        else if (letter == 'x') {
            windowStr << letter; // add value or absence of value
        }
        else {
            windowStr << '?'; // corrupted phenotype
            corrupted = true;
        }
    }
    os << windowStr.str() << "\n"
    << "Rows: " << kernel.getRows() << "\n"
       << "Row Strides:\n";
       int i = 1;
       for (auto& row : kernel.getRowStrides()) {
            os << "    Row " << i << ": " << row << "\n";
            ++i;
        }
    if (corrupted) {
        os << "Warning: Structure contains corrupted data.\n";
    }
    return os;
}

static void XavierKernelInit(std::vector<float>& weights, std::vector<char>& structure) {
    // Xavier initialization for tanh activations
    std::mt19937 gen(std::random_device{}()); // RNG
    float distRange = std::sqrt(6.0f / float(std::count(structure.begin(), structure.end(), 'v')) + 1.0f); // Xavier init for 2x2 kernel 'seeds' with 1 output
    
    std::uniform_real_distribution<float> dist(-distRange, distRange);

    std::generate(weights.begin(), weights.end(), [&]() { return dist(gen); });
}

Kernel::Kernel(float mutRate) 
    : structure{'v', 'v', 'n', 'v', 'v', 'n'}, // pattern for 2x2 kernel with next layer
    mutationRate(mutRate),
    layer(1),
    rows(2),
    rowStrides(2, 2) 
    {
        weights.resize(std::count(peek().first.begin(), peek().first.end(), 'v')); // 2x2 kernel has 4 weights
        ::XavierKernelInit(weights, structure);  

    }

void Kernel::generateWindow(int rowStride) {
    if (window.size() > 0) {
        window.clear(); 
    }
    else {
        std::cout << "\033[1;31mGenerating Index:Value pairs in window.\033[0m\n";
    }
    int rowMultiplier = 0;
    int rowOffset = 0;
    int indexCounter = 0;   
    for (auto& letter : peek().first) { // char vector
        if (letter == 'n') {
            rowMultiplier++;
            rowOffset = rowMultiplier * rowStride; // increment row for next layer of weights
        }
        else if (letter == 'v') {
            window.push_back(std::make_pair(indexCounter + rowOffset, peek().second[indexCounter])); // store index and value as pair in window
            indexCounter++;
        }
        else if (letter == 'x') {
            window.push_back(std::make_pair(indexCounter++ + rowOffset, 0.0f)); // store index and value as pair in indexValue
        }
        else {
            throw std::runtime_error("Corrupted structure in kernel.");
        }
    }
    for (auto& pair : window) {
        std::cout << "Index: " << pair.first << '\n' << "Value: " << pair.second << '\n'; 
    }

    std::cout << "\033[1;32mWindow generated.\n\033[0m";
}

TrainingKernel::TrainingKernel(float mutRate) 
    : Kernel(mutRate), gradients(0), migrationHistory(1, 1),
      health(1.0f), fitness(1.0f), contentment(1.0f)

    {
        // def mutate() first

    }


