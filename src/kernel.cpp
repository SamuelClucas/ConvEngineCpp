#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include "kernel.h"

std::ostream& operator<<(std::ostream& os, Kernel& kernel) {
    os << "Kernel Window:\n";

    std::ostringstream windowStr;
    assert(kernel.getWindow().size() > 0);
    for (int i = 0; i < kernel.getHeight() * kernel.getWidth(); ++i){
        windowStr << "Index: " << kernel.getWindow()[i].first << '\n'
        << "Weight: " << kernel.getWindow()[i].second << '\n';
    }

    os << windowStr.str();
    return os;
}

static void XavierKernelInit(std::vector<float>& weights, std::array<int, 2>& widthByHeight) {
    // Xavier initialization for tanh activations
    std::mt19937 gen(std::random_device{}()); // RNG
    float distRange = std::sqrt(6.0f / float(widthByHeight[0] * widthByHeight[1]) + 1.0f); // Xavier init for width*height inputs into 1 output
    
    std::uniform_real_distribution<float> dist(-distRange, distRange);
    weights.resize(widthByHeight[0] * widthByHeight[1]);
    std::generate(weights.begin(), weights.end(), [&]() { return dist(gen); });
}

Kernel::Kernel(int width, int height, int idx) 
    : widthByHeight{width, height} , index(idx)
    {
        ::XavierKernelInit(weights, widthByHeight);  

    }

void Kernel::generateWindow(int rowStride) {
    if (window.size() > 0) {
        window.clear(); 
    }
    int rowOffset = 0;
    for ( int i = 0; i < this->getHeight(); ++i){
        rowOffset = i * rowStride;
        for (int j = 0; j < this->getWidth(); ++j){
            window.push_back(std::make_pair(j + rowOffset, this->weights[j + i]));
        }
    }
}

TrainingKernel::TrainingKernel(int width, int height, int idx) 
    : Kernel(width, height, idx), gradients(0)
    {
    }




