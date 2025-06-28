#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <cassert>
#include <vector>

// if using tanh activations, use Xavier kernel initialisation
class Kernel { 
    protected:
    std::array<int, 2> widthByHeight;
    std::vector<float> weights; 

    std::vector<std::pair<int, float>> window; // stores index and value as pair

    public:
    Kernel(int width, int height, int idx);

    ~Kernel() = default;
    int index;

    // Getters
    const int& getWidth() const {return widthByHeight[0];};
    const int& getHeight() const {return widthByHeight[1];};
    std::vector<float> getWeights() {return weights;};
    std::vector<std::pair<int, float>> getWindow() { 
        return window; // return index:value pairs 
    }

    void generateWindow(int rowStride); // vector of index:value pairs for convolution


};



class TrainingKernel : public Kernel { 
    public:
    std::vector<float> gradients;;

    TrainingKernel(int width, int height, int idx); 

    ~TrainingKernel() = default;
    

};



std::ostream& operator<<(std::ostream& os, Kernel& kernel);