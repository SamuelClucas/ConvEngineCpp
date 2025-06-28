#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <cassert>
#include <vector>

// if using tanh activations, use Xavier kernel initialisation
class Kernel { // less mutable members, more const
    protected:
    std::vector<char> structure; // determines kernel shape: v for value present; x for not present; n for next layer, row major order.
    std::vector<float> weights; // kernel weights
    int layer; 
    int rows;
    std::vector<int> rowStrides;

    std::vector<std::pair<int, float>> window; // stores index and value as pair
   
    const float mutationRate; // default mutation rate for Darwinian evolution

    public:
    Kernel(float mutationRate = 1.0f);

    ~Kernel() = default;

    std::vector<int> getRowStrides() const {return rowStrides;};
    int getRows() const {return rows;};

    // transcribe DNA into mRNA (indices for each step in convolution)
    void generateWindow(int rowStride);
    
    // getters 
    std::vector<std::pair<int, float>> getWindow() { 
        return window; // return transcription as vector of pairs
    }

    std::pair<const std::vector<char>&, const std::vector<float>&> peek() const { // return const reference (no copies)
        return std::pair<const std::vector<char>&, const std::vector<float>&>(
            static_cast<const std::vector<char>&>(structure),
            static_cast<const std::vector<float>&>(weights)
        );
    }

    std::pair<std::vector<char>, std::vector<float>> copy() const { // return as copy
        return std::pair<std::vector<char>, std::vector<float>>(structure, weights);
    }

    std::pair<std::vector<char>&, std::vector<float>&> get() { // return mutable reference
        return std::pair<std::vector<char>&, std::vector<float>&>(structure, weights);
    }

    void step(); 

};



class TrainingKernel : public Kernel { // more mutable members for training, but also for Darwinian evolution
    public:
    std::vector<float> gradients;;

    struct Mutations {
        int locus; // index in genome
        char from; // what was there before
        char to; // what is there now
        float lossBefore;
        float lossAfter; // loss before and after mutation
    };
    
    std::vector<Mutations> mutationHistory; // sequence of mutations applied to this kernel
    std::vector<int> migrationHistory; // sequence of layers this kernel has been in

    float health; // loss integral averaged over the kernel
    float fitness; // current loss value
    float contentment; // ratio of health to fitness, or just loss derivative itself

    TrainingKernel(float mutationRate = 1.0f); // define new constructor for mutable members

    ~TrainingKernel() = default;

    void migrate() {
        // move to another layer if health or contentment too low
    }

    void mutate();
    

};



std::ostream& operator<<(std::ostream& os, const Kernel& kernel);