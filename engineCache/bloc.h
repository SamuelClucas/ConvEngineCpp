#pragma once

#include <vector>
#include <iostream>
#include <ostream>
#include <cmath>

#include "sample.h"
#include "kernel.h"
#include "dataset.h"
#include "feature_map.h"

template<typename T, typename I, typename K>
class bloc {
public:
    // For now, to simplify initial development

    enum LType {
        convolve,
        embedding,
        maxpool,
        upsampleFuse,
    };

    struct Layer {
        // Int indices to universal look up vectors
        int kernelStart;
        int kernelEnd;
        int mapOutStart; // Each map knows its family by indices
        int mapOutEnd;
        LType desc; // layer type descriptor
        int maxPoolWidth;
        int maxPoolHeight;
    };

    // Constructor & Destructor
    bloc(int sampleChannels, int kernelsPerChannel); // Calculates size of maps and kernels
    ~bloc() = default;

    // Core parameters
    int inputChannels_;
    int layers_;
    std::vector<int> kernelsPerLayer_;

    // Universal look up vectors
    std::vector<FeatureMap<T, K>> maps;
    std::vector<K> kernels;
    std::vector<int> mapsPerLayer_;
    std::vector<Layer> layers; // Each layer has kernel and map indices

    // Helpers
    std::vector<K> loadKernelVector(int& layer);
    std::vector<int> getMapIndices(int layer); // Returns vector of map indices for a given layer

    void storeInputConvolution(std::array<std::vector<FeatureMap<T, K>>, 3>&& convolutions);
    void storeConvolution(std::vector<FeatureMap<T, K>>& convolutions);
    void maxpoolDown(int width, int height); // handles maxpool of bloc

    void forwardPass(I& inputSample);

    void addEmbeddingLayer(int sampleChannels, int kernelsPerChannel);
    void addConvolveLayer(int kernelsInLayer);
    void addMaxpoolLayer(int width, int height);


    inline Layer& getPrevLayer() {return layers[layers.size()-1];};
    inline int countMapsOutPrevLayer(){return layers[layers.size()-1].mapOutEnd - layers[layers.size()-1].mapOutStart + 1;};
};

// === Implementation ===
template<typename T, typename I, typename K>
void bloc<T, I, K>::addMaxpoolLayer(int width, int height){
    Layer maxpool{getPrevLayer().kernelEnd, 
        getPrevLayer().kernelEnd, 
        getPrevLayer().mapOutEnd + 1, 
        getPrevLayer().mapOutEnd + 1 + countMapsOutPrevLayer() - 1,
        LType::maxpool,
        width,
        height
    };
    layers.push_back(maxpool);
    std::cout << "Layer meta: " << '\n' 
        << "Kernel start: " 
        << maxpool.kernelStart << '\n'
        << "Kernel end: " 
        << maxpool.kernelEnd << '\n'
        << "Map start: " 
        << maxpool.mapOutStart << '\n'
        << "Map end: " 
        << maxpool.mapOutEnd<< '\n';
}
template<typename T, typename I, typename K>
void bloc<T, I, K>::addConvolveLayer(int kernelsInLayer){
    Layer convolve{getPrevLayer().kernelEnd + 1, 
        getPrevLayer().kernelEnd + kernelsInLayer, // kernels in layer starts from 1, no -1 necessary 
        getPrevLayer().mapOutEnd + 1, 
        getPrevLayer().mapOutEnd + 1 + (kernelsInLayer * countMapsOutPrevLayer()) - 1,
        LType::convolve
    };
    layers.push_back(convolve);
    std::cout << "Layer meta: " << '\n' 
        << "Kernel start: " 
        << convolve.kernelStart << '\n'
        << "Kernel end: " 
        << convolve.kernelEnd << '\n'
        << "Map start: " 
        << convolve.mapOutStart << '\n'
        << "Map end: " 
        << convolve.mapOutEnd<< '\n';
}


template<typename T, typename I, typename K>
void bloc<T, I, K>::forwardPass(I& inputSample) {
    // Generate layer 1 maps from sample.
    // Each "father" kernel is indexed from input layer; maps however start from layer 1, output from first image convolution.
    // TO BE CLEAR: when accessing maps through family.mother, maps[0] == layer 1 *after* input convolution
    
    for (int l = 0; l < layers.size(); ++l){
            switch (layers[l].desc) {
                case LType::embedding: {
                    std::vector<K> ks = loadKernelVector(l);
                    storeInputConvolution(inputSample.convolve(ks));
                    layers_ += 1;
                    break;
                }
                case LType::convolve: {
                    std::vector<int> idxs = getMapIndices(l);
                    std::vector<K> ks = loadKernelVector(l);
                    for (int j = 0; j < idxs.size(); ++j){
                        std::vector<FeatureMap<T, K>> out = maps[idxs[j]].convolve(ks);
                        storeConvolution(out);
                    }
                    layers_ += 1;
                    break;
                }
                case LType::maxpool: {
                    std::vector<int> idxs = getMapIndices(l);
                    std::vector<FeatureMap<T, K>> output;
                    for (int j = 0; j < idxs.size(); ++j){
                        output.push_back(maps[idxs[j]].maxpool(layers[l].maxPoolWidth, layers[l].maxPoolHeight));
                    }
                    storeConvolution(output);
                    layers_ += 1;
                    break;
                }
                case LType::upsampleFuse: {
                    std::cout << "Not yet added..." << '\n';
                    break;
                }
                default: {
                    std::cout << "Switch case error" << std::endl;
                    break;
                }
            }
            
        }
}

template<typename T, typename I, typename K>
void bloc<T, I, K>::storeInputConvolution(std::array<std::vector<FeatureMap<T, K>>, 3>&& convolutions) {
    // Always stores Red, then Green, then Blue
    int before = maps.size();
    for (int c = 0; c < inputChannels_; ++c) {
        for (int i = 0; i < convolutions[c].size(); ++i) {
            convolutions[c][i].setIndex(maps.size()); // Sets index, used as mother when .convolve called on map
            maps.push_back(convolutions[c][i]);
        }
    }
    int after = maps.size();
    int delta = after - before;
    mapsPerLayer_.push_back(delta);

}

template<typename T, typename I, typename K>
void bloc<T, I, K>::storeConvolution(std::vector<FeatureMap<T, K>>& convolutions) {
    // FeatureMap::convolve is channel agnostic
    int before = maps.size();
    for (int i = 0; i < convolutions.size(); ++i) {
        convolutions[i].setIndex(maps.size());
        int motherIdx = convolutions[i].family.mother;
        if (motherIdx < 0 || motherIdx >= maps.size()) {
            std::cerr << "ERROR: Invalid mother index: " << motherIdx
                    << " (maps.size() = " << maps.size() << ") at convolution " << i << std::endl;
            std::abort();
        }
        maps[convolutions[i].family.mother].pushChild(convolutions[i].index);
        maps.push_back(convolutions[i]); 
    }
    int after = maps.size();
    int delta = after - before;
    mapsPerLayer_.push_back(delta);
}

template<typename T, typename I, typename K>
std::vector<int> bloc<T, I, K>::getMapIndices(int layer) {
    assert(layer > 0); // layer 0 has no previous, is special (input)
    assert(layer < layers.size());

    std::vector<int> load;
    for (int i = layers[layer-1].mapOutStart; i <= layers[layer-1].mapOutEnd; ++i) {
        load.push_back(i);
    }
    return load;
}


template<typename T, typename I, typename K>
std::vector<K> bloc<T, I, K>::loadKernelVector(int& layer) {
    assert(layer <= layers_);
    // Loads the vector of kernels for the given layer.

    std::vector<K> load;
    for (int i = layers[layer].kernelStart; i < layers[layer].kernelEnd + 1; ++i){
        K kern(3, 3, i);
        load.push_back(kern);
        kernels.push_back(kern);
    }

    return load;
}

template<typename T, typename I, typename K>
bloc<T, I, K>::bloc(int sampleChannels, int kernelsPerChannel)
    : inputChannels_(sampleChannels),
      layers_(1),
      kernelsPerLayer_()
{
    Layer embedding{0,
        sampleChannels * kernelsPerChannel - 1,
        0, 
        sampleChannels * kernelsPerChannel - 1,
        LType::embedding
    };
    kernelsPerLayer_.push_back(sampleChannels * kernelsPerChannel);
    mapsPerLayer_.push_back(sampleChannels * kernelsPerChannel);
    layers.push_back(embedding);
    std::cout << "Layer meta: " << '\n' 
        << "Kernel start: " 
        << 0 << '\n'
        << "Kernel end: " 
        << sampleChannels * kernelsPerChannel - 1 << '\n'
        << "Map start: " 
        << 0 << '\n'
        << "Map end: " 
        << sampleChannels * kernelsPerChannel - 1 << '\n';
}
