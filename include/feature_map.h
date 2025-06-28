#pragma once

#include <cassert>
#include <ostream>
#include <vector>
#include <math.h>
#include "sample.h"
#include "kernel.h"

template<typename T, typename K> // floating point number T, Kernel K
class FeatureMap { 
    public:
        struct Family { // stores indices as ints (see look-up vectors in bloc.h)
            int mother; // parent feature map
            int father; // kernel that convolved mother
            std::vector<int> partners; // kernels in layer
            std::vector<int> children; // children convolutions
        };

        Family family;

        const std::vector<T> data; // convolved output data
        const int flatLength;
        const int rowStride;
        const int rows;
        int index;

        FeatureMap(std::vector<T>& convolution, int width, int father); // father = -1 for maxpool layers

        ~FeatureMap() = default;

        inline const float& operator()(int col, int row) const { // map(column, row)
            assert(col < rowStride);
            assert(row < flatLength / rowStride);
            return data[row * rowStride + col];
        }

        std::vector<FeatureMap<T, K>> convolve(std::vector<K>& kernels);
        FeatureMap<T, K> upsample(const FeatureMap& target);
        FeatureMap<T, K> fuse(const FeatureMap& second);
        FeatureMap<T, K> maxpool(int width, int height);

        inline void setMother(int mother) {family.mother = mother;};
        inline void setFather(int father) {family.father = father;};
        inline void pushChild(int childidx) {family.children.push_back(childidx);};
        inline void setIndex(int idx){index=idx;};
};

template<typename T, typename K>
FeatureMap<T, K>::FeatureMap(std::vector<T>& convolution, int width, int fatherIndex)
    : data(convolution), 
    flatLength(convolution.size()),
    rowStride(width),
    rows(flatLength/rowStride)
    
    {
    assert(flatLength > 0);
    setFather(fatherIndex);
    }

template<typename T, typename K> // T floating point, K kernel
std::vector<FeatureMap<T, K>> FeatureMap<T, K>::convolve(std::vector<K>& kernels){
    std::vector<FeatureMap<T, K>> features;
    int steps = 0;
    int width = 0;
    int height = 0;
    T buf = 0.0f;
    int addRows = 0;
    for (size_t k = 0; k < kernels.size(); ++k) { // could use range based for loop but eh
        K& kernel = kernels[k];
        kernel.generateWindow(this->rowStride); 
        std::vector<T> convResult;
        width = kernel.getWidth();
        steps = this->rowStride - width + 1;
        height = this->rows - kernel.getHeight() + 1;
        
        for (int j = 0; j < height; ++j){
            addRows = this->rowStride * j;

            for (int i = 0; i < steps; ++i){
                for (auto& pair : kernel.getWindow()) { // iterate over index:value pairs in window
                    buf += pair.second * this->data[pair.first + i + addRows];
                }
                    convResult.push_back(buf); 
                    buf = 0;
                }
            }
            assert(convResult.size() == steps * height);
            features.push_back(FeatureMap<T, K>(convResult, steps, kernel.index)); // sets father, layer—indexing happens in bloc.h pushback into maps
            features[features.size()-1].setMother(this->index);
            this->family.partners.push_back(kernel.index); // children are set in bloc.h in maps.push_back
    }

    return features;
}

template<typename T, typename K>
FeatureMap<T, K> FeatureMap<T, K>::upsample(const FeatureMap<T, K>& target){

    std::vector<float> output(target.rows * target.rowStride);

    for (int i = 0; i < target.rows; ++i){
    for (int j = 0; j < target.rowStride; ++j){
        float input_y = i * (static_cast<float>(this->rows) / target.rows);
        float input_x = j * (static_cast<float>(this->rowStride) / target.rowStride);

        int y0 = static_cast<int>(std::floor(input_y));
        int y1 = std::min(y0 + 1, this->rows - 1);
        int x0 = static_cast<int>(std::floor(input_x));
        int x1 = std::min(x0 + 1, this->rowStride - 1);

        float wy = input_y - y0;
        float wx = input_x - x0;

        float v00 = data[y0 * this->rowStride + x0];
        float v01 = data[y0 * this->rowStride + x1];
        float v10 = data[y1 * this->rowStride + x0];
        float v11 = data[y1 * this->rowStride + x1];

        float top = v00 * (1 - wx) + v01 * wx;
        float bottom = v10 * (1 - wx) + v11 * wx;
        float newpixel = top * (1 - wy) + bottom * wy;

        output[i * target.rowStride + j] = newpixel;
    }
    }
    assert(output.size() == target.flatLength);
    FeatureMap<T, K> FM(output, target.rowStride);
    return FM;
}

template<typename T, typename K>
FeatureMap<T, K> FeatureMap<T, K>::fuse(const FeatureMap<T, K>& second){
    assert(second.data.size() == this->data.size());
    std::vector<T> output;
    for (int i = 0; i < this->data.size(); ++i){
        float mean = (this->data[i] + second.data[i]) / 2.0f;
        std::cout << mean << ", ";
        output.push_back(mean);
    }
    FeatureMap<T, K> fusion(output, this->rowStride);
    return fusion;
}

template<typename T, typename K>
FeatureMap<T, K> FeatureMap<T, K>::maxpool(int width, int height) { 
    // Use vector instead of array for dynamic sizing
    std::vector<int> window(width * height);
    
    // Populate window offsets (relative positions)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            window[i * width + j] = i * this->rowStride + j;
        }
    }

    std::vector<T> output;
    std::vector<T> buf; // Changed to type T and proper initialization
    
    int stepsAcross = this->rowStride - width + 1;
    int stepsDown = this->rows - height + 1;

    for (int j = 0; j < stepsDown; ++j) {
        int baseIndex = j * this->rowStride; // Starting position for this row
        for (int i = 0; i < stepsAcross; ++i) {
            buf.clear(); // Clear buffer for each pooling operation
            int startPos = baseIndex + i;
            
            // Collect values in the pooling window
            for (int k = 0; k < window.size(); ++k) {
                buf.push_back(this->data[startPos + window[k]]);
            }
            
            // Get the maximum value (dereference the iterator)
            output.push_back(*std::max_element(buf.begin(), buf.end()));
        }
    }
    
    // Fixed assertion
    assert(output.size() == stepsAcross * stepsDown);
    
    FeatureMap<T, K> FM(output, stepsAcross, -1); // maxpool layers get -1 as father, not kernel.index
    FM.setMother(this->index);
    return FM;
}