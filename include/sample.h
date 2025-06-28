#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <concepts>
#include <ostream>
#include <sstream>
#include <numeric>
#include <array>
#include "kernel.h"
#include "label.h"


// yet to implement 
#include "feature_map.h"

enum class Channel {red, green, blue};

// --- // Base Image—later could make child of Sample class
template<typename T>
class Image {
    public:
        const Label<T> label;
        const int flatLength, rowStride, rows;

        Image(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);
        virtual ~Image() = default; // virtual destructor for diamond polymorphism inheritance with any Training*Image class

};
// definition
template<typename T>
Image<T>::Image(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions) 
    : flatLength(sample.rows * sample.cols),
    rowStride(sample.cols),
    rows(sample.rows),
    label(filename, numObjs, positions) {}

// --- //




// --- // RGBImage
template<typename T, typename K>
class RGBImage : public Image<T> {
    private:
    std::array<std::vector<T>, 3> pixels; // pixel intensities for each channel

    public:
    const int channels;

    RGBImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);
    ~RGBImage() = default;

    const std::vector<T>& getChannel(Channel c) const { 
        switch (c) {
            case ::Channel::red:   return this->pixels[0];
            case ::Channel::green: return this->pixels[1];
            case ::Channel::blue:  return this->pixels[2];
            default: throw std::out_of_range("Invalid channel");
        }
    }

    FeatureMap<T, K> convolve(const std::array<K, 3>&& kernel); // returns convolved output for three channels

};

// definition
template<typename T, typename K>
RGBImage<T, K>::RGBImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions)
    : Image<T>(sample, filename, numObjs, positions), channels(sample.channels())
{ 
    const uchar* im = sample.data;
    std::cout << "Rows * Cols: " << this->flatLength << std::endl;
    std::cout << "Channels: " << this->channels << std::endl;

    // populate pixel vectors
    assert(this->channels == 3);

    for (int i = 0; i < this->flatLength; ++i){
        this->pixels[0].push_back(static_cast<T>(im[3 * i + 0]) / T(255));
        this->pixels[1].push_back(static_cast<T>(im[3 * i + 1]) / T(255));
        this->pixels[2].push_back(static_cast<T>(im[3 * i + 2]) / T(255));
        }

    assert(this->pixels[0].size() == this->flatLength && this->pixels[1].size() == this->flatLength && this->pixels[2].size() == this->flatLength);

    std::cout << this->label.filename << " initialised." << std::endl;
}
// --- //


// --- // Greyscale Image
template<typename T, typename K>
class GreyscaleImage : public Image<T> {
    private:
    std::vector<T> grey;

    public:
    GreyscaleImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);
    ~GreyscaleImage() = default;

    inline const std::vector<T>& getVector() const { return this->grey; } // non-mutating access

    virtual const T& operator()(int col, int row) const{ // image(Channel, Column, Row)
        assert(col < this->rowStride);
        assert(row < this->rows);
        return this->grey[row * this->rowStride + col];
    }

    FeatureMap<T, K> convolve(const K& kernel);
};

// definition: 

// important to realise, should the network evolve to select one channel for feature map generation, 
// this class is used to store that channel during inference. This way cv::Mat (colour or not) can be input
    // — if cv::Mat::channels() > 1, specify which channel for GreyscaleImage.
template<typename T, typename K>
GreyscaleImage<T, K>::GreyscaleImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions) 
    : Image<T>(sample, filename, numObjs, positions)
    {
        assert(sample.channels() == 1);
        const uchar* im = sample.data;

        for (int i = 0; i < this->flatLength; ++i){
            this->grey.push_back(static_cast<T>(im[i]) / T(255));
        }
        assert(this->grey.size() == this->flatLength);
}

// --- // TrainingImage
template<typename T>
class TrainingImage {
    public:
        const Label<T> label;
        const int flatLength, rowStride, rows;

        TrainingImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);
        virtual ~TrainingImage() = default; // virtual destructor for diamond polymorphism inheritance with any Training*Image class

};
// definition
template<typename T>
TrainingImage<T>::TrainingImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions) 
    : flatLength(sample.rows * sample.cols),
    rowStride(sample.cols),
    rows(sample.rows),
    label(filename, numObjs, positions) {}
// 


// --- // TrainingRGBImage
template<typename T, typename K>
class TrainingRGBImage : public TrainingImage<T>  {
    public:
    using TrainingImage<T>::TrainingImage;
    TrainingRGBImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);

    std::array<std::vector<FeatureMap<T, K>>, 3> convolve(std::vector<K>& kernels); // returns convolved output for three channels
    std::array<std::vector<T>, 3> pixels; // pixel intensities for each channel

    const int channels;

    virtual const T& operator()(Channel c, int col, int row) const { // instance(Channel, Column, Row) Column, Row 0-indexed
        int idx = row * this->rowStride + col;
        assert(col < this->rowStride); // 0-indexed
        assert(row < this->rows);
        assert(idx < this->flatLength);
        switch (c) {
            case ::Channel::red:   return this->pixels[0][idx];
            case ::Channel::green: return this->pixels[1][idx];
            case ::Channel::blue:  return this->pixels[2][idx];
            default: throw std::out_of_range("Invalid channel");
        }
    }

};

template<typename T, typename K>
TrainingRGBImage<T, K>::TrainingRGBImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions)
:   TrainingImage<T>(sample, filename, numObjs, positions), 
channels(sample.channels())
{
    const uchar* im = sample.data;
    std::cout << "Rows * Cols: " << this->flatLength << std::endl;
    std::cout << "Channels: " << this->channels << std::endl;

    // populate pixel vectors
    assert(this->channels == 3);

    for (int i = 0; i < this->flatLength; ++i){
        this->pixels[0].push_back(static_cast<T>(im[3 * i + 0]) / T(255));
        this->pixels[1].push_back(static_cast<T>(im[3 * i + 1]) / T(255));
        this->pixels[2].push_back(static_cast<T>(im[3 * i + 2]) / T(255));
        }

    assert(this->pixels[0].size() == this->flatLength && this->pixels[1].size() == this->flatLength && this->pixels[2].size() == this->flatLength);

    std::cout << this->label.filename << " initialised." << std::endl;
}

// --- // convolve method for TrainingRGBImage
template<typename T, typename K>
std::array<std::vector<FeatureMap<T, K>>, 3> TrainingRGBImage<T, K>::convolve(std::vector<K>& kernels) {
    assert(kernels.size() % 3 == 0); // Equal number per channel
    size_t kernelsPerChannel = kernels.size() / 3;
    std::array<std::vector<FeatureMap<T, K>>, 3> features;

    // For each channel: 0=R, 1=G, 2=B
    for (int c = 0; c < 3; ++c) {
        for (size_t k = 0; k < kernelsPerChannel; ++k) {
            K& kernel = kernels[c * kernelsPerChannel + k];
            kernel.generateWindow(this->rowStride);

            int width = kernel.getWidth();
            int height = kernel.getHeight();
            int steps = this->rowStride - width + 1;
            int rows = this->rows - height + 1;

            std::vector<T> convResult;
            for (int j = 0; j < rows; ++j) {
                int addRows = this->rowStride * j;
                for (int i = 0; i < steps; ++i) {
                    T buf = 0;
                    for (auto& pair : kernel.getWindow()) {
                        buf += pair.second * this->pixels[c][pair.first + i + addRows];
                    }
                    convResult.push_back(buf);
                }
            }
            assert(steps > 0 && rows > 0);
            features[c].push_back(FeatureMap<T, K>(convResult, steps, kernel.index)); // father kernel, layer == 0 (first convolution)
        }
    }
    return features;
}


// --- // TrainingGreyscaleImage
template<typename T, typename K>
class TrainingGreyscaleImage : public TrainingImage<T> {
    public:
    using TrainingImage<T>::TrainingImage;
    TrainingGreyscaleImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions);

    std::vector<T> grey;

    T& operator()(int col, int row) { // image(Channel, Column, Row)
        assert(col < this->rowStride);
        assert(row < this->rows);
        return this->grey[row * this->rowStride + col];
    }

    FeatureMap<T, K> convolve(const K& kernel);

};

template<typename T, typename K>
TrainingGreyscaleImage<T, K>::TrainingGreyscaleImage(const cv::Mat& sample, const std::string& filename, const int& numObjs, const std::vector<T>& positions) 
: TrainingImage<T>(sample, filename, numObjs, positions)
{
    assert(sample.channels() == 1);
    const uchar* im = sample.data;

    for (int i = 0; i < this->flatLength; ++i){
        this->grey.push_back(static_cast<T>(im[i]) / T(255));
    }
    assert(this->grey.size() == this->flatLength);

    std::cout << this->label.filename << " initialised." << std::endl;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Image<T>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n";
    return os;
}
template<typename T>
std::ostream& operator<<(std::ostream& os, const TrainingImage<T>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n";
    return os;
}
template<typename T, typename K>
std::ostream& operator<<(std::ostream& os, const RGBImage<T, K>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n"
        << "Channels: " << im.channels << "\n"
        << "Red intensities: ";
        for(const auto& idx : im.getChannel(::Channel::red)) {
            os << idx << " ";
        } 
        os << "\n"
        << "Green intensities: ";
        for(const auto& idx : im.getChannel(::Channel::green)) {
            os << idx << " ";
        } 
        os << "\n"
        << "Blue intensities: "; 
        for(const auto& idx : im.getChannel(::Channel::blue)) {
            os << idx << " ";
        } 
        os << "\n";
    return os;
}
template<typename T, typename K>
std::ostream& operator<<(std::ostream& os, const TrainingRGBImage<T, K>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n"
        << "Channels: " << im.channels << "\n"
        << "Red intensities: ";
        for(const auto& idx : im.pixels[0]) { // red channel
            os << idx << " ";
        } 
        os << "\n"
        << "Green intensities: ";
        for(const auto& idx : im.pixels[1]) {
            os << idx << " ";
        } 
        os << "\n"
        << "Blue intensities: "; 
        for(const auto& idx : im.pixels[2]) {
            os << idx << " ";
        } 
        os << "\n";
    return os;
}
template<typename T, typename K>
std::ostream& operator<<(std::ostream& os, const GreyscaleImage<T, K>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n"
        << "Pixel intensities: ";
        for(const auto& idx : im.getVector()) {
            os << idx << " ";
        } 
        os << "\n";
    return os;
}
template<typename T, typename K>
std::ostream& operator<<(std::ostream& os, const TrainingGreyscaleImage<T, K>& im){
    os << "Image: " << im.label.filename << "\n"
       << "Flat Length: " << im.flatLength << "\n"
       << "Row Stride: " << im.rowStride << "\n"
       << "Rows: " << im.rows << "\n"
       << "Label Objects: " << im.label.numObjects() << "\n"
        << "Pixel intensities: ";
        for(const auto& idx : im.grey) {
            os << idx << " ";
        } 
        os << "\n";
    return os;
}