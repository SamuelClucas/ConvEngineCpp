#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <ostream>
#include <sstream>
#include <concepts>
#include <type_traits>
#include "label.h"
#include "sample.h"

// Designed to work with Image-derived types
// I: Image type (e.g., TrainingImage, GreyscaleImage)
// T: Type for coordinates (e.g., float, double)
template<template<typename, typename> class I, typename T, typename K>
concept ImageType = 
    std::is_same_v<I<T, K>, TrainingGreyscaleImage<T, K>> ||
    std::is_same_v<I<T, K>, TrainingRGBImage<T, K>>;

template<template<typename, typename> class I, typename T, typename K>
requires ImageType<I, T, K>
class Dataset {
    private:
    std::vector<I<T, K>> samples; 

    public:
    // Constructor
    Dataset(const std::string& images_dir,
        const std::string& labels_file,
        int target_height,
        int target_width,
        int labelPositionsLength);
    
    ~Dataset() = default;

    inline const int getObjectsPresent() const { return samples.empty() ? 0 : samples[0].label.numObjects(); }
    inline int size() const { return samples.size(); } // number of samples in dataset
    inline std::vector<I<T, K>>& getSamplesVector() { return samples; } // mutable access
    inline I<T, K>& operator[](int idx) { return samples.at(idx); } 
    

};

// Overloaded << to print dataset information
template<template<typename, typename> class I, typename T, typename K>
requires ImageType<I, T, K>
std::ostream& operator<<(std::ostream& os, const Dataset<I, T, K>& dataset) {
    assert(dataset.getSamplesVector().size() > 0);
    os << "Dataset with " << dataset.getSamplesVector().size() << " samples.\n";
    int i = 0;
    for (const auto& sample : dataset.getSamplesVector()) {
        os << "Sample " << i++
           << ": " << sample.label.filename
           << " | Present: " << sample.label.numObjects()
           << " | Coordinates: ";
        for (const auto& coord : sample.label.positions) {
            os << coord << " ";
        }
        os << "\nImage size: " << sample.flatLength * sample.channels
         << " | Row stride: " << sample.rowStride
         << " | Linear length: " << sample.flatLength
         << " | Channels: " << sample.channels
         << " | Rows: " << sample.label.filename
         << std::endl;
    }
    return os;
}


template<template<typename, typename> class I, typename T, typename K>
requires ImageType<I, T, K>
Dataset<I, T, K>::Dataset(
    const std::string& images_dir,
    const std::string& labels_file,
    int target_height,
    int target_width,
    int coordsLength
)
{

    // read labels.txt
    std::ifstream infile(labels_file);
    if (!infile) {
        std::cerr << "Could not open labels file: " << labels_file << std::endl;
    }

    std::string line;
    std::string img_name;
    std::vector<T> coords(coordsLength);

    bool present;
    bool err = false;

    while (std::getline(infile, line)) { // get lines as std::string
        err = false;
        
        if (line.empty()) continue; 

        std::istringstream iss(line); // split line string
        
        if (!(iss >> img_name >> present)) {
            std::cerr << "Malformed line: " << line << std::endl;
            err = true;
            continue;
        }
        else {
            // create label
            if (present == 1) {
                T coord;
                for (int i = 0; i < coords.size(); ++i) {
                    if (!(iss >> coord)) {
                        std::cerr << "Malformed line: " << line << std::endl;
                        err = true;
                        break;
                        }
                    coords[i] = coord;
                }
            }
            else {
                std::fill(coords.begin(), coords.end(), 0);
                }
            

            // load image
            std::string full_img_path = images_dir + img_name;
            cv::Mat img = cv::imread(full_img_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Could not load image: " << full_img_path << std::endl;
                err = true;
                continue;
            }
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // app will use OpenGL texture, which uses RGB
            // downsize to model expectation

            float scale = std::min(
                static_cast<float>(target_width) / float(img.cols),
                static_cast<float>(target_height) / float(img.rows)
            );
            int new_width = std::max(1, static_cast<int>(img.cols * scale));
            int new_height = std::max(1, static_cast<int>(img.rows * scale));
            int x_offset = std::max(0, (target_width - new_width) / 2);
            int y_offset = std::max(0, (target_height - new_height) / 2);
    
            // Defensive clamp:
            if (new_width > target_width) new_width = target_width;
            if (new_height > target_height) new_height = target_height;
            if (x_offset + new_width > target_width) x_offset = target_width - new_width;
            if (y_offset + new_height > target_height) y_offset = target_height - new_height;
    
            if (new_width <= 0 || new_height <= 0 ||
                x_offset < 0 || y_offset < 0 ||
                x_offset + new_width > target_width ||
                y_offset + new_height > target_height) {
                std::cerr << "Bad crop/resize params for " << full_img_path << std::endl;
                err = true;
                continue;
            }
    
            cv::Mat tmp, resized(target_height, target_width, img.type(), cv::Scalar(0));
            cv::resize(img, tmp, cv::Size(new_width, new_height));
            resized.setTo(cv::Scalar(0));
            //std::cout << "Loaded sample from: " << full_img_path << std::endl;
            tmp.copyTo(resized(cv::Rect(x_offset, y_offset, new_width, new_height)));
            if(!err){
                I<T, K>  finalSample(resized, img_name, present, coords);
                // add to dataset
                samples.push_back(finalSample);
            }
        }
    }
    std::cout << samples.size() << " training samples loaded." << std::endl; // add() just for fun, not needed
}
