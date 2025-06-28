#include <iostream>
#include <string>
#include "dataset.h"
#include "label.h"
#include "sample.h"
#include "kernel.h"
#include "feature_map.h"
#include "bloc.h"

int main() {
    std::string images_dir("lib/samples/");
    std::string labels_file("lib/labels.txt");
    int target_height = 300;
    int target_width = 300;
    int labelPositionsLength = 8;

    Dataset<TrainingRGBImage, float, TrainingKernel> dataset(images_dir, 
        labels_file,
        target_height,
        target_width,
        labelPositionsLength);


    bloc<float, TrainingRGBImage<float, TrainingKernel>, TrainingKernel> bloc(3, 3); // input channels, kernels per channel for initial convolve


    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addMaxpoolLayer(2, 2);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addMaxpoolLayer(2, 2);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);
    bloc.addConvolveLayer(3);

    std::cout << "Sample getter check: " << '\n' << dataset.getSamplesVector()[0] << std::endl;

    bloc.forwardPass(dataset.getSamplesVector()[0]);
    return 0;
}

