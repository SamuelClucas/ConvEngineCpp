#pragma once

#include <random>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <array>

// --- Detection result: stores output from predict() ---
struct DetectionResult {
    bool objectPresent = false;
    std::vector<float> vertices;   // [x1,y1,x2,y2,x3,y3,x4,y4], normalized
    float confidence = 0.0f;
};

class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector() = default;

    // ---- Inference API ----
    DetectionResult predict(const cv::Mat& input) const;
    void drawPredictedPolygon(cv::Mat& image,
                              const std::vector<float>& norm_coords,
                              const cv::Scalar& color = cv::Scalar(0,255,0)) const;

    // ---- Model Saving/Loading ----
    void loadModel(const std::string& path);
    void saveModel(const std::string& path) const;

    // ---- Feature Extraction ----
    // (All main convolution layers now use vectors of feature maps!)
    void extractFeatures(const cv::Mat& input, std::vector<float>& features) const;
    void downsizeInput(const cv::Mat& input, cv::Mat& output) const;
    void flattenFeatureMap(const cv::Mat& input, std::vector<float>& flattened) const;
    void conv9maxPool(const cv::Mat& input, std::vector<cv::Mat>& outputs) const;
    void conv25(const std::vector<cv::Mat>& inputs, std::vector<cv::Mat>& outputs) const;
    void conv4(const std::vector<cv::Mat>& inputs, std::vector<cv::Mat>& outputs) const;
    void addAndMedian(const std::vector<cv::Mat>& inputs, cv::Mat& output) const;
    void upscale(const cv::Mat& input, cv::Mat& output, int target_rows, int target_cols) const;
    void addResidual(const cv::Mat& input, cv::Mat& output, float alpha = 1.0f) const;

    // ---- Fully Connected Layers ----
    struct HiddenLayer {
        std::vector<std::vector<float>> weights;  // [output_dim][input_dim]
        std::vector<float> biases;                // [output_dim]
    };

    std::vector<HiddenLayer> hidden_layers;

    std::vector<float> fc_weights_confidence;     // [last_hidden_dim]
    float fc_bias_confidence = 0.0f;
    std::vector<std::vector<float>> fc_weights_bbox; // [8][last_hidden_dim]
    std::vector<float> fc_bias_bbox;              // [8]

    void initializeFullyConnected(int fc_input_dim, int n_hidden = 8);
    void forwardFC(const std::vector<float>& input,
                   bool& object_present,
                   std::vector<float>& bbox_coords,
                   float& confidence) const;

    // ---- Input Shape ----
    int getInputHeight() const { return input_height; }
    int getInputWidth()  const { return input_width; }
    // In object_detector.h, public section:
    int numKernels() const { return static_cast<int>(kernels.size()); }


private:
    // ---- Hyperparameters ----
    static constexpr int input_height = 300;
    static constexpr int input_width  = 300;

    // ---- Convolution kernels ----
    struct Kernel {
        std::array<std::array<float, 9>, 3> matrix9;    // 3x3x3 for RGB conv
        std::array<float, 4> matrix4_single;            // 2x2 for single-channel
        std::array<float, 25> matrix25_single;          // 5x5 for single-channel
    };
    static constexpr int num_kernels = 16;  // Static so can be used in struct/class
    std::vector<Kernel> kernels;            // Will be resized in constructor

    std::vector<Kernel> kernel_grads; // Same shape as kernels


    // ---- Weight initialization state ----
    mutable bool fc_initialized = false;
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<float> dist2;

    // ---- Math helpers ----
    float dot(const std::vector<float>& a, const std::vector<float>& b) const;
    void matvec(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec,
                const std::vector<float>& bias, std::vector<float>& out) const;
};
