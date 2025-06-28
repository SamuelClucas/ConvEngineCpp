#include "object_detector.h"
#include "dataset_loader.cpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <tuple>

// Hyperparameters
constexpr int BATCH_SIZE = 8;
constexpr int EPOCHS = 40;
constexpr float LR_FC = 0.0008f;
constexpr float LR_KERN = 0.00008f;
constexpr float CONF_THRESH = 0.75f;

struct ForwardCache {
    // Store all intermediate values for backprop
    std::vector<std::vector<cv::Mat>> conv_maps;  // [kernel][feature map]
    std::vector<std::vector<cv::Mat>> pooled_maps;
    std::vector<float> flattened;
};

struct Gradients {
    std::vector<ObjectDetector::Kernel> dKernels;
    std::vector<ObjectDetector::HiddenLayer> dHiddenLayers;
    std::vector<float> dFCWeightsConf;
    float dFCBiasConf = 0.0f;
    std::vector<std::vector<float>> dFCWeightsBBox;
    std::vector<float> dFCBiasBBox;

    void clear(const ObjectDetector& detector) {
        dKernels = std::vector<ObjectDetector::Kernel>(detector.kernels.size());
        dHiddenLayers = detector.hidden_layers;
        for (auto& l : dHiddenLayers) {
            for (auto& w : l.weights) std::fill(w.begin(), w.end(), 0.0f);
            std::fill(l.biases.begin(), l.biases.end(), 0.0f);
        }
        dFCWeightsConf.assign(detector.fc_weights_confidence.size(), 0.0f);
        dFCBiasConf = 0.0f;
        dFCWeightsBBox = detector.fc_weights_bbox;
        for (auto& w : dFCWeightsBBox) std::fill(w.begin(), w.end(), 0.0f);
        dFCBiasBBox.assign(detector.fc_bias_bbox.size(), 0.0f);
    }
};

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float sigmoid_deriv(float x) { float s = sigmoid(x); return s * (1 - s); }
inline float clamp(float v, float mn, float mx) { return std::max(mn, std::min(mx, v)); }

float binary_cross_entropy(float pred, float target) {
    pred = clamp(pred, 1e-6f, 1.f - 1e-6f);
    return -target * std::log(pred) - (1 - target) * std::log(1 - pred);
}
float mse_loss(float pred, float target) { return 0.5f * (pred - target) * (pred - target); }
float mse_grad(float pred, float target) { return pred - target; }

// ---- Feature Extraction with forward cache ----
void extractFeaturesWithCache(const ObjectDetector& model, const cv::Mat& input, std::vector<float>& features, ForwardCache& cache) {
    // 1. Downsize
    cv::Mat downsized(ObjectDetector::input_height, ObjectDetector::input_width, input.type(), cv::Scalar(0));
    model.downsizeInput(input, downsized);

    // 2. Convolution + Maxpool (store both pre- and post-pool maps for backprop)
    int N = model.kernels.size();
    int h = downsized.rows, w = downsized.cols, pad = 1;
    std::vector<cv::Mat> conv_maps(N), pooled_maps(N);
    for (int k = 0; k < N; ++k) {
        conv_maps[k] = cv::Mat::zeros(h, w, CV_32FC1);
        for (int y = pad; y < h - pad; ++y) {
            for (int x = pad; x < w - pad; ++x) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ki = (ky + 1) * 3 + (kx + 1);
                        cv::Vec3b pixel = downsized.at<cv::Vec3b>(y + ky, x + kx);
                        for (int c = 0; c < 3; ++c)
                            sum += pixel[c] * model.kernels[k].matrix9[c][ki];
                    }
                }
                conv_maps[k].at<float>(y, x) = sum;
            }
        }
    }
    // Maxpool (2x2) for each kernel
    int outRows = h / 2, outCols = w / 2;
    for (int k = 0; k < N; ++k) {
        pooled_maps[k] = cv::Mat::zeros(outRows, outCols, CV_32FC1);
        for (int y = 0; y < outRows; ++y) {
            for (int x = 0; x < outCols; ++x) {
                int sy = y * 2, sx = x * 2;
                float m = -1e9f;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        m = std::max(m, conv_maps[k].at<float>(sy + dy, sx + dx));
                pooled_maps[k].at<float>(y, x) = m;
            }
        }
    }
    // Store for backward
    cache.conv_maps = { conv_maps };
    cache.pooled_maps = { pooled_maps };

    // 3. Flatten all pooled maps into one big feature vector
    features.clear();
    for (int k = 0; k < N; ++k) {
        for (int y = 0; y < pooled_maps[k].rows; ++y) {
            for (int x = 0; x < pooled_maps[k].cols; ++x) {
                features.push_back(pooled_maps[k].at<float>(y, x));
            }
        }
    }
    cache.flattened = features;
}

// ---- Backward Pass: FC + Conv9 kernels ----
void forward_backward(
    ObjectDetector& model,
    const cv::Mat& input,
    const Label& label,
    Gradients& grads,
    float& total_conf_loss,
    float& total_bbox_loss
) {
    // ---- Forward pass with cache ----
    std::vector<float> features;
    ForwardCache cache;
    extractFeaturesWithCache(model, input, features, cache);

    // --- FC Forward (identical to before) ---
    std::vector<std::vector<float>> layer_inputs;
    std::vector<float> cur = features, cur_raw;
    layer_inputs.push_back(cur);

    std::vector<std::vector<float>> layer_z;
    for (const auto& layer : model.hidden_layers) {
        cur_raw.resize(layer.biases.size());
        std::vector<float> next(layer.biases.size());
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            float sum = layer.biases[i];
            for (size_t j = 0; j < cur.size(); ++j)
                sum += layer.weights[i][j] * cur[j];
            cur_raw[i] = sum;
            next[i] = std::max(0.0f, sum); // ReLU
        }
        layer_z.push_back(cur_raw);
        cur = std::move(next);
        layer_inputs.push_back(cur);
    }

    // Heads
    float conf_raw = model.fc_bias_confidence;
    for (size_t i = 0; i < model.fc_weights_confidence.size(); ++i)
        conf_raw += model.fc_weights_confidence[i] * cur[i];
    float conf_pred = sigmoid(conf_raw);

    std::vector<float> bbox_raw(8, 0.0f), bbox_pred(8, 0.0f);
    for (int i = 0; i < 8; ++i) {
        float sum = model.fc_bias_bbox[i];
        for (size_t j = 0; j < model.fc_weights_bbox[i].size(); ++j)
            sum += model.fc_weights_bbox[i][j] * cur[j];
        bbox_raw[i] = sum;
        bbox_pred[i] = sigmoid(sum);
    }

    float conf_target = static_cast<float>(label.present);
    float conf_loss = binary_cross_entropy(conf_pred, conf_target);
    float bbox_loss = 0.0f;
    if (label.present && label.box.size() == 8) {
        for (int i = 0; i < 8; ++i)
            bbox_loss += mse_loss(bbox_pred[i], label.box[i]);
    }
    total_conf_loss += conf_loss;
    total_bbox_loss += bbox_loss;

    // --- FC Backward ---
    float d_conf = (conf_pred - conf_target);
    grads.dFCBiasConf += d_conf;
    for (size_t i = 0; i < model.fc_weights_confidence.size(); ++i)
        grads.dFCWeightsConf[i] += d_conf * cur[i];

    std::vector<float> d_cur_conf(cur.size(), 0.0f);
    for (size_t i = 0; i < model.fc_weights_confidence.size(); ++i)
        d_cur_conf[i] += d_conf * model.fc_weights_confidence[i];

    std::vector<float> d_cur_bbox(cur.size(), 0.0f);
    for (int i = 0; i < 8; ++i) {
        float d_bbox = 0.0f;
        if (label.present && label.box.size() == 8)
            d_bbox = mse_grad(bbox_pred[i], label.box[i]) * sigmoid_deriv(bbox_raw[i]);
        grads.dFCBiasBBox[i] += d_bbox;
        for (size_t j = 0; j < model.fc_weights_bbox[i].size(); ++j) {
            grads.dFCWeightsBBox[i][j] += d_bbox * cur[j];
            d_cur_bbox[j] += d_bbox * model.fc_weights_bbox[i][j];
        }
    }

    // Backprop FC → hidden layers
    std::vector<float> d_cur = d_cur_conf;
    for (size_t i = 0; i < d_cur.size(); ++i) d_cur[i] += d_cur_bbox[i];

    for (int l = (int)model.hidden_layers.size() - 1; l >= 0; --l) {
        const auto& z = layer_z[l];
        std::vector<float> d_pre(z.size());
        for (size_t i = 0; i < z.size(); ++i)
            d_pre[i] = (z[i] > 0 ? 1.0f : 0.0f) * d_cur[i]; // ReLU grad
        auto& dW = grads.dHiddenLayers[l].weights;
        auto& dB = grads.dHiddenLayers[l].biases;
        const auto& prev_input = layer_inputs[l];
        for (size_t i = 0; i < dW.size(); ++i) {
            dB[i] += d_pre[i];
            for (size_t j = 0; j < dW[i].size(); ++j)
                dW[i][j] += d_pre[i] * prev_input[j];
        }
        std::vector<float> d_next(dW[0].size(), 0.0f);
        for (size_t j = 0; j < dW[0].size(); ++j)
            for (size_t i = 0; i < dW.size(); ++i)
                d_next[j] += d_pre[i] * model.hidden_layers[l].weights[i][j];
        d_cur = d_next;
    }

    // --- Backprop through flattening and maxpool to convolution kernels ---
    // Our "d_cur" is the gradient of loss w.r.t each feature in "features"
    // Map this back to each location in each pooled map:
    int N = model.kernels.size();
    int outRows = cache.pooled_maps[0][0].rows, outCols = cache.pooled_maps[0][0].cols;
    std::vector<cv::Mat> d_pooled(N);
    int flat_idx = 0;
    for (int k = 0; k < N; ++k) {
        d_pooled[k] = cv::Mat::zeros(outRows, outCols, CV_32FC1);
        for (int y = 0; y < outRows; ++y)
            for (int x = 0; x < outCols; ++x)
                d_pooled[k].at<float>(y, x) = d_cur[flat_idx++];
    }
    // Backprop maxpool: only pass gradient to the max location
    int h = cache.conv_maps[0][0].rows, w = cache.conv_maps[0][0].cols;
    int pad = 1;
    std::vector<cv::Mat> d_conv(N, cv::Mat::zeros(h, w, CV_32FC1));
    for (int k = 0; k < N; ++k) {
        for (int y = 0; y < outRows; ++y) {
            for (int x = 0; x < outCols; ++x) {
                int sy = y * 2, sx = x * 2;
                float maxval = -1e9f;
                int max_iy = sy, max_ix = sx;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        float v = cache.conv_maps[0][k].at<float>(sy + dy, sx + dx);
                        if (v > maxval) {
                            maxval = v;
                            max_iy = sy + dy;
                            max_ix = sx + dx;
                        }
                    }
                }
                d_conv[k].at<float>(max_iy, max_ix) += d_pooled[k].at<float>(y, x);
            }
        }
    }
    // Backprop through conv: compute dL/dKernel for each kernel param (for 3x3 only)
    for (int k = 0; k < N; ++k) {
        for (int y = pad; y < h - pad; ++y) {
            for (int x = pad; x < w - pad; ++x) {
                float grad_output = d_conv[k].at<float>(y, x);
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ki = (ky + 1) * 3 + (kx + 1);
                        cv::Vec3b pixel = input.at<cv::Vec3b>(y + ky, x + kx);
                        for (int c = 0; c < 3; ++c)
                            grads.dKernels[k].matrix9[c][ki] += grad_output * pixel[c];
                    }
                }
            }
        }
    }
}

// Update parameters (same as before, but now includes kernels)
void update_params(ObjectDetector& model, const Gradients& grads, float lr_fc, float lr_kern, int batch_size) {
    // Update FC weights
    for (size_t l = 0; l < model.hidden_layers.size(); ++l) {
        for (size_t i = 0; i < model.hidden_layers[l].weights.size(); ++i) {
            for (size_t j = 0; j < model.hidden_layers[l].weights[i].size(); ++j)
                model.hidden_layers[l].weights[i][j] -= lr_fc * grads.dHiddenLayers[l].weights[i][j] / batch_size;
            model.hidden_layers[l].biases[i] -= lr_fc * grads.dHiddenLayers[l].biases[i] / batch_size;
        }
    }
    for (size_t i = 0; i < model.fc_weights_confidence.size(); ++i)
        model.fc_weights_confidence[i] -= lr_fc * grads.dFCWeightsConf[i] / batch_size;
    model.fc_bias_confidence -= lr_fc * grads.dFCBiasConf / batch_size;
    for (size_t i = 0; i < model.fc_weights_bbox.size(); ++i)
        for (size_t j = 0; j < model.fc_weights_bbox[i].size(); ++j)
            model.fc_weights_bbox[i][j] -= lr_fc * grads.dFCWeightsBBox[i][j] / batch_size;
    for (size_t i = 0; i < model.fc_bias_bbox.size(); ++i)
        model.fc_bias_bbox[i] -= lr_fc * grads.dFCBiasBBox[i] / batch_size;

    // Update kernels (for 3x3)
    for (size_t k = 0; k < model.kernels.size(); ++k)
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 9; ++i)
                model.kernels[k].matrix9[c][i] -= lr_kern * grads.dKernels[k].matrix9[c][i] / batch_size;
}

int main() {
    std::string images_dir = "lib/samples/";
    std::string labels_file = "lib/labels.txt";
    constexpr int IN_H = 300, IN_W = 300;
    auto dataset = load_dataset(images_dir, labels_file, IN_H, IN_W);

    ObjectDetector model;
    model.initializeFullyConnected(256 * model.numKernels() / 16, 8);

    std::mt19937 rng{std::random_device{}()};
    std::shuffle(dataset.begin(), dataset.end(), rng);

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float total_conf_loss = 0, total_bbox_loss = 0;
        int n_samples = 0;
        Gradients grads;
        grads.clear(model);

        for (size_t i = 0; i < dataset.size(); ++i) {
            forward_backward(model, dataset[i].first, dataset[i].second, grads, total_conf_loss, total_bbox_loss);
            n_samples++;

            if (n_samples % BATCH_SIZE == 0 || i == dataset.size() - 1) {
                update_params(model, grads, LR_FC, LR_KERN, n_samples);
                grads.clear(model);
                n_samples = 0;
            }
        }
        std::cout << "[Epoch " << epoch << "] Mean confidence loss: " << total_conf_loss / dataset.size()
                  << ", Mean bbox loss: " << total_bbox_loss / dataset.size()
                  << ", Num samples: " << dataset.size() << std::endl;

        if (!dataset.empty()) {
            DetectionResult pred = model.predict(dataset[0].first);
            std::cout << "Sanity predict, sample 0: objectPresent=" << pred.objectPresent
                      << " confidence=" << pred.confidence << " vertices=";
            for (float v : pred.vertices) std::cout << v << " ";
            std::cout << std::endl;
        }
    }
    model.saveModel("object_detector_model.txt");
    std::cout << "Training complete, model saved to object_detector_model.txt" << std::endl;
    return 0;
}
