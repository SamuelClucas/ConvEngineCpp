#include "object_detector.h"
#include <fstream>
#include <algorithm>
#include <cmath>

// --------- UTILITY FUNCTIONS ---------

// Dot product (common C++ idiom: size must match)
float ObjectDetector::dot(const std::vector<float>& a, const std::vector<float>& b) const {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

// --------- CONSTRUCTOR ---------

ObjectDetector::ObjectDetector()
    : rng(std::random_device{}()), dist2(1.0f, 2.0f) // Seed RNG, set default dist
{
    // Initialize kernels with random values (could use smarter init if desired)
    for (auto& kernel : kernels) {
        for (int c = 0; c < 3; ++c){
            for (int i = 0; i < 9; ++i){
                kernel.matrix9[c][i] = dist2(rng);
            }
        }
        for (int i = 0; i < 4; ++i){
            kernel.matrix4_single[i] = dist2(rng);
        }
        for (int i = 0; i < 25; ++i){
            kernel.matrix25_single[i] = dist2(rng);
        }
    }
    // Weights will be lazily initialized in FC if/when needed.
}

void ObjectDetector::addAndMedian(const std::vector<cv::Mat>& inputs, cv::Mat& output) const {
    assert(!inputs.empty());
    int rows = inputs[0].rows, cols = inputs[0].cols;
    for (const auto& mat : inputs) {
        assert(mat.type() == CV_32FC1 && mat.rows == rows && mat.cols == cols);
    }
    output = cv::Mat::zeros(rows, cols, CV_32FC1);
    std::vector<float> buffer(inputs.size());
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (size_t i = 0; i < inputs.size(); ++i)
                buffer[i] = inputs[i].at<float>(y, x);
            std::nth_element(buffer.begin(), buffer.begin() + buffer.size()/2, buffer.end());
            float median = buffer[buffer.size()/2];
            output.at<float>(y, x) = median;
        }
    }
    // No residual here by default; call addResidual outside if needed!
}

// --------- INFERENCE API ---------

DetectionResult ObjectDetector::predict(const cv::Mat& input) const {
    std::vector<float> features;
    extractFeatures(input, features);

    DetectionResult result;
    forwardFC(features, result.objectPresent, result.vertices, result.confidence);
    return result;
}

void ObjectDetector::drawPredictedPolygon(cv::Mat& image, const std::vector<float>& norm_coords, const cv::Scalar& color) const {
    if (norm_coords.size() != 8) return;
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; ++i) {
        int x = static_cast<int>(norm_coords[2 * i] * image.cols);
        int y = static_cast<int>(norm_coords[2 * i + 1] * image.rows);
        pts.emplace_back(x, y);
    }
    for (int i = 0; i < 4; ++i)
        cv::line(image, pts[i], pts[(i + 1) % 4], color, 2);
    for (const auto& pt : pts)
        cv::circle(image, pt, 4, color, -1);

    // Optional: semi-transparent fill
    std::vector<std::vector<cv::Point>> contour = { pts };
    cv::Mat overlay = image.clone();
    cv::fillPoly(overlay, contour, color);
    cv::addWeighted(overlay, 0.3, image, 0.7, 0, image);
}

// --------- MODEL I/O ---------

void ObjectDetector::saveModel(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing.\n";
        return;
    }

    // Save hidden layers
    out << hidden_layers.size() << "\n";
    for (const auto& layer : hidden_layers) {
        out << layer.weights.size() << " " << (layer.weights.empty() ? 0 : layer.weights[0].size()) << "\n";
        for (const auto& neuron : layer.weights) {
            for (float w : neuron) out << w << " ";
            out << "\n";
        }
        for (float b : layer.biases) out << b << " ";
        out << "\n";
    }

    // Confidence head
    out << fc_weights_confidence.size() << "\n";
    for (float w : fc_weights_confidence) out << w << " ";
    out << "\n";
    out << fc_bias_confidence << "\n";

    // Bbox head
    out << fc_weights_bbox.size() << " " << (fc_weights_bbox.empty() ? 0 : fc_weights_bbox[0].size()) << "\n";
    for (const auto& neuron : fc_weights_bbox) {
        for (float w : neuron) out << w << " ";
        out << "\n";
    }
    for (float b : fc_bias_bbox) out << b << " ";
    out << "\n";

    out.close();
}


void ObjectDetector::loadModel(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << path << " for reading.\n";
        return;
    }

    size_t n_layers;
    in >> n_layers;
    hidden_layers.resize(n_layers);
    for (auto& layer : hidden_layers) {
        size_t out_dim, in_dim;
        in >> out_dim >> in_dim;
        layer.weights.assign(out_dim, std::vector<float>(in_dim));
        layer.biases.assign(out_dim, 0.0f);
        for (size_t i = 0; i < out_dim; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                in >> layer.weights[i][j];
        for (size_t i = 0; i < out_dim; ++i)
            in >> layer.biases[i];
    }

    // Confidence head
    size_t conf_dim;
    in >> conf_dim;
    fc_weights_confidence.assign(conf_dim, 0.0f);
    for (size_t i = 0; i < conf_dim; ++i)
        in >> fc_weights_confidence[i];
    in >> fc_bias_confidence;

    // Bbox head
    size_t bbox_out, bbox_in;
    in >> bbox_out >> bbox_in;
    fc_weights_bbox.assign(bbox_out, std::vector<float>(bbox_in));
    for (size_t i = 0; i < bbox_out; ++i)
        for (size_t j = 0; j < bbox_in; ++j)
            in >> fc_weights_bbox[i][j];
    fc_bias_bbox.assign(bbox_out, 0.0f);
    for (size_t i = 0; i < bbox_out; ++i)
        in >> fc_bias_bbox[i];

    in.close();
}


// --------- NETWORK PIPELINE BUILDING BLOCKS ---------

void ObjectDetector::downsizeInput(const cv::Mat& input, cv::Mat& output) const {
    // Pro tip: Center-crop and resize in one go
    float scale = std::min(
        static_cast<float>(ObjectDetector::input_width) / input.cols,
        static_cast<float>(ObjectDetector::input_height) / input.rows
    );
    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);
    int x_offset = (ObjectDetector::input_width  - new_width) / 2;
    int y_offset = (ObjectDetector::input_height - new_height) / 2;
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height));
    output.setTo(cv::Scalar(0)); // fill background with black
    resized.copyTo(output(cv::Rect(x_offset, y_offset, new_width, new_height)));
}

// 3x3 conv + 2x2 maxpool (RGB input, N float outputs)
// outputs will be resized/filled inside the function
void ObjectDetector::conv9maxPool(const cv::Mat& input, std::vector<cv::Mat>& outputs) const {
    assert(input.type() == CV_8UC3);
    int h = input.rows, w = input.cols, kSize = 3, pad = 1;
    const int N = num_kernels;
    outputs.clear();
    outputs.resize(N, cv::Mat::zeros(h, w, CV_32FC1)); // Raw conv outputs for each kernel

    // Convolution
    for (int k = 0; k < N; ++k) { // For each kernel
        for (int y = pad; y < h - pad; ++y) {
            for (int x = pad; x < w - pad; ++x) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ki = (ky + 1) * 3 + (kx + 1);
                        cv::Vec3b pixel = input.at<cv::Vec3b>(y + ky, x + kx);
                        for (int c = 0; c < 3; ++c) {
                            sum += pixel[c] * kernels[k].matrix9[c][ki];
                        }
                    }
                }
                outputs[k].at<float>(y, x) = sum;
            }
        }
    }

    // Maxpooling (2x2) for each feature map
    int outRows = h / 2, outCols = w / 2;
    for (int k = 0; k < N; ++k) {
        cv::Mat pooled = cv::Mat::zeros(outRows, outCols, CV_32FC1);
        for (int y = 0; y < outRows; ++y) {
            for (int x = 0; x < outCols; ++x) {
                int sy = y * 2, sx = x * 2;
                float m = -1e9f;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        m = std::max(m, outputs[k].at<float>(sy + dy, sx + dx));
                pooled.at<float>(y, x) = m;
            }
        }
        outputs[k] = pooled; // Replace raw conv output with pooled output
    }

    // Optional: Add residual connection if you wish (would need to handle N outputs)
    // For now, let's skip this unless you want to add skip connections for each channel

    // If you want to apply skip connection, you'll need to upsample input to pooled size
    // and add to each output map.
}


// -- 5x5 conv (single channel in/out)
void ObjectDetector::conv25(const std::vector<cv::Mat>& inputs, std::vector<cv::Mat>& outputs) const {
    assert(!inputs.empty());
    int N_in = inputs.size();
    int N_kernels = kernels.size();
    int h = inputs[0].rows, w = inputs[0].cols, kSize = 5, pad = 2;
    for (const auto& m : inputs)
        assert(m.type() == CV_32FC1 && m.rows == h && m.cols == w);

    outputs.clear();
    outputs.resize(N_in * N_kernels); // output[i * N_kernels + k]

    for (int i = 0; i < N_in; ++i) {
        for (int k = 0; k < N_kernels; ++k) {
            cv::Mat out = cv::Mat::zeros(h, w, CV_32FC1);
            for (int y = pad; y < h - pad; ++y) {
                for (int x = pad; x < w - pad; ++x) {
                    float sum = 0.0f;
                    for (int ky = -2; ky <= 2; ++ky) {
                        for (int kx = -2; kx <= 2; ++kx) {
                            int ki = (ky + 2) * 5 + (kx + 2);
                            float v = inputs[i].at<float>(y + ky, x + kx);
                            sum += v * kernels[k].matrix25_single[ki];
                        }
                    }
                    out.at<float>(y, x) = sum;
                }
            }
            // Optionally, add residual here per your design
            outputs[i * N_kernels + k] = out;
        }
    }
}


// -- 2x2 conv (single channel in/out)
void ObjectDetector::conv4(const std::vector<cv::Mat>& inputs, std::vector<cv::Mat>& outputs) const {
    assert(!inputs.empty());
    int N_in = inputs.size();
    int N_kernels = kernels.size();
    int h = inputs[0].rows, w = inputs[0].cols;
    for (const auto& m : inputs)
        assert(m.type() == CV_32FC1 && m.rows == h && m.cols == w);

    int out_h = h - 1, out_w = w - 1; // 2x2 kernel, valid padding
    outputs.clear();
    outputs.resize(N_in * N_kernels); // output[i * N_kernels + k]

    for (int i = 0; i < N_in; ++i) {
        for (int k = 0; k < N_kernels; ++k) {
            cv::Mat out = cv::Mat::zeros(out_h, out_w, CV_32FC1);
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < 2; ++ky) {
                        for (int kx = 0; kx < 2; ++kx) {
                            int ki = ky * 2 + kx;
                            float v = inputs[i].at<float>(y + ky, x + kx);
                            sum += v * kernels[k].matrix4_single[ki];
                        }
                    }
                    out.at<float>(y, x) = sum;
                }
            }
            // Optionally: add residual if you want (use addResidual per your style)
            outputs[i * N_kernels + k] = out;
        }
    }
}


void ObjectDetector::upscale(const cv::Mat& input, cv::Mat& output, int target_rows, int target_cols) const {
    assert(input.type() == CV_32FC1);
    cv::resize(input, output, cv::Size(target_cols, target_rows), 0, 0, cv::INTER_LINEAR);
}

void ObjectDetector::flattenFeatureMap(const cv::Mat& input, std::vector<float>& flattened) const {
    assert(input.type() == CV_32FC1);
    flattened.clear();
    flattened.reserve(input.rows * input.cols);
    for (int y = 0; y < input.rows; ++y) {
        const float* rowPtr = input.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x)
            flattened.push_back(rowPtr[x]);
    }
}

// --------- FC FORWARD PASS ---------

void ObjectDetector::forwardFC(const std::vector<float>& input, bool& object_present,
    std::vector<float>& bbox_coords, float& confidence) const
{
    std::vector<float> cur = input, cur_raw;
    // Forward through hidden layers
    for (const auto& layer : hidden_layers) {
        cur_raw.resize(layer.biases.size());
        std::vector<float> next(layer.biases.size());
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            float sum = layer.biases[i];
            for (size_t j = 0; j < cur.size(); ++j)
                sum += layer.weights[i][j] * cur[j];
                cur_raw[i] = sum;
                next[i] = std::max(0.0f, sum); // ReLU
        }
    cur = std::move(next);
    }
    // Heads
    float conf_raw = fc_bias_confidence;
    for (size_t i = 0; i < fc_weights_confidence.size(); ++i)
        conf_raw += fc_weights_confidence[i] * cur[i];
        confidence = 1.0f / (1.0f + std::exp(-conf_raw));
        object_present = (confidence >= 0.75f);

        bbox_coords.resize(8);
    for (int i = 0; i < 8; ++i) {
        float sum = fc_bias_bbox[i];
        for (size_t j = 0; j < fc_weights_bbox[i].size(); ++j)
            sum += fc_weights_bbox[i][j] * cur[j];
            bbox_coords[i] = 1.0f / (1.0f + std::exp(-sum));
        }
}

// --------- FC WEIGHT INIT ---------

void ObjectDetector::initializeFullyConnected(int fc_input_dim, int n_hidden) {
    // Layer sizes: input -> 256 -> 192 -> ... -> 32 (or whatever)
    std::vector<int> layer_sizes = {fc_input_dim, 256, 192, 160, 128, 96, 64, 48, 32}; // 8 hidden
    if (n_hidden != 8) {
        // Custom logic if you want more/less
        layer_sizes.resize(n_hidden + 1);
        for (int i = 1; i <= n_hidden; ++i) layer_sizes[i] = std::max(32, layer_sizes[i-1] / 2);
    }
    hidden_layers.clear();
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int l = 0; l < n_hidden; ++l) {
        HiddenLayer h;
        h.weights.resize(layer_sizes[l+1], std::vector<float>(layer_sizes[l]));
        h.biases.resize(layer_sizes[l+1], 0.0f);
        for (auto& row : h.weights)
            for (float& w : row)
                w = dist(rng);
        hidden_layers.push_back(std::move(h));
    }
    int last = layer_sizes.back();
    fc_weights_confidence.resize(last);
    for (float& w : fc_weights_confidence) w = dist(rng);
    fc_bias_confidence = 0.0f;

    fc_weights_bbox.resize(8, std::vector<float>(last));
    for (auto& row : fc_weights_bbox)
        for (float& w : row)
            w = dist(rng);
    fc_bias_bbox.resize(8, 0.0f);
}

void ObjectDetector::addResidual(const cv::Mat& input, cv::Mat& output, float alpha) const {
    cv::Mat input_ready;

    // Match channels
    if (input.channels() == 3 && output.channels() == 1)
        cv::cvtColor(input, input_ready, cv::COLOR_BGR2GRAY);
    else
        input_ready = input.clone();

    // Match type
    if (input_ready.type() != output.type())
        input_ready.convertTo(input_ready, output.type());

    // Match size
    if (input_ready.size() != output.size())
        cv::resize(input_ready, input_ready, output.size());

    // Finally, add
    output += alpha * input_ready;
}


void ObjectDetector::extractFeatures(const cv::Mat& input, std::vector<float>& features) const {
    // Defensive: Ensure RGB input and correct size
    cv::Mat downsized(ObjectDetector::input_height, ObjectDetector::input_width, input.type(), cv::Scalar(0));
    downsizeInput(input, downsized);

    // ---- Pipeline ----

    // 1. conv9maxPool: RGB input, N_kernels output maps
    std::vector<cv::Mat> feat1;
    conv9maxPool(downsized, feat1);  // feat1.size() == num_kernels

    // 2. conv25: For each feat1 map, get N_kernels more maps (depthwise)
    std::vector<cv::Mat> feat2;
    for (const auto& map : feat1) {
        std::vector<cv::Mat> out_maps;
        conv25(map, out_maps);       // out_maps.size() == num_kernels
        feat2.insert(feat2.end(), out_maps.begin(), out_maps.end());
    }
    // Now feat2.size() == num_kernels * num_kernels

    // 3. conv4: For each feat2 map, get N_kernels more maps (depthwise)
    std::vector<cv::Mat> feat3;
    for (const auto& map : feat2) {
        std::vector<cv::Mat> out_maps;
        conv4({map}, out_maps);      // out_maps.size() == num_kernels
        feat3.insert(feat3.end(), out_maps.begin(), out_maps.end());
    }
    // Now feat3.size() == num_kernels^3

    // 4. Fuse all feat3 maps with median or mean (to keep feature vector compact)
    cv::Mat fused;
    addAndMedian(feat3, fused); // Could also sum, average, etc.

    // 5. Downsample to 16x16
    cv::Mat shrunken;
    cv::resize(fused, shrunken, cv::Size(16, 16));
    flattenFeatureMap(shrunken, features);
    std::cout << "Shrink to 16x16, features.size(): " << features.size() << std::endl;

    // (Optional: flatten all feat3 maps instead of fusing if you want a *huge* feature vector!)
}

// Matrix-vector multiply: output = mat * vec + bias
void ObjectDetector::matvec(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec, 
    const std::vector<float>& bias, std::vector<float>& out) 
    const{
    assert(!mat.empty() && mat[0].size() == vec.size());
    out.resize(mat.size());
    for (size_t i = 0; i < mat.size(); ++i) {
        float sum = bias.empty() ? 0.0f : bias[i];
        for (size_t j = 0; j < vec.size(); ++j)
            sum += mat[i][j] * vec[j];
            out[i] = sum;
    }
}
