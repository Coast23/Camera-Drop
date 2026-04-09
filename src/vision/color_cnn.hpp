#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace camdrop::vision {

struct ColorCnnOptions {
    int ort_threads = 1;
    size_t batch_size = 1024;
    std::string input_name = "input";
    std::string output_name = "logits";
};

class ColorCnnClassifier {
public:
    explicit ColorCnnClassifier(const std::string& model_path, ColorCnnOptions options = {});

    const ColorCnnOptions& options() const { return options_; }
    std::vector<uint8_t> PredictPayloadColors(const cv::Mat& deskewed);

private:
    ColorCnnOptions options_;
    Ort::Env env_;
    Ort::Session session_;
    std::vector<int16_t> payload_x_;
    std::vector<int16_t> payload_y_;
};

}  // namespace camdrop::vision
