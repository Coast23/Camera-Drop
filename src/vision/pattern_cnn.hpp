#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace camdrop::vision {

struct PatternCnnOptions {
    int ort_threads = 1;
    size_t batch_size = 8192;
    std::string input_name = "input";
    std::string output_name = "logits";
};

class PatternCnnClassifier {
public:
    explicit PatternCnnClassifier(const std::string& model_path, PatternCnnOptions options = {});

    const PatternCnnOptions& options() const { return options_; }
    std::vector<uint8_t> PredictPayloadPatterns(const cv::Mat& deskewed);

private:
    PatternCnnOptions options_;
    Ort::Env env_;
    Ort::Session session_;
    std::vector<int16_t> payload_x_;
    std::vector<int16_t> payload_y_;
};

}  // namespace camdrop::vision
